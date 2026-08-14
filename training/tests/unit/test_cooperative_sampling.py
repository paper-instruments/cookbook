from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fireworks.training.sdk.deployment import DeploymentSampler

from training.utils.rl.cooperative_sampling import (
    _CooperativeByteStream,
    _CooperativeYieldBudget,
    CooperativeSamplingAsyncClient,
    install_cooperative_sampling_transport,
)


class _OneChunkStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.closed = False

    async def __aiter__(self):
        yield self.content

    async def aclose(self) -> None:
        self.closed = True


def test_client_shares_byte_budget_across_response_streams() -> None:
    sources = iter([_OneChunkStream(b"aaaa"), _OneChunkStream(b"bbbb")])

    async def run() -> tuple[list[bytes], list[_OneChunkStream], int]:
        client = CooperativeSamplingAsyncClient(
            cooperative_byte_budget=8,
            cooperative_time_budget_seconds=60,
        )
        used_sources: list[_OneChunkStream] = []

        async def send(request, *, stream, auth, follow_redirects):
            source = next(sources)
            used_sources.append(source)
            return httpx.Response(200, request=request, stream=source)

        client.send = send
        sleep = AsyncMock()
        with patch(
            "training.utils.rl.cooperative_sampling.asyncio.sleep",
            sleep,
        ):
            contents = []
            for _ in range(2):
                response = await client.post("https://example.test")
                contents.append(await response.aread())
        await client.aclose()
        return contents, used_sources, sleep.await_count

    contents, used_sources, yield_count = asyncio.run(run())

    assert contents == [b"aaaa", b"bbbb"]
    assert yield_count == 1
    assert all(source.closed for source in used_sources)


def test_yield_budget_expires_before_byte_limit() -> None:
    budget = _CooperativeYieldBudget(
        byte_budget=64 * 1024,
        time_budget_seconds=0.001,
    )

    assert not budget.consume(1, now=0.0)
    assert not budget.consume(1, now=0.0009)
    assert budget.consume(1, now=0.0011)
    assert not budget.consume(1, now=0.0012)


def test_cooperative_stream_closes_when_consumer_stops_early() -> None:
    source = _OneChunkStream(b"first-second")
    stream = _CooperativeByteStream(
        source,
        budget=_CooperativeYieldBudget(
            byte_budget=6,
            time_budget_seconds=60,
        ),
        chunk_bytes=6,
    )

    async def run() -> None:
        async for chunk in stream:
            assert chunk == b"first-"
            break
        # Closing an async generator after break is scheduled by the runtime.
        await asyncio.sleep(0)

    asyncio.run(run())

    assert source.closed


def test_client_sends_success_response_as_live_stream() -> None:
    source = _OneChunkStream(b"data: [DONE]\n\n")
    seen: dict[str, object] = {}

    async def run() -> tuple[httpx.Response, bytes]:
        client = CooperativeSamplingAsyncClient()

        async def send(request, *, stream, auth, follow_redirects):
            seen.update(
                request=request,
                stream=stream,
                auth=auth,
                follow_redirects=follow_redirects,
            )
            return httpx.Response(200, request=request, stream=source)

        client.send = send
        response = await client.post(
            "https://example.test/inference/v1/completions",
            json={"prompt": [1, 2, 3]},
        )
        content = b"".join([chunk async for chunk in response.aiter_bytes()])
        await client.aclose()
        return response, content

    response, content = asyncio.run(run())
    request = seen["request"]

    assert seen["stream"] is True
    assert isinstance(request, httpx.Request)
    assert json.loads(request.content) == {"prompt": [1, 2, 3]}
    assert request.headers["content-type"] == "application/json"
    assert isinstance(response.stream, _CooperativeByteStream)
    assert content == b"data: [DONE]\n\n"


def test_client_consumes_error_response_before_returning() -> None:
    source = _OneChunkStream(b'{"error":"overloaded"}')

    async def run() -> httpx.Response:
        client = CooperativeSamplingAsyncClient()

        async def send(request, *, stream, auth, follow_redirects):
            return httpx.Response(503, request=request, stream=source)

        client.send = send
        response = await client.post("https://example.test", json={"prompt": []})
        await client.aclose()
        return response

    response = asyncio.run(run())

    assert response.content == b'{"error":"overloaded"}'
    assert source.closed


def test_deployment_sampler_consumes_cooperative_live_response() -> None:
    payload = (
        b'data: {"choices":[{"text":"hi","finish_reason":"stop",'
        b'"raw_output":{"completion_token_ids":[40,50]}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    source = _OneChunkStream(payload)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, stream=source)

    async def run() -> dict:
        sampler = DeploymentSampler(
            inference_url="https://example.test",
            model="model",
            api_key="secret",
        )
        client = CooperativeSamplingAsyncClient(
            transport=httpx.MockTransport(handler),
            cooperative_byte_budget=16,
            cooperative_time_budget_seconds=60,
        )
        sampler._async_client = client
        try:
            result, _ = await sampler.async_completions_stream(
                prompt=[1, 2, 3],
                raw_output=True,
            )
            return result
        finally:
            await client.aclose()
            sampler.close()

    result = asyncio.run(run())

    assert result["choices"][0]["raw_output"]["completion_token_ids"] == [40, 50]
    assert source.closed


def test_install_rejects_an_already_live_client() -> None:
    class Sampler:
        _base_verify = False
        _async_client = httpx.AsyncClient()

    sampler = Sampler()
    try:
        with pytest.raises(RuntimeError, match="before sampling"):
            install_cooperative_sampling_transport(sampler)
    finally:
        asyncio.run(sampler._async_client.aclose())
