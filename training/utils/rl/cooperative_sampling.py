"""Event-loop-friendly HTTP transport for high-concurrency sampling.

``DeploymentSampler.async_completions_stream`` consumes an SSE response, but
fireworks-ai 1.2.9 obtains that response through ``AsyncClient.post``. HTTPX's
high-level verb helpers buffer the complete response unless ``stream=True`` is
passed through ``send``. Large token/logprob/routing responses are therefore
delivered to the sampler as one buffered parsing burst on the rollout event
loop.

This cookbook-local compatibility boundary keeps the SDK's retry, metrics, and
response-assembly behavior while changing the sampler's private async client
to serialize large token prompts in a worker thread, return successful
responses as live streams, and cooperatively yield between bounded response
chunks. Remove it once the pinned SDK provides equivalent behavior itself.
"""

from __future__ import annotations

import json as json_lib
import asyncio
from typing import Any

import httpx

_COOPERATIVE_CHUNK_BYTES = 64 * 1024


class _CooperativeByteStream(httpx.AsyncByteStream):
    """Bound CPU work per event-loop turn when the network coalesces data."""

    def __init__(
        self,
        stream: httpx.AsyncByteStream,
        *,
        chunk_bytes: int = _COOPERATIVE_CHUNK_BYTES,
    ) -> None:
        if chunk_bytes < 1:
            raise ValueError("chunk_bytes must be positive")
        self._stream = stream
        self._chunk_bytes = chunk_bytes
        self._closed = False

    async def __aiter__(self):
        try:
            async for chunk in self._stream:
                view = memoryview(chunk)
                for start in range(0, len(view), self._chunk_bytes):
                    # An async iterator is not automatically a scheduling point
                    # while its current transport chunk is already buffered.
                    await asyncio.sleep(0)
                    yield bytes(view[start : start + self._chunk_bytes])
        finally:
            # DeploymentSampler stops at the SSE [DONE] sentinel. Since that
            # can precede network EOF, release the live response stream when
            # its iterator is closed early instead of leaking the connection.
            await self.aclose()

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            await self._stream.aclose()


class CooperativeSamplingAsyncClient(httpx.AsyncClient):
    """HTTPX client whose POST helper preserves streaming and loop fairness."""

    def __init__(
        self,
        *args: Any,
        cooperative_chunk_bytes: int = _COOPERATIVE_CHUNK_BYTES,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if cooperative_chunk_bytes < 1:
            raise ValueError("cooperative_chunk_bytes must be positive")
        self._cooperative_chunk_bytes = cooperative_chunk_bytes

    async def post(self, url: Any, **kwargs: Any) -> httpx.Response:
        """Build a POST request off-loop and send successful responses live."""

        json_body = kwargs.pop("json", None)
        if json_body is not None:
            if kwargs.get("content") is not None:
                raise TypeError("cannot provide both content and json")
            kwargs["content"] = await asyncio.to_thread(_encode_json, json_body)
            headers = httpx.Headers(kwargs.get("headers"))
            headers.setdefault("content-type", "application/json")
            kwargs["headers"] = headers

        auth = kwargs.pop("auth", httpx.USE_CLIENT_DEFAULT)
        follow_redirects = kwargs.pop(
            "follow_redirects",
            httpx.USE_CLIENT_DEFAULT,
        )
        request = self.build_request("POST", url, **kwargs)
        response = await self.send(
            request,
            auth=auth,
            follow_redirects=follow_redirects,
            stream=True,
        )

        # The SDK may inspect and retry an error status without consuming its
        # body. Read it here so its connection is released before retrying.
        if response.is_error:
            await response.aread()
            return response

        response.stream = _CooperativeByteStream(
            response.stream,
            chunk_bytes=self._cooperative_chunk_bytes,
        )
        return response


def install_cooperative_sampling_transport(sampler: Any) -> None:
    """Install the compatibility client before a DeploymentSampler is used."""

    current = getattr(sampler, "_async_client", None)
    if current is not None and not current.is_closed:
        raise RuntimeError(
            "cooperative sampling transport must be installed before sampling"
        )

    verify = bool(getattr(sampler, "_base_verify", True))
    sampler._async_client = CooperativeSamplingAsyncClient(
        verify=verify,
        timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0),
        limits=httpx.Limits(
            max_connections=256,
            max_keepalive_connections=64,
        ),
    )


def _encode_json(value: Any) -> bytes:
    """Match HTTPX's compact JSON encoding outside the rollout loop."""

    return json_lib.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
