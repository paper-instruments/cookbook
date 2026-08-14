"""Event-loop-friendly HTTP transport for high-concurrency sampling.

``DeploymentSampler.async_completions_stream`` consumes an SSE response, but
fireworks-ai 1.2.9 obtains that response through ``AsyncClient.post``. HTTPX's
high-level verb helpers buffer the complete response unless ``stream=True`` is
passed through ``send``. Large token/logprob/routing responses are therefore
delivered to the sampler as one buffered parsing burst on the rollout event
loop.

This cookbook-local compatibility boundary keeps the SDK's retry, metrics, and
response-assembly behavior while changing the sampler's private async client
to return successful responses as live streams and cooperatively yield after a
bounded amount of response work. Remove it once the pinned SDK provides
equivalent behavior itself.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

_COOPERATIVE_BYTE_BUDGET = 64 * 1024
_COOPERATIVE_TIME_BUDGET_SECONDS = 0.001


class _CooperativeYieldBudget:
    """Share response-processing time fairly across active streams."""

    def __init__(self, *, byte_budget: int, time_budget_seconds: float) -> None:
        if byte_budget < 1:
            raise ValueError("byte_budget must be positive")
        if time_budget_seconds <= 0:
            raise ValueError("time_budget_seconds must be positive")
        self._byte_budget = byte_budget
        self._time_budget_seconds = time_budget_seconds
        self._bytes_since_yield = 0
        self._last_yield_at: float | None = None

    def consume(self, byte_count: int, *, now: float) -> bool:
        if self._last_yield_at is None:
            self._last_yield_at = now
        self._bytes_since_yield += byte_count
        if (
            self._bytes_since_yield < self._byte_budget
            and now - self._last_yield_at < self._time_budget_seconds
        ):
            return False

        # Reset before yielding so another active stream observes the new budget.
        self._bytes_since_yield = 0
        self._last_yield_at = now
        return True


class _CooperativeByteStream(httpx.AsyncByteStream):
    """Bound CPU work per event-loop turn when the network coalesces data."""

    def __init__(
        self,
        stream: httpx.AsyncByteStream,
        *,
        budget: _CooperativeYieldBudget,
        chunk_bytes: int,
    ) -> None:
        if chunk_bytes < 1:
            raise ValueError("chunk_bytes must be positive")
        self._stream = stream
        self._budget = budget
        self._chunk_bytes = chunk_bytes
        self._closed = False

    async def __aiter__(self):
        loop = asyncio.get_running_loop()
        try:
            async for chunk in self._stream:
                view = memoryview(chunk)
                for start in range(0, len(view), self._chunk_bytes):
                    piece = bytes(view[start : start + self._chunk_bytes])
                    if self._budget.consume(len(piece), now=loop.time()):
                        await asyncio.sleep(0)
                    yield piece
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
        cooperative_byte_budget: int = _COOPERATIVE_BYTE_BUDGET,
        cooperative_time_budget_seconds: float = _COOPERATIVE_TIME_BUDGET_SECONDS,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._cooperative_byte_budget = cooperative_byte_budget
        self._cooperative_yield_budget = _CooperativeYieldBudget(
            byte_budget=cooperative_byte_budget,
            time_budget_seconds=cooperative_time_budget_seconds,
        )

    async def post(self, url: Any, **kwargs: Any) -> httpx.Response:
        """Send successful responses as live streams."""

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
            budget=self._cooperative_yield_budget,
            chunk_bytes=self._cooperative_byte_budget,
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
