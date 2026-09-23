from __future__ import annotations

import time
import logging
from contextlib import nullcontext
from importlib import import_module
from types import SimpleNamespace

import pytest

from training.utils.timer import Timer, flush_timing, training_phase, wall_timer


def test_wall_timer_measures_without_recording_step_metric(monkeypatch) -> None:
    timer = Timer()
    timer.reset()
    timestamps = iter((10.0, 12.5))
    monkeypatch.setattr(time, "perf_counter", lambda: next(timestamps))

    with wall_timer() as span:
        assert span.elapsed == 0.0

    assert span.elapsed == 2.5
    assert timer.log_dict() == {}


@pytest.mark.parametrize("fail", [False, True])
def test_training_phase_accounts_for_wall_and_cpu_even_on_failure(
    monkeypatch, caplog, fail
) -> None:
    Timer().reset()
    wall = iter((10.0, 17.0))
    thread = iter((2.0, 3.0))
    process = iter((4.0, 7.0))
    monkeypatch.setattr(
        import_module("training.utils.timer"),
        "time",
        SimpleNamespace(
            monotonic=lambda: next(wall),
            thread_time=lambda: next(thread),
            process_time=lambda: next(process),
        ),
    )
    caplog.set_level(logging.INFO)
    error = RuntimeError("backend failure")
    with pytest.raises(RuntimeError) if fail else nullcontext() as caught:
        with training_phase("fwd_bwd", batch=2, chunk=3):
            if fail:
                raise error
    if fail:
        assert caught.value is error
    assert flush_timing() == {
        "perf/fwd_bwd_time": 7.0,
        "perf/fwd_bwd_thread_cpu_time": 1.0,
        "perf/fwd_bwd_process_cpu_time": 3.0,
    }
    assert flush_timing() == {}
    assert "phase=fwd_bwd" in caplog.messages[0]
    assert "batch=2 chunk=3" in caplog.messages[-1]
    assert f"status={'error' if fail else 'ok'}" in caplog.messages[-1]
