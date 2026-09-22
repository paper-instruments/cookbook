"""CPU tests for async_rl_loop with remote services mocked."""

from __future__ import annotations

import asyncio
import inspect
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import tinker

from training.recipes import async_rl_loop
from training.utils.checkpoints import ResumeInfo
from training.utils.rl.rollout import RolloutRun, RolloutSample


class _StopAfterProvisioning(RuntimeError):
    pass


class _StopAtRolloutFactory(RuntimeError):
    pass


def test_evaluation_rollout_context_is_explicit_and_compatible() -> None:
    seen: list[tuple[int, bool]] = []

    async def rollout(_row, *, sample_index: int, evaluation: bool = False):
        seen.append((sample_index, evaluation))
        return None

    evaluation_rollout = async_rl_loop.make_evaluation_rollout_fn(rollout)
    asyncio.run(evaluation_rollout({}, sample_index=2, cursor_index=7))

    assert seen == [(2, True)]


def test_evaluation_rollout_omits_unsupported_context() -> None:
    seen: list[int] = []

    async def rollout(_row, *, sample_index: int):
        seen.append(sample_index)
        return None

    evaluation_rollout = async_rl_loop.make_evaluation_rollout_fn(rollout)
    asyncio.run(evaluation_rollout({}, sample_index=3, cursor_index=7))

    assert seen == [3]


class TestConfigDefaults:
    def test_config_has_no_runner_state(self) -> None:
        cfg = async_rl_loop.Config(log_path="gs://logs")

        assert not hasattr(cfg, "runner")
        assert not hasattr(async_rl_loop, "RunnerIO")
        assert "write_running_progress" not in inspect.getsource(async_rl_loop.main)

    def test_config_has_no_conditional_initial_sync(self) -> None:
        cfg = async_rl_loop.Config(log_path="gs://logs")

        assert not hasattr(cfg, "weight_sync_before_training")

    def test_config_cleanup_defaults_on(self) -> None:
        cfg = async_rl_loop.Config(log_path="gs://logs")

        assert cfg.cleanup_on_exit is True

    def test_config_recovery_defaults_preserve_existing_behavior(self) -> None:
        cfg = async_rl_loop.Config(log_path="gs://logs")

        assert cfg.warm_start_from_adapter is None
        assert cfg.dcp_save_interval == 0
        assert cfg.weight_sync_timeout == 600

    def test_config_pipeline_chunks_default_to_one(self) -> None:
        cfg = async_rl_loop.Config(log_path="gs://logs")

        assert cfg.pipeline_chunks_per_step == 1

    def test_config_defaults_to_rollout_anchored_cispo(self) -> None:
        cfg = async_rl_loop.Config(log_path="gs://logs")

        assert cfg.kl_beta == 0
        assert cfg.cispo_clip_low_threshold == 0
        assert cfg.cispo_clip_high_threshold == 5
        assert cfg.anchor_logp == "rollout"
        assert cfg.router_replay is True
        assert cfg.router_replay_completion_only is True
        assert not hasattr(cfg, "policy_loss")
        assert not hasattr(cfg, "loss_path")


@pytest.mark.parametrize("field", ["eps_clip", "eps_clip_high", "tis"])
def test_config_rejects_obsolete_grpo_knobs(field) -> None:
    with pytest.raises(TypeError, match=field):
        async_rl_loop.Config(log_path="gs://logs", **{field: 0.2})


@pytest.mark.parametrize(
    "trainer",
    [
        async_rl_loop.TrainerConfig(reference_training_shape_id="ref-shape"),
        async_rl_loop.TrainerConfig(reference_job_id="ref-job"),
    ],
)
def test_main_rejects_unused_reference_trainer_config(trainer) -> None:
    cfg = async_rl_loop.Config(log_path="gs://logs", kl_beta=0, trainer=trainer)

    with pytest.raises(ValueError, match="does not use a reference trainer"):
        async_rl_loop.main(
            cfg,
            rows=[],
            rollout_fn_factory=lambda _setup: lambda _sample: None,
        )


@pytest.mark.parametrize(
    "config_overrides, error",
    [
        ({"kl_beta": 0.001}, "requires kl_beta=0"),
        ({"kl_beta": float("nan")}, "requires kl_beta=0"),
        ({"anchor_logp": "old_policy"}, "anchor_logp='rollout'"),
        ({"anchor_logp": "unknown"}, "anchor_logp='rollout'"),
        ({"cispo_clip_low_threshold": -0.1}, "0 <= low < high"),
        ({"cispo_clip_low_threshold": 6}, "0 <= low < high"),
        ({"cispo_clip_high_threshold": 0}, "0 <= low < high"),
        ({"cispo_clip_low_threshold": float("nan")}, "must be finite"),
        ({"cispo_clip_high_threshold": float("inf")}, "must be finite"),
    ],
)
def test_main_rejects_invalid_cispo_config_before_provisioning(
    monkeypatch: pytest.MonkeyPatch, config_overrides, error
) -> None:
    cfg = async_rl_loop.Config(log_path="gs://logs", **config_overrides)
    build_service = MagicMock(side_effect=AssertionError("must fail before provisioning"))
    monkeypatch.setattr(async_rl_loop, "build_service_client", build_service)

    with pytest.raises(ValueError, match=error):
        async_rl_loop.main(
            cfg,
            rows=[],
            rollout_fn_factory=lambda _setup: lambda _sample: None,
        )
    build_service.assert_not_called()


@pytest.mark.parametrize(
    "config_overrides, error",
    [
        (
            {
                "lora_rank": 8,
                "warm_start_from_adapter": "accounts/a/models/adapter",
                "init_from_checkpoint": "step-5",
            },
            "mutually exclusive",
        ),
        (
            {"lora_rank": 0, "warm_start_from_adapter": "accounts/a/models/adapter"},
            "requires lora_rank > 0",
        ),
    ],
)
def test_main_validates_adapter_warm_start(config_overrides, error) -> None:
    cfg = async_rl_loop.Config(log_path="gs://logs", **config_overrides)

    with pytest.raises(ValueError, match=error):
        async_rl_loop.main(
            cfg,
            rows=[],
            rollout_fn_factory=lambda _setup: lambda _sample: None,
        )


# ---------------------------------------------------------------------------
# SDK service construction
# ---------------------------------------------------------------------------


def _build_service_kwargs(
    monkeypatch: pytest.MonkeyPatch, cfg: async_rl_loop.Config
) -> dict:
    calls = []

    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(async_rl_loop, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(async_rl_loop, "validate_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        async_rl_loop,
        "resolve_router_replay_enabled",
        lambda **kwargs: kwargs["requested"],
    )
    monkeypatch.setattr(
        async_rl_loop, "load_deployment_tokenizer", lambda *args, **kwargs: object()
    )

    def fake_build_service_client(**kwargs):
        calls.append(kwargs)
        raise _StopAfterProvisioning

    monkeypatch.setattr(
        async_rl_loop, "build_service_client", fake_build_service_client
    )

    with pytest.raises(_StopAfterProvisioning):
        async_rl_loop.main(
            cfg,
            rows=[{"prompt": "1+1"}],
            rollout_fn_factory=lambda _setup: lambda _sample: None,
        )

    assert len(calls) == 1
    return calls[0]


def test_main_requests_cleanup_for_sdk_created_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = async_rl_loop.Config(
        log_path="/tmp/async_rl_test_logs",
        deployment=async_rl_loop.DeployConfig(tokenizer_model="Qwen/Qwen3-1.7B"),
    )

    kwargs = _build_service_kwargs(monkeypatch, cfg)

    assert kwargs["cleanup_trainer_on_close"] is True
    assert (
        kwargs["cleanup_deployment_on_close"]
        == async_rl_loop.CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO
    )


def test_main_can_disable_cleanup_on_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = async_rl_loop.Config(
        log_path="/tmp/async_rl_test_logs",
        cleanup_on_exit=False,
        deployment=async_rl_loop.DeployConfig(tokenizer_model="Qwen/Qwen3-1.7B"),
    )

    kwargs = _build_service_kwargs(monkeypatch, cfg)

    assert kwargs["cleanup_trainer_on_close"] is False
    assert kwargs["cleanup_deployment_on_close"] is None


@pytest.mark.parametrize("factory_raises", [False, True])
def test_main_owns_one_managed_sampling_client(
    monkeypatch: pytest.MonkeyPatch,
    factory_raises: bool,
) -> None:
    events: list[str] = []
    tokenizer = object()
    deployment_sampler = SimpleNamespace(
        base_url="https://deployment.test/v1",
        model="accounts/test/deployments/model",
    )
    sampling_client = MagicMock(deployment_sampler=deployment_sampler)
    sampling_client.close.side_effect = lambda: events.append("sampling_client.close")
    service = MagicMock(
        trainer_job_id="trainer-test",
        reference_client_job_id=None,
        reference_trainer_job_id=None,
        deployment_id="deployment-test",
        max_context_length=131_072,
    )
    service.create_training_client.return_value = object()
    service.create_sampling_client.return_value = sampling_client
    service.create_deployment_sampler.side_effect = AssertionError(
        "the managed sampling client must own the sampler"
    )
    service.close.side_effect = lambda: events.append("service.close")
    policy = MagicMock()
    policy.save_weights_for_sampler.return_value = SimpleNamespace(path="snapshot-test")
    checkpoints = MagicMock()
    checkpoints.resume.return_value = None
    checkpoint_kwargs = {}
    seen_setups: list[async_rl_loop.RolloutSetup] = []

    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    replacements = {
        "setup_wandb": lambda *_args, **_kwargs: None,
        "wandb_finish": lambda **_kwargs: None,
        "log_metrics": lambda *_args, **_kwargs: None,
        "validate_config": lambda *_args, **_kwargs: None,
        "resolve_router_replay_enabled": lambda **_kwargs: False,
        "read_api_extra_headers_env": lambda: {},
        "load_deployment_tokenizer": lambda _deployment: tokenizer,
        "build_service_client": lambda **_kwargs: service,
        "TrainingCheckpoints": lambda *_args, **kwargs: (
            checkpoint_kwargs.update(kwargs) or checkpoints
        ),
    }
    for name, replacement in replacements.items():
        monkeypatch.setattr(async_rl_loop, name, replacement)
    monkeypatch.setattr(
        async_rl_loop.ReconnectableClient,
        "from_training_client",
        lambda *_args, **_kwargs: policy,
    )

    def rollout_factory(setup):
        seen_setups.append(setup)
        if factory_raises:
            raise _StopAtRolloutFactory

        async def rollout(_sample):
            return None

        return rollout

    cfg = async_rl_loop.Config(
        log_path="/tmp/async_rl_test_logs",
        kl_beta=0,
        completions_per_prompt=2,
        router_replay=False,
        save_final_checkpoint=False,
        deployment=async_rl_loop.DeployConfig(
            tokenizer_model="Qwen/Qwen3-1.7B",
        ),
    )

    on_dataloader_saved = MagicMock()
    if factory_raises:
        with pytest.raises(_StopAtRolloutFactory):
            async_rl_loop.main(
                cfg,
                rows=[],
                rollout_fn_factory=rollout_factory,
                on_dataloader_saved=on_dataloader_saved,
            )
    else:
        async_rl_loop.main(
            cfg,
            rows=[],
            rollout_fn_factory=rollout_factory,
            on_dataloader_saved=on_dataloader_saved,
        )

    assert len(seen_setups) == 1
    assert seen_setups[0].sampler is sampling_client
    assert seen_setups[0].inference_base_url == deployment_sampler.base_url
    assert seen_setups[0].model == deployment_sampler.model
    service.create_sampling_client.assert_called_once_with(tokenizer=tokenizer)
    service.create_deployment_sampler.assert_not_called()
    service.hotload_sampler_snapshot.assert_called_once_with("snapshot-test")
    assert checkpoint_kwargs["on_dataloader_saved"] is on_dataloader_saved
    checkpoints.resume.assert_called_once_with(
        init_from_checkpoint=None,
        warm_start_from_adapter=None,
        require_dataloader_state=True,
        resume_recipe_state=False,
    )
    assert events == ["sampling_client.close", "service.close"]


@pytest.mark.parametrize(
    ("custom_advantages", "fail_forward", "continuation"),
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, False, True),
    ],
    ids=["default", "mean_only", "backend_failure", "continuation"],
)
def test_main_accumulates_native_cispo_chunks_before_one_optimizer_and_hotload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    caplog,
    custom_advantages: bool,
    fail_forward: bool,
    continuation: bool,
) -> None:
    caplog.set_level("INFO")
    events: list[str] = []
    metrics: list[dict] = []
    datums_seen: list[tinker.Datum] = []
    sampled_positions: list[tuple[int, int]] = []
    step_offset = 4 if continuation else 0
    route = "AQIDBA=="
    sampling_client = MagicMock(
        deployment_sampler=SimpleNamespace(
            base_url="https://deployment.test/v1",
            model="accounts/test/deployments/model",
        )
    )
    service = MagicMock(
        trainer_job_id="trainer-test",
        reference_trainer_job_id=None,
        deployment_id="deployment-test",
        max_context_length=131_072,
    )
    service.create_sampling_client.return_value = sampling_client
    service.create_reference_client.side_effect = AssertionError("CISPO has no reference")
    service.hotload_sampler_snapshot.side_effect = lambda path: events.append(
        f"hotload:{path}"
    )
    policy = MagicMock()
    policy.forward.side_effect = AssertionError("CISPO must not pre-score the batch")
    policy.forward_backward_custom.side_effect = AssertionError("CISPO must be native")
    backend_error = RuntimeError("forward_backward failed")

    def forward_backward(datums, loss_fn, *, loss_fn_config):
        events.append("forward_backward")
        assert len(datums) == 2
        assert loss_fn == "cispo"
        assert loss_fn_config == {
            "clip_low_threshold": 0.0,
            "clip_high_threshold": 5.0,
            "ratio_log_cap": 20.0,
        }
        datums_seen.extend(datums)
        if fail_forward:
            raise backend_error
        return SimpleNamespace(
            metrics={"loss:sum": 1.0},
            loss_fn_outputs=[
                {
                    "logprobs": tinker.TensorData(
                        data=[-999.0, -0.75, -999.0, -0.75, -0.75],
                        dtype="float32",
                        shape=[5],
                    )
                }
                for _ in datums
            ],
        )

    def optim_step(_params, *, grad_accumulation_normalization):
        assert grad_accumulation_normalization == "num_loss_tokens"
        events.append("optim_step")
        return SimpleNamespace(metrics={})

    def save_weights(name, **_kwargs):
        events.append(f"save:{name}")
        return SimpleNamespace(path=name)

    policy.forward_backward.side_effect = forward_backward
    policy.optim_step.side_effect = optim_step
    policy.save_weights_for_sampler.side_effect = save_weights
    checkpoints = MagicMock()
    checkpoints.resume.return_value = (
        ResumeInfo(step=4, data_consumed=2, source_job_id="previous-job")
        if continuation
        else None
    )
    setup_wandb = MagicMock()
    build_service = MagicMock(return_value=service)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    replacements = {
        "setup_wandb": setup_wandb,
        "wandb_finish": lambda **_kwargs: None,
        "log_metrics": lambda values, **_kwargs: metrics.append(dict(values)),
        "validate_config": lambda *_args, **_kwargs: None,
        "resolve_router_replay_enabled": lambda **_kwargs: True,
        "read_api_extra_headers_env": lambda: {},
        "load_deployment_tokenizer": lambda _deployment: object(),
        "build_service_client": build_service,
        "TrainingCheckpoints": lambda *_args, **_kwargs: checkpoints,
    }
    for name, replacement in replacements.items():
        monkeypatch.setattr(async_rl_loop, name, replacement)
    monkeypatch.setattr(
        async_rl_loop.ReconnectableClient,
        "from_training_client",
        lambda *_args, **_kwargs: policy,
    )

    def rollout_factory(setup):
        assert setup.sample_kwargs["include_routing_matrix"] is True
        assert setup.sample_kwargs["echo"] is False

        async def rollout(row, *, sample_index, cursor_index, epoch):
            sampled_positions.append((cursor_index, epoch))
            return RolloutRun(
                segments=[
                    RolloutSample(
                        tokens=[10, 11, 20 + row["id"], 30 + sample_index, 40, 50],
                        logprobs=[0.0, 0.0, -0.2, 0.0, -0.4, -0.6],
                        loss_mask=[0, 0, 1, 0, 1, 1],
                        reward=float(sample_index),
                        raw_logprobs=[0.0, 0.0, -1.0, 0.0, -1.0, -1.0],
                        routing_matrices=[route] * 5,
                    )
                ]
            )

        return rollout

    cfg = async_rl_loop.Config(
        log_path=str(tmp_path),
        completions_per_prompt=2,
        prompt_groups_per_step=4,
        pipeline_chunks_per_step=4,
        max_concurrency_rollout_sample=8,
        grad_accumulation_normalization="num_loss_tokens",
        shuffle=False,
        epochs=2 if continuation else 1,
        init_from_checkpoint="previous-job:step-4" if continuation else None,
        resume_recipe_state=continuation,
        save_final_checkpoint=False,
        deployment=async_rl_loop.DeployConfig(tokenizer_model="Qwen/Qwen3-1.7B"),
    )
    advantage_calls: list[list[float]] = []

    def mean_centered_advantages(rewards):
        advantage_calls.append(list(rewards))
        mean_reward = sum(rewards) / len(rewards)
        return [reward - mean_reward for reward in rewards]

    kwargs = {
        "rows": [{"id": i} for i in range(3 if continuation else 4)],
        "rollout_fn_factory": rollout_factory,
        **({"advantage_fn": mean_centered_advantages} if custom_advantages else {}),
    }
    if fail_forward:
        with pytest.raises(RuntimeError) as caught:
            async_rl_loop.main(cfg, **kwargs)
        assert caught.value is backend_error
        policy.optim_step.assert_not_called()
        assert events == ["save:step-0", "hotload:step-0", "forward_backward"]
        phase_logs = [m for m in caplog.messages if m.startswith("Training phase end")]
        assert len(phase_logs) == 3
        assert "phase=datum_build" in phase_logs[1] and "status=ok" in phase_logs[1]
        assert "phase=fwd_bwd" in phase_logs[2] and "status=error" in phase_logs[2]
        assert not any("async/realized_training_chunks" in item for item in metrics)
        return

    result = async_rl_loop.main(cfg, **kwargs)

    assert len(advantage_calls) == (4 if custom_advantages else 0)
    assert all(sorted(rewards) == [0.0, 1.0] for rewards in advantage_calls)
    assert result["steps"] == step_offset + 1
    checkpoints.resume.assert_called_once_with(
        init_from_checkpoint=cfg.init_from_checkpoint,
        warm_start_from_adapter=None,
        require_dataloader_state=True,
        resume_recipe_state=continuation,
    )
    assert sorted(sampled_positions) == (
        [(2, 0)] * 2 + [(3, 1)] * 2 + [(4, 1)] * 2 + [(5, 1)] * 2
        if continuation
        else [(i, 0) for i in range(4) for _ in range(2)]
    )
    assert events == [
        f"save:step-{step_offset}",
        f"hotload:step-{step_offset}",
        *(["forward_backward"] * 4),
        "optim_step",
        f"save:step-{step_offset + 1}",
        f"hotload:step-{step_offset + 1}",
    ]
    policy.forward.assert_not_called()
    policy.forward_backward_custom.assert_not_called()
    assert build_service.call_args.kwargs["reference_required"] is False
    service.create_reference_client.assert_not_called()
    assert len(datums_seen) == 8
    for datum in datums_seen:
        inputs = datum.loss_fn_inputs
        assert set(inputs) == {"target_tokens", "logprobs", "advantages"}
        assert list(inputs["logprobs"].data) == pytest.approx([0, -0.2, 0, -0.4, -0.6])
        targets = list(inputs["target_tokens"].data)
        assert targets == [11, targets[1], targets[2], 40, 50]
        assert datum.model_input.to_ints() == [10, 11, targets[1], targets[2], 40]
        assert datum.model_input.routing_matrices == ["", route, route, route, route]
        advantage_scale = 0.5 if custom_advantages else 1 / math.sqrt(2)
        advantage = (1 if targets[2] == 31 else -1) * advantage_scale
        assert list(inputs["advantages"].data) == pytest.approx(
            [0, advantage, 0, advantage, advantage]
        )
    expected_rows = [2, 0, 1, 2] if continuation else list(range(4))
    assert sorted(
        tuple(datum.loss_fn_inputs["target_tokens"].data[1:3]) for datum in datums_seen
    ) == sorted((20 + row, 30 + sample) for row in expected_rows for sample in range(2))
    [step_metrics] = [item for item in metrics if "async/realized_training_chunks" in item]
    assert step_metrics["async/realized_training_chunks"] == 4
    phases = (
        "chunk_combine",
        "datum_build",
        "fwd_bwd",
        "postprocess",
        "optim_prepare",
        "optim_step",
    )
    assert all(step_metrics[f"perf/{phase}_time"] >= 0 for phase in phases)
    assert step_metrics["perf/train_worker_time"] == pytest.approx(
        sum(step_metrics[f"perf/{phase}_time"] for phase in phases)
        + step_metrics["perf/train_worker_unphased_time"]
    )
    assert step_metrics["perf/train_accounting_error_time"] == pytest.approx(
        0, abs=1e-7
    )
    assert step_metrics["perf/train_worker_unphased_time"] >= 0
    chunk_logs = [m for m in caplog.messages if m.startswith("Training chunk ")]
    assert len(chunk_logs) == 4
    assert all("groups=1 datums=2 target_positions=10" in m for m in chunk_logs)
    assert step_metrics["train/loss:sum"] == 4
    assert step_metrics["train/raw_inference_logprob_coverage"] == 1
    assert step_metrics["train/inference_k1"] == pytest.approx(0.25)
    assert step_metrics["train/inference_k3"] == pytest.approx(
        math.exp(0.25) - 0.25 - 1, abs=1e-6
    )
    assert setup_wandb.call_args.args[1]["algorithm"] == "cispo"
    assert setup_wandb.call_args.args[1]["trainer_loss"] == "builtin"


def test_periodic_cursor_persistence_failure_stops_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    deployment_sampler = SimpleNamespace(
        base_url="https://deployment.test/v1",
        model="accounts/test/deployments/model",
    )
    sampling_client = MagicMock(deployment_sampler=deployment_sampler)
    sampling_client.close.side_effect = lambda: events.append("sampling_client.close")
    service = MagicMock(
        trainer_job_id="trainer-test",
        reference_client_job_id=None,
        reference_trainer_job_id=None,
        deployment_id="deployment-test",
        max_context_length=131_072,
    )
    service.create_training_client.return_value = object()
    service.create_sampling_client.return_value = sampling_client
    service.close.side_effect = lambda: events.append("service.close")
    policy = MagicMock()
    policy.save_weights_for_sampler.return_value = SimpleNamespace(path="snapshot-test")
    policy.optim_step.return_value = object()
    checkpoints = MagicMock()
    checkpoints.resume.return_value = None
    persistence_error = async_rl_loop.DataloaderStatePersistenceError("commit failed")
    checkpoints.save.side_effect = persistence_error

    class _Batch:
        batch_id = 1

        async def chunks(self):
            if False:
                yield None

    class _Coordinator:
        global_step = 1

        def __init__(self, **_kwargs):
            self._batch = _Batch()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def next_batch(self):
            batch, self._batch = self._batch, None
            return batch

        async def run_blocking(
            self, _operation, function, *args, optimizer_batch=None, **kwargs
        ):
            return function(*args, **kwargs)

        def raise_if_failed(self, _batch=None):
            return None

        def publish(self, _batch):
            return SimpleNamespace(
                resolved_rows=1,
                trained_against_version=0,
                step_time=0.1,
            )

        def snapshot(self):
            return {}

    class _Telemetry:
        def __init__(self, **_kwargs):
            pass

        def start(self, _snapshot):
            return None

        def finish_step(self, **_kwargs):
            return None

        async def aclose(self):
            events.append("telemetry.aclose")

    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    replacements = {
        "setup_wandb": lambda *_args, **_kwargs: None,
        "wandb_finish": lambda **_kwargs: None,
        "log_metrics": lambda *_args, **_kwargs: None,
        "validate_config": lambda *_args, **_kwargs: None,
        "resolve_router_replay_enabled": lambda **_kwargs: False,
        "read_api_extra_headers_env": lambda: {},
        "load_deployment_tokenizer": lambda _deployment: object(),
        "build_service_client": lambda **_kwargs: service,
        "TrainingCheckpoints": lambda *_args, **_kwargs: checkpoints,
        "AsyncRLCoordinator": _Coordinator,
        "AsyncRLTelemetry": _Telemetry,
    }
    for name, replacement in replacements.items():
        monkeypatch.setattr(async_rl_loop, name, replacement)
    monkeypatch.setattr(
        async_rl_loop.ReconnectableClient,
        "from_training_client",
        lambda *_args, **_kwargs: policy,
    )

    cfg = async_rl_loop.Config(
        log_path="/tmp/async_rl_test_logs",
        kl_beta=0,
        completions_per_prompt=2,
        prompt_groups_per_step=1,
        dcp_save_interval=1,
        router_replay=False,
        save_final_checkpoint=False,
        deployment=async_rl_loop.DeployConfig(
            tokenizer_model="Qwen/Qwen3-1.7B",
        ),
    )

    with pytest.raises(async_rl_loop.DataloaderStatePersistenceError) as exc_info:
        async_rl_loop.main(
            cfg,
            rows=[{"id": "row-1", "prompt": "1+1"}],
            rollout_fn_factory=lambda _setup: (lambda _sample: None),
        )

    assert exc_info.value is persistence_error
    checkpoints.save.assert_called_once_with(
        "step-1",
        resumable=True,
        promotable=False,
        data_consumed=1,
    )
    assert events == [
        "telemetry.aclose",
        "sampling_client.close",
        "service.close",
    ]


def test_main_requests_trainer_cleanup_for_empty_job_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = async_rl_loop.Config(
        log_path="/tmp/async_rl_test_logs",
        trainer=async_rl_loop.TrainerConfig(job_id=""),
        deployment=async_rl_loop.DeployConfig(tokenizer_model="Qwen/Qwen3-1.7B"),
    )

    kwargs = _build_service_kwargs(monkeypatch, cfg)

    assert kwargs["cleanup_trainer_on_close"] is True
