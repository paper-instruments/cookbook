"""Pure-logic tests for the async_rl_loop runtime helpers.

Covers the deterministic pieces that don't require tinker, the Fireworks
SDK, or a deployment.
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from training.recipes import async_rl_loop


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

    def test_config_exposes_only_grpo_knobs(self) -> None:
        cfg = async_rl_loop.Config(log_path="gs://logs")

        assert cfg.kl_beta == 0.001
        assert cfg.eps_clip == 0.2
        assert cfg.eps_clip_high is None
        assert cfg.anchor_logp == "old_policy"
        assert cfg.router_replay is True
        assert cfg.router_replay_completion_only is True
        assert not hasattr(cfg, "policy_loss")
        assert not hasattr(cfg, "loss_path")


def test_main_has_direct_client_grpo_customization_boundary() -> None:
    source = inspect.getsource(async_rl_loop.main)

    assert "make_grpo_loss_fn(" in source
    assert "policy.forward_backward_custom(" in source
    assert 'cfg.anchor_logp == "old_policy"' in source
    assert "To switch to built-in PPO or another loss" in source
    assert "adding dispatch" in source
    assert "skills/fireworks-training/references/rl-custom-loss.md" in source
    assert "build_loss_fn" not in source
    assert "loss_path" not in source


@pytest.mark.parametrize(
    "trainer",
    [
        async_rl_loop.TrainerConfig(reference_training_shape_id="ref-shape"),
        async_rl_loop.TrainerConfig(reference_job_id="ref-job"),
    ],
)
def test_main_rejects_unused_reference_trainer_config(trainer) -> None:
    cfg = async_rl_loop.Config(log_path="gs://logs", kl_beta=0, trainer=trainer)

    with pytest.raises(ValueError, match="require kl_beta > 0"):
        async_rl_loop.main(
            cfg,
            rows=[],
            rollout_fn_factory=lambda _setup: lambda _sample: None,
        )


@pytest.mark.parametrize(
    "config_overrides",
    [{"eps_clip": -0.1}, {"eps_clip_high": -0.1}, {"kl_beta": -0.1}],
)
def test_main_rejects_invalid_grpo_config(config_overrides) -> None:
    cfg = async_rl_loop.Config(log_path="gs://logs", **config_overrides)

    with pytest.raises(ValueError, match="must be non-negative"):
        async_rl_loop.main(
            cfg,
            rows=[],
            rollout_fn_factory=lambda _setup: lambda _sample: None,
        )


def test_main_rejects_unknown_anchor_logp() -> None:
    cfg = async_rl_loop.Config(log_path="gs://logs", anchor_logp="unknown")

    with pytest.raises(ValueError, match="anchor_logp must be"):
        async_rl_loop.main(
            cfg,
            rows=[],
            rollout_fn_factory=lambda _setup: lambda _sample: None,
        )


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
    )
    assert events == ["sampling_client.close", "service.close"]


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
