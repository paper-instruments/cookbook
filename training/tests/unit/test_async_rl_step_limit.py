"""Execute the real recipe, producer, coordinator and cursor persistence offline."""

from __future__ import annotations

import asyncio
import json
from functools import partial
from types import SimpleNamespace

import pytest
import tinker

from training.recipes import async_rl_loop
from training.utils.checkpoints import TrainingCheckpoints
from training.utils.dataloader import CursorDataLoader
from training.utils.rl.rollout import RolloutRun, RolloutSample


class _OfflineRemote:
    trainer_job_id = "phase-job"
    reference_trainer_job_id = None
    deployment_id = "deployment-test"
    max_context_length = 1024

    def __init__(self):
        self.events = []
        self.checkpoint_rows = []
        self.optimizer_step = 0
        self.accumulated_groups = 0
        self.update_sizes = []
        self.sampled = []
        self.active_rollouts = 0
        self.drop_rows = set()
        self.block_from_row = None
        self.fail_saves = set()
        self.save_count = 0
        self.finished = False

    def create_training_client(self, *_args, **_kwargs):
        return self

    def create_sampling_client(self, **_kwargs):
        return SimpleNamespace(
            deployment_sampler=SimpleNamespace(base_url="http://offline", model="test"),
            close=lambda: self.events.append("sampler_closed"),
        )

    def close(self):
        assert self.active_rollouts == 0
        self.events.append("service_closed")

    def forward_backward(self, datums, loss_fn, *, loss_fn_config):
        assert loss_fn == "cispo"
        self.accumulated_groups += len(datums) // 2
        self.events.append("forward_backward")
        return SimpleNamespace(
            metrics={},
            loss_fn_outputs=[
                {"logprobs": tinker.TensorData(data=[-0.2, -0.2], dtype="float32", shape=[2])}
                for _datum in datums
            ],
        )

    def optim_step(self, _params, **_kwargs):
        self.optimizer_step += 1
        self.update_sizes.append(self.accumulated_groups)
        self.accumulated_groups = 0
        self.events.append("optimizer")
        return SimpleNamespace(metrics={})

    def save_weights_for_sampler(self, name, **_kwargs):
        self.events.append(f"sampler:{name}")
        self.checkpoint_rows.append({
            "name": name,
            "checkpointType": "CHECKPOINT_TYPE_INFERENCE_BASE",
            "promotable": True,
        })
        return SimpleNamespace(path=name)

    def hotload_sampler_snapshot(self, path):
        self.events.append(f"hotload:{path}")

    def save_state(self, name):
        self.events.append(f"save:{name}")
        if name in self.fail_saves:
            raise RuntimeError("remote checkpoint failed")
        assert self.accumulated_groups == 0
        self.save_count += 1
        self.checkpoint_rows.append({
            "name": f"step-{self.optimizer_step}",
            "createTime": f"2026-09-30T00:00:{self.save_count:02d}Z",
            "checkpointType": "CHECKPOINT_TYPE_TRAINING",
            "promotable": False,
        })

    def list_checkpoints(self, _job):
        return list(self.checkpoint_rows)

    def resolve_checkpoint_path(self, name, **_kwargs):
        return name

    def load_state_with_optimizer(self, path):
        self.events.append(f"load:{path}")
        self.optimizer_step = int(path.removeprefix("step-"))

    def rollout_factory(self, _setup):
        async def rollout(row, *, sample_index, cursor_index):
            self.sampled.append(cursor_index)
            self.active_rollouts += 1
            try:
                if self.block_from_row is not None and cursor_index >= self.block_from_row:
                    await asyncio.Event().wait()
                await asyncio.sleep(0)
                if row["id"] in self.drop_rows:
                    return None
                return RolloutRun(segments=[RolloutSample(
                    tokens=[10, row["id"] + 20, 11],
                    logprobs=[0, -0.2, -0.2],
                    loss_mask=[0, 1, 1],
                    reward=float(sample_index),
                )])
            finally:
                self.active_rollouts -= 1
        return rollout


@pytest.fixture
def offline(monkeypatch, tmp_path):
    remote = _OfflineRemote()
    monkeypatch.setenv("FIREWORKS_API_KEY", "offline-test-key")

    def finish(**_kwargs):
        remote.finished = True

    for name, replacement in {
        "setup_wandb": lambda *_args, **_kwargs: None,
        "wandb_finish": finish,
        "log_metrics": lambda *_args, **_kwargs: None,
        "validate_config": lambda *_args, **_kwargs: None,
        "resolve_router_replay_enabled": lambda **_kwargs: False,
        "read_api_extra_headers_env": lambda: {},
        "load_deployment_tokenizer": lambda *_args: object(),
        "build_service_client": lambda **_kwargs: remote,
        "TrainingCheckpoints": partial(TrainingCheckpoints, save_stabilize_s=0),
    }.items():
        monkeypatch.setattr(async_rl_loop, name, replacement)
    monkeypatch.setattr(
        async_rl_loop.ReconnectableClient,
        "from_training_client",
        lambda *_args, **_kwargs: remote,
    )

    def run(row_count=385, **overrides):
        values = dict(
            log_path=str(tmp_path),
            completions_per_prompt=2,
            prompt_groups_per_step=64,
            pipeline_chunks_per_step=4,
            shuffle=False,
            router_replay=False,
            max_steps=5,
            require_full_batches=True,
            save_final_checkpoint=False,
            deployment=async_rl_loop.DeployConfig(tokenizer_model="offline"),
        )
        dynamic_filter_fn = overrides.pop("dynamic_filter_fn", None)
        on_dataloader_saved = overrides.pop("on_dataloader_saved", None)
        values.update(overrides)
        return async_rl_loop.main(
            async_rl_loop.Config(**values),
            rows=[{"id": index} for index in range(row_count)],
            rollout_fn_factory=remote.rollout_factory,
            dynamic_filter_fn=dynamic_filter_fn,
            on_dataloader_saved=on_dataloader_saved,
        )

    return remote, run, tmp_path


def test_stops_at_five_full_updates_and_saves_before_resource_close(offline):
    remote, run, path = offline
    result = run(init_from_checkpoint="previous-job:step-21", dcp_save_interval=2)

    assert result["steps"] == 5
    assert result["stop_reason"] == "max_steps_reached"
    assert remote.update_sizes == [64] * 5
    assert sorted(set(remote.sampled)) == list(range(320))
    assert remote.events.index("save:step-5") < remote.events.index("service_closed")
    saved = json.loads((path / "dataloader.json").read_text())
    assert saved["step-26"]["recipe_step"] == 5
    assert saved["step-26"]["dataloader_state"]["cursor"] == 320
    assert remote.finished


@pytest.mark.parametrize("row_count, expected_steps", [(31, 0), (95, 1), (287, 4)])
def test_shortage_never_sends_partial_batch_to_trainer(offline, row_count, expected_steps):
    remote, run, path = offline
    result = run(row_count=row_count)

    assert result["steps"] == expected_steps
    assert result["stop_reason"] == "dataset_exhausted_before_max_steps"
    assert remote.update_sizes == [64] * expected_steps
    assert remote.events.count("forward_backward") == 4 * expected_steps
    assert remote.accumulated_groups == 0
    saved = json.loads((path / "dataloader.json").read_text())
    assert saved[f"step-{expected_steps}"]["dataloader_state"]["cursor"] == expected_steps * 64


def test_filtering_and_dropped_rows_refill_full_batches(offline):
    remote, run, path = offline
    remote.drop_rows = set(range(0, 80, 4))

    def keep_group(group):
        return group.row_meta["row_id"] % 4 != 1

    result = run(row_count=600, dynamic_filter_fn=keep_group)

    assert result["steps"] == 5
    assert remote.update_sizes == [64] * 5
    assert len(set(remote.sampled)) > 320
    saved = json.loads((path / "dataloader.json").read_text())["step-5"]
    assert saved["dataloader_state"]["cursor"] > 320


def test_shutdown_cancels_speculation_without_committing_future_rows(offline):
    remote, run, path = offline
    remote.block_from_row = 320
    result = run(max_head_offpolicy_versions=1)

    assert result["steps"] == 5
    assert any(row >= 320 for row in remote.sampled)
    assert remote.active_rollouts == 0
    saved = json.loads((path / "dataloader.json").read_text())["step-5"]
    assert saved["dataloader_state"]["cursor"] == 320
    assert saved["dataloader_state"]["resolved_indices"] == []


def test_renamed_checkpoint_resumes_phase_offset_and_already_complete_is_noop(offline):
    remote, run, path = offline
    rows = [{"id": index} for index in range(385)]
    remote.optimizer_step = 22
    ckpt = TrainingCheckpoints(remote, remote, trainer_id="phase-job", log_path=str(path), save_stabilize_s=0)
    ckpt.save(
        "step-1", resumable=True, promotable=False, recipe_step=1,
        dataloader_state=CursorDataLoader(rows, start_cursor=64).snapshot(),
    )
    result = run(init_from_checkpoint="phase-job:step-22")

    assert result["steps"] == 5
    assert remote.update_sizes == [64] * 4
    assert min(remote.sampled) == 64
    assert max(remote.sampled) == 319
    saved = json.loads((path / "dataloader.json").read_text())
    assert saved["step-26"]["recipe_step"] == 5
    remote.sampled.clear()
    remote.events.clear()
    remote.update_sizes.clear()

    assert run()["steps"] == 5
    assert not remote.sampled
    assert not remote.update_sizes
    assert remote.events == ["load:step-26", "sampler_closed", "service_closed"]


@pytest.mark.parametrize("failure", ["remote", "cursor"])
def test_terminal_checkpoint_failure_is_not_success(offline, failure):
    remote, run, _path = offline
    kwargs = {}
    if failure == "remote":
        remote.fail_saves.add("step-5")
    else:
        def fail_cursor():
            raise OSError("cursor durability failed")
        kwargs["on_dataloader_saved"] = fail_cursor

    with pytest.raises(RuntimeError, match="checkpoint failed|persist dataloader state"):
        run(dcp_save_interval=5, **kwargs)

    assert remote.update_sizes == [64] * 5
    assert not remote.finished
    assert remote.events[-2:] == ["sampler_closed", "service_closed"]


def test_nonterminal_checkpoint_failure_warns_and_training_continues(offline, caplog):
    remote, run, _path = offline
    remote.fail_saves.add("step-2")
    assert run(dcp_save_interval=2)["steps"] == 5
    assert "[step 2] dcp_save failed" in caplog.text
    assert remote.update_sizes == [64] * 5


def test_default_still_streams_and_trains_eof_partial_batch(offline):
    remote, run, path = offline
    result = run(row_count=95, max_steps=None, require_full_batches=False)
    assert result["steps"] == 2
    assert "stop_reason" not in result
    assert remote.update_sizes == [64, 31]
    assert not (path / "dataloader.json").exists()


@pytest.mark.parametrize("max_steps", [0, -1, True, 1.5, "5"])
def test_invalid_limit_fails_before_provisioning(monkeypatch, max_steps):
    def unexpected_provision(**_kwargs):
        pytest.fail("invalid limit provisioned remote resources")
    monkeypatch.setattr(async_rl_loop, "build_service_client", unexpected_provision)
    with pytest.raises(ValueError, match="max_steps"):
        async_rl_loop.main(
            async_rl_loop.Config(log_path="unused", max_steps=max_steps),
            rows=[], rollout_fn_factory=lambda _setup: None,
        )
