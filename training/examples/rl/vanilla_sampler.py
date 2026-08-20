"""Example-owned sampler helper for async RL rollouts."""

from __future__ import annotations

from typing import Any

from fireworks.training.sdk.deployment import DeploymentSampler

from training.recipes.async_rl_loop import RolloutSetup


def build_deployment_sampler(setup: RolloutSetup) -> Any:
    """Return the recipe sampler or construct a deployment-backed sampler.

    The training recipe assembles the setup once at startup and hands it
    to the rollout factory. ``max_concurrency_rollout_sample`` caps rollout
    callbacks; the optional sampling concurrency controller separately limits
    concurrent LLM requests.
    """
    if setup.sampler is not None:
        return setup.sampler
    return DeploymentSampler(
        inference_url=setup.inference_base_url,
        model=setup.model,
        api_key=setup.api_key,
        tokenizer=setup.tokenizer,
    )
