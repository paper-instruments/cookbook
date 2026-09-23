# RL loss execution

`training/recipes/rl_loop.py` is an intentionally opinionated client-side GRPO
recipe. It computes
group-normalized advantages and calls `make_grpo_loss_fn(...)` directly through
`forward_backward_custom(...)`.

There is no loss selector, registry, runtime import, or fallback. The public
algorithm knobs are `kl_beta`, `eps_clip`, `eps_clip_high`, and `tis`; the
synchronous recipe also exposes `anchor_logp="old_policy" | "rollout"`.

In this fork, `async_rl_loop.py` instead calls built-in `cispo` directly, with
the rollout anchor and no reference KL. See [its loss contract](rl-async.md#loss-path).
It does not retain a fallback to custom GRPO. Its callers must use the specialized
CISPO configuration rather than passing PPO/TIS controls.

## Default client path

The recipe performs an optional reference forward when `kl_beta > 0`, snapshots
old-policy logprobs, and calls:

```python
policy.forward_backward_custom(
    data,
    make_grpo_loss_fn(
        advantages=advantages,
        ref_logprobs=ref_logprobs,
        prompt_len=prompt_lens,
        inf_logprobs=rollout_logprobs,
        old_policy_logprobs=old_policy_logprobs,
        kl_beta=cfg.kl_beta,
        eps_clip=cfg.eps_clip,
        eps_clip_high=cfg.eps_clip_high,
        tis_config=cfg.tis,
    ),
)
```

This one closure owns PPO clipping, behavioral TIS, and optional reference KL.
Set `kl_beta=0` to skip reference provisioning.

The synchronous recipe defaults to `anchor_logp="old_policy"`: snapshot trainer logprobs for the
PPO anchor and compute TIS against rollout behavior logprobs. Setting
`anchor_logp="rollout"` skips the snapshot, anchors PPO directly on rollout
logprobs, and makes the TIS ratio identity.

## Switching or adding a loss

Fork the recipe at its documented direct `forward_backward_custom` call.
For the exact built-in switch and new-algorithm workflow, read
[`rl-custom-loss.md`](rl-custom-loss.md).

The server built-in `"ppo"` path cannot apply reference KL. A built-in fork
must require `kl_beta=0`, prepare the kernel's datum contract explicitly, and
call `forward_backward(...)`. Do not keep both paths behind a config selector.

## Multimodal datum contract

Vision RL uses the canonical Tinker expanded sequence coordinates. For an
unshifted sequence of length `N`, including every image slot:

- `datum.model_input.length == N - 1`;
- `target_tokens`, `weights`, forward logprobs, and backward gradients all have
  length `N - 1`;
- image positions in `target_tokens` are zero wire placeholders; and
- image positions have zero weight/advantage and contribute no loss.

Do not strip image positions or compress tensors into text-only coordinates.
