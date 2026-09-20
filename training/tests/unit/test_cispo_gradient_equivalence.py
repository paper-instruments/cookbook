"""CPU gradient proofs, not tests of Fireworks' private native CISPO kernel.

The CISPO reference consumes actual builtin datums and models the historical
``clip_low_threshold=0, clip_high_threshold=5, ratio_log_cap=20`` contract.
Matching first derivatives requires identical old/current forwards, token TIS
with cap 5, no IcePop, KL zero, and no optimizer mutation between chunks.
"""

from __future__ import annotations

import math

import torch
import tinker

from training.utils.rl.grpo import make_grpo_loss_fn
from training.utils.rl.losses import build_grpo_datums
from training.utils.rl.tis import SAFETY_CLAMP, TISConfig


def test_identical_forwards_match_for_signed_advantages_and_masked_tokens():
    log_ratios = [0.0, math.log(0.1), math.log(4.999), 30.0,
                  math.log(5.0), math.log(5.001), math.log(10.0), -30.0]
    mask = [0, 1, 1, 0, 1, 1, 1, 0]
    cases = [_case(advantage, log_ratios, mask) for advantage in (1.5, -0.75, 0.0)]

    grpo = _grpo_gradients(cases)
    cispo = _cispo_reference_gradients(cases)

    for old_gradient, new_gradient in zip(grpo, cispo, strict=True):
        torch.testing.assert_close(old_gradient, new_gradient, rtol=1e-6, atol=0)
        assert torch.count_nonzero(old_gradient[[0, 3, 7]]) == 0
    assert grpo[0][-2] < 0 and grpo[1][-2] > 0
    assert torch.count_nonzero(grpo[2]) == 0


def test_four_uneven_chunks_preserve_the_global_nonzero_gradient_mean():
    chunks = [
        [_case(1.5, [0.0, 0.0, 2.0], [0, 1, 1])],
        [_case(-0.75, [0.0, -2.0, 0.0, 1.0, 3.0], [0, 1, 0, 1, 1])],
        [_case(0.0, [0.0, 0.0], [0, 1])],
        [_case(2.0, [0.0, 0.0, 0.0, -1.0], [0, 0, 1, 1])],
    ]
    cases = [case for chunk in chunks for case in chunk]
    chunk_token_counts = [
        sum(sum(case[0].loss_fn_inputs["weights"].data) for case in chunk if case[1] != 0)
        for chunk in chunks
    ]
    assert chunk_token_counts == [2, 3, 0, 2]
    # SDK documents nonzero-gradient tokens, not all eight sampled tokens.
    denominator = sum(chunk_token_counts)
    sampled_tokens = sum(sum(case[0].loss_fn_inputs["weights"].data) for case in cases)
    assert sampled_tokens == 8

    # Isolate accumulation math from float32 reduction-order rounding; the
    # other tests retain float32 coverage, including its underflow counterexample.
    parameters = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    full_batch_loss = _shared_parameter_loss(cases, parameters, _grpo_loss)
    (full_batch_loss / denominator).backward()
    expected = parameters.grad.clone()

    for loss_fn in (_grpo_loss, _cispo_reference_loss):
        parameters = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        full_batch_loss = _shared_parameter_loss(cases, parameters, loss_fn)
        (full_batch_loss / denominator).backward()
        torch.testing.assert_close(parameters.grad, expected, rtol=1e-12, atol=0)

        parameters = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        chunk_gradients = []
        for chunk in chunks:
            previous = (
                torch.zeros_like(parameters)
                if parameters.grad is None else parameters.grad.clone()
            )
            _shared_parameter_loss(chunk, parameters, loss_fn).backward()
            chunk_gradients.append(parameters.grad - previous)

        torch.testing.assert_close(
            parameters.grad / denominator, expected, rtol=1e-12, atol=0,
        )

        # Uneven lengths and a zero-advantage chunk distinguish these mistakes.
        chunk_means = [
            gradient / max(count, 1)
            for gradient, count in zip(chunk_gradients, chunk_token_counts, strict=True)
        ]
        assert not torch.allclose(torch.stack(chunk_means).mean(dim=0), expected)
        assert not torch.allclose(parameters.grad / sampled_tokens, expected)


def test_ratio_log_floor_is_required_for_extreme_first_step_equivalence():
    cases = [_case(1.0, [-30.0, -20.0, -19.0, 20.0, 30.0])]
    grpo = _grpo_gradients(cases)[0]
    capped = _cispo_reference_gradients(cases)[0]
    uncapped = _cispo_reference_gradients(cases, ratio_log_cap=None)[0]

    torch.testing.assert_close(grpo, capped, rtol=1e-6, atol=0)
    torch.testing.assert_close(grpo[1:], uncapped[1:], rtol=1e-6, atol=0)
    assert abs(grpo[0]) > 20_000 * abs(uncapped[0])
    torch.testing.assert_close(grpo[-2:], torch.tensor([-5.0, -5.0]))


def test_forward_mismatch_breaks_equivalence_through_ppo_and_tis_clipping():
    cases = [_case(1.0, [0.0]), _case(-1.0, [0.0]), _case(1.0, [math.log(10.0)])]
    old_rows = [(case[2] + offset).tolist() for case, offset in zip(cases, (-1.0, 1.0, -0.01))]

    grpo = torch.cat(_grpo_gradients(cases, old_rows=old_rows))
    cispo = torch.cat(_cispo_reference_gradients(cases))

    torch.testing.assert_close(grpo[:2], torch.zeros(2))
    torch.testing.assert_close(cispo, torch.tensor([-1.0, 1.0, -5.0]))
    assert grpo[2] < cispo[2] - 0.04


def test_float32_tied_ppo_branches_can_underflow_a_subnormal_advantage():
    advantage = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)).item()
    cases = [_case(advantage, [0.0])]

    grpo = _grpo_gradients(cases)[0]
    cispo = _cispo_reference_gradients(cases)[0]

    # maximum splits a tie's gradient in half; each branch can underflow.
    assert grpo.item() == 0.0
    assert cispo.item() == -advantage


def _case(advantage, log_ratios, mask=None):
    mask = [1] * len(log_ratios) if mask is None else mask
    length = len(mask)
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(list(range(10, 10 + length))),
        loss_fn_inputs={
            "target_tokens": list(range(11, 11 + length)),
            "weights": list(mask),
        },
    )
    current = torch.full((length,), -32.0)
    behavior = (current - torch.tensor(log_ratios)).tolist()
    prompt_len = next(index for index, active in enumerate(mask) if active) + 1
    return datum, advantage, current, behavior, prompt_len


def _grpo_gradients(cases, *, old_rows=None):
    logprobs = [case[2].clone().requires_grad_() for case in cases]
    return torch.autograd.grad(_grpo_loss(cases, logprobs, old_rows=old_rows), logprobs)


def _grpo_loss(cases, logprobs, *, old_rows=None):
    data, advantages, currents, behavior, prompt_lens = zip(*cases, strict=True)
    loss_fn = make_grpo_loss_fn(
        advantages=list(advantages), ref_logprobs=[], prompt_len=list(prompt_lens),
        inf_logprobs=list(behavior),
        old_policy_logprobs=old_rows or [current.tolist() for current in currents],
        kl_beta=0.0, eps_clip=0.2, tis_config=TISConfig(cap=5.0),
    )
    loss, _ = loss_fn(list(data), logprobs)
    return loss


def _cispo_reference_gradients(cases, *, ratio_log_cap=SAFETY_CLAMP):
    logprobs = [case[2].clone().requires_grad_() for case in cases]
    loss = _cispo_reference_loss(cases, logprobs, ratio_log_cap=ratio_log_cap)
    return torch.autograd.grad(loss, logprobs)


def _cispo_reference_loss(cases, logprobs, *, ratio_log_cap=SAFETY_CLAMP):
    data, advantages, _currents, behavior, prompt_lens = zip(*cases, strict=True)
    datums = build_grpo_datums(
        data=list(data), advantages=list(advantages),
        old_policy_logprobs=list(behavior), inf_logprobs=list(behavior),
        prompt_lens=list(prompt_lens), tis_config=TISConfig(cap=1.0),
    )
    loss = torch.tensor(0.0)
    for original, datum, current in zip(data, datums, logprobs, strict=True):
        assert datum.model_input is original.model_input
        assert datum.loss_fn_inputs["target_tokens"].data == original.loss_fn_inputs["target_tokens"].data
        behavior_logprobs = torch.tensor(datum.loss_fn_inputs["logprobs"].data, dtype=current.dtype)
        masked_advantages = torch.tensor(datum.loss_fn_inputs["advantages"].data, dtype=current.dtype)
        log_ratio = current - behavior_logprobs
        if ratio_log_cap is not None:
            log_ratio = log_ratio.clamp(-ratio_log_cap, ratio_log_cap)
        weight = log_ratio.exp().clamp(0.0, 5.0).detach()
        loss = loss - (weight * current * masked_advantages).sum()
    return loss


def _shared_parameter_loss(cases, parameters, loss_fn):
    # At zero parameters, old/current scores match; both parameters influence
    # every datum, with distinct token Jacobians rather than independent leaves.
    logprobs = [
        case[2].to(parameters.dtype) + parameters[0] + parameters[1] * torch.arange(len(case[2]))
        for case in cases
    ]
    return loss_fn(cases, logprobs)
