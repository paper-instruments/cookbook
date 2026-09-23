import json

import pytest

from training.utils.dataloader import CursorDataLoader, CursorState


def test_cursor_starts_from_resume_point():
    loader = CursorDataLoader(["a", "b", "c"], start_cursor=1)

    assert next(loader).value == "b"
    assert loader.data_consumed == 1


def test_cursor_advances_in_order_only():
    loader = CursorDataLoader(["a", "b", "c"])

    loader.mark_resolved(1)
    assert loader.data_consumed == 0

    loader.mark_resolved(0)
    assert loader.data_consumed == 2

    loader.mark_resolved(2)
    assert loader.data_consumed == 3


def test_cursor_ignores_already_resolved_indices():
    loader = CursorDataLoader(["a", "b"], start_cursor=1)

    loader.mark_resolved(0)
    assert loader.data_consumed == 1


def test_cursor_can_resume_past_available_rows():
    loader = CursorDataLoader(["a"], start_cursor=3)

    assert loader.data_consumed == 3
    assert list(loader) == []


def test_cursor_owns_epochs_without_reusing_row_objects():
    loader = CursorDataLoader([{"id": 1}], epochs=2)

    first = next(loader)
    first.value["seen"] = True
    second = next(loader)

    assert first.index == 0
    assert second.index == 1
    assert second.value == {"id": 1}


def test_cursor_shuffle_is_deterministic_per_epoch():
    rows = list(range(10))
    first = [
        item.value for item in CursorDataLoader(rows, epochs=2, shuffle=True, seed=7)
    ]
    second = [
        item.value for item in CursorDataLoader(rows, epochs=2, shuffle=True, seed=7)
    ]
    unshuffled = [item.value for item in CursorDataLoader(rows, epochs=2)]

    assert first == second
    assert first != unshuffled
    assert sorted(first[:10]) == rows
    assert sorted(first[10:]) == rows


def test_cursor_resume_reconstructs_epoch_shuffle_position():
    rows = list(range(6))
    full = [
        item.value for item in CursorDataLoader(rows, epochs=3, shuffle=True, seed=4)
    ]
    resumed = [
        item.value
        for item in CursorDataLoader(
            rows, start_cursor=8, epochs=3, shuffle=True, seed=4
        )
    ]

    assert resumed == full[8:]


def test_sparse_progress_survives_repeated_restarts_across_shuffled_epochs():
    rows = [{"id": i, "dataset_revision": "tasks@v1"} for i in range(5)]
    expected = list(CursorDataLoader(rows, epochs=2, shuffle=True, seed=7))
    loader = CursorDataLoader(rows, epochs=2, shuffle=True, seed=7)
    # Position zero remains a straggler across several optimizer updates.
    for index in (1, 3, 5, 8):
        loader.mark_resolved(index)
    snapshot = loader.snapshot()
    assert snapshot.cursor == 0
    assert snapshot.resolved_indices == (1, 3, 5, 8)
    loader.mark_resolved(2)
    assert snapshot.resolved_indices == (1, 3, 5, 8)

    restored = CursorState.from_dict(json.loads(json.dumps(snapshot.to_dict())))
    resumed = CursorDataLoader(
        rows, epochs=2, shuffle=True, seed=7, resume_state=restored
    )
    assert resumed.remaining_items == 6
    assert list(resumed) == [
        item for item in expected if item.index not in (1, 3, 5, 8)
    ]
    resumed.mark_resolved(0)
    resumed.mark_resolved(2)
    second = resumed.snapshot()
    assert second.cursor == 4
    again = CursorDataLoader(
        rows, start_cursor=4, epochs=2, shuffle=True, seed=7, resume_state=second
    )
    assert list(again) == [expected[i] for i in (4, 6, 7, 9)]
    for index in (4, 6, 7, 9):
        again.mark_resolved(index)
    assert again.data_consumed == 10
    assert again.remaining_items == 0


@pytest.mark.parametrize("change", ["order", "revision", "seed", "shuffle", "epochs"])
def test_sparse_resume_rejects_changed_dataset_identity_or_order(change):
    rows = [{"id": i, "dataset_revision": "tasks@v1"} for i in range(3)]
    kwargs = {"epochs": 2, "shuffle": True, "seed": 7}
    state = CursorDataLoader(rows, **kwargs).snapshot()
    if change == "order":
        rows.reverse()
    elif change == "revision":
        rows[0]["dataset_revision"] = "tasks@v2"
    else:
        kwargs[change] = {"seed": 8, "shuffle": False, "epochs": 3}[change]
    with pytest.raises(ValueError, match="Resume dataset/order"):
        CursorDataLoader(rows, resume_state=state, **kwargs)


@pytest.mark.parametrize("indices", [[0], [2, 1], [1, 1], [-1]])
def test_sparse_state_rejects_ambiguous_positions(indices):
    data = CursorDataLoader([0, 1, 2]).snapshot().to_dict()
    data["resolved_indices"] = indices
    with pytest.raises(ValueError, match="invalid sparse"):
        CursorState.from_dict(data)


def test_sparse_resume_rejects_out_of_range_positions():
    state = CursorDataLoader([0, 1]).snapshot().to_dict()
    state["resolved_indices"] = [2]
    with pytest.raises(ValueError, match="positions do not match"):
        CursorDataLoader([0, 1], resume_state=CursorState.from_dict(state))
