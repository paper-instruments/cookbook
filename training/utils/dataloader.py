"""Cursor-based dataloader utilities."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class CursorState:
    """Checkpoint-owned progress, including resolved rows beyond the first hole."""

    cursor: int
    resolved_indices: tuple[int, ...]
    dataset_fingerprint: str

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "cursor": self.cursor,
            "resolved_indices": list(self.resolved_indices),
            "dataset_fingerprint": self.dataset_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: dict) -> CursorState:
        cursor = value.get("cursor")
        indices = value.get("resolved_indices")
        fingerprint = value.get("dataset_fingerprint")
        if (
            value.get("version") != 1
            or type(cursor) is not int
            or cursor < 0
            or not isinstance(indices, list)
            or any(type(i) is not int or i <= cursor for i in indices)
            or indices != sorted(set(indices))
            or not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(c not in "0123456789abcdef" for c in fingerprint)
        ):
            raise ValueError("invalid sparse dataloader state")
        return cls(cursor, tuple(indices), fingerprint)


@dataclass(frozen=True)
class CursorItem(Generic[T]):
    index: int
    value: T


class CursorDataLoader(Generic[T]):
    def __init__(
        self,
        items: list[T],
        start_cursor: int = 0,
        *,
        epochs: int = 1,
        shuffle: bool = False,
        seed: int = 0,
        resume_state: CursorState | None = None,
    ):
        if start_cursor < 0:
            raise ValueError("start_cursor must be >= 0")
        if epochs < 0:
            raise ValueError("epochs must be >= 0")
        self.items = items
        self.epochs = epochs
        self.shuffle = shuffle
        self.seed = seed
        self.cursor = start_cursor
        self.next_index = start_cursor
        self._resolved: set[int] = set()
        self._permutations: dict[int, list[int]] = {}
        self._dataset_fingerprint: str | None = None
        if resume_state is not None:
            if resume_state.dataset_fingerprint != self._fingerprint():
                raise ValueError(
                    "Resume dataset/order, seed, shuffle or epochs changed"
                )
            if (
                resume_state.cursor != start_cursor
                or resume_state.cursor > self.total_items
                or any(i >= self.total_items for i in resume_state.resolved_indices)
            ):
                raise ValueError("Resume dataloader positions do not match dataset")
            self._resolved.update(resume_state.resolved_indices)

    def __iter__(self):
        return self

    def __next__(self) -> CursorItem[T]:
        while self.next_index < self.cursor or self.next_index in self._resolved:
            self.next_index += 1
        if self.next_index >= self.total_items:
            raise StopIteration
        idx = self.next_index
        self.next_index += 1
        return CursorItem(
            index=idx, value=copy.deepcopy(self.items[self._row_index(idx)])
        )

    @property
    def data_consumed(self) -> int:
        return self.cursor

    @property
    def total_items(self) -> int:
        return len(self.items) * self.epochs

    @property
    def remaining_items(self) -> int:
        return max(0, self.total_items - self.cursor - len(self._resolved))

    def snapshot(self) -> CursorState:
        return CursorState(
            self.cursor, tuple(sorted(self._resolved)), self._fingerprint()
        )

    @property
    def epoch_id(self) -> int:
        return self.cursor // len(self.items) if self.items else 0

    @property
    def sample_offset(self) -> int:
        return self.cursor % len(self.items) if self.items else 0

    def mark_resolved(self, index: int) -> None:
        if index < self.cursor:
            return
        if index >= self.total_items:
            raise ValueError("resolved index out of range")
        self._resolved.add(index)
        while self.cursor in self._resolved:
            self._resolved.remove(self.cursor)
            self.cursor += 1

    def _row_index(self, index: int) -> int:
        if not self.items:
            raise IndexError("empty dataloader")
        epoch = index // len(self.items)
        offset = index % len(self.items)
        if not self.shuffle:
            return offset
        if epoch not in self._permutations:
            perm = list(range(len(self.items)))
            random.Random(self.seed + epoch).shuffle(perm)
            self._permutations[epoch] = perm
        return self._permutations[epoch][offset]

    def _fingerprint(self) -> str:
        if self._dataset_fingerprint is None:
            encoded = json.dumps(
                [self.items, self.epochs, self.shuffle, self.seed],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
            self._dataset_fingerprint = hashlib.sha256(encoded).hexdigest()
        return self._dataset_fingerprint
