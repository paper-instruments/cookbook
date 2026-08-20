#!/usr/bin/env python3
"""Resolve and verify the cookbook's declared minimum Fireworks SDK."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

SDK_REQUIREMENT_RE = re.compile(
    r"^fireworks-ai(?:\[[^]]+\])?(?P<specifiers>[^;]*)(?:;.*)?$"
)
MINIMUM_RE = re.compile(r"(?:^|,)\s*>=\s*(?P<version>[^,\s]+)\s*(?=,|$)")
VCS_RE = re.compile(
    r"^\s*@\s*git\+(?P<url>https://[^@\s]+)@(?P<commit>[0-9a-f]{40})\s*$"
)


def declared_sdk_requirement(pyproject: Path) -> str:
    with pyproject.open("rb") as handle:
        dependencies = tomllib.load(handle)["project"]["dependencies"]

    sdk_requirements = []
    for dependency in dependencies:
        requirement = dependency.strip()
        match = SDK_REQUIREMENT_RE.fullmatch(requirement)
        if match:
            sdk_requirements.append(requirement)

    if len(sdk_requirements) != 1:
        raise ValueError(
            "project.dependencies must contain exactly one fireworks-ai requirement"
        )

    return sdk_requirements[0]


def _declared_sdk_contract(requirement: str) -> tuple[str, str, str | None]:
    match = SDK_REQUIREMENT_RE.fullmatch(requirement)
    assert match is not None
    specifiers = match.group("specifiers")
    vcs = VCS_RE.fullmatch(specifiers)
    if vcs:
        return "commit", vcs.group("commit"), vcs.group("url")

    minimums = [match.group("version") for match in MINIMUM_RE.finditer(specifiers)]
    if len(minimums) != 1:
        raise ValueError(
            "the fireworks-ai requirement must contain one explicit >= minimum or "
            "one immutable VCS commit"
        )
    return "version", minimums[0], None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--assert-installed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requirement = declared_sdk_requirement(args.pyproject)
    contract, expected, expected_url = _declared_sdk_contract(requirement)
    if not args.assert_installed:
        print(expected)
        return 0

    try:
        installed = distribution("fireworks-ai")
    except PackageNotFoundError:
        print("fireworks-ai is not installed")
        return 1

    if contract == "version":
        if installed.version != expected:
            print(
                f"installed fireworks-ai {installed.version} does not equal "
                f"declared minimum {expected}"
            )
            return 1
        print(f"verified declared minimum fireworks-ai=={expected}")
        return 0

    direct_url_text = installed.read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text else {}
    installed_url = direct_url.get("url")
    installed_commit = direct_url.get("vcs_info", {}).get("commit_id")
    if installed_url != expected_url or installed_commit != expected:
        print(
            "installed fireworks-ai source does not equal declared VCS requirement: "
            f"expected {expected_url}@{expected}, got {installed_url}@{installed_commit}"
        )
        return 1
    print(f"verified declared fireworks-ai commit {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
