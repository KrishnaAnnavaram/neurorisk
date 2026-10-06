"""Explicit feature sets. Model inputs come ONLY from these whitelists.

Columns that a caller adds to a DataFrame (for example an old "risk score" column) can never
reach a model, because `select` copies only the listed columns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import pandas as pd

from .schema import BY_NAME, ID_COLUMN, IGNORED_COLUMNS, LABEL_COLUMN

FORBIDDEN = frozenset({ID_COLUMN, LABEL_COLUMN, *IGNORED_COLUMNS})


class FeatureSetError(ValueError):
    """A feature set definition is invalid."""


@dataclass(frozen=True)
class FeatureSet:
    name: str
    role: str
    description: str
    domains: dict[str, tuple[str, ...]]

    @property
    def columns(self) -> list[str]:
        return [c for cols in self.domains.values() for c in cols]

    def domain_of(self, column: str) -> str:
        for domain, cols in self.domains.items():
            if column in cols:
                return domain
        raise KeyError(column)


def _check(fs: FeatureSet) -> None:
    seen: set[str] = set()
    for col in fs.columns:
        if col in FORBIDDEN:
            raise FeatureSetError(f"feature set {fs.name}: {col!r} is an ID, label or ignored column")
        spec = BY_NAME.get(col)
        if spec is None or not spec.is_feature:
            raise FeatureSetError(f"feature set {fs.name}: {col!r} is not a feature column of the schema")
        if col in seen:
            raise FeatureSetError(f"feature set {fs.name}: {col!r} is listed twice")
        seen.add(col)
    if not seen:
        raise FeatureSetError(f"feature set {fs.name} has no columns")


def parse_feature_sets(raw: dict) -> dict[str, FeatureSet]:
    """Resolve `extends` chains and validate each set against the schema."""
    resolved: dict[str, FeatureSet] = {}

    def build(name: str, stack: tuple[str, ...] = ()) -> FeatureSet:
        if name in resolved:
            return resolved[name]
        if name in stack:
            raise FeatureSetError(f"circular 'extends' at {name}")
        if name not in raw:
            raise FeatureSetError(f"unknown feature set {name!r}")
        entry = raw[name]
        domains: dict[str, tuple[str, ...]] = {}
        parent = entry.get("extends")
        if parent:
            domains.update(build(parent, stack + (name,)).domains)
        for domain, cols in entry.get("domains", {}).items():
            if domain in domains:
                raise FeatureSetError(f"feature set {name}: domain {domain!r} is defined twice")
            domains[domain] = tuple(cols)
        fs = FeatureSet(name, entry.get("role", ""), entry.get("description", ""), domains)
        _check(fs)
        resolved[name] = fs
        return fs

    for key in raw:
        build(key)
    return resolved


def load_feature_sets(path: Path | None = None) -> dict[str, FeatureSet]:
    if path is None:
        text = resources.files("neurorisk").joinpath("configs/feature_sets.json").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    return parse_feature_sets(json.loads(text))


def get_feature_set(name: str, path: Path | None = None) -> FeatureSet:
    sets = load_feature_sets(path)
    if name not in sets:
        raise FeatureSetError(f"unknown feature set {name!r}; choose from {sorted(sets)}")
    return sets[name]


def select(df: pd.DataFrame, fs: FeatureSet) -> pd.DataFrame:
    """Return a new frame with exactly the feature set's columns, in the defined order."""
    missing = [c for c in fs.columns if c not in df.columns]
    if missing:
        raise FeatureSetError(f"data lacks columns for feature set {fs.name}: {missing}")
    return df.loc[:, fs.columns].copy()
