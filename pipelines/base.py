#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common dataclasses and base class used by category-specific pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional

TimingHook = Callable[[str, float], None]


@dataclass(frozen=True)
class PipelineRunParams:
    """File-level inputs supplied to a pipeline invocation."""

    esoa_csv: Path
    annex_csv: Optional[Path] = None
    pnf_csv: Optional[Path] = None
    out_csv: Path = Path("esoa_matched.csv")


@dataclass(frozen=True)
class PipelineContext:
    """Resolved project paths that pipelines can rely on."""

    project_root: Path
    inputs_dir: Path
    outputs_dir: Path


@dataclass
class PipelineOptions:
    """Optional switches controlling pipeline behaviour."""

    skip_excel: bool = False
    extra: Dict[str, object] = field(default_factory=dict)

    def flag(self, key: str, default: bool = False) -> bool:
        """Helper to fetch boolean extras with a default."""
        value = self.extra.get(key, default)
        return bool(value)


@dataclass
class PipelinePreparedInputs:
    """Paths emitted by a pipeline's preparation stage."""

    esoa_csv: Path
    annex_csv: Optional[Path] = None
    pnf_csv: Optional[Path] = None
    artifacts: Dict[str, Path] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Unified return object for category pipelines."""

    matched_csv: Path
    prepared: PipelinePreparedInputs
    extras: Dict[str, Path] = field(default_factory=dict)


class BasePipeline:
    """Abstract pipeline contract that category-specific implementations must follow."""

    item_ref_code: str = ""
    display_name: str = ""
    description: str = ""

    def pre_run(
        self,
        context: PipelineContext,
        params: PipelineRunParams,
        options: PipelineOptions,
        *,
        timing_hook: Optional[TimingHook] = None,
    ) -> Mapping[str, Path]:
        """Optional hook executed before preparation. Return any artifacts that should be tracked."""
        return {}

    def prepare_inputs(
        self,
        context: PipelineContext,
        params: PipelineRunParams,
        options: PipelineOptions,
        *,
        timing_hook: Optional[TimingHook] = None,
    ) -> PipelinePreparedInputs:
        """Transform raw inputs into normalized CSVs ready for matching."""
        raise NotImplementedError

    def match(
        self,
        context: PipelineContext,
        prepared: PipelinePreparedInputs,
        options: PipelineOptions,
        *,
        timing_hook: Optional[TimingHook] = None,
    ) -> PipelineResult:
        """Execute the core matching stage given prepared CSVs."""
        raise NotImplementedError

    def post_run(
        self,
        context: PipelineContext,
        result: PipelineResult,
        options: PipelineOptions,
        *,
        timing_hook: Optional[TimingHook] = None,
    ) -> None:
        """Optional hook executed after matching to enrich outputs."""

    def run(
        self,
        context: PipelineContext,
        params: PipelineRunParams,
        options: Optional[PipelineOptions] = None,
        *,
        timing_hook: Optional[TimingHook] = None,
    ) -> PipelineResult:
        """End-to-end orchestration that mirrors prepare + match, including optional hooks."""
        opts = options or PipelineOptions()
        artifacts: Dict[str, Path] = {}
        artifacts.update(self.pre_run(context, params, opts, timing_hook=timing_hook))
        prepared = self.prepare_inputs(context, params, opts, timing_hook=timing_hook)
        prepared.artifacts.update(artifacts)
        result = self.match(context, prepared, opts, timing_hook=timing_hook)
        self.post_run(context, result, opts, timing_hook=timing_hook)
        return result


__all__ = [
    "BasePipeline",
    "PipelineContext",
    "PipelineOptions",
    "PipelinePreparedInputs",
    "PipelineResult",
    "PipelineRunParams",
    "TimingHook",
]
