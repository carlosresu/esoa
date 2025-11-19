#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline registry that maps ITEM_REF_CODE categories to pipeline implementations."""

from __future__ import annotations

from typing import Dict, Iterable, Type

from .base import BasePipeline


PIPELINE_REGISTRY: Dict[str, Type[BasePipeline]] = {}


def register_pipeline(cls: Type[BasePipeline]) -> Type[BasePipeline]:
    """Decorator used by pipeline implementations to register themselves."""
    code = getattr(cls, "item_ref_code", None)
    if not code:
        raise ValueError(f"Pipeline {cls.__name__} must define item_ref_code.")
    PIPELINE_REGISTRY[code] = cls
    return cls


def get_pipeline(item_ref_code: str) -> BasePipeline:
    """Instantiate a pipeline given an ITEM_REF_CODE."""
    try:
        cls = PIPELINE_REGISTRY[item_ref_code]
    except KeyError as exc:
        raise KeyError(f"No pipeline registered for ITEM_REF_CODE={item_ref_code!r}.") from exc
    return cls()


def list_pipelines() -> Iterable[BasePipeline]:
    """Yield instantiated pipelines for each registered ITEM_REF_CODE."""
    for cls in PIPELINE_REGISTRY.values():
        yield cls()


# Import pipeline implementations so they register with the module-level mapping.
from .drugs.pipeline import DrugsAndMedicinePipeline  # noqa: E402,F401
from .labs.pipeline import LaboratoryAndDiagnosticPipeline  # noqa: E402,F401


__all__ = ["PIPELINE_REGISTRY", "register_pipeline", "get_pipeline", "list_pipelines"]
