# pipeline_types.py
"""Shared types for pipeline modules (passthrough + RIFE).

Single source of truth for PipelineResult to prevent field drift across
pipeline.py and run_rife.py.
"""
from dataclasses import dataclass


@dataclass
class PipelineResult:
    returncode: int
    stdout: str
    stderr: str
