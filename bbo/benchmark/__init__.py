"""Benchmark-level entrypoints built on top of the BBO core."""

from .runner import BenchmarkRunConfig, run_benchmark, run_named_benchmark

__all__ = ["BenchmarkRunConfig", "run_benchmark", "run_named_benchmark"]
