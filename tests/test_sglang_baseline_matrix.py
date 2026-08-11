from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_matrix_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "sglang" / "matrix.py"
    spec = importlib.util.spec_from_file_location("run_sglang_baseline_matrix", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_split_selector_values_accepts_space_and_comma_groups() -> None:
    matrix = _load_matrix_module()

    assert matrix.split_selector_values([["nanobot", "random_search"], ["optuna_tpe,pycma"]]) == (
        "nanobot",
        "random_search",
        "optuna_tpe",
        "pycma",
    )
