from __future__ import annotations

import pytest

import bbo.tasks.dbtune as dbtune
from bbo.tasks import ALL_TASK_NAMES, TASK_FAMILIES, TASK_REGISTRY, create_task

_DBTUNE_SURROGATE_TASKS = {
    "knob_http_surrogate_sysbench_5",
    "knob_http_surrogate_sysbench_all",
    "knob_http_surrogate_job_5",
    "knob_http_surrogate_job_all",
    "knob_http_surrogate_pg_5",
    "knob_http_surrogate_pg_20",
}


def test_task_families_use_dbtune_labels_for_dbtune_tasks() -> None:
    assert "dbtune_surrogate_service" in TASK_FAMILIES
    assert set(TASK_FAMILIES["dbtune_surrogate_service"]) == _DBTUNE_SURROGATE_TASKS
    assert "dbtune_mariadb" not in TASK_FAMILIES
    assert "dbtune_surrogate" not in TASK_FAMILIES
    assert "http_surrogate" not in TASK_FAMILIES
    assert "database" not in TASK_FAMILIES
    assert "surrogate" not in TASK_FAMILIES


def test_dbtune_registry_only_keeps_http_surrogate_tasks() -> None:
    dbtune_tasks = {name for name in ALL_TASK_NAMES if name.startswith("knob_")}

    assert dbtune_tasks == _DBTUNE_SURROGATE_TASKS
    assert all(not name.startswith("knob_http_mariadb_") for name in ALL_TASK_NAMES)
    assert all(not name.startswith("knob_http_mariadb_") for name in TASK_REGISTRY)


def test_dbtune_package_entrypoint_does_not_export_mariadb_tasks() -> None:
    forbidden = {
        "DBTUNE_MARIADB_TASK_IDS",
        "DBTUNE_MARIADB_TASK_NAMES",
        "DATABASE_TASK_NAMES",
        "create_dbtune_mariadb_task",
        "create_http_database_task",
    }

    assert forbidden.isdisjoint(dbtune.__all__)
    for name in forbidden:
        assert not hasattr(dbtune, name)


def test_create_task_rejects_removed_mariadb_dbtune_task() -> None:
    with pytest.raises(ValueError, match="Unknown task"):
        create_task("knob_http_mariadb_sysbench_read_write_5", max_evaluations=1)
