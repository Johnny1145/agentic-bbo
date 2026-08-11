#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CONFIG_DIR = SCRIPT_DIR / "configs"


SGLANG_ARG_PROFILES: dict[str, list[str]] = {
    "opro": [
        "--opro-backend",
        "openai",
        "--opro-model",
        "{sglang_model}",
        "--opro-openai-api-key-env",
        "LOCAL_LLM_API_KEY",
        "--opro-openai-base-url",
        "{sglang_base_url}",
        "--opro-openai-timeout-seconds",
        "{sglang_timeout_seconds}",
        "--opro-openai-max-retries",
        "{sglang_max_retries}",
        "--no-opro-openai-include-seed",
        "--no-opro-openai-include-store",
        "--opro-prompt-pairs",
        "{opro_prompt_pairs}",
    ],
    "llambo": [
        "--llambo-backend",
        "openai",
        "--llambo-model",
        "{sglang_model}",
        "--llambo-openai-api-key-env",
        "LOCAL_LLM_API_KEY",
        "--llambo-openai-base-url",
        "{sglang_base_url}",
        "--llambo-openai-timeout-seconds",
        "{sglang_timeout_seconds}",
        "--llambo-openai-max-retries",
        "{sglang_max_retries}",
        "--no-llambo-openai-include-seed",
        "--no-llambo-openai-include-store",
        "--no-llambo-openai-use-structured-outputs",
    ],
    "pablo": [
        "--pablo-provider",
        "sglang",
        "--pablo-model",
        "{sglang_model}",
        "--pablo-base-url",
        "{sglang_base_url}",
        "--pablo-api-key-env",
        "LOCAL_LLM_API_KEY",
    ],
    "palbo": [
        "--pablo-provider",
        "sglang",
        "--pablo-model",
        "{sglang_model}",
        "--pablo-base-url",
        "{sglang_base_url}",
        "--pablo-api-key-env",
        "LOCAL_LLM_API_KEY",
    ],
    "nanobot": [
        "--agent-provider",
        "openai",
        "--agent-model",
        "{sglang_model}",
        "--agent-api-base",
        "{sglang_base_url}",
        "--agent-api-key-env",
        "LOCAL_LLM_API_KEY",
        "--agent-prompt-style",
        "workspace",
        "--agent-tool-mode",
        "workspace_json",
        "--agent-history-limit",
        "{agent_history_limit}",
        "--agent-max-tool-calls",
        "{agent_max_tool_calls}",
        "--agent-timeout-seconds",
        "{agent_timeout_seconds}",
        "--agent-max-retries",
        "{agent_max_retries}",
        "--agent-enable-memory",
        "--no-agent-enable-code-interpreter",
        "--agent-code-backend",
        "local_disabled",
        "--agent-web-search-provider",
        "disabled",
    ],
    "agentic_nanobot": [
        "--agent-provider",
        "openai",
        "--agent-model",
        "{sglang_model}",
        "--agent-api-base",
        "{sglang_base_url}",
        "--agent-api-key-env",
        "LOCAL_LLM_API_KEY",
        "--agent-prompt-style",
        "workspace",
        "--agent-tool-mode",
        "workspace_json",
        "--agent-history-limit",
        "{agent_history_limit}",
        "--agent-max-tool-calls",
        "{agent_max_tool_calls}",
        "--agent-timeout-seconds",
        "{agent_timeout_seconds}",
        "--agent-max-retries",
        "{agent_max_retries}",
        "--agent-enable-memory",
        "--no-agent-enable-code-interpreter",
        "--agent-code-backend",
        "local_disabled",
        "--agent-web-search-provider",
        "disabled",
    ],
    "agentic_openai_compatible": [
        "--agent-model",
        "{sglang_model}",
        "--agent-api-base",
        "{sglang_base_url}",
        "--agent-api-key-env",
        "LOCAL_LLM_API_KEY",
        "--agent-history-limit",
        "{agent_history_limit}",
        "--agent-max-tool-calls",
        "{agent_max_tool_calls}",
        "--agent-timeout-seconds",
        "{agent_timeout_seconds}",
        "--agent-max-retries",
        "{agent_max_retries}",
    ],
    "openai_compatible_agent": [
        "--agent-model",
        "{sglang_model}",
        "--agent-api-base",
        "{sglang_base_url}",
        "--agent-api-key-env",
        "LOCAL_LLM_API_KEY",
        "--agent-history-limit",
        "{agent_history_limit}",
        "--agent-max-tool-calls",
        "{agent_max_tool_calls}",
        "--agent-timeout-seconds",
        "{agent_timeout_seconds}",
        "--agent-max-retries",
        "{agent_max_retries}",
    ],
}


@dataclass(frozen=True)
class Runtime:
    repo_root: Path
    script_dir: Path
    config_dir: Path
    dry_run: bool
    passthrough: tuple[str, ...]
    run_root_override: str | None
    algorithms_override: tuple[str, ...] | None
    tasks_override: tuple[str, ...] | None
    seeds_override: tuple[str, ...] | None
    jobs_override: int | None
    resume: bool | None
    overwrite: bool | None
    env: dict[str, str]
    checked_sglang: set[tuple[str, str]]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run SGLang baseline matrices from family config files. "
            "This is an outer runner; it does not replace historical workflow files."
        )
    )
    parser.add_argument("--family", action="append", help="Family config name, e.g. synthetic. Repeatable.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every config in scripts/sglang/configs.",
    )
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--dry-run", "--list", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--algorithm",
        action="append",
        nargs="+",
        help="Algorithm selector. Space/comma separated; repeatable.",
    )
    parser.add_argument(
        "--task",
        action="append",
        nargs="+",
        help="Task/benchmark selector. Space/comma separated; repeatable.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        nargs="+",
        help="Seed selector. Space/comma separated; repeatable.",
    )
    parser.add_argument("--jobs", type=int, help="Override family jobs setting.")
    parser.add_argument("--run-root", help="Override RUN_ROOT for a single selected family.")
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--overwrite", action="store_true", default=None)
    parser.add_argument("--list-families", action="store_true", help="Show available family configs and exit.")
    parser.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are appended to generated commands.",
    )
    args = parser.parse_args(argv)

    config_dir = args.config_dir.resolve()
    if args.list_families:
        for path in discover_config_paths(config_dir):
            print(path.stem)
        return 0

    selected_paths = selected_config_paths(config_dir, families=args.family, include_all=args.all)
    if not selected_paths:
        parser.error("select at least one --family or pass --all")
    if args.run_root and len(selected_paths) != 1:
        parser.error("--run-root can only be used with one selected family")

    passthrough = tuple(item for item in args.passthrough if item != "--")
    env = baseline_env(os.environ)
    runtime = Runtime(
        repo_root=REPO_ROOT,
        script_dir=SCRIPT_DIR,
        config_dir=config_dir,
        dry_run=bool(args.dry_run),
        passthrough=passthrough,
        run_root_override=args.run_root,
        algorithms_override=split_selector_values(args.algorithm),
        tasks_override=split_selector_values(args.task),
        seeds_override=split_selector_values(args.seed),
        jobs_override=args.jobs,
        resume=args.resume,
        overwrite=args.overwrite,
        env=env,
        checked_sglang=set(),
    )

    for path in selected_paths:
        config = load_config(path)
        adapter = adapter_for(config, runtime)
        adapter.run()
    return 0


def discover_config_paths(config_dir: Path) -> list[Path]:
    return sorted(path for path in config_dir.glob("*.toml") if path.is_file())


def selected_config_paths(config_dir: Path, *, families: Sequence[str] | None, include_all: bool) -> list[Path]:
    if include_all:
        return discover_config_paths(config_dir)
    paths: list[Path] = []
    for family in families or []:
        path = config_dir / f"{family}.toml"
        if not path.exists():
            raise SystemExit(f"Unknown family config `{family}` at {path}")
        paths.append(path)
    return paths


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    config.setdefault("_path", str(path))
    return config


def split_selector_values(values: Sequence[Any] | None) -> tuple[str, ...] | None:
    if not values:
        return None
    selected: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            nested = split_selector_values(value)
            if nested:
                selected.extend(nested)
            continue
        for item in str(value).replace(",", " ").split():
            if item:
                selected.append(item)
    return tuple(selected)


def baseline_env(source: Mapping[str, str]) -> dict[str, str]:
    env = dict(source)
    agent_timeout_was_set = "AGENT_TIMEOUT_SECONDS" in env
    env.setdefault("SGLANG_BASE_URL", "http://127.0.0.1:18300/v1")
    env.setdefault("SGLANG_MODEL", "qwen3.5-9b")
    env.setdefault("SGLANG_TIMEOUT_SECONDS", "600")
    env.setdefault("SGLANG_MAX_RETRIES", "0")
    env.setdefault("LOCAL_LLM_API_KEY", "EMPTY")
    env.setdefault("OPENAI_API_KEY", env["LOCAL_LLM_API_KEY"])
    env.setdefault("SGLANG_API_KEY", env["LOCAL_LLM_API_KEY"])
    env.setdefault("PABLO_SGLANG_ENABLE_THINKING", "1")
    env.setdefault("AGENT_HISTORY_LIMIT", "200")
    env.setdefault("AGENT_MAX_TOOL_CALLS", "64")
    env.setdefault("AGENT_TIMEOUT_SECONDS", "900")
    env.setdefault("AGENT_MAX_RETRIES", "4")
    env.setdefault("OPRO_PROMPT_PAIRS", "20")
    env.setdefault("OPRO_GENERATION_ROUNDS", "4")
    env.setdefault("OPRO_CANDIDATES", "1")
    env.setdefault("EXECUTOR", "process")
    env.setdefault("SERVICE_WORKERS", "1")
    env.setdefault("MOLECULE_N_INITIAL_POINTS", "50")
    env.setdefault("MOLECULE_OPTIMIZER_BUDGET", "200")
    env.setdefault("MOLECULE_AGENT_TIMEOUT_SECONDS", env["AGENT_TIMEOUT_SECONDS"] if agent_timeout_was_set else "300")
    return env


def adapter_for(config: Mapping[str, Any], runtime: Runtime) -> "FamilyAdapter":
    family = str(config.get("family") or "")
    adapters: dict[str, type[FamilyAdapter]] = {
        "synthetic": SyntheticAdapter,
        "molecule": MoleculeAdapter,
        "bboplace": BBOPlaceAdapter,
        "dbtune": DBTuneAdapter,
    }
    try:
        return adapters[family](config, runtime)
    except KeyError as exc:
        raise SystemExit(f"Unsupported family `{family}` in {config.get('_path')}") from exc


class FamilyAdapter:
    def __init__(self, config: Mapping[str, Any], runtime: Runtime) -> None:
        self.config = config
        self.runtime = runtime
        self.family = str(config["family"])
        self.env_prefix = str(config.get("env_prefix") or self.family.upper())
        self.run_root = self.resolve_run_root()

    def run(self) -> None:
        algorithms = self.selected_algorithms()
        for algorithm in algorithms:
            print(f"=== {self.display_name()} {algorithm} ===", flush=True)
            self.run_algorithm(algorithm)

    def display_name(self) -> str:
        return str(self.config.get("display_name") or self.family)

    def run_algorithm(self, algorithm: str) -> None:
        command = self.command_config(algorithm)
        if command is not None:
            self.run_command(algorithm, command)
            return
        generic = self.generic_matrix_config(algorithm)
        if generic is not None:
            self.run_bbo_matrix(algorithm, generic)
            return
        raise SystemExit(f"Unsupported {self.family} algorithm in config: {algorithm}")

    def run_command(self, algorithm: str, command: Mapping[str, Any]) -> None:
        kind = str(command.get("kind", "workflow"))
        if kind != "workflow":
            raise SystemExit(f"Unsupported command kind `{kind}` for {self.family}.{algorithm}")
        if bool(command.get("requires_sglang", False)):
            self.require_sglang()
        cmd = self.workflow_command(algorithm, command)
        self.exec(cmd)

    def run_bbo_matrix(self, algorithm: str, generic: Mapping[str, Any]) -> None:
        profile = str(generic.get("sglang_args_profile") or algorithm)
        sglang_args = self.sglang_args(profile)
        if sglang_args:
            self.require_sglang()
        tasks = self.selected_tasks(generic)
        seeds = self.selected_seeds()
        for task in tasks:
            for seed in seeds:
                cmd = self.bbo_run_command(algorithm, task, seed, generic, sglang_args)
                self.exec(cmd)

    def workflow_command(self, algorithm: str, command: Mapping[str, Any]) -> list[str]:
        cmd = ["uv", "run"]
        cmd.extend(self.uv_extra_flags(command.get("uv_extras", [])))
        cmd.extend(["python", self.render(str(command["script"]), algorithm=algorithm)])
        cmd.extend(self.render_args(command.get("args", []), algorithm=algorithm))
        if bool(command.get("append_seed_flags", True)):
            cmd.extend(repeat_flag("--seed", self.selected_seeds()))
        if bool(command.get("append_state_flags", True)):
            cmd.extend(self.state_flags())
        if bool(command.get("append_task_flags", True)):
            selection = self.task_selection(command)
            if selection.mode == "selected":
                cmd.extend(repeat_flag(selection.flag, selection.values))
        for repeat in command.get("repeat_flags", []):
            cmd.extend(repeat_flag_from_config(repeat, self.runtime.env))
        cmd.extend(self.runtime.passthrough)
        return cmd

    def bbo_run_command(
        self,
        algorithm: str,
        task: str,
        seed: str,
        generic: Mapping[str, Any],
        sglang_args: Sequence[str],
    ) -> list[str]:
        cmd = ["uv", "run"]
        cmd.extend(self.uv_extra_flags(generic.get("uv_extras", []), algorithm=algorithm, owner=generic))
        cmd.extend(
            [
                "python",
                "-m",
                "bbo.run",
                "--task",
                task,
                "--algorithm",
                algorithm,
                "--seed",
                seed,
                "--max-evaluations",
                self.resolve_config_value(generic, "max_evaluations", algorithm=algorithm),
                "--results-root",
                self.render(str(generic.get("results_root", "{run_root}/generic")), algorithm=algorithm),
            ]
        )
        if bool(generic.get("no_plots", True)):
            cmd.append("--no-plots")
        cmd.extend(self.render_args(generic.get("args", []), algorithm=algorithm))
        cmd.extend(sglang_args)
        if self.resume_enabled():
            cmd.append("--resume")
        cmd.extend(self.runtime.passthrough)
        return cmd

    def uv_extra_flags(
        self,
        extras: Any,
        *,
        algorithm: str | None = None,
        owner: Mapping[str, Any] | None = None,
    ) -> list[str]:
        values = [str(item) for item in list(extras or [])]
        if owner is not None and algorithm is not None:
            values.extend(str(item) for item in owner.get("uv_extras_by_algorithm", {}).get(algorithm, []))
        flags: list[str] = []
        for extra in values:
            flags.extend(["--extra", extra])
        return flags

    def render_args(self, args: Iterable[Any], *, algorithm: str) -> list[str]:
        return [self.render(str(arg), algorithm=algorithm) for arg in args]

    def render(self, value: str, *, algorithm: str | None = None, extra: Mapping[str, str] | None = None) -> str:
        context = self.base_context(algorithm=algorithm)
        if extra:
            context.update(extra)
        try:
            return value.format(**context)
        except KeyError as exc:
            raise SystemExit(f"Missing render key `{exc.args[0]}` for {self.family}: {value}") from exc

    def base_context(self, *, algorithm: str | None = None) -> dict[str, str]:
        env = self.runtime.env
        context = {
            "repo_root": str(self.runtime.repo_root),
            "script_dir": str(self.runtime.script_dir),
            "run_root": str(self.run_root),
            "family": self.family,
            "algorithm": algorithm or "",
            "jobs": str(self.selected_jobs()),
            "sglang_base_url": env["SGLANG_BASE_URL"],
            "sglang_model": env["SGLANG_MODEL"],
            "sglang_timeout_seconds": env["SGLANG_TIMEOUT_SECONDS"],
            "sglang_max_retries": env["SGLANG_MAX_RETRIES"],
            "agent_history_limit": env.get("AGENT_HISTORY_LIMIT", "200"),
            "agent_max_tool_calls": env.get("AGENT_MAX_TOOL_CALLS", "64"),
            "agent_timeout_seconds": env.get("AGENT_TIMEOUT_SECONDS", "900"),
            "agent_max_retries": env.get("AGENT_MAX_RETRIES", "4"),
            "opro_prompt_pairs": env.get("OPRO_PROMPT_PAIRS", "20"),
            "opro_generation_rounds": env.get("OPRO_GENERATION_ROUNDS", "4"),
            "opro_candidates": env.get("OPRO_CANDIDATES", "1"),
            "executor": env.get("EXECUTOR", "process"),
            "service_workers": env.get("SERVICE_WORKERS", "1"),
            "molecule_n_initial_points": env.get("MOLECULE_N_INITIAL_POINTS", "50"),
            "molecule_optimizer_budget": env.get("MOLECULE_OPTIMIZER_BUDGET", "200"),
            "molecule_agent_timeout_seconds": env.get("MOLECULE_AGENT_TIMEOUT_SECONDS", "300"),
        }
        context.update(self.family_context())
        return context

    def family_context(self) -> dict[str, str]:
        return {}

    def resolve_config_value(self, config: Mapping[str, Any], key: str, *, algorithm: str) -> str:
        per_algorithm_env = config.get(f"{key}_env_by_algorithm", {})
        if isinstance(per_algorithm_env, Mapping) and algorithm in per_algorithm_env:
            env_key = str(per_algorithm_env[algorithm])
            if env_key in self.runtime.env:
                return self.runtime.env[env_key]
        per_algorithm = config.get(f"{key}_by_algorithm", {})
        if isinstance(per_algorithm, Mapping) and algorithm in per_algorithm:
            return self.render(str(per_algorithm[algorithm]), algorithm=algorithm)
        env_key = str(config.get(f"{key}_env") or "")
        if env_key and env_key in self.runtime.env:
            return self.runtime.env[env_key]
        default = config.get(key)
        if default is None:
            raise SystemExit(f"{self.family}.{key} must be set for {algorithm}")
        return self.render(str(default), algorithm=algorithm)

    def selected_algorithms(self) -> list[str]:
        selected = self.runtime.algorithms_override
        if selected is None:
            family_key = f"{self.env_prefix}_ALGORITHMS"
            raw = self.runtime.env.get(family_key) or self.runtime.env.get("ALGORITHMS")
            selected = split_env_words(raw) if raw else None
        if selected is None:
            selected = tuple(str(item) for item in self.config.get("default_algorithms", []))
        if not selected:
            raise SystemExit(f"No algorithms selected for {self.family}")
        return list(selected)

    def selected_seeds(self) -> list[str]:
        selected = self.runtime.seeds_override
        if selected is None:
            raw = self.runtime.env.get("SEEDS")
            selected = split_env_words(raw) if raw else None
        if selected is None:
            selected = tuple(str(item) for item in self.config.get("default_seeds", [1]))
        return list(selected)

    def selected_jobs(self) -> int:
        if self.runtime.jobs_override is not None:
            return self.runtime.jobs_override
        raw = self.runtime.env.get("JOBS")
        if raw:
            return int(raw)
        return int(self.config.get("default_jobs", 1))

    def selected_tasks(self, selector: Mapping[str, Any] | None = None) -> list[str]:
        selection = self.task_selection(selector or {})
        if selection.mode == "all":
            return [str(item) for item in self.config.get("tasks", [])]
        return list(selection.values)

    def task_selection(self, selector: Mapping[str, Any]) -> "TaskSelection":
        flag = str(selector.get("task_flag") or self.config.get("task_flag") or "--task")
        selected = self.runtime.tasks_override
        if selected is None:
            env_key = selector.get("task_env") or self.config.get("task_env") or f"{self.env_prefix}_TASKS"
            raw = self.runtime.env.get(str(env_key)) if env_key else None
            if raw is None and self.family != "dbtune":
                raw = self.runtime.env.get("TASKS")
            selected = split_env_words(raw) if raw else None
        if selected is None:
            default = selector.get("default_tasks", self.config.get("default_task_selection", "all"))
            if isinstance(default, str):
                if default == "all":
                    return TaskSelection(mode="all", flag=flag, values=())
                selected = split_env_words(default)
            else:
                selected = tuple(str(item) for item in default)
        if len(selected) == 1 and selected[0] == "all":
            return TaskSelection(mode="all", flag=flag, values=())
        return TaskSelection(mode="selected", flag=flag, values=tuple(selected))

    def command_config(self, algorithm: str) -> Mapping[str, Any] | None:
        commands = self.config.get("commands", {})
        if not isinstance(commands, Mapping):
            return None
        if algorithm in commands:
            command = commands[algorithm]
            return command if isinstance(command, Mapping) else None
        for _, command in commands.items():
            if not isinstance(command, Mapping):
                continue
            aliases = {str(item) for item in command.get("aliases", [])}
            if algorithm in aliases:
                return command
        return None

    def generic_matrix_config(self, algorithm: str) -> Mapping[str, Any] | None:
        generic = self.config.get("generic_bbo_run")
        if not isinstance(generic, Mapping):
            return None
        algorithms = {str(item) for item in generic.get("algorithms", [])}
        if algorithm in algorithms:
            return generic
        return None

    def state_flags(self) -> list[str]:
        flags: list[str] = []
        if self.overwrite_enabled():
            flags.append("--overwrite")
        if self.resume_enabled():
            flags.append("--resume")
        return flags

    def resume_enabled(self) -> bool:
        if self.runtime.resume is not None:
            return bool(self.runtime.resume)
        return self.runtime.env.get("RESUME", "0") == "1"

    def overwrite_enabled(self) -> bool:
        if self.runtime.overwrite is not None:
            return bool(self.runtime.overwrite)
        return self.runtime.env.get("OVERWRITE", "0") == "1"

    def sglang_args(self, profile: str) -> list[str]:
        template = SGLANG_ARG_PROFILES.get(profile, [])
        return [self.render(item, algorithm=profile) for item in template]

    def require_sglang(self) -> None:
        env = self.runtime.env
        key = (env["SGLANG_BASE_URL"], env["SGLANG_MODEL"])
        if self.runtime.dry_run or env.get("SKIP_SGLANG_CHECK", "0") == "1" or key in self.runtime.checked_sglang:
            return
        base_url = env["SGLANG_BASE_URL"].rstrip("/")
        if base_url.endswith("/chat/completions"):
            base_url = base_url.rsplit("/chat/completions", 1)[0]
        models_url = f"{base_url}/models" if base_url.endswith("/v1") else f"{base_url}/v1/models"
        try:
            with urllib.request.urlopen(models_url, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise SystemExit(f"SGLang endpoint is not reachable at {models_url}: {exc}") from exc
        ids = [str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict)]
        if env["SGLANG_MODEL"] not in ids:
            raise SystemExit(f"SGLang model `{env['SGLANG_MODEL']}` not found at {models_url}; available={ids}")
        self.runtime.checked_sglang.add(key)

    def resolve_run_root(self) -> Path:
        if self.runtime.run_root_override:
            return Path(self.runtime.run_root_override).expanduser().resolve()
        raw = self.runtime.env.get(f"{self.env_prefix}_RUN_ROOT") or self.runtime.env.get("RUN_ROOT")
        if raw:
            return Path(raw).expanduser().resolve()
        prefix = str(self.config.get("run_root_prefix") or f"sglang_{self.family}")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return self.runtime.repo_root / "workflow" / "script_runs" / f"{prefix}_{timestamp}"

    def exec(self, cmd: Sequence[str]) -> None:
        if self.runtime.dry_run:
            print("+ " + " ".join(shlex.quote(str(item)) for item in cmd), flush=True)
            return
        subprocess.run(list(cmd), cwd=self.runtime.repo_root, env=self.runtime.env, check=True)


@dataclass(frozen=True)
class TaskSelection:
    mode: str
    flag: str
    values: tuple[str, ...]


class SyntheticAdapter(FamilyAdapter):
    pass


class MoleculeAdapter(FamilyAdapter):
    def run_bbo_matrix(self, algorithm: str, generic: Mapping[str, Any]) -> None:
        if bool(generic.get("prepare_smiles_init", False)):
            self.prepare_smiles_init()
        super().run_bbo_matrix(algorithm, generic)

    def family_context(self) -> dict[str, str]:
        env = self.runtime.env
        csv_path = env.get("SMILES_INIT_SOURCE") or env.get("SMILES_INIT_CSV") or str(self.config.get("smiles_init_csv"))
        csv_path = csv_path.format(
            repo_root=str(self.runtime.repo_root),
            script_dir=str(self.runtime.script_dir),
            family=self.family,
        )
        txt_template = str(self.config.get("smiles_init_txt"))
        txt_path = env.get("SMILES_INIT_TXT") or txt_template.format(
            repo_root=str(self.runtime.repo_root),
            script_dir=str(self.runtime.script_dir),
            family=self.family,
        )
        return {
            "smiles_init_csv": csv_path,
            "smiles_init_txt": txt_path,
        }

    def prepare_smiles_init(self) -> None:
        if self.runtime.dry_run:
            return
        ctx = self.family_context()
        src = Path(ctx["smiles_init_csv"]).expanduser()
        dst = Path(ctx["smiles_init_txt"]).expanduser()
        if dst.exists() and dst.stat().st_size > 0 and self.runtime.env.get("REGENERATE_SMILES_INIT", "0") != "1":
            return
        if not src.exists():
            raise SystemExit(f"SMILES init CSV not found: {src}")
        seen: set[str] = set()
        values: list[str] = []
        with src.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "smile" not in reader.fieldnames:
                raise SystemExit(f"{src} must contain a `smile` column")
            for row in reader:
                smiles = (row.get("smile") or "").strip()
                if not smiles or smiles in seen:
                    continue
                seen.add(smiles)
                values.append(smiles)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("\n".join(values) + "\n", encoding="utf-8")
        print(f"wrote {len(values)} SMILES to {dst}", flush=True)


class BBOPlaceAdapter(FamilyAdapter):
    def family_context(self) -> dict[str, str]:
        return {"bboplace_base_urls": self.resolve_bboplace_base_urls()}

    def resolve_bboplace_base_urls(self) -> str:
        raw = self.runtime.env.get("BBOPLACE_BASE_URLS")
        if raw:
            return raw
        urls = [str(item).rstrip("/") for item in self.config.get("default_base_urls", [])]
        if self.runtime.dry_run:
            return urls[0] if urls else "http://127.0.0.1:8270"
        reachable: list[str] = []
        for url in urls:
            if probe_health(url):
                reachable.append(url)
        if not reachable:
            raise SystemExit("No BBOPlace evaluator /health endpoint is reachable. Set BBOPLACE_BASE_URLS.")
        return ",".join(reachable)


class DBTuneAdapter(FamilyAdapter):
    def selected_tasks(self, selector: Mapping[str, Any] | None = None) -> list[str]:
        selection = self.task_selection(selector or {})
        if selection.mode == "selected":
            return list(selection.values)
        tasks_by_family = self.config.get("tasks_by_family", {})
        if not isinstance(tasks_by_family, Mapping):
            raise SystemExit("DBTune config requires [tasks_by_family].")
        raw_families = self.runtime.env.get("DBTUNE_FAMILIES")
        families = split_env_words(raw_families) if raw_families else tuple(str(item) for item in self.config.get("default_families", []))
        tasks: list[str] = []
        for family in families:
            aliases = {
                "surrogate": "surrogate",
                "dbtune_surrogate": "surrogate",
                "dbtune_surrogate_service": "surrogate",
            }
            key = aliases.get(family)
            if not key or key not in tasks_by_family:
                raise SystemExit(f"Unknown DBTune family: {family}")
            tasks.extend(str(item) for item in tasks_by_family[key])
        return tasks

    def run_bbo_matrix(self, algorithm: str, generic: Mapping[str, Any]) -> None:
        self.runtime.env.setdefault(
            "BBO_UV_EXTRA_ARGS",
            str(generic.get("bbo_uv_extra_args", "--extra bo-tutorial --extra nanobot --extra pablo --extra optuna --extra surrogate")),
        )
        self.runtime.env.setdefault("DBTUNE_SYSBENCH_TIME_SEC", str(self.config.get("sysbench_time_sec", 180)))
        profile = str(generic.get("sglang_args_profile") or algorithm)
        sglang_args = self.sglang_args(profile)
        if sglang_args:
            self.require_sglang()
        for task in self.selected_tasks(generic):
            for seed in self.selected_seeds():
                cmd = [
                    "bash",
                    str(self.runtime.repo_root / "scripts" / "sglang" / "dbtune" / "run_problem.sh"),
                    task,
                    algorithm,
                    "--seed",
                    seed,
                    "--max-evaluations",
                    self.resolve_config_value(generic, "max_evaluations", algorithm=algorithm),
                    "--results-root",
                    str(self.run_root),
                ]
                if bool(generic.get("no_plots", True)):
                    cmd.append("--no-plots")
                cmd.extend(sglang_args)
                if self.resume_enabled():
                    cmd.append("--resume")
                cmd.extend(self.runtime.passthrough)
                self.exec(cmd)


def repeat_flag(flag: str, values: Iterable[str]) -> list[str]:
    args: list[str] = []
    for value in values:
        args.extend([flag, str(value)])
    return args


def repeat_flag_from_config(config: Mapping[str, Any], env: Mapping[str, str]) -> list[str]:
    flag = str(config["flag"])
    raw = env.get(str(config.get("env") or ""))
    if raw:
        values = split_env_words(raw)
    else:
        default = config.get("default", [])
        if isinstance(default, str):
            values = split_env_words(default)
        else:
            values = tuple(str(item) for item in default)
    return repeat_flag(flag, values)


def split_env_words(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item for item in value.replace(",", " ").split() if item)


def probe_health(url: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=2) as response:
            return 200 <= response.status < 500
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
