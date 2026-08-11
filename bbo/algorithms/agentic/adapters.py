"""Adapters that place existing ask/tell optimizers behind AgenticPolicy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ...core import Algorithm, Incumbent, TrialObservation
from .events import DeliberationEvent, DeliberationEventWriter
from .protocol import CommitCandidate, OptimizationContext, PolicyDecision


class AlgorithmPolicyAdapter:
    """Reuse an existing Algorithm with no method-specific control-flow rewrite."""

    def __init__(
        self,
        algorithm: Algorithm,
        *,
        method: str | None = None,
        run_dir: Path | str | None = None,
    ) -> None:
        self.algorithm = algorithm
        self._name = method or algorithm.name
        self.run_dir = None if run_dir is None else Path(run_dir)
        self._writer: DeliberationEventWriter | None = None
        self._evaluation_index = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def artifact_paths(self) -> dict[str, str]:
        paths = dict(getattr(self.algorithm, "artifact_paths", {}))
        if self._writer is not None:
            paths["deliberation_events_jsonl"] = str(self._writer.path)
        return paths
    @property
    def routing_table(self) -> dict[str, str]:
        return dict(getattr(self.algorithm, "routing_table", {}))


    def setup(self, context: OptimizationContext) -> None:
        setup_kwargs: dict[str, Any] = {}
        if context.description is not None:
            setup_kwargs["task_description"] = context.description
        if self.run_dir is not None:
            setup_kwargs["run_dir"] = self.run_dir
            self._writer = DeliberationEventWriter(self.run_dir / "deliberation_events.jsonl")
        self.algorithm.setup(context.task_spec, seed=context.seed, **setup_kwargs)
        self._evaluation_index = context.evaluation_index
        self._emit("context", {"history_size": len(context.history)})

    def deliberate(self, context: OptimizationContext) -> PolicyDecision:
        self._evaluation_index = context.evaluation_index
        suggestion = self.algorithm.ask()
        self._emit_tool_events(suggestion.metadata)
        source = self._source_metadata(suggestion.metadata)
        self._emit("propose", source)
        self._emit("commit", {"config": suggestion.config, **source})
        return CommitCandidate(
            config=dict(suggestion.config),
            metadata=dict(suggestion.metadata),
            budget=suggestion.budget,
        )

    def observe(self, observation: TrialObservation) -> None:
        self.algorithm.tell(observation)
        self._emit(
            "observation",
            {
                "status": observation.status.value,
                "objectives": observation.objectives,
                "trial_id": observation.suggestion.trial_id,
            },
        )
        self._evaluation_index += 1

    def incumbents(self) -> list[Incumbent]:
        return self.algorithm.incumbents()

    def snapshot(self) -> dict[str, Any]:
        snapshot = getattr(self.algorithm, "snapshot", None)
        if callable(snapshot):
            return dict(snapshot())
        return {"evaluation_index": self._evaluation_index}

    def restore(self, state: Mapping[str, Any]) -> None:
        restore = getattr(self.algorithm, "restore", None)
        if callable(restore):
            restore(dict(state))
        self._evaluation_index = int(state.get("evaluation_index", self._evaluation_index))

    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        if self._writer is None:
            return
        self._writer.append(
            DeliberationEvent(
                method=self.name,
                kind=kind,
                evaluation_index=self._evaluation_index,
                payload=payload,
            )
        )

    def _emit_tool_events(self, metadata: Mapping[str, Any]) -> None:
        """Normalize optimizer tool calls produced inside one agent ask."""

        call_id = metadata.get("agent_call_id")
        path_value = self.artifact_paths.get("agent_tool_calls_jsonl")
        if not call_id or not path_value or not Path(path_value).exists():
            return
        mapping = {
            "optimizer_predict": "probe",
            "optimizer_score": "probe",
            "optimizer_diagnostics": "probe",
            "optimizer_status": "probe",
            "optimizer_suggest": "propose",
            "optimizer_portfolio_suggest": "propose",
            "optimizer_set_backend": "reconfigure",
            "optimizer_set_bounds": "reconfigure",
            "optimizer_set_acquisition": "reconfigure",
            "optimizer_reset_policy": "reconfigure",
        }
        for line in Path(path_value).read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            record_call_id = record.get("agent_call_id") or record.get("call_id")
            kind = mapping.get(record.get("tool_name"))
            if record_call_id == call_id and kind and record.get("success", True):
                self._emit(kind, {
                    "tool_name": record["tool_name"],
                    "arguments": record.get("arguments", {}),
                })


    def _source_metadata(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        if self.name == "llambo":
            return {
                "phase": metadata.get("llambo_phase"),
                "backend": metadata.get("llambo_backend"),
                "candidate_count": metadata.get("llambo_candidate_count"),
            }
        if self.name == "pablo":
            role = metadata.get("pablo_role")
            if role:
                self._emit("role_call", {"role": role, "source": metadata.get("pablo_source")})
            return {
                "role": role,
                "source": metadata.get("pablo_source"),
                "round": metadata.get("pablo_round"),
            }
        return {"source": metadata.get("agent_source")}


__all__ = ["AlgorithmPolicyAdapter"]
