"""Attempt configuration, run manifests and trace recording for codex_lite runs."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from codex_lite import CODEX_LITE_TOKEN_CAP, TEMPERATURE as CODEX_LITE_TEMPERATURE, TOP_P as CODEX_LITE_TOP_P
from loop_driver import RUNNABLE_POLICY_IDS, solve_puzzle
from trace_store import JsonlTraceStore, fingerprint_payload


@dataclass(frozen=True)
class RolloutConfig:
    model: str
    seed_base: int
    policies: tuple[str, ...]
    time_budget: float
    tokenizer_path: str
    context_window: int | None
    request_timeout: float | None
    # Every request's sampling; the Kaggle launcher keeps codex_lite's defaults.
    temperature: float = CODEX_LITE_TEMPERATURE
    top_p: float = CODEX_LITE_TOP_P
    # (token id, additive logit bias) pairs; SGLang adds them before temperature, also when verifying drafts.
    logit_bias: tuple[tuple[int, float], ...] = ()
    # Lowers each attempt's completion budget (the launcher's --teacher-token-cap).
    teacher_token_cap: int | None = None

    def __post_init__(self) -> None:
        if not self.policies:
            raise ValueError("at least one rollout policy is required")
        unknown = set(self.policies) - RUNNABLE_POLICY_IDS
        if unknown:
            raise ValueError(f"unknown rollout policies: {sorted(unknown)}")
        if not self.tokenizer_path:
            raise ValueError("tokenizer_path is required by codex_lite")
        if not 0 < self.temperature <= 2 or not 0 < self.top_p <= 1:
            raise ValueError("temperature must lie in (0, 2] and top_p in (0, 1]")
        if any(type(token) is not int or token < 0 or not -100 <= bias <= 100 for token, bias in self.logit_bias):
            raise ValueError("logit bias needs nonnegative integer token ids and biases in [-100, 100]")
        cap = self.teacher_token_cap
        if cap is not None and (type(cap) is not int or not 1 <= cap <= CODEX_LITE_TOKEN_CAP):
            raise ValueError(f"teacher token cap must be an integer in [1, {CODEX_LITE_TOKEN_CAP}]")

    @property
    def samples(self) -> int:
        return len(self.policies)


def build_puzzle(challenge: dict[str, Any]) -> dict[str, Any]:
    return {
        "train": challenge["train"],
        "test": [{"input": item["input"]} for item in challenge["test"]],
    }


def order_task_keys(challenge_keys: set[str], requested: set[str] | None = None,
                    limit: int = 0) -> list[str]:
    selected = challenge_keys if requested is None else requested
    missing = sorted(selected - challenge_keys)
    if missing:
        raise ValueError("unknown task keys: " + ",".join(missing))
    ordered = sorted(selected)
    return ordered[:limit] if limit else ordered


def build_run_manifest(config: RolloutConfig, challenges_path: Path,
                       harness_paths: list[Path],
                       scheduler: dict[str, Any] | None = None) -> dict[str, Any]:
    rollout = asdict(config)
    if config.teacher_token_cap is None:
        rollout.pop("teacher_token_cap")
    if not config.logit_bias:
        rollout.pop("logit_bias")
    manifest = {
        "schema_version": 1,
        "trace_schema_version": 3,
        "rollout": rollout,
        "challenges_sha256": sha256_file(challenges_path),
        "harness_sha256": {
            path.name: sha256_file(path)
            for path in harness_paths
        },
    }
    if scheduler is not None:
        manifest["scheduler"] = scheduler
    return manifest


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a potentially multi-gigabyte artifact without copying it into RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class AttemptRecorder:
    """Persist the attempt rows emitted by ``solve_puzzle``."""

    def __init__(self, store: JsonlTraceStore, run_manifest: dict[str, Any]):
        self.store = store
        self.run_manifest = run_manifest

    def callback_for(self, task_id: str) -> Callable[[dict[str, Any]], None]:
        def persist(row: dict[str, Any]) -> None:
            trace_id = self.logical_trace_id(
                task_id,
                row["sample"],
                row["seed"],
                row["policy"],
            )
            self.store.append_attempt(trace_id, {**row, "task_id": task_id, "run": self.run_manifest})
        return persist

    def logical_trace_id(
        self, task_id: str, sample: int, seed: int, policy: str
    ) -> str:
        return f"{self.store.run_fingerprint}:{task_id}:{sample}:{seed}:{policy}"


def run_fingerprint(run_manifest: dict[str, Any]) -> str:
    """Fingerprint attempt semantics while allowing scheduler-only resume."""
    identity = {
        key: value
        for key, value in run_manifest.items()
        if key != "scheduler"
    }
    return fingerprint_payload(identity)


def create_attempt_recorder(trace_path: Path, run_manifest: dict[str, Any]) -> AttemptRecorder:
    store = JsonlTraceStore(trace_path, run_fingerprint(run_manifest))
    print(f"[trace] raw attempts -> {trace_path} | retained={store.retained}", flush=True)
    return AttemptRecorder(store, run_manifest)


def run_task_rollouts(client, config: RolloutConfig, task_id: str,
                      challenge: dict[str, Any],
                      trace_callback: Callable[[dict[str, Any]], None] | None = None,
                      attempt_limiter: Any | None = None,
                      sample_indices: list[int] | None = None,
                      provisional_callback: Callable[[int, list], None] | None = None) -> dict:
    return solve_puzzle(
        client,
        config.model,
        build_puzzle(challenge),
        policies=list(config.policies),
        time_budget=config.time_budget,
        seed_base=config.seed_base,
        label=task_id,
        tokenizer_path=config.tokenizer_path,
        context_window=config.context_window,
        request_timeout=config.request_timeout,
        trace_callback=trace_callback,
        attempt_limiter=attempt_limiter,
        sample_indices=sample_indices,
        provisional_callback=provisional_callback,
        temperature=config.temperature,
        top_p=config.top_p,
        logit_bias={str(token): bias for token, bias in config.logit_bias} or None,
        completion_token_cap=config.teacher_token_cap,
    )
