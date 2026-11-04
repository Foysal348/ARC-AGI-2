"""Run codex_lite attempts coverage-first and write an ARC Kaggle submission.

This module owns Kaggle's challenge/submission boundary and the two-attempt schedule. The
attempt loop, voting and admission live in ``codex_lite``, ``loop_driver``,
``rollout_runner`` and ``adaptive_concurrency``; local runs use the same modules.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptive_concurrency import (
    add_adaptive_concurrency_arguments,
    local_adaptive_runtime,
)
from rollout_runner import (
    RolloutConfig,
    order_task_keys,
    run_task_rollouts,
    build_run_manifest,
    create_attempt_recorder,
)
from codex_lite import (CODEX_LITE_CAP12K_POLICY, CODEX_LITE_CARRY_POLICY, CODEX_LITE_CARRYCAP_POLICY,
                        CODEX_LITE_IMAGES_POLICY, CODEX_LITE_NOFILLER_POLICY, CODEX_LITE_POLICY,
                        TEMPERATURE as CODEX_LITE_TEMPERATURE, TOP_P as CODEX_LITE_TOP_P)
from loop_driver import RUNNABLE_POLICY_TOKEN_CEILINGS

ATTEMPT_MODES = {
    # v21's Kaggle policy: a second attempt after an unverified first one carries its work.
    "codex_lite_carry": (CODEX_LITE_CARRY_POLICY, CODEX_LITE_CARRY_POLICY),
    # v22: carry plus a -1 logit bias on "Hmm" and "Alternatively" (codex_lite.CODEX_LITE_LOGIT_BIASES).
    "codex_lite_carry_nofiller": (CODEX_LITE_NOFILLER_POLICY, CODEX_LITE_NOFILLER_POLICY),
    # v27: v22 with the per-request output cap at 12,288 (codex_lite.CODEX_LITE_REQUEST_CAPS).
    "codex_lite_carry_nofiller_cap12k": (CODEX_LITE_CAP12K_POLICY, CODEX_LITE_CAP12K_POLICY),
    # Candidate (2026-10-27): v27 with 4,096 on a carried attempt's first request.
    "codex_lite_cap12k_carrycap": (CODEX_LITE_CARRYCAP_POLICY, CODEX_LITE_CARRYCAP_POLICY),
    # Kaggle since v28: v27 whose task message also shows the grids as images (needs the vision tower).
    "codex_lite_cap12k_images": (CODEX_LITE_IMAGES_POLICY, CODEX_LITE_IMAGES_POLICY),
    # v14-v20's policy: every second attempt starts fresh. The rollback and the carry baseline.
    "codex_lite": (CODEX_LITE_POLICY, CODEX_LITE_POLICY),
}


def parse_logit_bias(text: str) -> tuple[tuple[int, float], ...]:
    """``--logit-bias``'s JSON object as sorted (token id, bias) pairs."""
    value = json.loads(text or "{}")
    if not isinstance(value, dict):
        raise ValueError("--logit-bias must be a JSON object of token id to bias")
    return tuple(sorted((int(token), float(bias)) for token, bias in value.items()))


def policies_for_mode(mode: str) -> tuple[str, str]:
    try:
        return ATTEMPT_MODES[mode]
    except KeyError as exc:
        raise ValueError(
            f"unknown attempt mode {mode!r}; choose from {sorted(ATTEMPT_MODES)}"
        ) from exc

FALLBACK_GRID = [[0]]
# The runner's own modules: hashed into every trace manifest and shipped by the Kaggle runtime builder.
HARNESS_FILES = ("kaggle_runner.py", "rollout_runner.py", "loop_driver.py", "trace_store.py", "adaptive_concurrency.py",
                 "codex_lite.py", "codex_lite_interface.py", "arc_tools.py")


@dataclass(frozen=True)
class AttemptPhase:
    """One global phase in the fixed two-attempt Kaggle schedule."""

    name: str
    policy: str
    sample_index: int
    attempt_index: int


def attempt_phases(attempt_mode: str, *, primary_only: bool = False) -> tuple[AttemptPhase, ...]:
    """Return a coverage-first schedule for the selected two-attempt policy."""
    policies = policies_for_mode(attempt_mode)
    phases = (
        AttemptPhase("primary", policies[0], 0, 1),
        AttemptPhase("secondary", policies[1], 1, 2),
    )
    return phases[:1] if primary_only else phases


def primary_verified(result: dict[str, Any] | None) -> bool:
    """Whether a first attempt finished with a program that passed every visible training pair and an
    output for every test input (codex_lite carries the same attempts).

    On the public tasks such an attempt was correct 83 times in 100 (73 in 102 with REAP v2).
    With REAP v2 a second attempt after an unverified one gave 4.6 solves per million tokens,
    and after a verified one 2.6. A program passing training but raising on a test input
    (gpu-h200 cap12k `f560132c`, 2026-10-27) left no answer and counted as verified until 2026-10-29.
    """
    if not result or result.get("error") is not None:
        return False
    attempts = (result.get("stats") or {}).get("attempts") or []
    return any(attempt.get("n_train") and attempt.get("best_n_passed", 0) >= attempt["n_train"]
               and attempt.get("outputs") and all(grid is not None for grid in attempt["outputs"])
               for attempt in attempts)


class CoverageFirstAdmission:
    """Admit every first attempt before any second attempt.

    Second attempts start as soon as workers free up, but the adaptive limiter wakes every
    waiter at once, so a waiting second attempt could otherwise take a permit that a first
    attempt is still waiting for. A first attempt counts once: when it is admitted, or when it
    ends without admission.
    """

    def __init__(self, limiter: Any, first_attempts: int) -> None:
        self.limiter, self.waiting_first = limiter, first_attempts
        self.condition = threading.Condition()

    def permit(self, first: bool) -> "AdmissionPermit":
        return AdmissionPermit(self, first)

    def first_settled(self) -> None:
        with self.condition:
            self.waiting_first -= 1
            self.condition.notify_all()

    def await_first_attempts(self) -> None:
        with self.condition:
            self.condition.wait_for(lambda: self.waiting_first <= 0)


class AdmissionPermit:
    """The acquire/release pair that one attempt passes to ``solve_puzzle``."""

    def __init__(self, admission: CoverageFirstAdmission, first: bool) -> None:
        self.admission, self.first, self.settled = admission, first, False

    def acquire(self, timeout: float | None = None) -> None:
        if not self.first:
            self.admission.await_first_attempts()
        try:
            self.admission.limiter.acquire(timeout)
        finally:
            self.settle()

    def release(self, tokens: int) -> None:
        self.admission.limiter.release(tokens)

    def settle(self) -> None:
        if self.first and not self.settled:
            self.settled = True
            self.admission.first_settled()


def write_json(path: Path, payload: Any) -> None:
    """Atomically replace JSON so a kernel timeout leaves a valid submission."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def empty_submission(challenges: dict[str, dict]) -> dict[str, list[dict[str, list]]]:
    return {
        task_id: [
            {"attempt_1": FALLBACK_GRID, "attempt_2": FALLBACK_GRID}
            for _ in challenge["test"]
        ]
        for task_id, challenge in challenges.items()
    }


def is_valid_arc_grid(grid: Any) -> bool:
    """Return whether ``grid`` is a schema-valid ARC output grid."""
    if not isinstance(grid, list) or not 1 <= len(grid) <= 30:
        return False
    width = len(grid[0]) if isinstance(grid[0], list) else 0
    return bool(width and width <= 30) and all(
        isinstance(row, list)
        and len(row) == width
        and all(
            isinstance(cell, int) and not isinstance(cell, bool) and 0 <= cell <= 9
            for cell in row
        )
        for row in grid
    )


def apply_task_attempt_result(
    submission: dict[str, list[dict[str, list]]],
    task_id: str,
    rollout_submission: list[list[list[list[int]]]],
    *,
    attempt_index: int,
    mirror_missing: bool,
    protected_tests: set[int] | None = None,
    alternates: dict[int, list] | None = None,
) -> set[int]:
    """Persist one independently completed arm into a fixed Kaggle slot.

    The primary arm is mirrored into both slots immediately. When a secondary
    arm finishes after a missing primary, it is mirrored in the other direction.
    A second candidate from the same arm (a codex_lite alternate) fills the
    mirrored slot instead of a copy. ``protected_tests`` prevents a secondary
    result from replacing the primary prediction that already occupies the other
    slot; if the secondary repeats that prediction, the first differing grid of
    ``alternates[test]`` (the primary's alternates) or of its own candidates
    takes its slot instead.
    """
    if attempt_index not in (1, 2):
        raise ValueError("attempt_index must be 1 or 2")
    expected = len(submission[task_id])
    if len(rollout_submission) != expected:
        raise ValueError(
            f"{task_id}: rollout returned {len(rollout_submission)} test outputs; "
            f"expected {expected}"
        )

    protected = protected_tests or set()
    updated: set[int] = set()
    other_index = 2 if attempt_index == 1 else 1
    for test_index, candidates in enumerate(rollout_submission):
        valid = [candidate for candidate in candidates if is_valid_arc_grid(candidate)]
        if not valid:
            continue
        grid, slots = valid[0], submission[task_id][test_index]
        if test_index in protected and grid == slots[f"attempt_{other_index}"]:
            grid = next((candidate for candidate in [*(alternates or {}).get(test_index, []), *valid[1:]]
                         if is_valid_arc_grid(candidate) and candidate != grid), grid)
        slots[f"attempt_{attempt_index}"] = grid
        if mirror_missing and test_index not in protected:
            slots[f"attempt_{other_index}"] = next((candidate for candidate in valid[1:] if candidate != grid), grid)
        updated.add(test_index)
    return updated


class SubmissionSlots:
    """The runner's slot state: each arm's candidates go into its fixed slot (``apply_task_attempt_result``).

    The primary arm (attempt 1) records which tests it filled and its second candidates, which the
    secondary arm then respects. ``compare_runs`` replays stored attempts through the same object.
    """

    def __init__(self, submission: dict[str, list[dict[str, list]]]) -> None:
        self.submission = submission
        self.primary_valid_tests: dict[str, set[int]] = {task_id: set() for task_id in submission}
        # The primary arm's second candidates, offered when the secondary repeats the primary answer.
        self.primary_alternates: dict[str, dict[int, list]] = {task_id: {} for task_id in submission}

    def apply(self, task_id: str, attempt_index: int, rollout_submission: list) -> None:
        secondary = attempt_index == 2
        updated = apply_task_attempt_result(
            self.submission, task_id, rollout_submission, attempt_index=attempt_index, mirror_missing=True,
            protected_tests=self.primary_valid_tests[task_id] if secondary else None,
            alternates=self.primary_alternates[task_id] if secondary else None)
        if not secondary:
            self.primary_valid_tests[task_id].update(updated)
            self.primary_alternates[task_id] = {index: list(candidates[1:])
                                               for index, candidates in enumerate(rollout_submission)}


def build_parser() -> argparse.ArgumentParser:
    """Build the intentionally narrow CLI for the submitted execution path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="inducer")
    parser.add_argument("--seed-base", type=int, default=1234)
    parser.add_argument("--time-budget", type=float, default=600.0)
    parser.add_argument("--context-window", type=int, default=262144)
    parser.add_argument("--request-timeout", type=float, default=0)
    parser.add_argument("--temperature", type=float, default=CODEX_LITE_TEMPERATURE,
                        help="sampling temperature of every request (default codex_lite's)")
    parser.add_argument("--top-p", type=float, default=CODEX_LITE_TOP_P,
                        help="nucleus sampling of every request (default codex_lite's); top-k stays 20")
    parser.add_argument("--logit-bias", default="",
                        help='JSON object of token id to additive logit bias for every request, e.g. \'{"77264": -1}\'')
    parser.add_argument("--teacher-token-cap", type=int,
                        help="lower the total completion-token cap for each codex_lite attempt")
    parser.add_argument("--primary-only", action="store_true",
                        help="run only the first attempt per puzzle for a public smoke test")
    parser.add_argument("--secondary-unverified-first", action="store_true",
                        help="whenever a worker frees up for a second attempt, prefer a puzzle whose "
                             "finished first attempt did not pass every visible training pair (it "
                             "errored, or no program verified)")
    parser.add_argument("--trace-out", type=Path,
                        help="append intact physical attempts for independent post-verification")
    parser.add_argument(
        "--attempt-mode",
        choices=tuple(ATTEMPT_MODES),
        default="codex_lite_carry",
        help="two-attempt schedule (the Kaggle launcher passes codex_lite_cap12k_images; codex_lite starts every "
             "second attempt fresh)",
    )
    parser.add_argument(
        "--tokenizer-path",
        required=True,
        help="local Qwen3.8 model/tokenizer directory; every request's prompt IDs are checked",
    )
    add_adaptive_concurrency_arguments(parser)
    parser.add_argument("--challenges", type=Path, required=True)
    parser.add_argument("--submission", type=Path, default=Path("submission.json"))
    parser.add_argument("--progress", type=Path, default=Path("inducer_progress.json"))
    parser.add_argument("--concurrency", type=int, default=8, help="parallel puzzle workers")
    parser.add_argument("--limit", type=int, default=0, help="first N selected puzzles")
    parser.add_argument("--keys", default="", help="comma-separated puzzle IDs")
    return parser


def submission_rollout_config(args: argparse.Namespace) -> RolloutConfig:
    """Build the exact two-attempt policy schedule selected for submission."""
    return RolloutConfig(
        model=args.model,
        seed_base=args.seed_base,
        policies=policies_for_mode(args.attempt_mode),
        time_budget=args.time_budget,
        tokenizer_path=args.tokenizer_path,
        context_window=args.context_window,
        request_timeout=args.request_timeout or None,
        temperature=args.temperature,
        top_p=args.top_p,
        logit_bias=parse_logit_bias(args.logit_bias),
        teacher_token_cap=args.teacher_token_cap,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if (
        args.concurrency < 1
        or args.time_budget <= 0
        or args.context_window <= 0
        or args.request_timeout < 0
    ):
        parser.error(
            "concurrency, time-budget, and context-window must be positive; "
            "request-timeout must be nonnegative"
        )
    try:
        rollout_config = submission_rollout_config(args)
        phases = attempt_phases(args.attempt_mode, primary_only=args.primary_only)
        attempt_limiter, load_monitor, task_workers, scheduler_manifest = (
            local_adaptive_runtime(
                args,
                fallback_cost=max(
                    RUNNABLE_POLICY_TOKEN_CEILINGS[phase.policy] for phase in phases
                ),
            )
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.secondary_unverified_first:
        # Orders puzzles, not attempt semantics, so resumed traces keep their fingerprint.
        scheduler_manifest = {**(scheduler_manifest or {}), "secondary_unverified_first": True}

    challenges = json.loads(args.challenges.read_text())
    requested = {key.strip() for key in args.keys.split(",") if key.strip()} or None
    try:
        keys = order_task_keys(
            set(challenges),
            requested=requested,
            limit=0 if requested else args.limit,
        )
    except ValueError as exc:
        parser.error(str(exc))

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    submission = empty_submission(challenges)
    write_json(args.submission, submission)
    # Attempt threads persist provisional outputs; the main thread persists finished attempts.
    submission_lock = threading.Lock()
    slots = SubmissionSlots(submission)
    results: dict[str, dict[str, Any]] = {}
    started = time.time()
    recorder = None
    if args.trace_out:
        harness = Path(__file__).parent
        manifest = build_run_manifest(
            rollout_config, args.challenges,
            [harness / name for name in HARNESS_FILES],
            scheduler=scheduler_manifest,
        )
        if args.primary_only:
            manifest["attempt_phases"] = [phase.name for phase in phases]
        recorder = create_attempt_recorder(args.trace_out, manifest)

    policy_schedule = list(rollout_config.policies)
    print(
        f"[kaggle-inducer] selected={len(keys)}/{len(challenges)} "
        f"attempt_mode={args.attempt_mode} "
        f"policies={policy_schedule} task_workers={task_workers} "
        f"phases={[f'{phase.name}:{phase.policy}' for phase in phases]} "
        f"scheduler={scheduler_manifest}",
        flush=True,
    )

    def persist(task_id: str, phase: AttemptPhase, rollout_submission: list) -> None:
        """Fill this arm's slot (mirroring as for finished arms) and save at once."""
        with submission_lock:
            slots.apply(task_id, phase.attempt_index, rollout_submission)
            write_json(args.submission, submission)

    first_phase = phases[0]
    second_phase = phases[1] if len(phases) > 1 else None
    admission = (CoverageFirstAdmission(attempt_limiter, len(keys))
                 if attempt_limiter is not None and second_phase is not None else None)

    def run_one(task_id: str, phase: AttemptPhase) -> tuple[str, dict[str, Any]]:
        task_started = time.time()
        permit = admission.permit(phase is first_phase) if admission is not None else None
        try:
            result = run_task_rollouts(
                client,
                rollout_config,
                task_id,
                challenges[task_id],
                attempt_limiter=permit if permit is not None else attempt_limiter,
                sample_indices=[phase.sample_index],
                # An attempt still running at the notebook cutoff keeps its saved answer.
                provisional_callback=lambda _sample, outputs: persist(
                    task_id, phase, [[grid] for grid in outputs]
                ),
                **({"trace_callback": recorder.callback_for(task_id)} if recorder else {}),
            )
            return task_id, {
                "stats": result["stats"],
                "rollout_submission": result["submission"],
                "wall": round(time.time() - task_started, 3),
                "error": None,
            }
        except Exception as exc:
            traceback.print_exc()
            return task_id, {
                "stats": {},
                "wall": round(time.time() - task_started, 3),
                "error": repr(exc),
            }
        finally:
            if permit is not None:
                permit.settle()

    phase_done = {phase.name: 0 for phase in phases}
    completed_total = 0
    total_work = len(keys) * len(phases)
    waiting_first, waiting_second = list(keys), (list(keys) if second_phase is not None else [])

    def next_attempt() -> tuple[str, AttemptPhase] | None:
        """Every first attempt in task order, then second attempts as workers free up.

        There is no barrier between the two: a worker freed by a first attempt starts a second
        attempt while the last first attempts still run. A puzzle's second attempt waits only
        for its own first attempt, so the first answer and its mirror are saved before a second
        answer can take ``attempt_2``.
        """
        if waiting_first:
            return waiting_first.pop(0), first_phase
        ready = [key for key in waiting_second if first_phase.name in results.get(key, {})]
        if args.secondary_unverified_first:
            ready = [key for key in ready if not primary_verified(results[key][first_phase.name])] or ready
        if not ready:
            return None
        waiting_second.remove(ready[0])
        return ready[0], second_phase

    print(f"[kaggle-inducer] starting {first_phase.name} phase ({len(keys)} puzzle attempts)", flush=True)
    if load_monitor is not None:
        load_monitor.start()
    try:
        with ThreadPoolExecutor(max_workers=task_workers) as executor:
            running: dict[Future, AttemptPhase] = {}

            def fill() -> None:
                while len(running) < task_workers and (attempt := next_attempt()) is not None:
                    task_id, phase = attempt
                    if phase is second_phase and len(waiting_second) == len(keys) - 1:  # The first one.
                        unverified = sum(not primary_verified(result.get(first_phase.name))
                                         for result in results.values())
                        print(f"[kaggle-inducer] starting {phase.name} attempts as workers free up "
                              f"({phase_done[first_phase.name]}/{len(keys)} first attempts done, "
                              f"{unverified} unverified)", flush=True)
                    running[executor.submit(run_one, task_id, phase)] = phase

            fill()
            while running:
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in finished:
                    phase = running.pop(future)
                    task_id, result = future.result()
                    rollout_submission = result.pop("rollout_submission", None)
                    results.setdefault(task_id, {})[phase.name] = result
                    phase_done[phase.name] += 1
                    completed_total += 1

                    # The submission is replaced before progress metadata. A hard
                    # kernel stop therefore retains every independently completed
                    # valid arm, including a mirrored primary prediction.
                    if rollout_submission is not None:
                        persist(task_id, phase, rollout_submission)
                    write_json(args.progress, {
                        "partial": completed_total < total_work,
                        "phase": phase.name,
                        "phase_done": phase_done[phase.name],
                        "completed_attempts": completed_total,
                        "total_attempts": total_work,
                        "selected": len(keys),
                        "total_challenges": len(challenges),
                        "elapsed_seconds": round(time.time() - started, 3),
                        "policies": policy_schedule,
                        "attempt_mode": args.attempt_mode,
                        "scheduler": scheduler_manifest,
                        "adaptive_concurrency": (
                            attempt_limiter.snapshot() if attempt_limiter is not None else None
                        ),
                        "per_task": results,
                    })
                    status = "OK" if result["error"] is None else "ERR"
                    print(
                        f"[{phase.name} {phase_done[phase.name]}/{len(keys)} | "
                        f"{completed_total}/{total_work}] {status} {task_id} "
                        f"wall={result['wall']}s",
                        flush=True,
                    )
                fill()
    finally:
        if load_monitor is not None:
            load_monitor.stop()

    print(f"[kaggle-inducer] wrote {args.submission}", flush=True)


if __name__ == "__main__":
    main()
