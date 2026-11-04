#!/usr/bin/env python3
"""Post-verify inducer attempts, optionally materializing the hidden-correct SFT projection.

Verification is per physical attempt row.  The raw append-only sidecar is opened at
a fixed byte snapshot and never modified; a concurrent incomplete tail is ignored.
The outputs are a compact verification ledger, a report, and optionally an annotated trace
snapshot and one complete multi-turn SFT row per whole-puzzle exact match.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any, BinaryIO, Iterable

from trace_store import trace_policy, trace_to_sft_record


Grid = list[list[int]]


def valid_grid(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(row, list) and bool(row) for row in value)
        and len({len(row) for row in value}) == 1
        and all(
            isinstance(cell, int) and not isinstance(cell, bool) and 0 <= cell <= 9
            for row in value for cell in row
        )
    )


def exact_grid(left: Any, right: Any) -> bool:
    return valid_grid(left) and valid_grid(right) and left == right


def validate_solutions(value: Any) -> dict[str, list[Grid]]:
    if not isinstance(value, dict):
        raise ValueError("solutions must be a JSON object keyed by task ID")
    for task_id, outputs in value.items():
        if (
            not isinstance(task_id, str)
            or not task_id
            or not isinstance(outputs, list)
            or not outputs
            or not all(valid_grid(output) for output in outputs)
        ):
            raise ValueError(f"solutions contains an invalid entry for {task_id!r}")
    return value


def verify_attempt(attempt: dict[str, Any], solutions: dict[str, list[Grid]],
                   solutions_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = trace_policy(attempt)
    task_id = attempt.get("task_id")
    truths = solutions.get(task_id)
    outputs = attempt.get("outputs")
    if truths is None:
        status = "missing_solution"
        per_test = None
        solved = None
    else:
        per_test = [
            isinstance(outputs, list)
            and index < len(outputs)
            and exact_grid(outputs[index], truth)
            for index, truth in enumerate(truths)
        ]
        solved = (
            attempt.get("error") is None
            and isinstance(outputs, list)
            and len(outputs) == len(truths)
            and bool(per_test)
            and all(per_test)
        )
        status = "verified"

    verification = {
        "schema_version": 2,
        "status": status,
        "solutions_sha256": solutions_sha256,
        "hidden_per_test": per_test,
        "hidden_task_solved": solved,
    }
    annotated = copy.deepcopy(attempt)
    annotated["post_verification"] = verification
    annotated["hidden_per_test"] = per_test
    annotated["hidden_task_solved"] = solved
    ledger = {
        "schema_version": 1,
        "trace_id": attempt.get("trace_id"),
        "logical_trace_id": attempt.get("logical_trace_id"),
        "trace_revision": attempt.get("trace_revision", 0),
        "task_id": task_id,
        "sample": attempt.get("sample"),
        "seed": attempt.get("seed"),
        "policy": policy,
        "profile": copy.deepcopy(attempt.get("profile", policy)),
        "attempt_error": attempt.get("error"),
        **verification,
    }
    return annotated, ledger


def iter_snapshot_rows(stream: BinaryIO, snapshot_bytes: int) -> Iterable[dict[str, Any]]:
    line_number = 0
    while stream.tell() < snapshot_bytes:
        offset = stream.tell()
        raw = stream.readline(snapshot_bytes - offset)
        line_number += 1
        if not raw:
            break
        if not raw.endswith(b"\n"):
            break
        try:
            yield json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"trace snapshot line {line_number} is invalid JSON") from exc


def _temporary(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_suffix(path.suffix + ".tmp")


def post_verify(trace_in: Path, solutions_path: Path, ledger_out: Path,
                sft_out: Path | None, report_out: Path,
                annotated_trace_out: Path | None = None) -> dict[str, Any]:
    trace_in = trace_in.resolve()
    protected = {trace_in}
    outputs = [ledger_out, report_out, sft_out, annotated_trace_out]
    for output in filter(None, outputs):
        if output.resolve() in protected:
            raise ValueError("post-verification outputs must not overwrite the raw trace")

    solutions_bytes = solutions_path.read_bytes()
    solutions_sha256 = hashlib.sha256(solutions_bytes).hexdigest()
    solutions = validate_solutions(json.loads(solutions_bytes))

    ledger_tmp = _temporary(ledger_out)
    sft_tmp = _temporary(sft_out) if sft_out else None
    annotated_tmp = _temporary(annotated_trace_out) if annotated_trace_out else None
    attempts = passed = failed = missing = sft_records = 0
    policy_counts: Counter[str] = Counter()
    policy_passed: Counter[str] = Counter()

    with trace_in.open("rb") as source:
        snapshot_bytes = os.fstat(source.fileno()).st_size
        with ExitStack() as stack:
            ledger_stream = stack.enter_context(ledger_tmp.open("w", encoding="utf-8"))
            sft_stream = stack.enter_context(sft_tmp.open("w", encoding="utf-8")) if sft_tmp else None
            annotated_stream = (
                stack.enter_context(annotated_tmp.open("w", encoding="utf-8"))
                if annotated_tmp is not None else None
            )
            for attempt in iter_snapshot_rows(source, snapshot_bytes):
                attempts += 1
                policy = trace_policy(attempt)
                policy_counts[policy] += 1
                annotated, ledger = verify_attempt(attempt, solutions, solutions_sha256)
                ledger_stream.write(json.dumps(ledger, ensure_ascii=False) + "\n")
                if annotated_stream is not None:
                    annotated_stream.write(json.dumps(annotated, ensure_ascii=False) + "\n")
                solved = ledger["hidden_task_solved"]
                if solved is None:
                    missing += 1
                elif solved:
                    passed += 1
                    policy_passed[policy] += 1
                    if sft_stream is not None:
                        sft_stream.write(json.dumps(trace_to_sft_record(annotated), ensure_ascii=False) + "\n")
                        sft_records += 1
                else:
                    failed += 1
            for stream in (ledger_stream, sft_stream, annotated_stream):
                if stream is not None:
                    stream.flush()
                    os.fsync(stream.fileno())

    os.replace(ledger_tmp, ledger_out)
    if sft_tmp is not None:
        os.replace(sft_tmp, sft_out)
    if annotated_tmp is not None and annotated_trace_out is not None:
        os.replace(annotated_tmp, annotated_trace_out)

    report = {
        "schema_version": 2,
        "trace_path": str(trace_in),
        "trace_snapshot_bytes": snapshot_bytes,
        "solutions_path": str(solutions_path),
        "solutions_sha256": solutions_sha256,
        "attempts": attempts,
        "verified_attempts": passed + failed,
        "passed_attempts": passed,
        "failed_attempts": failed,
        "missing_solution_attempts": missing,
        "pass_rate": passed / (passed + failed) if passed + failed else None,
        "attempts_by_policy": dict(sorted(policy_counts.items())),
        "passed_by_policy": dict(sorted(policy_passed.items())),
        "sft_records": sft_records,
        "raw_trace_modified": False,
    }
    report_tmp = _temporary(report_out)
    report_tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(report_tmp, report_out)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-in", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--ledger-out", type=Path, required=True)
    parser.add_argument("--sft-out", type=Path, default=None)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--annotated-trace-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = post_verify(
        args.trace_in, args.solutions, args.ledger_out, args.sft_out,
        args.report_out, args.annotated_trace_out,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
