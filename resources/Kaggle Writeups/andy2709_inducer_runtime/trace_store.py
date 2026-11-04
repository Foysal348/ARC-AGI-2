"""Durable, lossless inducer attempt traces and SFT projection helpers."""
from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from typing import Any


def trace_policy(row: dict[str, Any]) -> str:
    """Read experiment identity across trace generations, without an allowlist.

    ``policy`` takes precedence when both fields exist. The original fields
    remain untouched; missing optional provenance is reported as unspecified.
    This is a data-reader helper, not a list of runnable inference policies.
    """
    for key in ("policy", "profile"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return "unspecified"


def fingerprint_payload(payload: dict[str, Any]) -> str:
    """Return a stable identity for a rollout configuration and harness revision."""
    import hashlib

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class JsonlTraceStore:
    """Append-only attempt store: reopening checks the run fingerprint and repairs a torn tail."""

    def __init__(self, path: Path, run_fingerprint: str):
        self.path = Path(path)
        self.run_fingerprint = run_fingerprint
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._logical_revisions: dict[str, int] = {}
        self._trace_ids = self._load_trace_ids()

    @property
    def retained(self) -> int:
        return len(self._trace_ids)

    def _load_trace_ids(self) -> set[str]:
        if not self.path.exists():
            return set()

        identities: set[str] = set()
        truncate_to: int | None = None
        append_newline = False
        with self.path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            line_number = 0
            while stream.tell() < size:
                offset = stream.tell()
                raw = stream.readline()
                line_number += 1
                final = stream.tell() == size
                try:
                    row = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    if final and not raw.endswith(b"\n"):
                        truncate_to = offset
                        break
                    raise ValueError(
                        f"{self.path}:{line_number}: invalid trace JSONL record"
                    ) from exc
                if row.get("run_fingerprint") != self.run_fingerprint:
                    raise ValueError(
                        f"{self.path}:{line_number}: trace belongs to a different run; "
                        "select a new trace path"
                    )
                trace_id = row.get("trace_id")
                if not isinstance(trace_id, str) or not trace_id:
                    raise ValueError(f"{self.path}:{line_number}: trace_id is missing")
                if trace_id in identities:
                    raise ValueError(f"{self.path}:{line_number}: duplicate trace_id {trace_id}")
                identities.add(trace_id)
                logical_id = row.get("logical_trace_id", trace_id)
                if not isinstance(logical_id, str) or not logical_id:
                    raise ValueError(f"{self.path}:{line_number}: logical_trace_id is invalid")
                revision = row.get("trace_revision", 0)
                if not isinstance(revision, int) or revision < 0:
                    raise ValueError(f"{self.path}:{line_number}: trace_revision is invalid")
                self._logical_revisions[logical_id] = max(
                    self._logical_revisions.get(logical_id, -1), revision
                )
                append_newline = final and not raw.endswith(b"\n")

        if truncate_to is not None:
            with self.path.open("r+b") as stream:
                stream.truncate(truncate_to)
                stream.flush()
                os.fsync(stream.fileno())
            print(f"[trace] removed torn final record from {self.path}", flush=True)
        elif append_newline:
            # A valid final record without a newline must be separated from the next append.
            with self.path.open("ab") as stream:
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
        return identities

    def append_attempt(self, logical_trace_id: str, row: dict[str, Any]) -> str:
        """Append a logical attempt, retaining later retries as explicit revisions."""
        with self._lock:
            revision = self._logical_revisions.get(logical_trace_id, -1) + 1
            trace_id = (
                logical_trace_id if revision == 0
                else f"{logical_trace_id}:retry{revision}"
            )
            if trace_id in self._trace_ids:
                raise ValueError(f"duplicate trace revision {trace_id}")
            payload = {
                **row,
                "trace_id": trace_id,
                "logical_trace_id": logical_trace_id,
                "trace_revision": revision,
                "run_fingerprint": self.run_fingerprint,
            }
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._trace_ids.add(trace_id)
            self._logical_revisions[logical_trace_id] = revision
            return trace_id


def _assistant_message(turn: dict[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": turn.get("content") or "",
    }
    reasoning = turn.get("reasoning")
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    tool_calls = turn.get("tool_calls") or []
    if tool_calls:
        message["tool_calls"] = copy.deepcopy(tool_calls)
    return message


def trace_to_sft_record(trace: dict[str, Any]) -> dict[str, Any]:
    """Convert one retained trace into one complete multi-turn SFT conversation."""
    policy = trace_policy(trace)
    turns = trace.get("turns") or []
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"trace {trace.get('trace_id', '<unknown>')} has no turns")
    final_turn = turns[-1]
    request_messages = final_turn.get("request_messages")
    if not isinstance(request_messages, list):
        raise ValueError(
            f"trace {trace.get('trace_id', '<unknown>')} final turn "
            "does not retain request_messages"
        )
    messages = copy.deepcopy(request_messages)
    messages.append(_assistant_message(final_turn))
    # request_messages is captured BEFORE this last response is executed.
    # Earlier observations are already in that snapshot; terminal observations
    # have no subsequent request to carry them into the SFT conversation.
    calls = final_turn.get("tool_calls") or []
    for index, result in enumerate(final_turn.get("tool_results") or []):
        if "response" not in result:
            continue
        call_id = result.get("tool_call_id")
        if call_id is None and index < len(calls):
            call_id = calls[index].get("id")
        response = result["response"]
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": response if isinstance(response, str) else json.dumps(response),
        })
    provenance = {
        key: copy.deepcopy(trace[key])
        for key in (
            "model", "run", "run_fingerprint", "logical_trace_id", "trace_revision",
            "post_verification", "source", "subset", "source_pool_sha256",
            "provenance", "reasoning_effort",
        )
        if key in trace
    }
    return {
        **provenance,
        "schema_version": 3,
        "source_schema_version": trace.get("schema_version"),
        "source_trace_id": trace.get("trace_id"),
        "task_id": trace.get("task_id"),
        "sample": trace.get("sample"),
        "seed": trace.get("seed"),
        "policy": policy,
        "profile": copy.deepcopy(trace.get("profile", policy)),
        "turn_count": len(turns),
        "messages": messages,
        "tools": copy.deepcopy(final_turn.get("request_tools")),
        "terminal_tool_results": copy.deepcopy(final_turn.get("tool_results") or []),
        "hidden_per_test": copy.deepcopy(trace.get("hidden_per_test")),
        "hidden_task_solved": trace.get("hidden_task_solved"),
        "attempt_error": trace.get("error"),
    }

