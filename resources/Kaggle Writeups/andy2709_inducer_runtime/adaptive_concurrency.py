"""Capacity-based adaptive admission control for inference episodes.

The controller deliberately treats an episode as a black box.  It grows the
number of admitted episodes while the inference engine has KV headroom, trims
the limit as KV fills, and cuts it when the engine develops a persistent
capacity queue. Kaggle and local evaluation share the same policy. Clear load polls grow admission without waiting for a long
episode to finish; the server's effective request ceiling includes Mamba state.

The pressure thresholds and admission smoothing are adapted from PrimeIntellect-ai/prime-rl's
Apache-2.0-licensed ``prime_rl/orchestrator/concurrency.py`` at commit
``0c58f007bcbebd2751a5a5c51da1948c02663d27`` (August 2026).
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from typing import Callable, Literal
from urllib.parse import urlsplit, urlunsplit

GROWTH_CLEAR_POLLS = 2
# At scale growth is proportional: one slot per 8 admitted, and one in 20 admitted episodes may be
# between turns (running commands) at a growth poll. Below 16 and 20 admitted these are one slot and
# none, so Kaggle's 8-running decisions are unchanged; 160 running on H200 needs both (2026-10-21).
GROWTH_STEP_DIVISOR = 8
GROWTH_BETWEEN_TURNS_DIVISOR = 20
KV_USAGE_GROW = 0.6
KV_USAGE_SOFT_CAP = 0.8
KV_USAGE_HARD_CAP = 0.9
KV_USAGE_TARGET = 0.7
KV_TRIM_COOLDOWN_POLLS = 6
QUEUE_RATIO = 0.5
QUEUE_PERSISTENCE_POLLS = 6
QUEUE_CUT_FRACTION = 0.9
PREEMPTION_CUT_FRACTION = 0.8
ADMISSION_WINDOW_S = 5.0
ALGORITHM = "capacity_admission_v3"

Signal = Literal["clear", "soft", "hard"]
SEVERITY: dict[Signal, int] = {"clear": 0, "soft": 1, "hard": 2}


@dataclass(frozen=True)
class AdaptiveConcurrencyConfig:
    min_inflight: int = 1
    max_inflight: int = 256
    initial_inflight: int | None = None
    poll_interval_s: float = 2.0

    def __post_init__(self) -> None:
        if self.min_inflight < 1 or self.max_inflight < self.min_inflight:
            raise ValueError("adaptive concurrency requires 1 <= min_inflight <= max_inflight")
        if self.initial_inflight is not None and not (
            self.min_inflight <= self.initial_inflight <= self.max_inflight
        ):
            raise ValueError("initial_inflight must lie within [min_inflight, max_inflight]")
        if self.poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")

    def manifest(self) -> dict:
        return {"algorithm": ALGORITHM, **asdict(self)}


@dataclass(frozen=True)
class EngineLoadSample:
    engine_id: str
    kv_usage: float
    running: int
    waiting: int
    kv_capacity_tokens: int | None = None
    max_model_len: int | None = None
    preemptions_delta: int = 0
    max_running_requests: int | None = None
    timestamp: float | None = None


class AdaptiveConcurrencyController:
    """Thread-safe episode limiter driven by fresh capacity observations.

    ``acquire``/``release`` wrap one complete reasoning or tool-using attempt,
    not an individual HTTP turn.  This is important for agentic episodes that
    alternate between inference and local Python execution.
    """

    def __init__(
        self,
        config: AdaptiveConcurrencyConfig,
        *,
        fallback_cost: int,
        min_burst: int = 8,
        log: Callable[[str], None] | None = None,
    ) -> None:
        if fallback_cost < 1:
            raise ValueError("fallback_cost must be positive")
        if min_burst < 1:
            raise ValueError("min_burst must be positive")
        self.config = config
        self.floor = config.min_inflight
        self.fallback_cost = fallback_cost
        self._log = log or logging.getLogger(__name__).info
        self.min_burst = min_burst

        self.engine_request_limit = config.max_inflight
        self.max_inflight = self._clamp_int(config.initial_inflight or self.floor)
        self.bootstrapped = config.initial_inflight is not None
        self.engine_max_len: int | None = None
        self.capacity_by_engine: dict[str, int] = {}

        self.turnover = 0.0
        self.signal: Signal = "clear"
        self.clear_polls = 0
        self._last_growth_poll: float | None = None
        self._last_timestamps: dict[str, float] = {}
        self.prev_waiting: dict[str, int] = {}
        self.queue_overload_polls = 0
        self.trim_cooldown = 0
        self.draining = False

        self._active = 0
        self._admission_window_start = time.monotonic()
        self._admissions_in_window = 0
        self._condition = threading.Condition()

    def acquire(self, timeout: float | None = None) -> None:
        """Acquire one episode permit, blocking until the adaptive cap allows it."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                now = time.monotonic()
                admission_budget, until_refill = self._admission_budget_locked(now)
                if self._active < self.max_inflight and admission_budget > 0:
                    break
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("timed out waiting for an adaptive concurrency permit")
                wait_for = remaining
                if admission_budget <= 0:
                    wait_for = until_refill if wait_for is None else min(wait_for, until_refill)
                self._condition.wait(wait_for)
            self._active += 1
            self._admissions_in_window += 1

    def release(self, tokens: int) -> None:
        """Release an episode; fresh load polls, not completions, control growth."""
        with self._condition:
            if self._active <= 0:
                raise RuntimeError("adaptive concurrency permit released without acquire")
            inflight = self._active
            self._active -= 1
            # Prime-RL refunds a natural completion's admission so replacing
            # steady-state work is immediate; only net pool expansion is smoothed.
            self._admissions_in_window = max(0, self._admissions_in_window - 1)
            self.turnover += 1 / inflight
            self.clear_polls = 0
            self.draining = self._active > self.max_inflight
            self._condition.notify_all()

    def invalidate_load(self) -> None:
        """Require a new clear window after a failed or missing load poll."""
        with self._condition:
            self.clear_polls = 0
            self._last_growth_poll = None

    def observe(self, samples: list[EngineLoadSample]) -> None:
        """Push one engine-load poll into the controller."""
        if not samples:
            self.invalidate_load()
            return
        if any(not math.isfinite(s.kv_usage) or s.kv_usage < 0 for s in samples):
            self.invalidate_load()
            raise ValueError("invalid KV usage in engine load")
        with self._condition:
            now = time.monotonic()
            fresh = all(
                s.timestamp is None or s.timestamp > self._last_timestamps.get(s.engine_id, -math.inf)
                for s in samples
            )
            if self._last_growth_poll is not None:
                gap = now - self._last_growth_poll
                if gap > 2 * self.config.poll_interval_s:
                    self.clear_polls = 0
                fresh = fresh and gap >= self.config.poll_interval_s * 0.9
            if fresh:
                self._last_growth_poll = now
                self._last_timestamps.update({s.engine_id: s.timestamp for s in samples if s.timestamp is not None})
            for sample in samples:
                if sample.kv_capacity_tokens:
                    self.capacity_by_engine[sample.engine_id] = sample.kv_capacity_tokens
                if sample.max_model_len:
                    self.engine_max_len = max(self.engine_max_len or 0, sample.max_model_len)
            # SGLang reports its effective ceiling after reserving Mamba slots.
            if all(s.max_running_requests is not None for s in samples):
                limit = sum(s.max_running_requests for s in samples)
                if limit < 1:
                    raise ValueError("engine has no request capacity")
                self.engine_request_limit = min(self.config.max_inflight, limit)
                if self.max_inflight > self.engine_request_limit:
                    self._apply_limit_locked(self.engine_request_limit, reason="effective server capacity")

            worst: Signal = "clear"
            max_usage = 0.0
            total_running = 0
            total_waiting = 0
            preempted = False
            for sample in samples:
                if sample.preemptions_delta > 0:
                    preempted = True
                    worst = "hard"
                if sample.waiting > 0 and self.prev_waiting.get(sample.engine_id, 0) > 0:
                    worst = max(worst, "soft", key=SEVERITY.__getitem__)
                max_usage = max(max_usage, sample.kv_usage)
                total_running += sample.running
                total_waiting += sample.waiting
            self.prev_waiting = {sample.engine_id: sample.waiting for sample in samples}

            if total_running > 0 and total_waiting > QUEUE_RATIO * total_running:
                self.queue_overload_polls += 1
            else:
                self.queue_overload_polls = 0
            queue_overload = self.queue_overload_polls >= QUEUE_PERSISTENCE_POLLS
            if queue_overload or max_usage > KV_USAGE_GROW:
                worst = max(
                    worst,
                    "hard" if queue_overload else "soft",
                    key=SEVERITY.__getitem__,
                )
            self.signal = worst

            inflight = self._active
            self.draining = inflight > self.max_inflight
            self.trim_cooldown = max(0, self.trim_cooldown - 1)

            if not self.bootstrapped and self.capacity is not None:
                self.bootstrapped = True
                cost = float(self.engine_max_len or self.fallback_cost)
                self._apply_limit_locked(self._clamp_int(self.capacity / cost), reason="initial KV capacity")

            if self.draining:
                self.clear_polls = 0
                return

            if preempted or queue_overload:
                if queue_overload:
                    self.queue_overload_polls = 0
                    target = self._clamp_int(inflight * QUEUE_CUT_FRACTION)
                    reason = "queue overload"
                else:
                    target = self._clamp_int(inflight * PREEMPTION_CUT_FRACTION)
                    reason = "preemptions"
                self._resize_down_locked(target, reason)
                self.clear_polls = 0
                return

            if max_usage > KV_USAGE_SOFT_CAP and inflight > 0 and self.trim_cooldown == 0:
                target = self._clamp_int(inflight * KV_USAGE_TARGET / max_usage)
                severity = "hard" if max_usage > KV_USAGE_HARD_CAP else "soft"
                self._resize_down_locked(target, f"kv headroom ({max_usage:.2f}, {severity} trim)")
                self.trim_cooldown = KV_TRIM_COOLDOWN_POLLS

            # A Python/tool pause is not permission to overbook its returning
            # request. Grow only when the admitted episodes are visible running
            # (all of them below 20), and leave at least 40% KV headroom before
            # probing more slots.
            between_turns = inflight // GROWTH_BETWEEN_TURNS_DIVISOR
            can_grow = (
                worst == "clear" and total_waiting == 0
                and self.trim_cooldown == 0 and not self.draining
                and inflight == self.max_inflight
                and inflight - between_turns <= total_running <= inflight
                and self.max_inflight < self.engine_request_limit
            )
            if not can_grow:
                self.clear_polls = 0
            elif fresh:
                self.clear_polls += 1
                if self.clear_polls >= GROWTH_CLEAR_POLLS:
                    step = max(1, self.max_inflight // GROWTH_STEP_DIVISOR)
                    self._apply_limit_locked(self._clamp_int(self.max_inflight + step),
                                             reason="available KV and request capacity")

    def snapshot(self) -> dict[str, int | float | str | bool | None]:
        with self._condition:
            return {
                "algorithm": ALGORITHM,
                "max_inflight": self.max_inflight,
                "active": self._active,
                "turnover": round(self.turnover, 6),
                "capacity_tokens": self.capacity,
                "effective_max_inflight": self.engine_request_limit,
                "signal": self.signal,
                "draining": self.draining,
            }

    @property
    def active(self) -> int:
        with self._condition:
            return self._active

    @property
    def capacity(self) -> int | None:
        return sum(self.capacity_by_engine.values()) or None

    def _resize_down_locked(self, target: int, reason: str) -> None:
        target = min(target, self.max_inflight)
        self._apply_limit_locked(target, reason=reason)

    def _admission_budget_locked(self, now: float) -> tuple[int, float]:
        elapsed = now - self._admission_window_start
        if elapsed >= ADMISSION_WINDOW_S:
            self._admission_window_start = now
            self._admissions_in_window = 0
            elapsed = 0.0
        burst_cap = max(self.min_burst, self.max_inflight // 10)
        return burst_cap - self._admissions_in_window, ADMISSION_WINDOW_S - elapsed

    def _apply_limit_locked(self, value: int, reason: str | None) -> None:
        if value == self.max_inflight:
            return
        old = self.max_inflight
        self.max_inflight = value
        self.clear_polls = 0
        self.draining = self._active > value
        self._condition.notify_all()
        if reason is not None:
            verb = "increased" if value > old else "decreased"
            self._log(
                f"[adaptive-concurrency] {verb} {old} -> {value} ({reason}); "
                f"active={self._active} turnover={self.turnover:.1f} signal={self.signal}"
            )

    def _clamp(self, value: float) -> float:
        return min(max(value, float(self.floor)), float(self.engine_request_limit))

    def _clamp_int(self, value: float) -> int:
        return int(self._clamp(value))


class LoadMonitor:
    """Poll an engine adapter and feed a controller from one daemon thread."""

    def __init__(
        self,
        controller: AdaptiveConcurrencyController,
        load_source: Callable[[], list[EngineLoadSample]],
        *,
        poll_interval_s: float,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.controller = controller
        self.load_source = load_source
        self.poll_interval_s = poll_interval_s
        self._log = log or logging.getLogger(__name__).warning
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reported_error = False

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("load monitor already started")
        self._thread = threading.Thread(target=self._run, name="adaptive-load-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.poll_interval_s + 1.0))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.controller.observe(self.load_source())
                self._reported_error = False
            except Exception as exc:
                self.controller.invalidate_load()
                if not self._reported_error:
                    self._log(
                        f"[adaptive-concurrency] load polling failed; holding current limit: {exc!r}"
                    )
                    self._reported_error = True
            self._stop.wait(self.poll_interval_s)


def _json_get(url: str, timeout: float) -> dict | list:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _origin(base_url: str) -> str:
    parts = urlsplit(base_url)
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parts.scheme, parts.netloc, path.rstrip("/"), "", ""))


class SGLangLoadSource:
    """Map SGLang 0.5.20 load snapshots onto Prime-RL's vLLM signals.

    SGLang's current ``/v1/loads`` endpoint exposes the same controller facts
    directly. ``token_usage`` is allocator-derived KV usage;
    ``num_running_reqs`` and ``num_waiting_reqs`` match vLLM's gauges. SGLang
    does not split waiting by reason, so the controller uses Prime-RL's
    documented fallback to the total waiting queue. A nonzero interval
    retraction count is the SGLang analogue of a vLLM preemption delta.
    """

    def __init__(self, base_url: str, *, timeout: float = 5.0) -> None:
        self.origin = _origin(base_url)
        self.timeout = timeout
        self._max_model_len: int | None = None
        self._retracted_by_engine: dict[str, int] = {}

    def __call__(self) -> list[EngineLoadSample]:
        if self._max_model_len is None:
            models = _json_get(f"{self.origin}/v1/models", self.timeout)
            if not isinstance(models, dict):
                raise ValueError("SGLang /v1/models did not return an object")
            lengths = [
                int(model.get("max_model_len") or 0)
                for model in models.get("data", [])
                if isinstance(model, dict)
            ]
            self._max_model_len = max(lengths, default=0) or None
        payload = _json_get(
            f"{self.origin}/v1/loads?include=core,queues", self.timeout
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("loads"), list):
            raise ValueError("SGLang /v1/loads did not return a load list")
        samples = []
        for index, row in enumerate(payload["loads"]):
            if not isinstance(row, dict):
                raise ValueError("SGLang /v1/loads contained a non-object load")
            waiting = int(row.get("num_waiting_reqs") or 0)
            used = int(row.get("num_used_tokens") or 0)
            capacity = int(row.get("max_total_num_tokens") or 0) or None
            queues = row.get("queues") if isinstance(row.get("queues"), dict) else {}
            engine_id = str(row.get("dp_rank", index))
            retracted = int(queues.get("retracted") or 0)
            previous_retracted = self._retracted_by_engine.get(engine_id)
            if previous_retracted is None:
                retracted_delta = 0
            elif retracted >= previous_retracted:
                retracted_delta = retracted - previous_retracted
            else:
                # SGLang's interval gauge periodically resets rather than
                # increasing forever; the post-reset value is all new work.
                retracted_delta = retracted
            self._retracted_by_engine[engine_id] = retracted
            samples.append(
                EngineLoadSample(
                    engine_id=engine_id,
                    kv_usage=float(
                        row.get("token_usage")
                        if row.get("token_usage") is not None
                        else used / capacity if capacity else 0.0
                    ),
                    running=int(row.get("num_running_reqs") or 0),
                    waiting=waiting,
                    kv_capacity_tokens=capacity,
                    max_model_len=self._max_model_len,
                    preemptions_delta=retracted_delta,
                    max_running_requests=int(row["max_running_requests"]) if row.get("max_running_requests") is not None else None,
                    timestamp=float(row["timestamp"]) if row.get("timestamp") is not None else None,
                )
            )
        return samples


def add_adaptive_concurrency_arguments(parser) -> None:
    """Add the shared local SGLang controller flags to an argument parser."""
    parser.add_argument(
        "--adaptive-concurrency",
        action="store_true",
        help="adapt the global number of in-flight attempts from SGLang load",
    )
    parser.add_argument("--adaptive-min-inflight", type=int, default=1)
    parser.add_argument(
        "--adaptive-initial-inflight",
        type=int,
        default=0,
        help="initial attempt cap (0 = --concurrency)",
    )
    parser.add_argument("--adaptive-max-inflight", type=int, default=256)
    parser.add_argument("--adaptive-poll-interval", type=float, default=5.0)


def local_adaptive_runtime(args, *, fallback_cost: int):
    """Build the common SGLang limiter/monitor and its provenance manifest.

    Returns ``(controller, monitor, task_workers, scheduler_manifest)``.  In
    fixed mode the first two values are ``None`` and existing task-concurrency
    behavior is preserved exactly. Each worker runs one attempt at a time.
    """
    if not args.adaptive_concurrency:
        return None, None, args.concurrency, None
    initial = args.adaptive_initial_inflight or args.concurrency
    config = AdaptiveConcurrencyConfig(
        min_inflight=args.adaptive_min_inflight,
        initial_inflight=initial,
        max_inflight=args.adaptive_max_inflight,
        poll_interval_s=args.adaptive_poll_interval,
    )
    controller = AdaptiveConcurrencyController(
        config,
        fallback_cost=fallback_cost,
        min_burst=1,
        log=lambda message: print(message, flush=True),
    )
    monitor = LoadMonitor(
        controller,
        SGLangLoadSource(args.base_url),
        poll_interval_s=config.poll_interval_s,
        log=lambda message: print(message, flush=True),
    )
    task_workers = max(args.concurrency, config.max_inflight)
    return controller, monitor, task_workers, {
        **config.manifest(),
        "load_source": "sglang_v1_loads",
    }
