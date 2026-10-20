
# ======================= CELL 0 (code)

import glob, importlib.util, json, math, os, re, socket, subprocess, sys, threading, time
from pathlib import Path
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import urllib.request, urllib.error

def parse_vllm_scheduler_log(text):
    """Summarize active-request vLLM scheduler telemetry from server logs."""

    prompt_rates, generation_rates, running_counts, kv_values = [], [], [], []
    for line in text.splitlines():
        prompt_match = re.search(
            r"Avg prompt throughput: ([0-9.]+) tokens/s", line)
        generation_match = re.search(
            r"Avg generation throughput: ([0-9.]+) tokens/s", line)
        running_match = re.search(r"Running: ([0-9]+) reqs", line)
        kv_match = re.search(r"GPU KV cache usage: ([0-9.]+)%", line)
        running = int(running_match.group(1)) if running_match else None
        if running is not None:
            running_counts.append(running)
        if prompt_match and (running is None or running > 0):
            value = float(prompt_match.group(1))
            if value > 0:
                prompt_rates.append(value)
        if generation_match and (running is None or running > 0):
            value = float(generation_match.group(1))
            if value > 0:
                generation_rates.append(value)
        if kv_match:
            kv_values.append(float(kv_match.group(1)))

    def mean(values):
        return (sum(values) / len(values)) if values else None

    mean_prompt = mean(prompt_rates)
    mean_generation = mean(generation_rates)
    return {
        "scheduler_log_samples": len(generation_rates),
        "prefill_tokens_per_second": (round(mean_prompt, 1)
                                       if mean_prompt is not None else None),
        "decode_tokens_per_second": (round(mean_generation, 1)
                                      if mean_generation is not None else None),
        "peak_prefill_tokens_per_second": (max(prompt_rates)
                                             if prompt_rates else None),
        "peak_decode_tokens_per_second": (max(generation_rates)
                                            if generation_rates else None),
        # Backward-compatible name retained for existing receipt readers.
        "peak_vllm_gen_throughput_tok_s": (max(generation_rates)
                                             if generation_rates else None),
        "mean_running_requests": (round(mean(running_counts), 1)
                                   if running_counts else None),
        "peak_running_requests": max(running_counts) if running_counts else None,
        "peak_kv_cache_usage_pct": max(kv_values) if kv_values else None,
    }


WORK = Path("/kaggle/working")
SUB = WORK / "submission.json"
RECEIPT = WORK / "freeform_receipt.json"
PORT = 1234
MODEL_NAME = "Qwen/Qwen3.8-27B-FP8"
AGGREGATOR_SOURCE = '#!/usr/bin/env python3\n"""Aggregate immutable replica candidates into a complete ARC pass@2 file."""\n\nfrom __future__ import annotations\n\nimport argparse\nfrom collections import Counter\nfrom datetime import datetime, timezone\nimport hashlib\nimport json\nimport math\nimport os\nfrom pathlib import Path\nimport re\nimport sys\nimport tempfile\nimport traceback\nfrom typing import Any\nimport uuid\n\n\nCOLLECTION_SCHEMA = "arc-qwen38-kaggle-collection-v1"\nCANDIDATE_SCHEMA = "arc-qwen38-kaggle-candidate-v1"\nSHA256_RE = re.compile(r"^[0-9a-f]{64}$")\n\n\ndef _stable_json(value: Any) -> str:\n    return json.dumps(value, sort_keys=True, separators=(",", ":"))\n\n\ndef _digest(value: Any) -> str:\n    return hashlib.sha256(_stable_json(value).encode()).hexdigest()\n\n\ndef _require_sha256(value: Any, *, context: str) -> str:\n    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):\n        raise RuntimeError(f"{context} must be a lowercase SHA-256 digest")\n    return value\n\n\ndef _quarantine_blocking_directory(path: Path) -> Path:\n    """Move a literal destination directory aside without deleting its contents.\n\n    Kaggle work directories can survive interrupted notebook runs. If a stale\n    run leaves a *directory* at ``submission.json``, ``os.replace`` cannot\n    publish the guaranteed floor and the failure happens before the rich-path\n    degradation handler exists. Preserve the debris under a unique sibling and\n    free the required file path. Symlinks are intentionally not followed: an\n    atomic file replace safely replaces the link itself.\n    """\n    if not path.is_dir() or path.is_symlink():\n        raise NotADirectoryError(f"not a blocking literal directory: {path}")\n    for _ in range(32):\n        quarantine = path.with_name(\n            f"{path.name}.blocking-dir.{uuid.uuid4().hex}.quarantine"\n        )\n        try:\n            os.rename(path, quarantine)\n        except FileExistsError:\n            continue\n        return quarantine\n    raise RuntimeError(f"could not allocate quarantine path for {path}")\n\n\ndef _write_json(path: Path, value: Any) -> None:\n    """Atomically publish JSON via a UNIQUE same-directory temp + fsync.\n\n    A fixed ``<name>.tmp`` sibling can be stale work-directory debris (e.g. a\n    leftover *directory* at that exact path), which makes the FIRST\n    guaranteed-floor publication raise ``IsADirectoryError`` and leave no\n    ``submission.json`` at all. A unique ``mkstemp`` temp in the same directory,\n    flushed + fsynced before ``os.replace`` (and a best-effort directory fsync\n    of the rename), makes the publish crash-durable and immune to stale\n    fixed-name debris. If the destination itself is a literal directory, it is\n    quarantined non-destructively only after the replacement payload has been\n    flushed. ``os.replace`` is atomic within one filesystem.\n    """\n    path.parent.mkdir(parents=True, exist_ok=True)\n    data = json.dumps(value, indent=2, sort_keys=True) + "\\n"\n    fd, tmp_name = tempfile.mkstemp(\n        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))\n    tmp = Path(tmp_name)\n    try:\n        with os.fdopen(fd, "w") as handle:\n            handle.write(data)\n            handle.flush()\n            os.fsync(handle.fileno())\n        if path.is_dir() and not path.is_symlink():\n            _quarantine_blocking_directory(path)\n        os.replace(tmp, path)\n    except BaseException:\n        tmp.unlink(missing_ok=True)\n        raise\n    try:  # durability of the rename itself; failure here does not lose the file\n        dir_fd = os.open(str(path.parent), os.O_RDONLY)\n        try:\n            os.fsync(dir_fd)\n        finally:\n            os.close(dir_fd)\n    except OSError:\n        pass\n\n\ndef load_collection_contract(\n    output_root: Path,\n    challenges_path: Path,\n    *,\n    challenges: dict[str, Any] | None = None,\n    challenge_sha256: str | None = None,\n) -> dict[str, Any] | None:\n    """Load and verify the immutable sweep contract when candidates exist."""\n\n    path = output_root / "collection_contract.json"\n    candidate_paths = sorted((output_root / "candidates").glob("*/*.json"))\n    if not path.is_file():\n        if candidate_paths:\n            raise RuntimeError(\n                "candidate files exist without collection_contract.json; "\n                "refusing to mix unattributed or stale runs"\n            )\n        return None\n    try:\n        contract = json.loads(path.read_text())\n    except (OSError, json.JSONDecodeError) as exc:\n        raise RuntimeError(f"invalid collection contract: {exc}") from exc\n    if contract.get("schema") != COLLECTION_SCHEMA:\n        raise RuntimeError(f"unexpected collection schema: {contract.get(\'schema\')!r}")\n    collection_id = contract.get("collection_id")\n    payload = {key: value for key, value in contract.items() if key != "collection_id"}\n    if collection_id != _digest(payload):\n        raise RuntimeError("collection contract digest does not verify")\n    if challenges is None or challenge_sha256 is None:\n        challenges, challenge_sha256 = load_public_challenges_with_digest(\n            challenges_path\n        )\n    if contract.get("challenge_sha256") != challenge_sha256:\n        raise RuntimeError(\n            "collection contract challenge digest does not match aggregation input"\n        )\n    task_ids = sorted(str(task_id) for task_id in challenges)\n    if contract.get("task_count") != len(task_ids):\n        raise RuntimeError("collection contract task_count does not match challenges")\n    if contract.get("task_ids_sha256") != _digest(task_ids):\n        raise RuntimeError("collection contract task_ids_sha256 does not match challenges")\n    model = contract.get("model")\n    if not isinstance(model, dict):\n        raise RuntimeError("collection contract has no model identity")\n    if model.get("repo_id") != "Qwen/Qwen3.8-27B":\n        raise RuntimeError("collection contract binds the wrong model repository")\n    if model.get("revision") != "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0":\n        raise RuntimeError("collection contract binds the wrong model revision")\n    _require_sha256(model.get("source_digest"), context="model.source_digest")\n    _require_sha256(\n        model.get("source_contract_sha256"),\n        context="model.source_contract_sha256",\n    )\n    _require_sha256(model.get("receipt_sha256"), context="model.receipt_sha256")\n    if model.get("weight_hashes_verified") is not True:\n        raise RuntimeError("collection contract lacks verified model weight hashes")\n    verified_file_count = model.get("verified_file_count")\n    if (\n        isinstance(verified_file_count, bool)\n        or not isinstance(verified_file_count, int)\n        or verified_file_count < 19\n    ):\n        raise RuntimeError("collection contract has an incomplete model file inventory")\n    runtime = contract.get("runtime")\n    if not isinstance(runtime, dict):\n        raise RuntimeError("collection contract has no runtime identity")\n    _require_sha256(\n        runtime.get("receipt_sha256"), context="runtime.receipt_sha256"\n    )\n    schedule = contract.get("schedule")\n    if not isinstance(schedule, dict):\n        raise RuntimeError("collection contract has no schedule")\n    if schedule.get("world_size") != 4:\n        raise RuntimeError("collection contract must use exactly four replicas")\n    for field in ("rollouts", "max_turns"):\n        value = schedule.get(field)\n        if isinstance(value, bool) or not isinstance(value, int) or value < 1:\n            raise RuntimeError(f"collection schedule {field} must be positive")\n    return contract\n\n\ndef validate_grid(grid: Any, *, context: str = "grid") -> None:\n    if not isinstance(grid, list) or not 1 <= len(grid) <= 30:\n        raise ValueError(f"{context}: invalid height")\n    if not all(isinstance(row, list) for row in grid):\n        raise ValueError(f"{context}: rows must be lists")\n    width = len(grid[0])\n    if not 1 <= width <= 30 or any(len(row) != width for row in grid):\n        raise ValueError(f"{context}: invalid rectangle")\n    if any(\n        isinstance(value, bool)\n        or not isinstance(value, int)\n        or not 0 <= value <= 9\n        for row in grid\n        for value in row\n    ):\n        raise ValueError(f"{context}: invalid color")\n\n\ndef _reject_duplicate_object_pairs(\n    pairs: list[tuple[str, Any]],\n) -> dict[str, Any]:\n    value: dict[str, Any] = {}\n    for key, item in pairs:\n        if key in value:\n            raise ValueError(f"duplicate JSON object key: {key!r}")\n        value[key] = item\n    return value\n\n\ndef validate_public_challenges(challenges: Any) -> dict[str, Any]:\n    """Accept only public ARC train labels and unlabeled test inputs."""\n\n    if not isinstance(challenges, dict) or not challenges:\n        raise ValueError("challenge JSON must be a non-empty task map")\n    for task_id, task in challenges.items():\n        if not isinstance(task_id, str) or not task_id:\n            raise ValueError("challenge task IDs must be non-empty strings")\n        context = f"challenge[{task_id!r}]"\n        if not isinstance(task, dict) or set(task) != {"train", "test"}:\n            raise ValueError(\n                f"{context}: task keys must be exactly train/test; "\n                "hidden labels or metadata are forbidden"\n            )\n        for split in ("train", "test"):\n            pairs = task[split]\n            if not isinstance(pairs, list) or not pairs:\n                raise ValueError(f"{context}.{split}: must be a non-empty list")\n            expected_keys = {"input", "output"} if split == "train" else {"input"}\n            for index, pair in enumerate(pairs):\n                pair_context = f"{context}.{split}[{index}]"\n                if not isinstance(pair, dict) or set(pair) != expected_keys:\n                    raise ValueError(\n                        f"{pair_context}: keys must be exactly "\n                        f"{sorted(expected_keys)!r}; hidden test labels are forbidden"\n                    )\n                validate_grid(pair["input"], context=f"{pair_context}.input")\n                if split == "train":\n                    validate_grid(pair["output"], context=f"{pair_context}.output")\n    return challenges\n\n\ndef load_public_challenges_with_digest(\n    path: Path,\n) -> tuple[dict[str, Any], str]:\n    try:\n        raw = path.read_bytes()\n        value = json.loads(\n            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_object_pairs\n        )\n    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:\n        raise RuntimeError(f"invalid public challenge JSON: {exc}") from exc\n    try:\n        challenges = validate_public_challenges(value)\n    except ValueError as exc:\n        raise RuntimeError(f"unsafe public challenge JSON: {exc}") from exc\n    return challenges, hashlib.sha256(raw).hexdigest()\n\n\ndef load_public_challenges(path: Path) -> dict[str, Any]:\n    return load_public_challenges_with_digest(path)[0]\n\n\ndef _valid_outputs(task_id: str, task: dict[str, Any], outputs: Any) -> bool:\n    if not isinstance(outputs, list) or len(outputs) != len(task["test"]):\n        return False\n    try:\n        for index, grid in enumerate(outputs):\n            validate_grid(grid, context=f"{task_id}.candidate[{index}]")\n    except ValueError:\n        return False\n    return True\n\n\ndef _identity(task: dict[str, Any]) -> list[list[list[int]]]:\n    return [pair["input"] for pair in task["test"]]\n\n\ndef _background(task: dict[str, Any]) -> list[list[list[int]]]:\n    counts = Counter(\n        value\n        for pair in task["train"]\n        for row in pair["output"]\n        for value in row\n    )\n    color = counts.most_common(1)[0][0]\n    same_shape = all(\n        (len(pair["input"]), len(pair["input"][0]))\n        == (len(pair["output"]), len(pair["output"][0]))\n        for pair in task["train"]\n    )\n    shapes = {\n        (len(pair["output"]), len(pair["output"][0])) for pair in task["train"]\n    }\n    fixed = next(iter(shapes)) if len(shapes) == 1 else None\n    rows = []\n    for pair in task["test"]:\n        if same_shape or fixed is None:\n            height, width = len(pair["input"]), len(pair["input"][0])\n        else:\n            height, width = fixed\n        rows.append([[color] * width for _ in range(height)])\n    return rows\n\n\ndef _geometric_grid(source: str, grid: list[list[int]]) -> list[list[int]]:\n    if source == "fallback-flip-lr":\n        return [list(reversed(row)) for row in grid]\n    if source == "fallback-flip-ud":\n        return list(reversed([row[:] for row in grid]))\n    if source == "fallback-transpose":\n        return [list(row) for row in zip(*grid)]\n    if source == "fallback-rotate-cw":\n        return [list(row) for row in zip(*reversed(grid))]\n    if source == "fallback-rotate-ccw":\n        return [list(row) for row in reversed(list(zip(*grid)))]\n    if source == "fallback-rotate-180":\n        return [list(reversed(row)) for row in reversed(grid)]\n    raise ValueError(f"unknown geometric fallback: {source}")\n\n\ndef _constant_color(task: dict[str, Any], color: int) -> list[list[list[int]]]:\n    same_shape = all(\n        (len(pair["input"]), len(pair["input"][0]))\n        == (len(pair["output"]), len(pair["output"][0]))\n        for pair in task["train"]\n    )\n    shapes = {\n        (len(pair["output"]), len(pair["output"][0])) for pair in task["train"]\n    }\n    fixed = next(iter(shapes)) if len(shapes) == 1 else None\n    rows = []\n    for pair in task["test"]:\n        if same_shape or fixed is None:\n            height, width = len(pair["input"]), len(pair["input"][0])\n        else:\n            height, width = fixed\n        rows.append([[color] * width for _ in range(height)])\n    return rows\n\n\ndef _fallback_outputs(\n    task: dict[str, Any], source: str\n) -> list[list[list[int]]]:\n    if source == "fallback-identity":\n        return _identity(task)\n    if source == "fallback-background":\n        return _background(task)\n    if source.startswith("fallback-constant-"):\n        return _constant_color(task, int(source.rsplit("-", 1)[-1]))\n    return [_geometric_grid(source, pair["input"]) for pair in task["test"]]\n\n\ndef _fallback_demo_score(\n    task: dict[str, Any], source: str\n) -> tuple[bool, float]:\n    if source == "fallback-identity":\n        matches = [pair["input"] == pair["output"] for pair in task["train"]]\n    elif source == "fallback-background":\n        counts = Counter(\n            value\n            for pair in task["train"]\n            for row in pair["output"]\n            for value in row\n        )\n        color = counts.most_common(1)[0][0]\n        same_shape = all(\n            (len(pair["input"]), len(pair["input"][0]))\n            == (len(pair["output"]), len(pair["output"][0]))\n            for pair in task["train"]\n        )\n        shapes = {\n            (len(pair["output"]), len(pair["output"][0]))\n            for pair in task["train"]\n        }\n        fixed = next(iter(shapes)) if len(shapes) == 1 else None\n        matches = []\n        for pair in task["train"]:\n            if same_shape or fixed is None:\n                height, width = len(pair["input"]), len(pair["input"][0])\n            else:\n                height, width = fixed\n            matches.append([[color] * width for _ in range(height)] == pair["output"])\n    elif source.startswith("fallback-constant-"):\n        color = int(source.rsplit("-", 1)[-1])\n        same_shape = all(\n            (len(pair["input"]), len(pair["input"][0]))\n            == (len(pair["output"]), len(pair["output"][0]))\n            for pair in task["train"]\n        )\n        shapes = {\n            (len(pair["output"]), len(pair["output"][0]))\n            for pair in task["train"]\n        }\n        fixed = next(iter(shapes)) if len(shapes) == 1 else None\n        matches = []\n        for pair in task["train"]:\n            if same_shape or fixed is None:\n                height, width = len(pair["input"]), len(pair["input"][0])\n            else:\n                height, width = fixed\n            matches.append([[color] * width for _ in range(height)] == pair["output"])\n    else:\n        matches = [\n            _geometric_grid(source, pair["input"]) == pair["output"]\n            for pair in task["train"]\n        ]\n    return all(matches), sum(matches) / len(matches)\n\n\ndef _trajectory_id(row: dict[str, Any]) -> tuple[int, int] | None:\n    rank = row.get("rank")\n    rollout = row.get("rollout")\n    if not isinstance(rank, int) or isinstance(rank, bool):\n        return None\n    if not isinstance(rollout, int) or isinstance(rollout, bool):\n        return None\n    return rank, rollout\n\n\ndef _trajectory_candidate_key(\n    row: dict[str, Any],\n) -> tuple[bool, float, int, int, str]:\n    turn = row.get("turn")\n    turn_value = turn if isinstance(turn, int) and not isinstance(turn, bool) else -1\n    return (\n        row.get("visible_pass") is True,\n        float(row.get("demo_score") or 0.0),\n        turn_value,\n        -int(row.get("program_cost") or 0),\n        str(row.get("run_id") or ""),\n    )\n\n\ndef _validate_selection_candidates(\n    task_id: str, task: dict[str, Any], candidates: list[dict[str, Any]]\n) -> None:\n    """Reject malformed in-memory candidates before ranking or clustering."""\n\n    seen_run_ids: set[str] = set()\n    for index, row in enumerate(candidates):\n        context = f"{task_id}.candidates[{index}]"\n        if not isinstance(row, dict):\n            raise RuntimeError(f"{context}: candidate must be an object")\n        if _trajectory_id(row) is None:\n            raise RuntimeError(f"{context}: candidate has no valid rank/rollout identity")\n        turn = row.get("turn")\n        if turn is not None and (\n            isinstance(turn, bool) or not isinstance(turn, int) or turn < 0\n        ):\n            raise RuntimeError(f"{context}: turn must be a non-negative integer")\n        run_id = row.get("run_id")\n        if not isinstance(run_id, str) or not run_id:\n            raise RuntimeError(f"{context}: run_id must be non-empty")\n        if run_id in seen_run_ids:\n            raise RuntimeError(f"{context}: duplicate run_id {run_id!r}")\n        seen_run_ids.add(run_id)\n        if not isinstance(row.get("visible_pass"), bool):\n            raise RuntimeError(f"{context}: visible_pass must be boolean")\n        demo_score = row.get("demo_score")\n        if (\n            isinstance(demo_score, bool)\n            or not isinstance(demo_score, (int, float))\n            or not math.isfinite(float(demo_score))\n            or not 0.0 <= float(demo_score) <= 1.0\n        ):\n            raise RuntimeError(f"{context}: demo_score must be finite in 0..1")\n        program_cost = row.get("program_cost")\n        if (\n            isinstance(program_cost, bool)\n            or not isinstance(program_cost, int)\n            or program_cost < 0\n        ):\n            raise RuntimeError(f"{context}: program_cost must be non-negative")\n        if not _valid_outputs(task_id, task, row.get("test_outputs")):\n            raise RuntimeError(f"{context}: candidate test outputs are invalid")\n\n\ndef _trajectory_representatives(\n    candidates: list[dict[str, Any]],\n) -> list[dict[str, Any]]:\n    """Choose one vote-bearing candidate per independent trajectory."""\n\n    best: dict[tuple[int, int], dict[str, Any]] = {}\n    for row in candidates:\n        trajectory = _trajectory_id(row)\n        if trajectory is None:\n            continue\n        current = best.get(trajectory)\n        if current is None or _trajectory_candidate_key(row) > _trajectory_candidate_key(\n            current\n        ):\n            best[trajectory] = row\n    return [best[key] for key in sorted(best)]\n\n\ndef _selection_key(row: dict[str, Any]) -> tuple[Any, ...]:\n    """Order hypotheses by visible evidence without conflating repairs.\n\n    ``support`` contains only the one vote-bearing state selected from each\n    independent ``(rank, rollout)`` trajectory.  Older repair states remain\n    useful pass@2 hypotheses, but their raw provenance can break ties only\n    after independent support and can never manufacture another vote.\n    """\n\n    return (\n        row["visible_pass"],\n        row["demo_score"],\n        row["visible_pass_support"],\n        row["support"],\n        row["model_derived"],\n        row["raw_visible_pass_trajectory_support"],\n        row["raw_trajectory_support"],\n        row["replica_support"],\n        row["rollout_support"],\n        -row["program_cost"],\n        row["representative_id"],\n    )\n\n\ndef load_candidates(\n    output_root: Path,\n    task_id: str,\n    task: dict[str, Any],\n    *,\n    collection_contract: dict[str, Any] | None = None,\n) -> list[dict[str, Any]]:\n    rows = []\n    directory = output_root / "candidates" / task_id\n    if not directory.is_dir():\n        return rows\n    if collection_contract is None:\n        raise RuntimeError(\n            f"{task_id}: candidate directory exists without a verified collection contract"\n        )\n    collection_id = collection_contract["collection_id"]\n    contract_sha256 = _digest(collection_contract)\n    seen_run_ids: set[str] = set()\n    schedule = collection_contract["schedule"]\n    world_size = int(schedule["world_size"])\n    rollouts = int(schedule["rollouts"])\n    max_turns = int(schedule["max_turns"])\n    for path in sorted(directory.glob("*.json")):\n        try:\n            row = json.loads(path.read_text())\n        except (OSError, json.JSONDecodeError) as exc:\n            raise RuntimeError(f"{path}: invalid candidate JSON: {exc}") from exc\n        if not isinstance(row, dict):\n            raise RuntimeError(f"{path}: candidate must be an object")\n        if row.get("schema") != CANDIDATE_SCHEMA:\n            raise RuntimeError(f"{path}: unexpected candidate schema")\n        declared_record_sha256 = _require_sha256(\n            row.get("candidate_record_sha256"),\n            context=f"{path}: candidate_record_sha256",\n        )\n        record_payload = {\n            key: value\n            for key, value in row.items()\n            if key != "candidate_record_sha256"\n        }\n        if declared_record_sha256 != _digest(record_payload):\n            raise RuntimeError(f"{path}: candidate record digest does not verify")\n        if row.get("collection_id") != collection_id:\n            raise RuntimeError(\n                f"{path}: candidate belongs to a different or unattributed collection"\n            )\n        if row.get("collection_contract_sha256") != contract_sha256:\n            raise RuntimeError(f"{path}: candidate collection-contract digest mismatch")\n        if row.get("challenge_sha256") != collection_contract["challenge_sha256"]:\n            raise RuntimeError(f"{path}: candidate challenge digest mismatch")\n        if row.get("model") != collection_contract["model"]:\n            raise RuntimeError(f"{path}: candidate model identity mismatch")\n        if row.get("runtime_receipt_sha256") != collection_contract["runtime"].get(\n            "receipt_sha256"\n        ):\n            raise RuntimeError(f"{path}: candidate runtime identity mismatch")\n        if row.get("task_id") != task_id:\n            raise RuntimeError(f"{path}: candidate task ID does not match its directory")\n        trajectory = _trajectory_id(row)\n        if trajectory is None:\n            raise RuntimeError(f"{path}: candidate has no valid rank/rollout identity")\n        rank, rollout = trajectory\n        turn = row.get("turn")\n        if not 0 <= rank < world_size:\n            raise RuntimeError(f"{path}: rank is outside the collection world size")\n        if not 0 <= rollout < rollouts:\n            raise RuntimeError(f"{path}: rollout is outside the collection schedule")\n        if isinstance(turn, bool) or not isinstance(turn, int) or not 0 <= turn < max_turns:\n            raise RuntimeError(f"{path}: turn is outside the collection schedule")\n        expected_name = f"rank{rank}-rollout{rollout}-turn{turn}.json"\n        if path.name != expected_name:\n            raise RuntimeError(\n                f"{path}: candidate filename does not match trajectory identity"\n            )\n        expected_run_id = f"{task_id}-rank{rank}-rollout{rollout}-turn{turn}"\n        run_id = row.get("run_id")\n        if run_id != expected_run_id:\n            raise RuntimeError(f"{path}: candidate run_id does not match trajectory")\n        if run_id in seen_run_ids:\n            raise RuntimeError(f"{path}: duplicate candidate run_id {run_id!r}")\n        seen_run_ids.add(run_id)\n        if row.get("source") != "qwen38-nf4-replica":\n            raise RuntimeError(f"{path}: unexpected candidate source")\n        if not isinstance(row.get("visible_pass"), bool):\n            raise RuntimeError(f"{path}: visible_pass must be boolean")\n        demo_score = row.get("demo_score")\n        if (\n            isinstance(demo_score, bool)\n            or not isinstance(demo_score, (int, float))\n            or not math.isfinite(float(demo_score))\n            or not 0.0 <= float(demo_score) <= 1.0\n        ):\n            raise RuntimeError(f"{path}: demo_score must be finite in 0..1")\n        program = row.get("program")\n        if not isinstance(program, str) or not program.strip():\n            raise RuntimeError(f"{path}: program must be non-empty")\n        program_sha256 = _require_sha256(\n            row.get("program_sha256"), context=f"{path}: program_sha256"\n        )\n        if program_sha256 != hashlib.sha256(program.encode()).hexdigest():\n            raise RuntimeError(f"{path}: program_sha256 does not match program bytes")\n        program_cost = row.get("program_cost")\n        if (\n            isinstance(program_cost, bool)\n            or not isinstance(program_cost, int)\n            or program_cost != len(program)\n        ):\n            raise RuntimeError(f"{path}: program_cost does not match program length")\n        if not _valid_outputs(task_id, task, row.get("test_outputs")):\n            raise RuntimeError(f"{path}: candidate test outputs are invalid")\n        test_outputs_sha256 = _require_sha256(\n            row.get("test_outputs_sha256"),\n            context=f"{path}: test_outputs_sha256",\n        )\n        if test_outputs_sha256 != _digest(row["test_outputs"]):\n            raise RuntimeError(\n                f"{path}: test_outputs_sha256 does not match test outputs"\n            )\n        rows.append(row)\n    return rows\n\n\ndef validate_submission_completeness(\n    challenges: dict[str, Any], submission: Any\n) -> None:\n    """Assert a submission is structurally complete and grid-valid.\n\n    The single reusable definition of "a submittable file": exactly the\n    challenge task set, one attempt object per test query, each object exactly\n    {attempt_1, attempt_2}, and every attempt a valid ARC grid (1..30 dims,\n    colours 0..9).  Reused by :func:`aggregate`, the last-resort fallback path,\n    and the offline robustness harness so the guarantee has one source of truth.\n    """\n\n    if not isinstance(submission, dict):\n        raise ValueError("submission must be a task->rows object")\n    if set(submission) != set(challenges):\n        missing = sorted(set(challenges) - set(submission))\n        extra = sorted(set(submission) - set(challenges))\n        raise ValueError(\n            f"submission task set mismatch (missing={missing}, extra={extra})"\n        )\n    for task_id, task in challenges.items():\n        rows = submission[task_id]\n        if not isinstance(rows, list) or len(rows) != len(task["test"]):\n            raise ValueError(f"{task_id}: wrong submission output count")\n        for index, row in enumerate(rows):\n            if not isinstance(row, dict) or set(row) != {"attempt_1", "attempt_2"}:\n                raise ValueError(f"{task_id}[{index}]: malformed attempt object")\n            for name in ("attempt_1", "attempt_2"):\n                validate_grid(row[name], context=f"{task_id}[{index}].{name}")\n\n\ndef guaranteed_fallback_submission(\n    challenges: dict[str, Any],\n) -> dict[str, list[dict[str, Any]]]:\n    """Build a structurally-complete submission that cannot fail to be valid.\n\n    This is the robustness floor for the BASE: it depends only on the already\n    validated public test *inputs* (never on train outputs, the collection\n    contract, model candidates, or the rich selection machinery), so it can be\n    written to disk before the model even loads and can never itself raise for a\n    validated challenge set.  attempt_1 is the test input (identity); attempt_2\n    is its left-right mirror (a distinct-but-valid second guess).  A better\n    submission from :func:`aggregate` overwrites it on success; if every richer\n    path fails, this remains and the run still submits a valid pass@2 file.\n    """\n\n    submission: dict[str, list[dict[str, Any]]] = {}\n    for task_id, task in challenges.items():\n        rows: list[dict[str, Any]] = []\n        for pair in task["test"]:\n            grid = pair["input"]\n            mirror = [list(reversed(row)) for row in grid]\n            rows.append({"attempt_1": grid, "attempt_2": mirror})\n        submission[task_id] = rows\n    # Fail loud if this floor is ever not valid: it means the input contract\n    # was violated upstream, which must surface, not be silently shipped.\n    validate_submission_completeness(challenges, submission)\n    return submission\n\n\ndef select_task(\n    task_id: str, task: dict[str, Any], candidates: list[dict[str, Any]]\n) -> tuple[list[dict[str, Any]], dict[str, Any]]:\n    fallback_sources = (\n        "fallback-identity",\n        "fallback-background",\n        "fallback-flip-lr",\n        "fallback-flip-ud",\n        "fallback-transpose",\n        "fallback-rotate-cw",\n        "fallback-rotate-ccw",\n        "fallback-rotate-180",\n    ) + tuple(f"fallback-constant-{color}" for color in range(10))\n    fallback_outputs = {\n        source: _fallback_outputs(task, source) for source in fallback_sources\n    }\n    _validate_selection_candidates(task_id, task, candidates)\n    submission_rows = []\n    query_receipts = []\n    trajectory_candidates = _trajectory_representatives(candidates)\n    for query_index in range(len(task["test"])):\n        # One trajectory can revise its output over several turns. Select its\n        # strongest state before clustering so it cannot vote for competing\n        # hypotheses at the same cutoff.\n        groups: dict[str, list[dict[str, Any]]] = {}\n        for candidate in trajectory_candidates:\n            grid = candidate["test_outputs"][query_index]\n            groups.setdefault(_digest(grid), []).append(candidate)\n        raw_groups: dict[str, list[dict[str, Any]]] = {}\n        for candidate in candidates:\n            grid = candidate["test_outputs"][query_index]\n            raw_groups.setdefault(_digest(grid), []).append(candidate)\n        ranked = []\n        for output_sha256, raw_members in raw_groups.items():\n            members = groups.get(output_sha256, [])\n            # A vote-bearing representative must come from the trajectory\n            # representatives themselves.  Otherwise a superseded state with\n            # the same test output could silently inflate attempt-1\'s visible\n            # score.  Correlated-only clusters have no such member, so their\n            # best raw state is used for attempt-2 ranking only.\n            representative = max(\n                members or raw_members,\n                key=_trajectory_candidate_key,\n            )\n            replicas = sorted(\n                {\n                    int(row["rank"])\n                    for row in members\n                    if isinstance(row.get("rank"), int)\n                }\n            )\n            rollouts = sorted(\n                {\n                    int(row["rollout"])\n                    for row in members\n                    if isinstance(row.get("rollout"), int)\n                }\n            )\n            trajectories = {\n                (int(row.get("rank", -1)), int(row.get("rollout", -1)))\n                for row in members\n            }\n            visible_trajectories = {\n                (int(row.get("rank", -1)), int(row.get("rollout", -1)))\n                for row in members\n                if row.get("visible_pass") is True\n            }\n            raw_trajectories = {\n                (int(row.get("rank", -1)), int(row.get("rollout", -1)))\n                for row in raw_members\n            }\n            raw_visible_trajectories = {\n                (int(row.get("rank", -1)), int(row.get("rollout", -1)))\n                for row in raw_members\n                if row.get("visible_pass") is True\n            }\n            correlated_only = not members\n            ranked.append(\n                {\n                    "grid": representative["test_outputs"][query_index],\n                    "output_sha256": output_sha256,\n                    "source": str(\n                        representative.get("source") or "qwen38-nf4-replica"\n                    ),\n                    "representative_id": str(representative.get("run_id") or ""),\n                    "visible_pass": representative.get("visible_pass") is True,\n                    "demo_score": float(representative.get("demo_score") or 0.0),\n                    "program_cost": int(representative.get("program_cost") or 0),\n                    "model_derived": True,\n                    "correlated_only": correlated_only,\n                    "attempt_1_eligible": not correlated_only,\n                    "hypothesis_origin": (\n                        "correlated-repair-state"\n                        if correlated_only\n                        else "independent-trajectory-vote"\n                    ),\n                    # Multiple repair turns within one rollout are correlated\n                    # and count as one vote. Consensus means independent\n                    # rank/rollout trajectories, not raw candidate files.\n                    "support": len(trajectories),\n                    "independent_support": len(trajectories),\n                    "raw_candidate_support": len(raw_members),\n                    "raw_trajectory_support": len(raw_trajectories),\n                    "raw_visible_pass_trajectory_support": len(\n                        raw_visible_trajectories\n                    ),\n                    "visible_pass_support": len(visible_trajectories),\n                    "replica_support": len(replicas),\n                    "rollout_support": len(rollouts),\n                    "supporting_replicas": replicas,\n                    "supporting_rollouts": rollouts,\n                    "supporting_ids": sorted(\n                        str(row.get("run_id") or "") for row in members\n                    ),\n                    "raw_supporting_ids": sorted(\n                        str(row.get("run_id") or "") for row in raw_members\n                    ),\n                }\n            )\n        # Visible evidence is primary.  Among equally fitting programs, the\n        # output with the most independent support wins attempt 1.  Attempt 2\n        # is the strongest distinct cluster, maximizing pass@2 diversity.\n        ranked.sort(key=_selection_key, reverse=True)\n        for source in fallback_sources:\n            grid = fallback_outputs[source][query_index]\n            fallback_pass, fallback_score = _fallback_demo_score(task, source)\n            ranked.append(\n                {\n                    "grid": grid,\n                    "output_sha256": _digest(grid),\n                    "source": source,\n                    "representative_id": source,\n                    "visible_pass": fallback_pass,\n                    "demo_score": fallback_score,\n                    "program_cost": 0,\n                    "model_derived": False,\n                    "correlated_only": False,\n                    "attempt_1_eligible": True,\n                    "hypothesis_origin": "deterministic-fallback",\n                    "support": 0,\n                    "independent_support": 0,\n                    "raw_candidate_support": 0,\n                    "raw_trajectory_support": 0,\n                    "raw_visible_pass_trajectory_support": 0,\n                    "visible_pass_support": 0,\n                    "replica_support": 0,\n                    "rollout_support": 0,\n                    "supporting_replicas": [],\n                    "supporting_rollouts": [],\n                    "supporting_ids": [],\n                    "raw_supporting_ids": [],\n                }\n            )\n        ranked.sort(key=_selection_key, reverse=True)\n\n        # A superseded repair state is deliberately ineligible for attempt 1:\n        # its trajectory cast its sole independent vote for another state.\n        # It remains eligible for attempt 2, where preserving a strong,\n        # distinct model-derived hypothesis is preferable to erasing it and\n        # falling back to an unrelated geometric guess.\n        attempt_1_pool = [row for row in ranked if row["attempt_1_eligible"]]\n        if not attempt_1_pool:\n            raise RuntimeError(\n                f"{task_id}.test[{query_index}]: no attempt-1-eligible hypothesis"\n            )\n        selected = [attempt_1_pool[0]]\n        seen = {attempt_1_pool[0]["output_sha256"]}\n        for row in ranked:\n            if row["output_sha256"] in seen:\n                continue\n            selected.append(row)\n            seen.add(row["output_sha256"])\n            if len(selected) == 2:\n                break\n        if len(selected) != 2:\n            raise RuntimeError(\n                f"{task_id}.test[{query_index}]: fewer than two distinct "\n                "evidence-ranked output hypotheses survived selection"\n            )\n        submission_rows.append(\n            {"attempt_1": selected[0]["grid"], "attempt_2": selected[1]["grid"]}\n        )\n        query_receipts.append(\n            {\n                "query_index": query_index,\n                "candidate_clusters": len(groups),\n                "raw_candidate_clusters": len(raw_groups),\n                "correlated_only_candidate_clusters": len(\n                    set(raw_groups) - set(groups)\n                ),\n                "selected": [\n                    {key: row[key] for key in row if key != "grid"}\n                    for row in selected\n                ],\n            }\n        )\n    return submission_rows, {\n        "candidate_count": len(candidates),\n        "trajectory_candidate_count": len(trajectory_candidates),\n        "visible_pass_candidates": sum(\n            row.get("visible_pass") is True for row in candidates\n        ),\n        "queries": query_receipts,\n    }\n\n\ndef aggregate(\n    challenges_path: Path, output_root: Path, submission_path: Path\n) -> tuple[dict[str, Any], dict[str, Any]]:\n    challenges, challenge_sha256 = load_public_challenges_with_digest(\n        challenges_path\n    )\n    collection_contract = load_collection_contract(\n        output_root,\n        challenges_path,\n        challenges=challenges,\n        challenge_sha256=challenge_sha256,\n    )\n    submission = {}\n    tasks = {}\n    for task_id, task in challenges.items():\n        candidates = load_candidates(\n            output_root,\n            task_id,\n            task,\n            collection_contract=collection_contract,\n        )\n        rows, receipt = select_task(task_id, task, candidates)\n        submission[task_id] = rows\n        tasks[task_id] = receipt\n    validate_submission_completeness(challenges, submission)\n    _write_json(submission_path, submission)\n    selected_sources = Counter(\n        selected["source"]\n        for task in tasks.values()\n        for query in task["queries"]\n        for selected in query["selected"]\n    )\n    receipt = {\n        "schema": "arc-qwen38-kaggle-pass2-aggregation-v1",\n        "observed_utc": datetime.now(timezone.utc).isoformat(),\n        "selection_scope": "per-test-query",\n        "collection_id": (\n            collection_contract["collection_id"] if collection_contract else None\n        ),\n        "collection_contract_sha256": (\n            _digest(collection_contract) if collection_contract else None\n        ),\n        "selection_policy": (\n            "attempt-1 excludes superseded repair states; then visible-pass, "\n            "demo-score, independent visible-pass support, independent "\n            "trajectory support, model-derived provenance, raw trajectory "\n            "provenance, replica/rollout spread, shorter-program; attempt-2 "\n            "is the strongest distinct hypothesis including zero-vote "\n            "superseded repair states"\n        ),\n        "challenge_sha256": challenge_sha256,\n        "submission_sha256": hashlib.sha256(submission_path.read_bytes()).hexdigest(),\n        "task_count": len(challenges),\n        "query_count": sum(len(task["test"]) for task in challenges.values()),\n        "candidate_count": sum(row["candidate_count"] for row in tasks.values()),\n        "tasks_with_candidate": sum(row["candidate_count"] > 0 for row in tasks.values()),\n        "tasks_with_visible_pass_candidate": sum(\n            row["visible_pass_candidates"] > 0 for row in tasks.values()\n        ),\n        "selected_source_counts": dict(sorted(selected_sources.items())),\n        "tasks": tasks,\n        "status": "pass",\n    }\n    _write_json(output_root / "aggregation_receipt.json", receipt)\n    return submission, receipt\n\n\ndef _emit_summary(summary: dict[str, Any]) -> None:\n    print(\n        "QWEN38_PASS2_AGGREGATION=" + json.dumps(summary, sort_keys=True),\n        flush=True,\n    )\n\n\ndef main() -> int:\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument("--challenges", type=Path, required=True)\n    parser.add_argument("--output-root", type=Path, required=True)\n    parser.add_argument("--submission", type=Path, required=True)\n    parser.add_argument(\n        "--guarantee-only",\n        action="store_true",\n        help=(\n            "Write only the dependency-free guaranteed-fallback submission and "\n            "exit 0. The kernel calls this first so a valid, complete "\n            "submission exists on disk before the model loads or any candidate "\n            "is produced."\n        ),\n    )\n    args = parser.parse_args()\n\n    # ROBUSTNESS FLOOR: a valid, complete submission must exist on disk no\n    # matter what fails later.  Build it from the validated public inputs and\n    # write it before anything richer is attempted.  If the challenges\n    # themselves are unreadable there is genuinely nothing to submit, so that\n    # (and only that) is allowed to raise.\n    challenges = load_public_challenges(args.challenges)\n    _write_json(args.submission, guaranteed_fallback_submission(challenges))\n\n    base_summary = {\n        "schema": "arc-qwen38-kaggle-pass2-aggregation-v1",\n        "task_count": len(challenges),\n        "query_count": sum(len(task["test"]) for task in challenges.values()),\n    }\n    if args.guarantee_only:\n        _emit_summary({**base_summary, "status": "guaranteed-fallback"})\n        return 0\n\n    try:\n        _, receipt = aggregate(args.challenges, args.output_root, args.submission)\n        # Defensive: whatever landed on disk must be a complete, valid file.\n        validate_submission_completeness(\n            challenges, json.loads(args.submission.read_text())\n        )\n    except Exception as exc:  # never let aggregation abort the submit\n        # Fail LOUD but keep submitting: rewrite the guaranteed floor and record\n        # the failure so a degraded run is visible, not silently shipped.\n        degraded = {\n            "schema": "arc-qwen38-kaggle-aggregation-degraded-v1",\n            "observed_utc": datetime.now(timezone.utc).isoformat(),\n            "error": f"{type(exc).__name__}: {exc}",\n            "traceback": traceback.format_exc(),\n            "fallback": "guaranteed-fallback-submission-retained",\n        }\n        try:\n            _write_json(\n                args.output_root / "aggregation_degraded_receipt.json", degraded\n            )\n        except OSError:\n            pass\n        _write_json(args.submission, guaranteed_fallback_submission(challenges))\n        print(\n            "LOUD: aggregation failed; retained guaranteed-fallback submission. "\n            f"{type(exc).__name__}: {exc}",\n            file=sys.stderr,\n            flush=True,\n        )\n        _emit_summary(\n            {\n                **base_summary,\n                "status": "degraded-guaranteed-fallback",\n                "error": f"{type(exc).__name__}: {exc}",\n            }\n        )\n        return 0\n\n    _emit_summary(\n        {\n            key: receipt[key]\n            for key in (\n                "schema",\n                "observed_utc",\n                "collection_id",\n                "submission_sha256",\n                "task_count",\n                "query_count",\n                "candidate_count",\n                "tasks_with_candidate",\n                "tasks_with_visible_pass_candidate",\n                "selected_source_counts",\n                "status",\n            )\n        }\n    )\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'
AGGREGATOR_SHA256 = "6327567d3ea70a084c017e50438f6d5ab438d1468d2c70af4608a40c87cf9811"
RUN_STARTED_S = time.time()

# ---- EXACT reproduced setting + budgeting knobs (surfaced for review/tuning) ----
# 57,344-token reasoning+output budget. vLLM enforces prompt + max_tokens <=
# max_model_len (65,536). The v8 pass@1 diagnostic truncated the HARDEST xhigh
# tasks at 49,152 (6/11 samples hit the cap with no grid), yet measured ARC
# prompts are only ~1-5K tokens (biggest: four 30x30 grids ~4.7K). The old ~16K
# prompt reserve was wasteful, so we give the model ~8K more room (reserve ~8K
# for the prompt) so hard tasks can finish. recover_grid backstops the true tail.
SETTINGS = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 57344,
            "reasoning_effort": "xhigh", "enable_thinking": True}
RERUN = os.environ.get("KAGGLE_IS_COMPETITION_RERUN")
# Concurrency: the v7 probe measured max-num-seqs 24 as SAFE and effective on the
# 4xL4 -- vLLM sustained ~24 concurrent with ZERO preemption at 97.2% KV, ~255
# tok/s sustained (~1.6x v6's 158 at C=8). 97.2% KV is the ceiling for these long
# ~27K-token xhigh samples, so 24 is about the max for full xhigh.
CONCURRENCY = 24           # in-flight tasks; = vLLM --max-num-seqs AND client workers
# COMFORTABLE PASS@1: one full-xhigh attempt per query. At ~255 tok/s sustained,
# 240 single attempts ~= 7h -- fits the 12h wall with ~5h of margin and ZERO
# timeout risk (full pass@2 was a marginal ~11-14h). attempt_2 stays the robust
# independently-shaped floor. This gives every hidden task one real xhigh answer
# and a clean end-to-end TIMING measurement on the true 240-task rerun.
SAMPLES_PER_TASK = 1
# The 1-hour saturation throughput probe is now OPT-IN via ARC_FREEFORM_PROBE
# (it already measured C=24). Default non-rerun push = a normal small diagnostic.
PROBE = (not RERUN) and bool(os.environ.get("ARC_FREEFORM_PROBE"))
LIMIT = None if RERUN else 8   # non-rerun = small diagnostic task set
# The floor is incremental and crash-durable, so reserve minutes rather than a
# fixed half-hour. New calls are admitted only when a rolling runtime quantile
# says they should finish before GENERATION_DEADLINE_S. A hard cutoff still
# terminates vLLM if a tail request violates that prediction.
HARD_WALL_S = RUN_STARTED_S + (11.95 * 3600 if RERUN else (1.0 if PROBE else 1.4) * 3600)
FINALIZATION_RESERVE_S = 3 * 60 if RERUN else 60
GENERATION_DEADLINE_S = HARD_WALL_S - FINALIZATION_RESERVE_S

def log(*a): print(*a, flush=True)

# ---------- 1. L4x4 gate ----------
def gpu_names():
    try:
        out = subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
                             check=True, capture_output=True, text=True).stdout
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:
        return []
names = gpu_names()
log("GPUs:", names)

# Sample driver-reported resident memory throughout model load + generation.
# A final snapshot alone misses the true peak; server metadata is not sufficient.
GPU_MEMORY_LOCK = threading.Lock()
GPU_MEMORY_STOP = threading.Event()
GPU_MEMORY_PEAKS = {}
GPU_MEMORY_SAMPLE_COUNT = [0]
GPU_MEMORY_ERRORS = []

def _sample_gpu_memory():
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout
        rows = []
        for raw in out.splitlines():
            fields = [field.strip() for field in raw.split(",")]
            if len(fields) != 4:
                continue
            try:
                index, used, total = int(fields[0]), int(fields[2]), int(fields[3])
            except ValueError:
                continue
            rows.append({"index": index, "name": fields[1],
                         "peak_used_mib": used, "total_mib": total})
        with GPU_MEMORY_LOCK:
            GPU_MEMORY_SAMPLE_COUNT[0] += 1
            for row in rows:
                old = GPU_MEMORY_PEAKS.get(row["index"])
                if old is None or row["peak_used_mib"] > old["peak_used_mib"]:
                    GPU_MEMORY_PEAKS[row["index"]] = row
    except Exception as exc:
        with GPU_MEMORY_LOCK:
            if len(GPU_MEMORY_ERRORS) < 8:
                GPU_MEMORY_ERRORS.append(type(exc).__name__ + ":" + str(exc)[:160])

def _gpu_memory_loop():
    _sample_gpu_memory()
    while not GPU_MEMORY_STOP.wait(5.0):
        _sample_gpu_memory()

def gpu_memory_summary():
    _sample_gpu_memory()
    with GPU_MEMORY_LOCK:
        devices = [dict(GPU_MEMORY_PEAKS[index]) for index in sorted(GPU_MEMORY_PEAKS)]
        count = int(GPU_MEMORY_SAMPLE_COUNT[0])
        errors = list(GPU_MEMORY_ERRORS)
    return {"status": "measured" if devices else "unavailable",
            "sampler": "nvidia-smi-driver-memory",
            "interval_seconds": 5.0,
            "sample_count": count, "devices": devices, "errors": errors}

def stop_gpu_memory_sampler():
    GPU_MEMORY_STOP.set()
    GPU_MEMORY_THREAD.join(timeout=10)

GPU_MEMORY_THREAD = threading.Thread(
    target=_gpu_memory_loop, name="qwen38-gpu-memory", daemon=True)
GPU_MEMORY_THREAD.start()

# ---------- 2. strict challenge snapshot + GUARANTEED FLOOR first ----------
# Import the single-source hardened finalizer embedded by the builder. Do not
# duplicate challenge/submission invariants in this kernel.
AGG_PATH = Path("/tmp/qwen38_freeform_aggregate_submission.py")
AGG_PATH.write_text(AGGREGATOR_SOURCE)
spec = importlib.util.spec_from_file_location("qwen38_freeform_aggregator", AGG_PATH)
AGG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AGG)

challenge_name = (
    "arc-agi_test_challenges.json" if RERUN
    else "arc-agi_evaluation_challenges.json"
)
challenge_hits = sorted(glob.glob(f"/kaggle/input/**/{challenge_name}", recursive=True))
if len(challenge_hits) != 1:
    raise RuntimeError(
        f"expected exactly one public challenge source for {challenge_name!r}; "
        f"found {challenge_hits!r}"
    )
ch_path = challenge_hits[0]
challenges, challenge_sha256 = AGG.load_public_challenges_with_digest(Path(ch_path))
task_ids = list(challenges)
if LIMIT is not None:
    task_ids = task_ids[:LIMIT]
log("challenges:", ch_path, "| tasks:", len(task_ids))

submission = AGG.guaranteed_fallback_submission(challenges)
fallback_submission = json.loads(json.dumps(submission))
def save_submission():
    # Never allow a malformed candidate to destroy the already-valid floor.
    AGG.validate_submission_completeness(challenges, submission)
    AGG._write_json(SUB, submission)
save_submission()            # a complete submission exists before any model work
log("guaranteed identity+mirror floor written:", SUB, "tasks:", len(submission))

# ---------- 3. install vLLM offline + serve FP8 TP4 ----------
def find_model_root():
    for cfg in glob.glob("/kaggle/input/**/config.json", recursive=True):
        d = Path(cfg).parent
        if list(d.glob("*.safetensors")): return str(d)
    return None
def find_wheelhouse():
    hits = glob.glob("/kaggle/input/**/vllm-*.whl", recursive=True) + glob.glob("/kaggle/input/**/vllm_*.whl", recursive=True)
    return str(Path(hits[0]).parent) if hits else None
model_root, wheelhouse = find_model_root(), find_wheelhouse()
served = False
if model_root and wheelhouse and len(names) == 4 and all("L4" in n.upper() for n in names):
    os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
    ins = subprocess.run([sys.executable,"-m","pip","install","--no-index","--no-warn-conflicts",
                          "--disable-pip-version-check","--find-links",wheelhouse,"vllm"],
                         capture_output=True, text=True)
    if ins.returncode == 0:
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"]="0"
        # 65,536 context (the proven L4x4 FP8 fit) so prompt + 32,768 output fits
        # (vLLM enforces prompt_tokens + max_tokens <= max_model_len). max-num-seqs
        # = the in-flight request concurrency (each task does 2 SEQUENTIAL samples).
        serve = [sys.executable,"-m","vllm.entrypoints.openai.api_server",
                 "--model", model_root, "--served-model-name", MODEL_NAME,
                 "--tensor-parallel-size","4","--max-model-len","65536",
                 "--gpu-memory-utilization","0.90","--enable-prefix-caching",
                 "--max-num-seqs", str(CONCURRENCY),
                 "--reasoning-parser","qwen3","--language-model-only",
                 "--dtype","bfloat16","--port",str(PORT)]
        log("serving:", " ".join(serve))
        srv_log = open(WORK/"vllm_server.log","w")
        server = subprocess.Popen(serve, stdout=srv_log, stderr=subprocess.STDOUT)
        def ready(deadline):
            while time.time() < deadline:
                if server.poll() is not None: return False
                try:
                    with socket.create_connection(("127.0.0.1",PORT),timeout=3):
                        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models",timeout=5) as r:
                            if r.status < 500: return True
                except Exception: pass
                time.sleep(5)
            return False
        served = ready(time.time()+1200)
        log("vLLM ready:", served)
    else:
        log("vLLM install failed:", ins.stderr[-1500:])
else:
    log("serving prerequisites not met (model/wheelhouse/L4x4). Floor-only submission stands.")

# ---------- 4. free-form direct-answer solve ----------
def render(g): return "\n".join(" ".join(str(v) for v in row) for row in g)
def build_prompt(task):
    parts = ["You are solving an abstract reasoning puzzle. Each example has an "
             "INPUT grid and its OUTPUT grid (digits 0-9 are colors; 0 is usually "
             "background). Infer the ONE rule that maps every input to its output, "
             "then apply it to the FINAL INPUT.\n"]
    for i, p in enumerate(task["train"]):
        parts.append("Example %d INPUT:\n%s\nExample %d OUTPUT:\n%s\n" %
                     (i+1, render(p["input"]), i+1, render(p["output"])))
    parts.append("FINAL INPUT:\n%s\n" % render(task["test_input"]))
    parts.append("Reason step by step however you like. Then give ONLY the final "
                 "output grid as rows of space-separated digits inside a fenced "
                 "block:\n```grid\n<rows>\n```\nOutput nothing after the block.")
    return "\n".join(parts)
def parse_grid(text):
    m = re.findall(r"```(?:grid|text)?\s*\n(.*?)```", text or "", re.DOTALL)
    cand = m[-1] if m else (text or "")
    rows = []
    for line in cand.strip().splitlines():
        toks = line.strip().split()
        if toks and all(t.isdigit() and len(t)==1 for t in toks):
            rows.append([int(t) for t in toks])
        elif rows:
            break
    return rows or None

def validated_model_grid(grid):
    if grid is None:
        return None
    try:
        AGG.validate_grid(grid, context="freeform model grid")
    except Exception as exc:
        log("rejecting malformed model grid:", type(exc).__name__, str(exc)[:200])
        return None
    return grid

def request_timeout_s(default_s):
    # Bound network waits by the actual generation wall rather than allowing a
    # 3600s urllib timeout to hold ThreadPoolExecutor threads past the deadline.
    remaining = GENERATION_DEADLINE_S - time.time()
    return max(5, min(default_s, remaining))

def ask(prompt, seed):
    # NOTE: vLLM rejects `reasoning_effort` (a llama.cpp-only knob) with HTTP 400.
    # On vLLM the xhigh-equivalent is enable_thinking=True + the --reasoning-parser,
    # with the thinking budget bounded by max_tokens. So we do NOT send it here.
    body = {"model": MODEL_NAME, "messages":[{"role":"user","content":prompt}],
            "temperature":SETTINGS["temperature"], "top_p":SETTINGS["top_p"],
            "top_k":SETTINGS["top_k"], "max_tokens":SETTINGS["max_tokens"], "seed":seed,
            "chat_template_kwargs":{"enable_thinking":SETTINGS["enable_thinking"]}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(req, timeout=request_timeout_s(3600)) as r:
        p = json.loads(r.read().decode())
    ch=p["choices"][0]; msg=ch["message"]
    return {"grid": validated_model_grid(parse_grid(msg.get("content") or "")),
            "finish": ch.get("finish_reason"),
            "content": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or "",
            "completion_tokens": (p.get("usage") or {}).get("completion_tokens"),
            "wall_s": round(time.time()-t0,1)}

def recover_grid(prompt, prior_reasoning):
    """A sample truncated at the token limit did all the reasoning but never
    emitted the grid. Feed the tail of that reasoning back and force ONLY the
    grid out, thinking OFF (cheap) — recover the answer instead of wasting it."""
    body = {"model": MODEL_NAME, "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "[my reasoning, truncated at the token limit]\n" + prior_reasoning[-8000:]},
                {"role": "user", "content": "You hit the token limit mid-reasoning. Do NOT reason further. "
                                            "Based on the reasoning above, output ONLY your best-guess final "
                                            "grid now as rows of space-separated digits in a fenced block:\n"
                                            "```grid\n<rows>\n```"}],
            "temperature": SETTINGS["temperature"], "top_p": SETTINGS["top_p"], "top_k": SETTINGS["top_k"],
            "max_tokens": 4096, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=request_timeout_s(600)) as r:
        p = json.loads(r.read().decode())
    return validated_model_grid(parse_grid(p["choices"][0]["message"].get("content") or ""))

per_task = {}
model_answered = set()
observed_walls = []
# --- throughput probe accounting ---
GEN_START_S = [None]            # set when the first sample is launched (post-serve)
SAMPLE_LOG = []                 # (t_complete, completion_tokens) per finished sample
PEAK_CONCURRENCY = [0]          # max in-flight requests actually reached

def predicted_sample_s():
    """Rolling near-wall admission cost, intentionally conservative at first.

    The packet reports roughly 40-minute long-tail samples at high depth. Until
    this run has enough measurements, use 40 minutes. Thereafter use a padded
    empirical p90, which adapts to the real saved-version/rerun workload rather
    than banking a fixed 30-minute reserve or launching blindly near the kill.
    """
    if len(observed_walls) < 8:
        return 40 * 60
    s = sorted(observed_walls)
    q90 = s[max(0, math.ceil(0.90 * len(s)) - 1)]
    return max(10 * 60, min(60 * 60, 1.10 * q90 + 60))

def safe_to_launch_sample():
    # The probe is a throughput measurement: keep the batch full until the hard
    # 1h cutoff kills the server (partial tail samples are expected and fine).
    if PROBE:
        return time.time() < GENERATION_DEADLINE_S
    return time.time() + predicted_sample_s() < GENERATION_DEADLINE_S

def solve_one(tid, qi, test_input, slot):
    prompt = build_prompt({"train": challenges[tid]["train"], "test_input": test_input})
    try:
        o = ask(prompt, seed=1000*qi + slot)
        if o.get("finish") == "length" and o.get("grid") is None and o.get("reasoning"):
            try:
                rg = recover_grid(prompt, o["reasoning"])
                if rg: o["grid"] = rg; o["finish"] = "length+recovered"
            except Exception:
                pass
    except urllib.error.HTTPError as exc:
        body = ""
        try: body = exc.read().decode()[:300]
        except Exception: pass
        o = {"grid": None, "finish": "http%d:%s" % (exc.code, body), "completion_tokens": None, "wall_s": None}
    except Exception as exc:
        o = {"grid": None, "finish": "error:%s" % type(exc).__name__, "completion_tokens": None, "wall_s": None}
    return tid, qi, slot, o

def record_sample(tid, qi, slot, o):
    """Publish a model answer without ever weakening submission validity."""
    grid = validated_model_grid(o.get("grid"))
    if grid is not None:
        key = (tid, qi)
        if key not in model_answered:
            # First genuine model hypothesis is attempt_1. Keep the independently
            # shaped fallback attempt_2 until another genuine model sample exists.
            submission[tid][qi]["attempt_1"] = grid
            model_answered.add(key)
        else:
            a1 = submission[tid][qi]["attempt_1"]
            if grid != a1:
                submission[tid][qi]["attempt_2"] = grid
        save_submission()
    wall = o.get("wall_s")
    if isinstance(wall, (int, float)) and wall > 0:
        observed_walls.append(float(wall))
    ct = o.get("completion_tokens")
    if isinstance(ct, (int, float)) and ct > 0:
        SAMPLE_LOG.append((time.time(), int(ct)))
    per_task.setdefault(tid, {}).setdefault(qi, []).append(
        {"slot": slot, "finish": o.get("finish"),
         "completion_tokens": o.get("completion_tokens"),
         "wall_s": wall, "grid": grid is not None})

def stop_server_for_deadline():
    if not served:
        return
    try:
        log("hard generation cutoff reached; terminating vLLM to release in-flight requests")
        server.terminate()
        server.wait(timeout=20)
    except Exception:
        try: server.kill()
        except Exception: pass

def run_phase(work, *, phase_name):
    """Run one scheduling phase with predictive admission and a hard wall."""
    iterator = iter(work)
    submitted = completed = 0
    exhausted = False
    ex = ThreadPoolExecutor(max_workers=CONCURRENCY)
    futs = {}
    try:
        def maybe_launch():
            nonlocal submitted, exhausted
            while len(futs) < CONCURRENCY and not exhausted and safe_to_launch_sample():
                try:
                    w = next(iterator)
                except StopIteration:
                    exhausted = True
                    return
                if GEN_START_S[0] is None:
                    GEN_START_S[0] = time.time()
                futs[ex.submit(solve_one, *w)] = w
                submitted += 1
            if len(futs) > PEAK_CONCURRENCY[0]:
                PEAK_CONCURRENCY[0] = len(futs)

        maybe_launch()
        while futs:
            if time.time() >= GENERATION_DEADLINE_S:
                stop_server_for_deadline()
                break
            done_set, _ = wait(
                list(futs), timeout=min(5, max(0.1, GENERATION_DEADLINE_S-time.time())),
                return_when=FIRST_COMPLETED,
            )
            if not done_set:
                continue
            for fut in done_set:
                w = futs.pop(fut)
                try:
                    tid, qi, slot, o = fut.result()
                    record_sample(tid, qi, slot, o)
                    completed += 1
                except Exception as exc:
                    log("sample failed:", w[0], type(exc).__name__, str(exc)[:160])
            if completed and completed % 20 == 0:
                log("  %s: %d completed / %d submitted; %.1fh left; predicted next %.1fmin" %
                    (phase_name, completed, submitted,
                     (GENERATION_DEADLINE_S-time.time())/3600,
                     predicted_sample_s()/60))
            maybe_launch()
        return {"submitted": submitted, "completed": completed, "exhausted": exhausted}
    finally:
        # At a hard cutoff vLLM has been terminated above, so running HTTP calls
        # fail promptly; cancel anything that never started and reap the pool.
        for fut in futs:
            fut.cancel()
        ex.shutdown(wait=True, cancel_futures=True)

def attempt2_risk_key(item):
    tid, qi, _ = item
    rows = per_task.get(tid, {}).get(qi, [])
    first = next((row for row in rows if row.get("slot") == 0), None)
    if first is None:
        return (4, 0, 0)
    finish = str(first.get("finish") or "")
    no_grid = not first.get("grid")
    recovered = "recovered" in finish or finish == "length"
    return (
        3 if no_grid else (2 if recovered else 1),
        int(first.get("completion_tokens") or 0),
        float(first.get("wall_s") or 0),
    )

if served and task_ids and PROBE:
    # 1-HOUR THROUGHPUT PROBE: keep the batch full at C=CONCURRENCY by cycling the
    # available queries with an increasing slot so we saturate the hardware and
    # measure real aggregate tokens/s + the safe KV ceiling. Ends at the 1h wall.
    queries = [(tid, qi, q["input"]) for tid in task_ids for qi, q in enumerate(challenges[tid]["test"])]
    log("PROBE: saturating C=%d over %d queries (cycling slots) until the 1h wall; cap=%d tok" %
        (CONCURRENCY, len(queries), SETTINGS["max_tokens"]))
    def _probe_work():
        slot = 0
        while True:
            for (tid, qi, inp) in queries:
                yield (tid, qi, inp, slot)
            slot += 1
    phase1 = run_phase(_probe_work(), phase_name="probe")
    phase2 = {"submitted": 0, "completed": 0, "exhausted": False}
    save_submission()
elif served and task_ids:
    queries = [(tid, qi, q["input"]) for tid in task_ids for qi, q in enumerate(challenges[tid]["test"])]
    log("solving: attempt_1 coverage for all %d queries; C=%d, cap=%d tok, generation wall in %.1fh" %
        (len(queries), CONCURRENCY, SETTINGS["max_tokens"],
         (GENERATION_DEADLINE_S - time.time())/3600))
    phase1 = run_phase(
        [(tid, qi, inp, 0) for (tid, qi, inp) in queries], phase_name="attempt_1")

    # Never spend on a second hypothesis until every query got its first model
    # launch. Then spend the bank on the shakiest label-free cases first:
    # missing answer > length/recovery > long/high-token normal completion.
    phase2 = {"submitted": 0, "completed": 0, "exhausted": False}
    if SAMPLES_PER_TASK < 2:
        log("pass@1: one xhigh attempt per query; attempt_2 stays the robust floor")
    elif phase1["exhausted"] and phase1["completed"] == len(queries) and safe_to_launch_sample():
        ordered = sorted(queries, key=attempt2_risk_key, reverse=True)
        phase2 = run_phase(
            [(tid, qi, inp, 1) for (tid, qi, inp) in ordered], phase_name="attempt_2")
    else:
        log("no attempt_2 bank: attempt_1 coverage incomplete or no safely-admissible sample remains")
    save_submission()
else:
    phase1 = phase2 = {"submitted": 0, "completed": 0, "exhausted": False}

# ---------- 5. finalize + budgeting receipt ----------
walls = [o["wall_s"] for t in per_task.values() for q in t.values() for o in q if o.get("wall_s")]
toks = [o["completion_tokens"] for t in per_task.values() for q in t.values() for o in q if o.get("completion_tokens")]

def _server_log_stats():
    """Parse scheduler telemetry, including tokens still in flight."""
    try:
        text = (WORK / "vllm_server.log").read_text(errors="ignore")
    except Exception:
        text = ""
    return parse_vllm_scheduler_log(text)

gpu_memory = gpu_memory_summary()

def _throughput():
    finished_tokens = sum(ct for _, ct in SAMPLE_LOG)
    window = None
    if GEN_START_S[0] is not None:
        window = max(1e-6, time.time() - GEN_START_S[0])
    finished_lower_bound = (round(finished_tokens / window, 1)
                            if window else None)
    server_stats = _server_log_stats()
    scheduler_rate = server_stats.get("decode_tokens_per_second")
    aggregate_rate = scheduler_rate or finished_lower_bound
    rate_source = ("vllm_scheduler_log_mean" if scheduler_rate
                   else ("finished_responses_lower_bound"
                         if finished_lower_bound else None))
    estimated_all_tokens = (round(aggregate_rate * window)
                            if aggregate_rate and window else finished_tokens)
    mean_ct = (sum(toks) / len(toks)) if toks else None
    projection_h = None
    if aggregate_rate and mean_ct:
        projection_h = round(
            240 * SAMPLES_PER_TASK * mean_ct / aggregate_rate / 3600, 2)
    return {
        "measured_aggregate_tokens_per_s": aggregate_rate,
        "throughput_measurement_source": rate_source,
        "lower_bound_finished_response_tokens_per_s": finished_lower_bound,
        "prefill_tokens_per_second": server_stats["prefill_tokens_per_second"],
        "decode_tokens_per_second": server_stats["decode_tokens_per_second"],
        "peak_concurrency_observed": PEAK_CONCURRENCY[0],
        "max_num_seqs_configured": CONCURRENCY,
        "wall_seconds": round(window, 1) if window else None,
        "generation_window_s": round(window, 1) if window else None,
        "finished_completion_tokens": finished_tokens,
        "estimated_generated_tokens_including_inflight": estimated_all_tokens,
        "generated_token_accounting": (
            "scheduler-rate estimate includes in-flight work; "
            "finished_completion_tokens is the exact completed-response floor"),
        "samples_finished": len(SAMPLE_LOG),
        "projection_samples_per_task": SAMPLES_PER_TASK,
        "projected_240task_wall_h_at_this_rate": projection_h,
        "projected_240task_pass2_wall_h_at_this_rate": (
            projection_h if SAMPLES_PER_TASK == 2 else None),
        "fits_12h_wall": (projection_h is not None and projection_h <= 11.5),
        "peak_per_device_memory_mib": gpu_memory["devices"],
        "gpu_memory_sampler": gpu_memory,
        **server_stats,
        "note": "Use the scheduler-log mean for saturated probes because it "
                "includes running requests; finished-response usage alone is "
                "only a lower bound. Projection uses 240*SAMPLES_PER_TASK*"
                "mean_completion_tokens / aggregate_rate.",
    }
throughput = _throughput()
log("THROUGHPUT:", json.dumps(throughput, indent=1))
receipt = {
    "schema": "arc-qwen38-freeform-direct-v1",
    "rerun": bool(RERUN), "probe": bool(PROBE), "gpus": names, "served": served,
    "settings": SETTINGS, "concurrency": CONCURRENCY, "samples_per_task": SAMPLES_PER_TASK,
    "throughput": throughput,
    "challenge_path": ch_path, "n_tasks_total": len(challenges), "n_tasks_attempted": len(per_task),
    "challenge_sha256": challenge_sha256,
    "aggregator_sha256": AGGREGATOR_SHA256,
    "submission_robustness": {
        "floor": "single-source guaranteed_fallback_submission (identity+mirror)",
        "challenge_validation": "single-source load_public_challenges_with_digest",
        "every_publish_validated": True,
        "atomic_writer": "single-source aggregate_submission._write_json",
    },
    "deadline_policy": {
        "hard_wall_s_from_start": round(HARD_WALL_S - RUN_STARTED_S, 1),
        "finalization_reserve_s": FINALIZATION_RESERVE_S,
        "generation_deadline_s_from_start": round(GENERATION_DEADLINE_S - RUN_STARTED_S, 1),
        "launch_admission": "40m prior until 8 samples, then padded empirical p90",
        "phase1": phase1,
        "phase2": phase2,
        "attempt2_priority": "missing-grid > length/recovered > completion_tokens > wall_s",
        "last_predicted_sample_s": round(predicted_sample_s(), 1),
    },
    "budgeting": {
        "mean_wall_s_per_sample": round(sum(walls)/len(walls),1) if walls else None,
        "max_wall_s_per_sample": max(walls) if walls else None,
        "mean_completion_tokens": round(sum(toks)/len(toks)) if toks else None,
        "max_completion_tokens": max(toks) if toks else None,
        "samples_completed": len(walls),
        "note": "use mean/max wall to project a 12h/240-task budget at this concurrency; a length-finish rate near 1.0 means the 32K cap is truncating and should be raised or reasoning bounded.",
    },
    "per_task": per_task,
}
RECEIPT.write_text(json.dumps(receipt, indent=1))
log("RECEIPT:", json.dumps({k:receipt[k] for k in ("served","n_tasks_attempted","budgeting")}, indent=1))
log("submission tasks:", len(json.load(open(SUB))))
stop_gpu_memory_sampler()
try:
    server.terminate(); server.wait(timeout=30)
except Exception:
    pass

