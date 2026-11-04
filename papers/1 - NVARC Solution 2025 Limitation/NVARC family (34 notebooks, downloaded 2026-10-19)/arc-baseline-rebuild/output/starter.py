import hashlib
import json
import os
import pickle
import random
import shutil
import sys
import time
from pathlib import Path


def sanitize_runtime_path():
    source_raw = os.environ.get("PROGRAM063_LEGACY_ROOT")
    overlay_raw = os.environ.get("PROGRAM063_OVERLAY_ROOT")
    if not source_raw or not overlay_raw:
        raise RuntimeError("Program No. 063 runtime path variables are missing")
    source_root = Path(source_raw).resolve()
    overlay_root = Path(overlay_raw).resolve()
    clean = []
    for item in sys.path:
        if not item:
            clean.append(item)
            continue
        try:
            resolved = Path(item).resolve()
        except OSError:
            continue
        if resolved == source_root or source_root in resolved.parents:
            continue
        if resolved == overlay_root:
            continue
        clean.append(item)
    sys.path[:] = [str(overlay_root)] + clean


sanitize_runtime_path()

import numpy as np
import torch
import argparse
import torch.multiprocessing as mp


STARTER_LOCK = Path("/kaggle/working/program063_starter.lock")
SEED_PROBE_TEXT = "arc-program-seed-probe-v1"


def env_flag(name):
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def seed_contract():
    raw_seed = os.environ.get("PROGRAM063_BASE_SEED")
    if raw_seed is None:
        raise RuntimeError("PROGRAM063_BASE_SEED is missing")
    try:
        base_seed = int(raw_seed)
    except ValueError as exc:
        raise RuntimeError(f"PROGRAM063_BASE_SEED is invalid: {raw_seed!r}") from exc
    python_hash_seed = os.environ.get("PYTHONHASHSEED")
    if python_hash_seed != "0":
        raise RuntimeError(
            f"PROGRAM063_PYTHONHASHSEED_INVALID expected='0' actual={python_hash_seed!r}"
        )
    policy = os.environ.get("PROGRAM063_SEED_POLICY")
    if policy != "rng_only_no_backend_change":
        raise RuntimeError(f"PROGRAM063_SEED_POLICY_INVALID={policy!r}")
    return base_seed, python_hash_seed, policy


def rng_state_report(stage, rank, task_id=None):
    base_seed, python_hash_seed, policy = seed_contract()
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)

    python_state = pickle.dumps(random.getstate(), protocol=4)
    numpy_state = pickle.dumps(np.random.get_state(), protocol=4)
    torch_cpu_state = torch.get_rng_state().cpu().numpy().tobytes()
    cuda_state_hashes = []
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            cuda_state_hashes.append(sha256_bytes(state.cpu().numpy().tobytes()))

    return {
        "stage": stage,
        "rank": rank,
        "task_id": task_id,
        "base_seed": base_seed,
        "python_hash_seed": python_hash_seed,
        "seed_policy": policy,
        "hash_probe": hash(SEED_PROBE_TEXT),
        "python_rng_sha256": sha256_bytes(python_state),
        "numpy_rng_sha256": sha256_bytes(numpy_state),
        "torch_cpu_rng_sha256": sha256_bytes(torch_cpu_state),
        "torch_cuda_rng_sha256": cuda_state_hashes,
        "torch_initial_seed": int(torch.initial_seed()),
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


class SeededQueueProxy:
    def __init__(self, queue, rank):
        self._queue = queue
        self._rank = rank

    def get(self, *args, **kwargs):
        item = self._queue.get(*args, **kwargs)
        if item is not None:
            report = rng_state_report("task_dequeue", self._rank, str(item))
            print("PROGRAM063_TASK_SEED_READY=" + json.dumps(report, sort_keys=True), flush=True)
        return item


def acquire_single_starter_lock():
    try:
        fd = os.open(STARTER_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        owner = STARTER_LOCK.read_text(encoding="utf-8", errors="replace") if STARTER_LOCK.exists() else "unknown"
        raise RuntimeError(f"PROGRAM063_DUPLICATE_STARTER_BLOCKED existing={owner}") from exc
    try:
        os.write(fd, f"pid={os.getpid()} started={time.time():.6f}\n".encode("utf-8"))
    finally:
        os.close(fd)
    print(f"PROGRAM063_STARTER_LOCK_ACQUIRED pid={os.getpid()} path={STARTER_LOCK}", flush=True)


def release_single_starter_lock():
    try:
        STARTER_LOCK.unlink()
    except FileNotFoundError:
        pass


def local_worker(rank, queue, end_time, test_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.set_default_device("cpu")
    worker_seed_report = rng_state_report("worker_init", rank)
    print("PROGRAM063_WORKER_SEED_READY=" + json.dumps(worker_seed_report, sort_keys=True), flush=True)

    if rank > 0:
        previous_marker = Path(f"/kaggle/worker{rank - 1}")
        while not previous_marker.exists():
            if time.time() >= end_time:
                raise TimeoutError(
                    f"Rank {rank} reached the global deadline while waiting for {previous_marker}"
                )
            time.sleep(2)

    # Import the byte-identical Program 024 solver only after this process has
    # received its fixed Python/NumPy/Torch seed state.
    from arc_solver import worker

    Path(f"/kaggle/worker{rank}").write_text(
        json.dumps({"rank": rank, "pid": os.getpid(), "gpu": rank}, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[Rank {rank}] start! pid={os.getpid()} gpu={rank}", flush=True)
    worker(rank, SeededQueueProxy(queue, rank), end_time, test_path)
    print(f"[Rank {rank}] done! pid={os.getpid()} gpu={rank}", flush=True)


def clean_runtime_state():
    output_dir = Path("/kaggle/inference_outputs")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for rank in range(4):
        marker = Path(f"/kaggle/worker{rank}")
        if marker.exists():
            marker.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, required=True)
    parser.add_argument("--challenge-path", type=str, required=True)
    args = parser.parse_args()
    base_seed, python_hash_seed, seed_policy = seed_contract()
    acquire_single_starter_lock()
    try:
        test_path = str(Path(args.challenge_path).resolve())
        if not os.path.isfile(test_path):
            raise FileNotFoundError(f"Challenge file not found: {test_path}")
        with open(test_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or not data:
            raise ValueError("Challenge JSON must be a non-empty object")
        visible_gpus = torch.cuda.device_count()
        if visible_gpus < 4:
            raise RuntimeError(
                f"Program No. 063 requires four visible GPUs; detected {visible_gpus}. "
                "Aborting before model loading."
            )
        torch_file = Path(torch.__file__).resolve()
        source_root = Path(os.environ["PROGRAM063_LEGACY_ROOT"]).resolve()
        if torch_file == source_root or source_root in torch_file.parents:
            raise RuntimeError(f"Program No. 063 loaded legacy Torch unexpectedly: {torch_file}")
        clean_runtime_state()
        rerun_mode = env_flag("KAGGLE_IS_COMPETITION_RERUN")
        selected_keys = []
        for key in sorted(data.keys()):
            if not rerun_mode and key not in ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]:
                continue
            selected_keys.append(key)
        if not selected_keys:
            raise RuntimeError("No ARC tasks were selected")
        manager = mp.Manager()
        queue = manager.Queue()
        for key in selected_keys:
            queue.put(key)
        for _ in range(4):
            queue.put(None)
        print(
            f"PROGRAM063_STARTER_READY parent_pid={os.getpid()} rerun_mode={rerun_mode} "
            f"tasks={len(selected_keys)} gpus={visible_gpus} torch={torch_file} "
            f"base_seed={base_seed} python_hash_seed={python_hash_seed} seed_policy={seed_policy}",
            flush=True,
        )
        mp.spawn(local_worker, args=(queue, args.end_time, test_path), nprocs=4)
        print(
            f"PROGRAM063_STARTER_COMPLETE parent_pid={os.getpid()} workers=4 tasks={len(selected_keys)}",
            flush=True,
        )
    finally:
        release_single_starter_lock()
