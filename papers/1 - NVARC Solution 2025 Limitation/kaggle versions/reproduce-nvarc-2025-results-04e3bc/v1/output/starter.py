import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import time


MODEL_PATH = Path(
    "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"
)
NPROCS = 4


def validate_model_path(model_path=MODEL_PATH):
    """Validate only the exact Kaggle model path confirmed by inspection."""
    model_path = Path(model_path)
    config_path = model_path / "config.json"

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Confirmed model directory does not exist: {model_path}"
        )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Model config.json is missing: {config_path}"
        )

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read model config: {config_path}"
        ) from exc

    model_type = config.get("model_type")
    architectures = config.get("architectures") or []

    if model_type != "qwen3":
        raise RuntimeError(
            f"Unexpected model_type={model_type!r} in {config_path}"
        )

    if "Qwen3ForCausalLM" not in architectures:
        raise RuntimeError(
            f"Unexpected architectures={architectures!r} in {config_path}"
        )

    weight_files = sorted(model_path.glob("*.safetensors"))
    if not weight_files:
        weight_files = sorted(model_path.glob("pytorch_model*.bin"))

    tokenizer_files = [
        model_path / name
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
        )
        if (model_path / name).is_file()
    ]

    if not weight_files:
        raise FileNotFoundError(
            f"No model weight files found in {model_path}"
        )

    if not tokenizer_files:
        raise FileNotFoundError(
            f"No tokenizer files found in {model_path}"
        )

    print(f"[Starter] Confirmed model path: {model_path}", flush=True)
    print(
        f"[Starter] Model type: {model_type}; "
        f"architectures: {architectures}",
        flush=True,
    )
    print(
        f"[Starter] Weight files: {len(weight_files)}; "
        f"tokenizer files: {len(tokenizer_files)}",
        flush=True,
    )

    return str(model_path)


def discover_gpu_tokens(nprocs=NPROCS):
    """
    Return concrete GPU identifiers without importing torch.

    Existing CUDA_VISIBLE_DEVICES is respected when it contains enough
    entries. Otherwise nvidia-smi indices are used.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()

    if visible and visible not in {"-1", "NoDevFiles"}:
        tokens = [
            token.strip()
            for token in visible.split(",")
            if token.strip()
        ]
        if len(tokens) >= nprocs:
            return tokens[:nprocs]

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "NVIDIA GPUs are not visible. Enable a 4-GPU Kaggle "
            "accelerator before running this notebook."
        ) from exc

    tokens = [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    ]

    if len(tokens) < nprocs:
        raise RuntimeError(
            f"This notebook requires {nprocs} visible GPUs, "
            f"but nvidia-smi reported {len(tokens)}: {tokens}"
        )

    return tokens[:nprocs]


def local_worker(rank, gpu_token, queue, end_time, model_path):
    # This must happen before importing torch, Unsloth, or arc_solver.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_token)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["ARC_MODEL_PATH"] = model_path

    # Serialize the first Unsloth import/patch across workers.
    if rank > 0:
        previous_marker = Path(f"/kaggle/worker{rank - 1}")
        while not previous_marker.exists():
            time.sleep(2)

    from arc_solver import worker

    marker_path = Path(f"/kaggle/worker{rank}")
    marker_path.write_text("Ok", encoding="utf-8")

    print(
        f"[Rank {rank}] start on CUDA_VISIBLE_DEVICES={gpu_token}",
        flush=True,
    )

    worker(rank, queue, end_time)

    print(f"[Rank {rank}] done!", flush=True)


def run_workers(queue, end_time, gpu_tokens, model_path):
    ctx = mp.get_context("spawn")
    processes = []

    for rank, gpu_token in enumerate(gpu_tokens):
        process = ctx.Process(
            target=local_worker,
            args=(rank, gpu_token, queue, end_time, model_path),
            name=f"arc-worker-{rank}",
        )
        process.start()
        processes.append(process)

    # Detect a failed worker promptly. This prevents later ranks from
    # waiting forever for an Unsloth marker that will never be created.
    while any(process.is_alive() for process in processes):
        failed = [
            process
            for process in processes
            if process.exitcode not in (None, 0)
        ]

        if failed:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()

            details = ", ".join(
                f"{process.name}={process.exitcode}"
                for process in failed
            )
            raise RuntimeError(f"Worker process failed: {details}")

        time.sleep(1)

    for process in processes:
        process.join()

    failed = [
        process
        for process in processes
        if process.exitcode != 0
    ]
    if failed:
        details = ", ".join(
            f"{process.name}={process.exitcode}"
            for process in failed
        )
        raise RuntimeError(f"Worker process failed: {details}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    args = parser.parse_args()

    model_path = validate_model_path()
    gpu_tokens = discover_gpu_tokens(NPROCS)

    print(f"[Starter] GPU assignments: {gpu_tokens}", flush=True)

    # Remove stale markers from earlier failed/rerun attempts.
    for rank in range(NPROCS):
        marker = Path(f"/kaggle/worker{rank}")
        if marker.exists():
            marker.unlink()

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    if rerun_mode:
        test_path = (
            "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/"
            "arc-agi_test_challenges.json"
        )
    else:
        test_path = (
            "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/"
            "arc-agi_evaluation_challenges.json"
        )

    with open(test_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    for key in sorted(data.keys()):
        if not rerun_mode:
            if key not in [
                "0934a4d8",
                "36a08778",
                "981571dc",
                "aa4ec2a5",
            ]:
                continue
        queue.put(key)

    for _ in range(NPROCS):
        queue.put(None)

    run_workers(
        queue=queue,
        end_time=args.end_time,
        gpu_tokens=gpu_tokens,
        model_path=model_path,
    )
