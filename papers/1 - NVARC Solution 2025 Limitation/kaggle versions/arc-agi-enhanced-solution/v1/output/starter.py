
"""
starter.py
==========
Spawns one worker process per GPU, distributes puzzle keys via a shared
queue, and coordinates sequential loading to avoid Unsloth import races.
"""

import os
import time
import json
import torch
import argparse
import torch.multiprocessing as mp


def gpu_worker(rank: int, queue, end_time: float):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.set_default_device("cpu")

    # Unsloth patches global state at import time — stagger worker starts
    # so only one process initialises Unsloth at a time.
    if rank > 0:
        sentinel = f"/kaggle/worker{rank - 1}.ready"
        while not os.path.exists(sentinel):
            time.sleep(5)

    from arc_solver import worker  # deferred import after stagger

    # Signal next worker that it is safe to import
    with open(f"/kaggle/worker{rank}.ready", "w") as fh:
        fh.write("ok")

    print(f"[GPU {rank}] initialised — starting puzzle loop")
    worker(rank, queue, end_time)
    print(f"[GPU {rank}] finished")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ARC-AGI-2 multi-GPU solver")
    parser.add_argument("--end-time", type=float, default=0.0,
                        help="Unix timestamp at which workers should stop")
    parser.add_argument("--gpus",     type=int,   default=4,
                        help="Number of GPU workers to spawn")
    args = parser.parse_args()

    is_rerun = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
    test_path = (
        "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
        if is_rerun else
        "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"
    )

    with open(test_path) as fh:
        tasks = json.load(fh)

    manager = mp.Manager()
    queue   = manager.Queue()

    for key in sorted(tasks.keys()):
        # In local-eval mode, restrict to a small validation subset
        if not is_rerun:
            if key not in {"0934a4d8", "36a08778", "981571dc", "aa4ec2a5"}:
                continue
        queue.put(key)

    # Sentinel values signal workers to stop
    for _ in range(args.gpus):
        queue.put(None)

    mp.spawn(gpu_worker, args=(queue, args.end_time), nprocs=args.gpus)
