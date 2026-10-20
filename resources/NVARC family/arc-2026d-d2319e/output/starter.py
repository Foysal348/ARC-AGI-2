import os
import time
import json
import torch
import argparse
import numpy as np
import torch.multiprocessing as mp

# ──────────────────────────────────────────────────────────────
#   Debug mode: শুধু এই puzzle গুলো manual run এ চলবে
#   Kaggle submit (rerun) এ automatically সব puzzle চলবে
# ──────────────────────────────────────────────────────────────
DEBUG_PUZZLES = ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]


def get_puzzle_priority(key, data):
    train   = data[key]['train']
    n_train = len(train)

    avg_input_size = np.mean([
        np.prod(np.shape(x['input'])) for x in train
    ])
    size_ratios = [
        max(np.prod(np.shape(x['output'])),
            np.prod(np.shape(x['input']))) /
        max(min(np.prod(np.shape(x['output'])),
                np.prod(np.shape(x['input']))), 1)
        for x in train
    ]
    max_ratio = max(size_ratios)
    avg_ratio = np.mean(size_ratios)

    n_colors = len(set(
        int(c)
        for x in train
        for arr in [x['input'], x['output']]
        for c in np.unique(arr).tolist()
    ))
    is_dynamic = any(
        np.shape(x['input']) != np.shape(x['output'])
        for x in train
    )

    if is_dynamic and max_ratio > 30:
        tier = 3
    elif is_dynamic and max_ratio > 5:
        tier = 2
    elif avg_input_size > 400:
        tier = 2
    else:
        tier = 1

    fine = (
        avg_input_size / 900  * 0.30 +
        (avg_ratio - 1) / 10  * 0.25 +
        n_colors / 10         * 0.20 +
        (0.25 if is_dynamic else 0.0)
    )

    if n_train >= 6 and tier < 3:
        fine *= 0.7

    return (tier, fine)


def local_worker(rank, queue, end_time):
    os.environ["CUDA_VISIBLE_DEVICES"]       = str(rank)
    os.environ["UNSLOTH_DISABLE_RL_PATCH"]   = "1"
    os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

    if rank > 0:
        while not os.path.exists(f"/kaggle/worker{rank-1}"):
            time.sleep(5)

    from arc_solver import worker

    with open(f"/kaggle/worker{rank}", "w") as f:
        f.write("Ok")

    print(f"[Rank {rank}] start!")
    worker(rank, queue, end_time)
    print(f"[Rank {rank}] done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0,
                        help="Unix timestamp when execution must stop. If 0, compute 12 hours from now.")
    args = parser.parse_args()

    if args.end_time == 0.0:
        end_time = time.time() + 12 * 3600
    else:
        end_time = args.end_time

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    if rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    with open(test_path, "r") as f:
        data = json.load(f)

    queue = mp.Manager().Queue()

    # ── EDA-based ordering ───────────────────────────────
    keys_sorted = sorted(
        data.keys(),
        key=lambda k: get_puzzle_priority(k, data)
    )

    # Tier breakdown log
    tiers = [get_puzzle_priority(k, data)[0] for k in keys_sorted]
    print(f"[Starter] Puzzle tiers — "
          f"Easy: {tiers.count(1)}, "
          f"Hard: {tiers.count(2)}, "
          f"Extreme: {tiers.count(3)}")
    print(f"[Starter] First: {keys_sorted[0]}, "
          f"Last: {keys_sorted[-1]}")

    # rerun = সব puzzle (ordered), manual = শুধু DEBUG_PUZZLES
    for key in keys_sorted:
        if not rerun_mode:
            if key not in DEBUG_PUZZLES:
                continue
        queue.put(key)

    # 4 sentinel — 4 worker এর জন্য
    for _ in range(4):
        queue.put(None)

    mp.spawn(local_worker, args=(queue, end_time), nprocs=4)
