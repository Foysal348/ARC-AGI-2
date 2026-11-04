import os
import time
import json
import argparse
import torch.multiprocessing as mp

# ──────────────────────────────────────────────────────────────
#   DEBUG_PUZZLES Environment Variable 
#   Example: export DEBUG_PUZZLES="0934a4d8,36a08778"
#   If not set or empty, runs ALL puzzles.
# ──────────────────────────────────────────────────────────────

#DEBUG_PUZZLES = ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]

debug_env = os.getenv("DEBUG_PUZZLES", "")
if debug_env.strip():
    DEBUG_PUZZLES = [x.strip() for x in debug_env.split(",")]
    print(f"Running in DEBUG mode for puzzles: {DEBUG_PUZZLES}")
else:
    DEBUG_PUZZLES = None
    print("Running for ALL puzzles (DEBUG_PUZZLES is None)")

def local_worker(rank, queue, end_time):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["UNSLOTH_DISABLE_RL_PATCH"] = "1"
    os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

    # Simple sync: wait for previous worker's ready file with 300s Timeout
    if rank > 0:
        start_wait = time.time()
        while not os.path.exists(f"/kaggle/worker{rank-1}"):
            time.sleep(5)
            if time.time() - start_wait > 300:  # 300 seconds (5 minutes) timeout
                raise TimeoutError(f"Worker {rank}: Timeout waiting for worker {rank-1} to create sync file.")

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

    if DEBUG_PUZZLES is not None:
        for key in sorted(data.keys()):
            if key in DEBUG_PUZZLES:
                queue.put(key)
    else:
        for key in sorted(data.keys()):
            queue.put(key)

    for _ in range(4):
        queue.put(None)

    mp.spawn(local_worker, args=(queue, end_time), nprocs=4)
