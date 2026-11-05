# Trace-generation engine handoff (2026-11-04)

For: a teammate generating ARC solution traces on Kaggle 4x L4 for the Gemma 4 developer agent competition. Everything is on Kaggle; nothing has to be downloaded.

## Engine: Domdolus Tolus (best public ARC-AGI-2 engine, LB 72.36-83.47 over 5 submissions)

| piece | Kaggle link |
|---|---|
| notebook (engine + harness, all patches inside) | https://www.kaggle.com/code/darrenamadeusmartin/domdolus-tolus-arc-agi-2 |
| model: Qwen3.8 Flash-Next, 2-bit GSQ Q2_0 experts (all experts, **not pruned**), FP8 attention / DeltaNet / HC / lm_head, version 2 | https://www.kaggle.com/models/darrenamadeusmartin/qwen38-flash-next-q2-gsq-vllm (PyTorch / default / **2**) |
| utility (offline vLLM 0.31.0 `site-vllm.tar`, `site-arc.tar`, harness repo) | https://www.kaggle.com/code/darrenamadeusmartin/flash-next-utility (notebook output, attached as input) |
| fast startup caches | https://www.kaggle.com/datasets/darrenamadeusmartin/fast-startup |
| their writeup (how it works) | https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/writeups/Domdolus-Tolus-Solution |
| original quantized weights | https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF |

Model size: about 92 GB (dense 6.9 GB, experts 34 GB, n-gram table 51 GB, kept in host RAM / `/dev/shm`). Check the model's LICENSE file for use of its outputs.

## How to run

1. Open the notebook → **Copy & Edit**. The copy keeps all four inputs and the pinned Docker image (Python 3.12 is required; the latest image will fail). Accelerator **GPU L4 x4**, internet off.
2. Tasks: cell 3 finds the competition JSON (`TEST_FILE`). For other tasks, put an ARC-format challenges JSON in an attached dataset and point `TEST_FILE` at it. `VALIDATION_MODE = True` runs the public evaluation set and scores it (6 h budget).
3. Run all. Startup is under 4 minutes. A watcher writes `submission.json` before the 12 h limit.

## Concurrency (the setting that matters for traces)

- Server: `MAX_SEQS = 128` (cell 43). KV pool 8.81 GiB per GPU = 4,064,286 tokens, plus a 30 GiB host-RAM tier.
- Harness: `CONCURRENCY = 70` in cell 46 (runs at a time, both tries). Their measured speed: 651 tok/s decode at 60 runs, 94 % prefix hit, 0 preemptions.
- For maximum traces: raise `CONCURRENCY` toward 96-128 (never above `MAX_SEQS`). Not measured above 70: watch the one-minute load line from `/metrics` (decode tok/s, preemptions) for the first 15 minutes and step back if preemptions appear or tok/s falls.
- Each run can use up to 200k tokens (`MAX_GEN_TOKENS`); attention window per run is cut back to 49,152 tokens each time it reaches 65,536.

## Where the traces are

- Per run: `transcript.md` + `images/` in the task's own folder, and `check_attempts.json`, under `WORK = "/tmp/arc_run"` (cell 3).
- `/tmp` is **not** saved by Kaggle, and `/kaggle/working` holds about 19 GB.
- Size: 12 h at 651 tok/s is about 28M decoded tokens, about 110 MB of text (under 50 MB gzipped). The `images/` folders are what fill the disk; images can be redrawn from the grid JSON, so do not keep them.
- Add this cell before the harness starts. It saves every text file (no images) as one `traces.tar.gz` every 30 minutes. Call `save_traces()` once more in the last cell and before the deadline watcher's `pkill -9`.

```python
import tarfile, threading, time, os, pathlib
SRC, DST = pathlib.Path("/tmp/arc_run"), "/kaggle/working/traces.tar.gz"
KEEP = (".md", ".json", ".jsonl", ".log", ".txt", ".py")
def save_traces():
    tmp = DST + ".part"
    with tarfile.open(tmp, "w:gz", compresslevel=6) as t:
        for f in SRC.rglob("*"):
            if f.is_file() and f.suffix in KEEP and "images" not in f.parts:
                t.add(f, arcname=str(f.relative_to(SRC)))
    os.replace(tmp, DST)          # an interrupted save never breaks the last good file
    print(f"[TRACES] saved {os.path.getsize(DST)/2**20:.1f} MiB")
def _loop():
    while True:
        time.sleep(1800)
        try: save_traces()
        except Exception as e: print("[TRACES] save failed:", e)
threading.Thread(target=_loop, daemon=True).start()
```

- More runs: each run version has its own output; make a Kaggle dataset from each run's output to keep them all together.
