# Push plan (not pushed yet; pushes only after the team's explicit approval)

| item | value |
|---|---|
| account | foysalemonshanto (17.63 quota-hours left on 2026-10-05 02:03 BST; weekly reset 2026-10-10 06:00 BST) |
| kernel | foysalemonshanto/fn-smoke-flashnext-4xl4, private, accelerator NvidiaL4 (4×L4), internet off |
| image | pinned Kaggle image gcr.io/kaggle-private-byod/python@sha256:37c64f7d… (Python 3.12; the default image moved to Python 3.13 on 2026-10-03 and these cp312 wheels would not install there) |
| inputs | competition arc-prize-2026-arc-agi-2; dataset dfranzen/pennyroyal-v253; model dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1 |
| mode | commit run = LOAD_ONLY probe (stages 0-5); competition rerun = fallback submission only |
| time | stage-0 stop: ~2-5 min; successful load: ~20-40 min; hard cap 45 min |
| quota | worst case ~1.5 quota-hours (45 min × 2) |
