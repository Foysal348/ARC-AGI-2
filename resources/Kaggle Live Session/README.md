# Kaggle live session: run code on a live 4xL4 from Claude Code

First working run: 2026-11-04 (BDT). Account foysalemonshanto, notebook `foysalemonshanto/live-l4-probe`.

## What it gives

Claude Code (on the local Windows PC) sends Python code to a **running Kaggle interactive session** (GPU L4 x4) and gets the outputs back in seconds. The kernel stays alive between calls, so a model loaded once stays loaded.

This replaces build → CPU kit → GPU commit → log for small measurements (patches, kernels, micro-benchmarks). It is the same idea as Domdolus Tolus's "Claude Code + Colab MCP on an L4", but on Kaggle's real 4xL4 machine.

## Pieces

| Piece | Where | Role |
|---|---|---|
| Kaggle MCP (official) | `.mcp.json` in the repo root: `https://www.kaggle.com/mcp`, header `Authorization: Bearer ${KAGGLE_API_TOKEN}` | quota, notebooks (save / batch session / status / output), writeups, datasets, models. **Batch only**: it has no tool that runs code in a live kernel. |
| `KAGGLE_API_TOKEN` | Windows **user** env variable, a new-style `KGAT_...` token | auth for the Kaggle MCP. The old 32-char `kaggle.json` key is **not** accepted (Unauthenticated). |
| Datalayer `jupyter-mcp-server[kaggle]` 2.2.4 | venv `_scratch/jmcp_venv` (Python 3.14) | provides `code_sandboxes.sandboxes.kaggle.client.KaggleKernelClient` (Jupyter kernel client over Kaggle's signed proxy). |
| `run_cell.py` | here and `_scratch/kaggle_live/run_cell.py` | runs code (a string or `--file`) in the live kernel, prints outputs. Auth = the signed proxy URL only (`token=None`), no API token. |
| `channels_url.txt` | `_scratch/kaggle_live/` (git-ignored) | the live session's websocket channels URL. **Secret** (signed JWT): never print it, never paste it in chat, never commit it. |

## One-time setup (done 2026-11-04)

1. `python -m venv _scratch/jmcp_venv` and `pip install "jupyter-mcp-server[kaggle]" jupyter_mcp_sandboxes`.
2. Write `.mcp.json` (above). VS Code Claude Code → `/mcp` shows `kaggle` Connected with 76 tools.
3. Token: the user creates a `KGAT_...` token (Kaggle → Settings → API → Generate New Token), copies it to the clipboard. Claude runs, without printing it:
   ```powershell
   $t = (Get-Clipboard | Out-String).Trim(); if ($t.StartsWith('KGAT_')) { [Environment]::SetEnvironmentVariable('KAGGLE_API_TOKEN', $t, 'User'); Set-Clipboard -Value ' ' }
   ```
   Then restart VS Code fully. Check: the Kaggle MCP tool `get_accelerator_quota` returns the quota.

## Easier official URL (found by the GPT deep research, 2026-11-04; not yet tested by us)

Kaggle officially supports external editors on the running session's Jupyter Server ("Only the backend Jupyter Server and its execution environment are shared between Kaggle and Colab", kaggle.com/docs/notebooks; staff announcement kaggle.com/product-announcements/567740). In the editor: **Run → Kaggle Jupyter Server → Manually connect → VS Code compatible URL** gives `https://kkb-production.jupyter-proxy.kaggle.net/k/<id>/<jwt>/proxy`. Save that in `channels_url.txt` instead of the DevTools websocket URL: `run_cell.py` (2026-11-04 update) sees the `/proxy` ending, lists `/api/kernels` and attaches to the most recently active kernel. Same secrecy rules.

Other public bridges found (not used): JoseGarciaMayen/remote-bridge, 0STG0T/kaggle-kernel-tools (Claude Code MCP), kgz (PyPI), kagglerun (PyPI), huytrao/kaggle-runtime-toolkit-agent-skills. Rules: Kaggle AUP allows ML use, forbids resource abuse / server farming / circumventing limits; avoid public tunnels.

## Every session (about 2 minutes)

1. **User, browser:** open the notebook (e.g. `kaggle.com/code/foysalemonshanto/live-l4-probe/edit`), Accelerator **GPU L4 x4**, start the session, run one cell.
2. **User, browser:** F12 → Network → filter button **Socket** (not the text box). Run a cell again if the list is empty. Pick an entry named `channels?session_id=...` with status **101** and Time **Pending** (still open; a "Finished" one is closed). Right-click → Copy → Copy URL.
3. **User:** paste into Notepad, Save As `D:\ARC AGI 2\_scratch\kaggle_live\channels_url.txt`, type **All files**.
4. **Claude:** check and repair the file without printing it. On 2026-11-04 the saved URL had lost `://`, every `/` and the `?`. It was rebuilt as
   `wss://kkb-production.jupyter-proxy.kaggle.net/k/<number>/<jwt>/proxy/api/kernels/<kernel uuid>/channels?session_id=...`
   and checked with `parse_kaggle_channels_url`.
5. **Claude:** run code:
   ```bash
   _scratch/jmcp_venv/Scripts/python _scratch/kaggle_live/run_cell.py "print(1)"
   _scratch/jmcp_venv/Scripts/python _scratch/kaggle_live/run_cell.py --file my_cell.py --timeout 1800
   ```
   Multi-GPU work: write a script to `/kaggle/working/` from the cell and run it with `torchrun --nproc_per_node=4` (example: `ar_cell.py`).
6. **User:** stop the session when done (⏻ in the editor). The session uses GPU quota for as long as it is open.

## First results (2026-11-04, live session)

- `nvidia-smi`: 4 × NVIDIA L4, 23,034 MiB each; torch 2.11.0+cu130, `cuda.device_count() = 4`.
- NCCL all-reduce, bf16, 4 GPUs (`ar_cell.py`, 50 iterations after 5 warm-up):

  | message | time per all-reduce | bus bandwidth |
  |---|---|---|
  | 16 KiB | 62.6 µs | 0.39 GB/s |
  | 256 KiB | 144.8 µs | 2.71 GB/s |
  | 4 MiB | 1,368.5 µs | 4.60 GB/s |
  | 64 MiB | 21,063.7 µs | 4.78 GB/s |

## What did NOT work (do not retry)

- **Kaggle MCP `create_notebook_session`** = a batch run of the saved notebook (finished in ~10 s, COMPLETE), not a live kernel. With a bare slug it returned "Invalid slug"; with `owner/slug` of a notebook that did not exist, "Permission 'kernelSessions.create' was denied". After `save_notebook` (QuickSave), the run went through but `nvidia-smi` printed nothing (machine shape `NvidiaL4` apparently not applied).
- **Old `kaggle.json` key as the Bearer token** → Unauthenticated. Use a `KGAT_...` token.
- **MCP "Reconnect" / "Clear authentication" in VS Code** did not log in by OAuth. The token header is what works.
- **DevTools text filter "WS"** shows Firebase `channel?gsessionid=...` fetches, not the kernel. Use the **Socket** type button.
- **Datalayer "live mode"** (batch agent + dataset bus, no browser) was not used: it writes the account's API token into a private notebook's source (against our security rule).

## Limits

- One URL per session; a new session needs a new URL from the browser.
- GPU quota: 30 h per week per account; the session burns it while open. Interactive sessions end after 12 h or when idle.
- Interactive sessions use Kaggle's current image (Python 3.13 seen in the batch log), not the old pinned image.
