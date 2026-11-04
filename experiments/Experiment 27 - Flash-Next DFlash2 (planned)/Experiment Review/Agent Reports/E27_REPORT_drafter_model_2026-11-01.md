# E27 drafter-Model agent (a606955a3965f1c67): no final report

The agent ended its turn with "Waiting." and never wrote a final report (no SubagentHandback message in `subagents/agent-a606955a3965f1c67.jsonl`). The facts below come from its tool outputs and the kernel files, not from a report.

- **Downloaded:** `incoai/Qwen3.8-Flash-Next-DFlash2` @ `a22a1744e85c17364dcc54a6a420af7135d4c7f6`.
  - `model.safetensors`: 1,206,140,008 B, sha256 `d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f`.
  - 81 BF16 tensors, header 8,800 B, data 1,206,131,200 B; no embed / lm_head.
  - Tensors seen: `candidate_selector.{hidden_projection [256,2560], predecessor_codebook [248320,256], successor_codebook [248320,256]}`, `fc [2560,12800]`, `hidden_norm`, `layers.N.attention_conv.*`, …
- **Kaggle Model:** `foysalemonshanto/qwen38-flash-next-dflash2` created, private.
  - The first instance create failed: "Should not specify other base model when setting ModelInstanceType to Unspecified". The fix was `externalBaseModelUrl` "".
  - A second upload of the plain `bf16` instance (task bo04xk460) stopped mid-`model.safetensors` when the agent ended. The instance does not exist (check-only: `instance exists: False`).
- **Repack kernel:** `foysalemonshanto/q27-e27-dflash2-embed-repack` (v364422784), COMPLETE in 2.1 min, REPACK OK. Contents of `repack_report.json`:
  - W4 `model-00015-of-00017.safetensors` and klee100 `model-00002-of-00011.safetensors` `model.language_model.embed_tokens.weight`: tensor sha256 `62288a3fca91d3575b68924a54ab73629b86b609284d3fcd059a309146c9be3a`, 1,271,398,400 B, mask row 248077 sha `860615491bdf6786178b8c14e96397f51e3895c556d0336e2b5ad8cc8e061a19`, norm 0.26273465156555176, zero rows 0. **byte_identical: true.**
  - Variant `bf16-embed` (target w4): `model.safetensors` 2,477,538,608 B sha `d2e0464d15936265bb75eda9d09149d27ac6986327ca269a0335cd3d49661d49`; `config.json` 1,236 B sha `2409575986f9ea24d455343eb469c0e6940def5edab1d1be503bb8eb4cc3b1b9`.
- **Local download of the repacked file:** interrupted at 0 B. D: has 5.1 GB free, so it was not retried; the Model is created from the kernel output on Kaggle.
- **Agent scripts** (scratchpad): `dflash2_assemble.py`, `dflash2_upload.py`, `dflash2_upload_variant.py`.
