"""E30 gentail_seam (GENTAIL_CKPT, design "Option B", PROTOTYPE 2026-11-01): env-gated source patch that publishes
the decode block-boundary Mamba state the RecoverSSM commit already writes, so a Duck turn's prefix hit reaches past
the previous prompt into the previous generation (K100Q replay: 70 % of the perfect 2.43 M re-prefill tokens).

Merlin v3.0.0 (vLLM 0.32 fork), commit ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5, tree 07c16a79 (the tree the Kaggle
payload installs). Files (pre-patch line numbers):
  vllm/v1/core/gentail_ckpt.py           NEW: GenTailController + maybe_install_gentail (all the logic)
  vllm/v1/core/sched/scheduler.py        class attr (:113), install after configure_recoverssm_pp_checkpoints
                                         (:392-394), on_schedule in update_after_schedule (:1935-1936), on_step_output
                                         after the rejection adjustment (:2458), drain next to _drain_deferred_frees
                                         (:2354-2355)
  vllm/v1/core/kv_cache_manager.py       class attr _gentail next to MAX_GRACE_PINS (:735), on_free / after_free in
                                         free (:727-733) and pop_blocks_for_free (:1080-1086)
Read, not patched: single_type_kv_cache_manager.py (MambaManager :1592-2521, FullAttentionManager :769-924,
  _retire_state_blocks :1673-1702, remove_skipped_blocks :1947-1978), block_pool.py (cache_partial_block :452-579,
  touch / free_blocks :759-830), kv_cache_coordinator.py (:119, :360-399, :805-962), ops/recoverssm.py (:27-115),
  model_states/recoverssm.py (:114-193), pp_utils.py (:81-175).
NOT touched: pp_utils.py, model_runner.py, gpu_worker.py (E25 nb4_ppack) and every E27 dflash_seam file: the file sets
  are disjoint, so the seams compose in any order (each checks only its own files).

Every inserted line carries "GENTAIL_CKPT". With GENTAIL_CKPT unset (or "0") maybe_install_gentail returns None before
it touches anything, every hook is `if self._gentail is not None`, so the original statements run unchanged (the
only extra work is one attribute test per hook site). The new module is imported once at scheduler init (stdlib
imports only at module level).

What the logic does: see the NEW_MODULE_SRC docstring (vllm/v1/core/gentail_ckpt.py). In short:
  record   per request the newest decode window whose commit plan (exact port of prepare_commit_plan_kernel's align
           branch, fed with the window's schedule-time committed count, query length and sampled count) stored the
           state at a block boundary b; hold that block in every Mamba group (one extra ref, so retirement or free
           drops only the table ref) -> PendingTail(request_id, request_generation, position, block_ids,
           producer_step, valid_tokens, token_hash).
  finish   finished with output, not aborted, same generation, producer_step + pp_size <= last_sched_seq + 1 (pp 2: T < L), b > num_prompt,
           b <= floor(valid/64)*64 - 64 (valid = min(committed, num_tokens) - reprefillable): cache the attention /
           MTP-draft full blocks up to A = floor(valid/64)*64, key A (partial key on block A // 4096, or, at a block
           end, verify the full block A/4096 - 1 is hashed at A), QUEUE the PendingTail behind the fence
           producer_step + pp_size. Never publish here.
  drain    fence passed (processed_step_seq), generation unchanged, b <= limit, token hash unchanged -> key the
           Mamba blocks at b, hand them to the finished turn's grace pin (else LRU), enforce the pin budget.
  release  superseded / skipped / preempted / reused-id holds also wait for the fence.

Env (all unset = the original Merlin behaviour):
  GENTAIL_CKPT=1                    on
  GENTAIL_CKPT_STRICT_TRUNC=0       keep a checkpoint after a stop truncation (default 1 = discard; A/B)
  GENTAIL_CKPT_KEEP_PROMPT_HOLD=1   keep the prompt-tail CoW hold a grid hold supersedes (memory A/B)
  GENTAIL_CKPT_PIN_BUDGET_FRAC=f    cap on all grace-pinned blocks, fraction of the pool (default 0.25)
  GENTAIL_CKPT_LOG_PUBLISH=1        one "publish" line per checkpoint (block ids for the gate-3 state reads)
  GENTAIL_CKPT_NOFENCE=1            NEGATIVE CONTROL ONLY (gate 4): publish at finish, no fence, no producer rule,
                                    no position limit, length-capped / truncated finishes allowed
Log lines (EngineCore stdout): "E30_GENTAIL {json}", events install / inactive / stats.

What this file holds:
  NEW_MODULE_SRC, EDITS {rel: [(anchor, replacement), ...]} (each anchor must occur exactly once before, each
  replacement exactly once after), PRE / POST shas, apply_gentail_patch / verify_gentail_patch / revert_gentail_patch
  (fail-safe: originals restored on any error), tree_state, server_env, parse_log, readback.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

MERLIN_COMMIT = "ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5"
MERLIN_TREE = "07c16a79d077f1e06c19d0eea84efd64f0b63540"
SCH_REL = "vllm/v1/core/sched/scheduler.py"
KVM_REL = "vllm/v1/core/kv_cache_manager.py"
NEW_REL = "vllm/v1/core/gentail_ckpt.py"
@@SHAS@@
MARKER = "GENTAIL_CKPT"
ENV = "GENTAIL_CKPT"
ENV_STRICT_TRUNC = "GENTAIL_CKPT_STRICT_TRUNC"
ENV_KEEP_PROMPT_HOLD = "GENTAIL_CKPT_KEEP_PROMPT_HOLD"
ENV_PIN_BUDGET = "GENTAIL_CKPT_PIN_BUDGET_FRAC"
ENV_NOFENCE = "GENTAIL_CKPT_NOFENCE"
ENV_LOG_PUBLISH = "GENTAIL_CKPT_LOG_PUBLISH"
ENV_STATS_EVERY = "GENTAIL_CKPT_STATS_EVERY"
ALL_ENVS = (ENV, ENV_STRICT_TRUNC, ENV_KEEP_PROMPT_HOLD, ENV_PIN_BUDGET, ENV_NOFENCE, ENV_LOG_PUBLISH, ENV_STATS_EVERY)
NB4_FILES = ("vllm/v1/worker/gpu/pp_utils.py", "vllm/v1/worker/gpu/model_runner.py", "vllm/v1/worker/gpu_worker.py")
DFLASH_FILES = ("vllm/model_executor/models/config.py", "vllm/models/qwen4_exp/nvidia/model.py",
                "vllm/model_executor/models/qwen3_dflash.py", "vllm/v1/worker/gpu/spec_decode/dflash/utils.py",
                "vllm/models/qwen4_exp/common/qsa_cache.py", "vllm/model_executor/layers/attention/attention.py",
                "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py", "vllm/model_executor/models/qwen3_dflash2.py")

NEW_MODULE_SRC = r'''@@MODULE@@'''

EDITS = {
    SCH_REL: [
        (
            "class Scheduler(SchedulerInterface):\n    def __init__(\n",
            "class Scheduler(SchedulerInterface):\n"
            "    # GENTAIL_CKPT (E30 gentail_seam): the generated-tail checkpoint controller\n"
            "    # (None unless GENTAIL_CKPT=1; set in __init__).\n"
            "    _gentail = None\n"
            "\n"
            "    def __init__(\n",
        ),
        (
            "            configure_recoverssm_pp_checkpoints(\n"
            "                self.kv_cache_manager.coordinator.single_type_managers\n"
            "            )\n",
            "            configure_recoverssm_pp_checkpoints(\n"
            "                self.kv_cache_manager.coordinator.single_type_managers\n"
            "            )\n"
            "        # GENTAIL_CKPT (E30 gentail_seam): publish the decode block-boundary\n"
            "        # Mamba state behind a PP fence; None unless GENTAIL_CKPT=1.\n"
            "        from vllm.v1.core.gentail_ckpt import maybe_install_gentail\n"
            "\n"
            "        self._gentail = maybe_install_gentail(self)\n",
        ),
        (
            "            request = self.requests[req_id]\n"
            "            request.num_computed_tokens += num_scheduled_token\n",
            "            request = self.requests[req_id]\n"
            "            if self._gentail is not None:  # GENTAIL_CKPT: commit-plan inputs\n"
            "                self._gentail.on_schedule(\n"
            "                    request, num_scheduled_token, self.sched_step_seq\n"
            "                )\n"
            "            request.num_computed_tokens += num_scheduled_token\n",
        ),
        (
            "            self.processed_step_seq += 1\n"
            "            self._drain_deferred_frees()\n",
            "            self.processed_step_seq += 1\n"
            "            self._drain_deferred_frees()\n"
            "            if self._gentail is not None:  # GENTAIL_CKPT: fenced publishes\n"
            "                self._gentail.drain(self.processed_step_seq)\n",
        ),
        (
            "            # Free encoder inputs only after the step has actually executed.\n",
            "            if self._gentail is not None:  # GENTAIL_CKPT: after the rejections\n"
            "                self._gentail.on_step_output(\n"
            "                    request,\n"
            "                    num_tokens_scheduled,\n"
            "                    len(generated_token_ids),\n"
            "                    self.processed_step_seq,\n"
            "                    output_is_stale,\n"
            "                )\n"
            "\n"
            "            # Free encoder inputs only after the step has actually executed.\n",
        ),
    ],
    KVM_REL: [
        (
            "    MAX_GRACE_PINS = 16\n",
            "    MAX_GRACE_PINS = 16\n"
            "    # GENTAIL_CKPT (E30 gentail_seam): the controller (set by the scheduler,\n"
            "    # None unless GENTAIL_CKPT=1); it raises MAX_GRACE_PINS on the instance.\n"
            "    _gentail = None\n",
        ),
        (
            "        self.release_pinned_hit(request.request_id)\n"
            "        if not self._grace_eligible(request):\n"
            "            self.coordinator.free(request.request_id)\n"
            "            return\n"
            "        blocks = self.coordinator.pop_blocks_for_free(request.request_id)\n"
            "        self._take_grace_pin(request, blocks)\n"
            "        self.block_pool.free_blocks(blocks)\n",
            "        self.release_pinned_hit(request.request_id)\n"
            "        if self._gentail is not None:  # GENTAIL_CKPT: before the table goes\n"
            "            self._gentail.on_free(request)\n"
            "        if not self._grace_eligible(request):\n"
            "            self.coordinator.free(request.request_id)\n"
            "            return\n"
            "        blocks = self.coordinator.pop_blocks_for_free(request.request_id)\n"
            "        self._take_grace_pin(request, blocks)\n"
            "        self.block_pool.free_blocks(blocks)\n"
            "        if self._gentail is not None:  # GENTAIL_CKPT: after the grace pin\n"
            "            self._gentail.after_free()\n",
        ),
        (
            "        self.release_pinned_hit(request.request_id)\n"
            "        blocks = self.coordinator.pop_blocks_for_free(request.request_id)\n"
            "        # A finished turn whose last step is still in flight: the grace takes\n"
            "        # its ref now; the table refs go when the fence drains.\n"
            "        if self._grace_eligible(request):\n"
            "            self._take_grace_pin(request, blocks)\n"
            "        return blocks\n",
            "        self.release_pinned_hit(request.request_id)\n"
            "        if self._gentail is not None:  # GENTAIL_CKPT: before the table goes\n"
            "            self._gentail.on_free(request)\n"
            "        blocks = self.coordinator.pop_blocks_for_free(request.request_id)\n"
            "        # A finished turn whose last step is still in flight: the grace takes\n"
            "        # its ref now; the table refs go when the fence drains.\n"
            "        if self._grace_eligible(request):\n"
            "            self._take_grace_pin(request, blocks)\n"
            "        if self._gentail is not None:  # GENTAIL_CKPT: after the grace pin\n"
            "            self._gentail.after_free()\n"
            "        return blocks\n",
        ),
    ],
}
PATCHED_RELS = (SCH_REL, KVM_REL)
if set(PATCHED_RELS + (NEW_REL,)) & set(NB4_FILES + DFLASH_FILES):
    raise RuntimeError("E30 gentail_seam overlaps the NB4 / E27 seam files")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _lf(t):
    return t.replace("\r\n", "\n")


def patch_text(text, edits):
    """Apply anchored edits: every anchor exactly once before, every replacement exactly once after."""
    for anchor, repl in edits:
        anchor, repl = _lf(anchor), _lf(repl)
        n = text.count(anchor)
        if n != 1:
            raise ValueError(f"GENTAIL anchor occurs {n} times (need 1): {anchor[:70]!r}")
        text = text.replace(anchor, repl)
        if text.count(repl) != 1:
            raise ValueError(f"GENTAIL replacement not unique after edit: {repl[:70]!r}")
    return text


def unpatch_text(text, edits):
    for anchor, repl in reversed(edits):
        anchor, repl = _lf(anchor), _lf(repl)
        n = text.count(repl)
        if n != 1:
            raise ValueError(f"GENTAIL replacement occurs {n} times (need 1): {repl[:70]!r}")
        text = text.replace(repl, anchor)
    return text


def new_module_bytes():
    return _lf(NEW_MODULE_SRC).encode("utf-8")


PRE = {SCH_REL: SCH_PRE, KVM_REL: KVM_PRE}
POST = {SCH_REL: SCH_POST, KVM_REL: KVM_POST}


def _file_state(tree, rel):
    f = Path(tree) / rel
    if rel == NEW_REL:
        if not f.exists():
            return "not_applied", None
        h = sha256_bytes(f.read_bytes())
        return ("applied" if h == NEW_SHA else "foreign"), h
    data = f.read_bytes()
    h = sha256_bytes(data)
    if h == POST[rel] and MARKER.encode() in data:
        return "applied", h
    if h == PRE[rel]:
        return "not_applied", h
    return "foreign", h


def verify_gentail_patch(tree):
    """Read-only: 'applied' / 'not_applied' (all three files agree), else 'foreign' / 'mixed'."""
    files = {}
    for rel in PATCHED_RELS + (NEW_REL,):
        st, h = _file_state(tree, rel)
        files[rel] = {"state": st, "sha": h}
    states = {v["state"] for v in files.values()}
    state = states.pop() if len(states) == 1 else ("foreign" if "foreign" in states else "mixed")
    return {"state": state, "files": files}


def tree_state(tree):
    s = verify_gentail_patch(tree)["state"]
    return {"not_applied": "ORIGINAL", "applied": "FULLY_PATCHED"}.get(s, "INVALID_MIXED")


def _py_compile(py, files):
    cp = subprocess.run([py, "-c", "import py_compile, sys\nfor f in sys.argv[1:]: py_compile.compile(f, doraise=True)"]
                        + [str(f) for f in files], capture_output=True, text=True, timeout=300)
    return {"rc": cp.returncode, "err": cp.stderr[-600:]}


def _drop_pyc(f):
    d = f.parent / "__pycache__"
    if d.is_dir():
        for pyc in d.glob(f.stem + ".*.pyc"):
            pyc.unlink()


def _write_atomic(f, data):
    tmp = f.with_name(f.name + ".e30tmp")
    tmp.write_bytes(data)
    os.replace(tmp, f)


def _restore(tree, originals):
    """Fail-safe: put back every original byte string (None = the file did not exist)."""
    for rel, data in originals.items():
        f = Path(tree) / rel
        if data is None:
            if f.exists():
                f.unlink()
        else:
            _write_atomic(f, data)
        _drop_pyc(f)


def apply_gentail_patch(tree, py=None):
    """Patch the installed tree (the folder holding vllm/). Refuses unless every file is at Merlin ad42e37's pre-sha
    (and gentail_ckpt.py absent), or everything is already applied. All new texts are built and sha-checked before
    anything is written; any failure while writing or verifying restores the original bytes."""
    tree = Path(tree)
    py = py or sys.executable
    v = verify_gentail_patch(tree)
    rec = {"tree": str(tree), "before": v}
    if v["state"] == "applied":
        rec["applied"] = "already_applied"
    elif v["state"] != "not_applied":
        raise RuntimeError("PATCH_GENTAIL_PRE_SHA_MISMATCH: " + json.dumps(v)[:900])
    else:
        originals = {rel: (tree / rel).read_bytes() for rel in PATCHED_RELS}
        originals[NEW_REL] = None
        new = {}
        for rel in PATCHED_RELS:
            out = patch_text(originals[rel].decode("utf-8"), EDITS[rel]).encode("utf-8")
            if sha256_bytes(out) != POST[rel]:
                raise RuntimeError(f"PATCH_GENTAIL_POST_SHA_MISMATCH {rel}: {sha256_bytes(out)}")
            new[rel] = out
        new[NEW_REL] = new_module_bytes()
        if sha256_bytes(new[NEW_REL]) != NEW_SHA:
            raise RuntimeError(f"PATCH_GENTAIL_NEW_SHA_MISMATCH: {sha256_bytes(new[NEW_REL])}")
        try:
            for rel, out in new.items():
                _write_atomic(tree / rel, out)
                _drop_pyc(tree / rel)
            after = verify_gentail_patch(tree)
            comp = _py_compile(py, [tree / rel for rel in new])
            if after["state"] != "applied" or comp["rc"]:
                raise RuntimeError("PATCH_GENTAIL_VERIFY_FAILED: " + json.dumps(
                    {"after": after, "py_compile": comp})[:900])
        except BaseException:
            _restore(tree, originals)
            raise
        rec["applied"] = "applied"
    rec["after"] = verify_gentail_patch(tree)
    rec["py_compile"] = _py_compile(py, [tree / rel for rel in PATCHED_RELS + (NEW_REL,)])
    if rec["after"]["state"] != "applied" or rec["py_compile"]["rc"]:
        raise RuntimeError("PATCH_GENTAIL_VERIFY_FAILED: " + json.dumps(
            {k: rec.get(k) for k in ("after", "py_compile")})[:900])
    rec["ok"] = True
    return rec


def revert_gentail_patch(tree, py=None):
    """Back to the exact Merlin files (anchors restored, sha == pre; gentail_ckpt.py removed). Refuses unless the tree
    is fully applied or already original. Fail-safe like apply."""
    tree = Path(tree)
    py = py or sys.executable
    v = verify_gentail_patch(tree)
    rec = {"tree": str(tree), "before": v}
    if v["state"] == "not_applied":
        rec["reverted"] = "not_applied"
    elif v["state"] != "applied":
        raise RuntimeError("REVERT_GENTAIL_SHA_MISMATCH: " + json.dumps(v)[:900])
    else:
        originals = {rel: (tree / rel).read_bytes() for rel in PATCHED_RELS + (NEW_REL,)}
        old = {}
        for rel in PATCHED_RELS:
            out = unpatch_text(originals[rel].decode("utf-8"), EDITS[rel]).encode("utf-8")
            if sha256_bytes(out) != PRE[rel]:
                raise RuntimeError(f"REVERT_GENTAIL_RESULT_MISMATCH {rel}: {sha256_bytes(out)}")
            old[rel] = out
        try:
            for rel, out in old.items():
                _write_atomic(tree / rel, out)
                _drop_pyc(tree / rel)
            (tree / NEW_REL).unlink()
            _drop_pyc(tree / NEW_REL)
            after = verify_gentail_patch(tree)
            if after["state"] != "not_applied":
                raise RuntimeError("REVERT_GENTAIL_VERIFY_FAILED: " + json.dumps(after)[:900])
        except BaseException:
            _restore(tree, originals)
            raise
        rec["reverted"] = "reverted"
    rec["after"] = verify_gentail_patch(tree)
    rec["py_compile"] = _py_compile(py, [tree / rel for rel in PATCHED_RELS])
    if rec["after"]["state"] != "not_applied" or rec["py_compile"]["rc"]:
        raise RuntimeError("REVERT_GENTAIL_VERIFY_FAILED: " + json.dumps(rec)[:900])
    rec["ok"] = True
    return rec


def server_env(on=True, strict_trunc=True, keep_prompt_hold=False, pin_budget_frac=None, nofence=False,
               log_publish=False, stats_every=None):
    """Env for the server (EngineCore inherits it). nofence=True is gate 4's NEGATIVE CONTROL only."""
    env = {ENV: "1" if on else "0"}
    if log_publish:
        env[ENV_LOG_PUBLISH] = "1"
    if stats_every is not None:
        env[ENV_STATS_EVERY] = str(int(stats_every))
    if not strict_trunc:
        env[ENV_STRICT_TRUNC] = "0"
    if keep_prompt_hold:
        env[ENV_KEEP_PROMPT_HOLD] = "1"
    if pin_budget_frac is not None:
        env[ENV_PIN_BUDGET] = str(float(pin_budget_frac))
    if nofence:
        env[ENV_NOFENCE] = "1"
    return env


def parse_log(text):
    out = []
    for line in text.splitlines():
        i = line.find("E30_GENTAIL ")
        if i < 0:
            continue
        try:
            out.append(json.loads(line[i + len("E30_GENTAIL "):]))
        except ValueError:
            continue
    return out


def readback(log_text):
    """Labels for a run: INSTALLED (+ its config), INACTIVE (+ reasons) or ABSENT; the last stats line."""
    recs = parse_log(log_text)
    inst = [r for r in recs if r.get("event") == "install"]
    inact = [r for r in recs if r.get("event") == "inactive"]
    stats = [r for r in recs if r.get("event") == "stats"]
    label = "INSTALLED" if inst else ("INACTIVE" if inact else "ABSENT")
    return {"label": label, "install": inst[-1] if inst else None, "inactive": inact[-1] if inact else None,
            "last_stats": stats[-1] if stats else None}
