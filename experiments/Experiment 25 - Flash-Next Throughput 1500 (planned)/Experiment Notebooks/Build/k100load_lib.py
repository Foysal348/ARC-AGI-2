"""E25 klee100 GPU load test: the pure parts (stdlib + numpy), embedded in the notebook and imported by the unit tests.

  build_model_dir   one model dir: symlinks to every klee100 file, the 10 FP8 PLE shards, the regenerated fixkeys overlay
  k6_check          every shard named in the index resolves to a file in the dir (K6), else no launch
  digest_plan       the files to stream-hash against the trusted HF LFS digest manifest (K7 + K3)
  verify_digests    streaming sha256 in parallel threads; any mismatch / missing / size difference is fatal before boot
  pin_ladder        --kv-cache-memory-bytes steps from the measured free memory per card (Merlin's pin rule ignores weights)
  k2_pin_from_k / k2_pins / ladder_run   K2's pin (base pin on unverified headroom, K-2) and the launch-OOM ladder (K-6)
  reference_gates / cli_flags_check      the pre-boot reference / prompt identity gates (K-4) and --help flags (K-9)
  method_map_check                       the GPU quant method map vs the checkpoint map (K-7)
  arith_cases / arith_score              the exact-answer arithmetic probe (K-12, recorded only)
  tf_label / yardstick / tf_reading / final_label   verdict labels; never an accuracy PASS
"""
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

GiB, MiB = 1 << 30, 1 << 20
OVERLAY_FILES = ("config.json", "quantization_config.json", "model.safetensors.index.json")
BF16_PLE_SHARD = "model-00001-of-00011.safetensors"          # excluded from the Kaggle Model (BF16 PLE, replaced by FP8)


# ===================================================================================== model dir (K6)
def build_model_dir(dst, k100_dir, ple_dir, overlay_dir, ple_files, link=None):
    """dst <- symlinks to every klee100 file (except the 3 overlay files), symlinks to the FP8 PLE shards, copies of the
    overlay files. Never follows a name outside the three source dirs. Returns the record (no hashing here)."""
    link = link or os.symlink
    dst, k100_dir, ple_dir, overlay_dir = Path(dst), Path(k100_dir), Path(ple_dir), Path(overlay_dir)
    if dst.exists():
        for p in dst.iterdir():
            if p.is_dir() and not p.is_symlink():
                raise RuntimeError(f"unexpected directory in the model dir: {p}")
            p.unlink()
    dst.mkdir(parents=True, exist_ok=True)
    rec = {"dir": str(dst), "klee100_links": [], "ple_links": [], "overlay_copies": {}, "skipped": [], "errors": []}
    for p in sorted(k100_dir.iterdir()):
        if p.is_dir():
            rec["skipped"].append(p.name + "/")
            continue
        if p.name in OVERLAY_FILES:
            rec["skipped"].append(p.name + " (overlay replaces it)")
            continue
        try:
            link(str(p), str(dst / p.name))
            rec["klee100_links"].append(p.name)
        except OSError as e:
            rec["errors"].append(f"link {p.name}: {e!r}"[:200])
    for n in ple_files:
        src = ple_dir / n
        if not src.is_file():
            rec["errors"].append(f"missing FP8 PLE shard {n}")
            continue
        if (dst / n).exists():
            rec["errors"].append(f"name clash {n}")
            continue
        try:
            link(str(src), str(dst / n))
            rec["ple_links"].append(n)
        except OSError as e:
            rec["errors"].append(f"link {n}: {e!r}"[:200])
    for n in OVERLAY_FILES:
        b = (overlay_dir / n).read_bytes()
        (dst / n).write_bytes(b)
        rec["overlay_copies"][n] = {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
    rec["ok"] = not rec["errors"]
    return rec


def k6_check(ckpt):
    """K6: every shard file named in model.safetensors.index.json must exist in the dir (a symlink is resolved: a dangling
    link is missing). Also: the BF16 PLE shard must not be referenced, and config.json carries the FP8 PLE dtype."""
    ckpt = Path(ckpt)
    out = {"dir": str(ckpt)}
    try:
        idx = json.loads((ckpt / "model.safetensors.index.json").read_text())
        files = sorted(set(idx["weight_map"].values()))
    except Exception as e:
        return dict(out, ok=False, why=f"index unreadable: {type(e).__name__}: {e}"[:300], missing=None)
    missing = [f for f in files if not (ckpt / f).is_file()]
    out.update(index_files=len(files), index_tensors=len(idx["weight_map"]), missing=missing,
               bf16_ple_referenced=BF16_PLE_SHARD in files)
    try:
        cfg = json.loads((ckpt / "config.json").read_text())
        out["ple_embedding_dtype"] = (cfg.get("text_config") or {}).get("ple_embedding_dtype")
    except Exception as e:
        out["ple_embedding_dtype"] = f"unreadable: {e!r}"[:120]
    why = []
    if missing:
        why.append(f"K6_INDEXED_SHARD_MISSING: {missing[:5]} ({len(missing)} of {len(files)})")
    if out["bf16_ple_referenced"]:
        why.append("the index references the BF16 PLE shard " + BF16_PLE_SHARD)
    if out["ple_embedding_dtype"] != "float8_e4m3fn":
        why.append(f"config.json text_config.ple_embedding_dtype = {out['ple_embedding_dtype']!r}")
    out["ok"] = not why
    out["why"] = "; ".join(why) or None
    return out


# ===================================================================================== digests (K7 + K3)
def digest_plan(manifest, k100_dir, ple_dir, extra=("tokenizer.json",)):
    """(name, path, size, sha256, source) of every klee100 shard used (all except the BF16 PLE shard), the FP8 PLE shards and
    the small LFS files in `extra` (tokenizer.json; the original index is replaced by the overlay and checked separately)."""
    k100_dir, ple_dir = Path(k100_dir), Path(ple_dir)
    kd, md = manifest["repos"]["klee100"], manifest["repos"]["merlin_fp8_ple"]
    plan = []
    for f, v in sorted(kd["shards"].items()):
        if f == BF16_PLE_SHARD:
            continue
        plan.append((f, str(k100_dir / f), int(v["size"]), v["sha256"], "klee100 " + kd["revision"][:8]))
    for f in extra:
        v = (kd.get("other_lfs_files") or {}).get(f)
        if v:
            plan.append((f, str(k100_dir / f), int(v["size"]), v["sha256"], "klee100 " + kd["revision"][:8]))
    for f, v in sorted(md["shards"].items()):
        plan.append((f, str(ple_dir / f), int(v["size"]), v["sha256"], "merlin_fp8_ple " + md["revision"][:8]))
    return plan


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def verify_digests(plan, threads=8, hasher=sha256_file, log=None):
    """Streaming sha256 of every planned file in `threads` threads (hashlib releases the GIL). Fail closed: a missing
    file, a size difference or a digest mismatch -> ok False. Sizes are checked before hashing (fast fail)."""
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    res, pre = {}, []
    for name, path, size, want, src in plan:
        p = Path(path)
        if not p.is_file():
            res[name] = {"ok": False, "why": "missing", "path": path}
        elif p.stat().st_size != size:
            res[name] = {"ok": False, "why": f"size {p.stat().st_size} != {size}", "path": path}
        else:
            pre.append((name, path, size, want, src))

    def one(item):
        name, path, size, want, src = item
        t = time.time()
        try:
            got, n = hasher(path)
        except Exception as e:
            return name, {"ok": False, "why": f"read error {type(e).__name__}: {e}"[:200], "path": path}
        ok = got == want and n == size
        r = {"ok": ok, "sha256": got, "bytes": n, "sec": round(time.time() - t, 1), "source": src}
        if not ok:
            r["why"] = f"sha256 {got[:16]} != trusted {want[:16]}" if got != want else f"read {n} bytes != {size}"
        if log:
            log("digest", name, "OK" if ok else "MISMATCH", r["sec"], "s")
        return name, r
    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as ex:
        for name, r in ex.map(one, pre):
            res[name] = r
    bad = sorted(n for n, r in res.items() if not r["ok"])
    gb = sum(r.get("bytes", 0) for r in res.values()) / 1e9
    sec = round(time.time() - t0, 1)
    return {"ok": not bad and len(res) == len(plan) and bool(plan), "bad": bad, "n_planned": len(plan), "n_checked": len(res),
            "gb": round(gb, 2), "sec": sec, "gb_per_s": round(gb / sec, 3) if sec else None, "threads": threads, "files": res}


# ===================================================================================== KV pin ladder
def stage_of_card(i, tp=2):
    """Merlin TP2 x PP2 with CUDA_VISIBLE_DEVICES=0,1,2,3 (PCI_BUS_ID): global rank r on device r, PP stage = r // TP
    (NB0 boot B: cards 0 / 1 held the 17.33 GiB PP0 weights, cards 2 / 3 the 16.13 GiB PP1 weights)."""
    return int(i) // tp


def free_by_card(readings):
    """nvidia-smi rows [[index, used_mib, total_mib], ...] (several readings) -> {index: total - max used}."""
    used, tot = {}, {}
    for row in readings or []:
        for r in row or []:
            if not isinstance(r, (list, tuple)) or len(r) < 3:
                continue
            i, u, t = int(r[0]), float(r[1]), float(r[2])
            used[i] = max(used.get(i, 0.0), u)
            tot[i] = t
    return {i: tot[i] - used[i] for i in sorted(tot)}


def pin_ladder(free_mib, base_pin, fracs=(0.9, 0.8, 0.7), safety_mib=1024, reserve_mib_pp1=0, n_cards=4, tp=2,
               align=2 * MiB, min_gain_mib=256):
    """Steps for --kv-cache-memory-bytes from the free memory per card measured at the base pin (all readings of the probe
    boot, the highest use counts). headroom(card) = free(card) - safety - reserve(card) (reserve = the MTP draft's weights
    + its workspace / graphs on the PP1 cards, which the MTP-off probe does not hold); pin(frac) = base + frac x min over
    cards of headroom, rounded down to `align`. Steps that add < min_gain_mib are dropped. Fail closed on missing cards."""
    out = {"base_pin": int(base_pin), "fracs": list(fracs), "safety_mib": safety_mib, "reserve_mib_pp1": reserve_mib_pp1,
           "free_mib_per_card": {str(k): v for k, v in (free_mib or {}).items()}}
    if not free_mib or len(free_mib) != n_cards or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                                                        for v in free_mib.values()):
        return dict(out, ok=False, steps=[], why=f"free memory per card not measured on all {n_cards} cards: {free_mib!r}"[:300])
    head = {i: f - safety_mib - (reserve_mib_pp1 if stage_of_card(i, tp) == 1 else 0) for i, f in free_mib.items()}
    hmin = min(head.values())
    out.update(headroom_mib_per_card={str(k): round(v, 1) for k, v in head.items()}, headroom_min_mib=round(hmin, 1),
               binding_card=min(head, key=head.get))
    steps = []
    for fr in fracs:
        add = fr * hmin
        if add < min_gain_mib:
            continue
        pin = int((int(base_pin) + int(add * MiB)) // align * align)
        steps.append({"frac": fr, "kv_bytes": pin, "gain_mib": round((pin - base_pin) / MiB, 1), "gib": round(pin / GiB, 3)})
    out["steps"] = steps
    out["ok"] = bool(steps)
    out["why"] = None if steps else f"no step adds >= {min_gain_mib} MiB (min headroom {hmin:.0f} MiB)"
    return out


def free_valid4(free_mib, n_cards=4):
    """K-2: a usable free-memory reading = exactly the cards {0 .. n_cards-1}, every value finite and >= 0."""
    if not isinstance(free_mib, dict) or len(free_mib) != n_cards:
        return False
    try:
        keys = {int(k) for k in free_mib}
    except (TypeError, ValueError):
        return False
    return keys == set(range(n_cards)) and all(_fin(v) and v >= 0 for v in free_mib.values())


def k2_pin_from_k(free_mib, pin, reserve_mib_pp1, min_free_mib=512, floor=0, tp=2, align=2 * MiB, lower_pin=None):
    """K2 (MTP K3) keeps K's pin when K's measured free memory per card (MTP off) covers the MTP reserve on the PP1 cards +
    min_free_mib on every card; otherwise the pin drops by the largest deficit (never below `floor` = Merlin's BF16 pin
    KV_DEFAULT). `lower_pin` (K died of OOM in its teacher-forced call): K2 uses that lower ladder step instead.
    K-2: a missing / partial (cards != {0,1,2,3}) / non-finite reading, or no valid K pin -> K2 uses `floor` (the base BF16
    pin), rule FALLBACK_BASE_PIN_UNVERIFIED_HEADROOM, headroom_verified False (never K's pin on unverified headroom)."""
    valid = free_valid4(free_mib)
    if lower_pin is not None:
        p = max(int(floor), int(lower_pin))
        return {"kv_bytes": p, "gib": round(p / GiB, 3), "rule": "K_TF_OOM: the next lower ladder step", "deficit_mib": None,
                "headroom_verified": valid}
    if not valid or not isinstance(pin, int) or isinstance(pin, bool):
        p = int(floor)
        return {"kv_bytes": p, "gib": round(p / GiB, 3), "rule": "FALLBACK_BASE_PIN_UNVERIFIED_HEADROOM", "deficit_mib": None,
                "headroom_verified": False, "free_mib_seen": repr(free_mib)[:200], "k_pin": pin if isinstance(pin, int) else repr(pin)[:40]}
    fm = {int(i): float(v) for i, v in free_mib.items()}
    need = {i: min_free_mib + (reserve_mib_pp1 if stage_of_card(i, tp) == 1 else 0) for i in fm}
    deficit = max(need[i] - fm[i] for i in fm)
    if deficit <= 0:
        return {"kv_bytes": int(pin), "gib": round(pin / GiB, 3), "rule": "SAME_PIN_AS_K", "deficit_mib": round(deficit, 1),
                "need_mib": need, "headroom_verified": True}
    p = max(int(floor), int((int(pin) - int(math.ceil(deficit) * MiB)) // align * align))
    return {"kv_bytes": p, "gib": round(p / GiB, 3), "rule": "LOWERED: K's free memory does not hold the MTP reserve",
            "deficit_mib": round(deficit, 1), "need_mib": need, "headroom_verified": True}


def k2_pins(pin2, base_pin, step_mib=512):
    """K-6: K2's pin ladder = dedupe [pin2, max(base, pin2 - 512 MiB), base] (order kept; never below the base pin)."""
    out = []
    for p in (int(pin2), max(int(base_pin), int(pin2) - step_mib * MiB), int(base_pin)):
        if p not in out:
            out.append(p)
    return out


def ladder_run(pins, launch, can_try=None, known_oom=(), body_retry=None):
    """K-6 / R1-4: launch(pin, i) -> the boot's step record, one pin at a time. The next pin only after LAUNCH_FAILED with
    an OOM at launch and no device fault; a device fault (Xid / illegal memory access / misaligned address) stops at once;
    a step skipped by the budget (SKIPPED_BUDGET, step() checks each try) or the abort stops. R1-4: a rung >= the lowest
    pin that already OOMed (here or `known_oom`) is skipped; an extra rung (after the first launch) needs can_try(i, pin)
    (K: left() >= EST_K + K2's minimum) else SKIPPED_K2_RESERVE. Launch OOM and body OOM are recorded separately.
    Returns (the last record, the tries)."""
    rec, tries, launched, used_body_retry = None, [], 0, False
    ko = [int(x) for x in (known_oom or ()) if x]
    floor = min(ko) if ko else None
    for i, p in enumerate(pins):
        if floor is not None and int(p) >= floor:
            tries.append({"kv_bytes": int(p), "status": "SKIPPED_ABOVE_OOM_PIN", "stop": "SKIPPED_ABOVE_OOM_PIN", "oom_pin": floor})
            continue
        if launched and can_try is not None and not can_try(i, p):
            tries.append({"kv_bytes": int(p), "status": "SKIPPED_K2_RESERVE", "stop": "SKIPPED_K2_RESERVE"})
            break
        r = launch(p, i) or {}
        launched += 1
        st = r.get("status")
        t = {"kv_bytes": int(p), "name": r.get("boot_name"), "launch_name": r.get("launch_name"), "port": r.get("port"), "status": st,
             "oom_at_launch": bool(r.get("oom_at_launch")), "oom_in_body": bool(r.get("oom_in_body")),
             "device_fault": bool(r.get("device_fault"))}
        tries.append(t)
        rec = r
        if t["device_fault"]:
            t["stop"] = "DEVICE_FAULT"
            break
        if st in ("SKIPPED_BUDGET", "SKIPPED_ABORT") or str(st).startswith("SKIPPED_"):
            t["stop"] = st
            t["launched"] = False
            break
        if not (st == "LAUNCH_FAILED" and t["oom_at_launch"]):
            # V-3: ONE controlled retry at the next lower rung after a body OOM before the core (no device fault, the GPUs
            # released, the budget covering a core + the later boots' reserves): body_retry(record) decides
            if body_retry is not None and not used_body_retry and i < len(pins) - 1 and body_retry(r):
                used_body_retry = True
                t["stop"] = "BODY_OOM_RETRY_NEXT_PIN"
                floor = min(x for x in (floor, int(r.get("kv_bytes") or p)) if x is not None)
                continue
            t["stop"] = "NOT_A_LAUNCH_OOM"
            break
        floor = min(x for x in (floor, int(r.get("kv_bytes") or p)) if x is not None)
        t["stop"] = "NEXT_PIN" if i < len(pins) - 1 else "LADDER_EXHAUSTED"
    if rec is None:
        rec = {"status": (tries[-1]["status"] if tries else "NOT_LAUNCHED"), "why": "no rung launched"}
    return rec, tries


def k2_plan(left_sec, k_ready_sec, body_min_sec=16 * 60, body_max_sec=34 * 60, factor=1.5, teardown_sec=180, margin_sec=300):
    """R1-2: K2's body cap from K's measured ready time: cap = left - 1.5 x K_ready - margin (300) - teardown (180) - 60 s
    slack (step()'s rounding), clipped to [16, 34] min; est (min) = cap + 1.5 x K_ready + teardown, rounded up. fits False:
    step() skips it (SKIPPED_BUDGET)."""
    cap = min(float(body_max_sec), float(left_sec) - factor * float(k_ready_sec) - margin_sec - teardown_sec - 60)
    fits = cap >= body_min_sec
    cap = max(cap, float(body_min_sec))
    return {"body_cap_sec": int(cap), "est_min": int(math.ceil((cap + factor * float(k_ready_sec) + teardown_sec) / 60)),
            "fits": fits, "k_ready_sec": float(k_ready_sec), "left_sec": round(float(left_sec))}


def death_label(watchdog, oom, xid, died_in):
    """R1-10 / R2-2 / R2-3: what a server death is called. The boot watchdog fired -> BOOT_CAP_KILL; a device fault ->
    DEVICE_FAULT_IN_<item>; an OOM line -> TF_OOM only when it died in a TF request, else BODY_OOM_<item>; otherwise
    TF_SERVER_DIED / SERVER_DIED_IN_<item>."""
    item = str(died_in or "unknown")
    if watchdog:
        return "BOOT_CAP_KILL"
    if xid:
        return "DEVICE_FAULT_IN_" + item
    if oom:
        return "TF_OOM" if item.startswith("tf") else "BODY_OOM_" + item
    return "TF_SERVER_DIED" if item.startswith("tf") else "SERVER_DIED_IN_" + item


def tf_incomplete(m):
    """A teacher-forced meta whose response lacked usable prompt logprobs: valid < expected (both known)."""
    v, e = (m or {}).get("valid"), (m or {}).get("expected")
    return isinstance(v, int) and isinstance(e, int) and v < e


def tf_core_done(tf, bands=("2k", "8k", "40k")):
    """R2-3: boot K's core TF requests (2k / 8k / 40k, fresh) completed without an error, a skip or missing positions."""
    return all(isinstance(((tf or {}).get(b) or {}).get("fresh"), dict) and "status" not in tf[b]["fresh"]
               and not tf[b]["fresh"].get("error") and not tf_incomplete(tf[b]["fresh"]) for b in bands)


SWEEP_LEVELS = (16, 24, 32)                           # V-14: K3's stream sweep (no 8-stream window)


def sweep_fit(P, F, prompt_tokens, level, gen_tokens, block=4096, frac=0.95):
    """V-14: is `level` streams of `prompt_tokens` (+ `gen_tokens` generated during admission + window) resident in the
    measured pool? per request = ceil((prompt + gen) / block) + F blocks; fits when level x per request <= 0.95 x P (the
    0.95 admission rule). P / F unknown -> fits False (never a guess)."""
    if not isinstance(P, int) or P <= 0 or not isinstance(F, int) or F < 0:
        return {"fits": False, "why": f"pool / F not measured (P {P}, F {F})"}
    per = int(math.ceil((prompt_tokens + gen_tokens) / block)) + F
    bound = int(frac * P // per)
    return {"fits": level <= bound, "level": level, "blocks_per_request": per, "resident_bound_095": bound, "P": P, "F": F,
            "context_tokens": int(prompt_tokens + gen_tokens), "need_blocks": level * per, "limit_blocks": round(frac * P, 1)}


def sweep_stop(w, level):
    """V-14: a band stops at the first level that preempts, runs fewer than `level` streams, or is CAPACITY_LIMITED."""
    w = w or {}
    rm = w.get("running_max")
    if (w.get("preemptions") or 0) > 0:
        return "PREEMPTION"
    if w.get("status") == "CAPACITY_LIMITED":
        return "CAPACITY_LIMITED"
    if isinstance(rm, (int, float)) and rm < level:
        return "RUNNING_BELOW_LEVEL"
    return None


def k3_body_oom_retry_ok(rec, budget_ok):
    """V-3: K3 died of OOM in its body BEFORE the core was done, no device fault, the GPU-release gate passed, and
    budget_ok() (a K3 core + the later boots' reserves still fit) -> one retry at the next lower rung."""
    st = str((rec or {}).get("status") or "")
    return (isinstance(rec, dict) and (st == "TF_OOM" or st.startswith("BODY_OOM_")) and rec.get("core_done") is not True
            and not rec.get("device_fault") and rec.get("gpu_released") is not False and bool(budget_ok()))


def k3_core_done(tf, cap):
    """V-16: K3's core is done only with TF 2k + 8k complete AND the pool verified (capacity_ok: worker == gauge, MATCH) AND
    the pool blocks and F measured (WORKER_ONLY / MISMATCH / no F -> not done)."""
    cap = cap or {}
    return bool(tf_core_done(tf, bands=("2k", "8k")) and cap.get("capacity_ok") is True
                and cap.get("pool_blocks") is not None and cap.get("F") is not None)


def k_oom_after_core(k):
    """R2-3: K died of OOM (TF_OOM / BODY_OOM_*) after its core measurements, with clean evidence: K2 still runs (lower pin)."""
    st = str((k or {}).get("status") or "")
    return (isinstance(k, dict) and (st == "TF_OOM" or st.startswith("BODY_OOM_")) and k.get("correctness") == "CLEAN"
            and k.get("config_ok") is True and k.get("core_done") is True)


def worker_free(readings):
    """R1-3: rpc_cache_info results ({"ranks": [{"rank", "mem": {"free_mib"}}]}) -> {rank: lowest free MiB} (the worker's
    own torch free memory; global rank r runs on card r)."""
    out = {}
    for rd in readings or []:
        for w in ((rd.get("ranks") or []) if isinstance(rd, dict) else []):
            if not isinstance(w, dict) or not isinstance(w.get("rank"), int) or isinstance(w.get("rank"), bool):
                continue
            f = (w.get("mem") or {}).get("free_mib")
            if _fin(f):
                out[w["rank"]] = min(out.get(w["rank"], f), f)
    return out


def combine_free(smi, worker, k0_gap=None, default_gap_mib=1024):
    """R1-3: per card min(nvidia-smi free, the worker's free of the same rank) (NB1 R2: nvidia-smi read 734-990 MiB
    higher); no worker reading -> nvidia-smi minus K0's measured gap, else minus 1,024 MiB. Returns (free, source) per card."""
    out, src = {}, {}
    for i, v in (smi or {}).items():
        w = (worker or {}).get(i)
        g = (k0_gap or {}).get(i, (k0_gap or {}).get(str(i)))
        if _fin(w):
            out[i], src[i] = min(v, w), "min(nvidia-smi, worker)"
        elif _fin(g):
            out[i], src[i] = v - max(0.0, g), "nvidia-smi - K0 gap"
        else:
            out[i], src[i] = v - default_gap_mib, f"nvidia-smi - {default_gap_mib} MiB (no worker reading)"
    return out, src


def pool_from_usage(us, worker_p=None, min_distinct=4, lo=20, hi=3000):
    """R2-7: the pool (blocks) from kv_cache_usage samples: None below `min_distinct` distinct values (a small false divisor
    passes 2-3 samples); the worker's P is tested first; else the first p in lo..hi that fits every sample."""
    vals = sorted({round(u, 9) for u in (us or []) if _fin(u) and 0 < u < 1})
    if len(vals) < min_distinct:
        return None

    def fits(p):
        return all(abs(u * p - round(u * p)) < 0.02 for u in vals)
    if isinstance(worker_p, int) and not isinstance(worker_p, bool) and worker_p > 0 and fits(worker_p):
        return worker_p
    return next((p for p in range(lo, hi + 1) if fits(p)), None)


def probe_engine_view(raw):
    """R1-5: the 3 short arithmetic probes gate on ENGINE faults only (a request error, an empty or garbage answer); the
    arithmetic itself is recorded (probes_correct n/3), never a gate (it is what K-12 measures). No logprobs on this path."""
    cases = list((raw or {}).get("cases") or [])
    faults = [c.get("q") for c in cases if c.get("error") or not str(c.get("content") or "").strip() or c.get("garbage")]
    return {"ok": bool(cases) and not faults, "engine_faults": faults,
            "probes_correct": f"{sum(1 for c in cases if c.get('ok'))}/{len(cases)}",
            "arith_all_correct": (raw or {}).get("ok"), "cases": cases, "gate": "engine faults only (R1-5)"}


# ===================================================================================== pre-boot gates (K-4, K-9)
def reference_gates(ref, pid, tf_bands=("2k", "8k", "40k", "90k"), sc_bands=("2k", "8k", "40k")):
    """K-4: REFERENCE_DATASET (ref_load ok is True) and PROMPT_IDENTITY_READY (every TF / SC band present with `used` in
    {rendered, nb0_ids}): hard gates before install / K0. R2-11: the RC2 oracle identity is record-only (oracle_identity)."""
    g = {}
    rok = isinstance(ref, dict) and ref.get("ok") is True
    g["REFERENCE_DATASET"] = (rok, None if rok else str((ref or {}).get("why") or (ref or {}).get("bad") or "reference not loaded")[:400]
                              if isinstance(ref, dict) or ref is None else repr(ref)[:200])
    pid = pid if isinstance(pid, dict) else {}
    bad = []
    for kind, bands in (("tf", tf_bands), ("sc", sc_bands)):
        d = pid.get(kind) or {}
        for b in bands:
            u = (d.get(b) or {}).get("used")
            if u not in ("rendered", "nb0_ids"):
                bad.append(f"{kind} {b}: {u}")
    g["PROMPT_IDENTITY_READY"] = (bool(pid) and not bad, "; ".join(bad)[:400] if bad else (None if pid else "prompt identity not run"))
    return g


def oracle_identity(pid):
    """R2-11: the RC2 oracle prompt identity, RECORDED only (RC2's rows are diagnostic, need=(); RC2_REF holds 16-hex
    prompt shas; the oracle cannot be substituted)."""
    om = ((pid if isinstance(pid, dict) else {}).get("rc2_oracle_prompts_match")) or {}
    return {"all_match": bool(om) and all(v is True for v in om.values()), "per_ref": om, "record_only": True}


def cli_flags_check(h):
    """K-9: vllm `serve --help=all` result -> ok False (a required flag is missing: no launch, FAIL_CLI_FLAGS) / True / None
    (help did not run or errored: recorded, the launch continues)."""
    if not isinstance(h, dict):
        return {"ok": None, "why": "help not run", "recorded": repr(h)[:120]}
    if h.get("ran") is not True:
        return {"ok": None, "why": "HELP_ERROR (recorded; the launch continues): " + str(h.get("error"))[:200], "tail": str(h.get("tail"))[-200:]}
    miss = list(h.get("missing") or [])
    return {"ok": not miss, "missing": miss, "why": ("missing flags: " + " ".join(miss)) if miss else None}


# ===================================================================================== GPU method map (K-7)
K100_N_LAYERS = 48
ROUTED_METHODS = ("HummingMoEMethod",)                # the klee100 CPU kit: W2 / W3 routed experts -> humming
MTP_METHODS = ("AutoGPTQMoEMethod",)                  # the CPU kit: MTP W4 g128 experts -> AutoGPTQMoEMethod (Marlin on sm_89)
MTP_BITS_GROUP = (4, 128)


def k100_expected_map(n=K100_N_LAYERS):
    """The checkpoint's routed-expert map: layers 0-11 and 36-47 W3 g128, 12-35 W2 g64."""
    return {L: ((3, 128) if (L < 12 or L >= 36) else (2, 64)) for L in range(n)}


PP_PARTITION = (25, 23)                               # Merlin fn-quad-24g-sm89: VLLM_PP_LAYER_PARTITION 25/23


def parse_partition(v, default=PP_PARTITION):
    """VLLM_PP_LAYER_PARTITION "25,23" -> (25, 23); anything else -> the default."""
    try:
        t = tuple(int(x) for x in str(v).replace("/", ",").split(",") if x.strip())
        return t if len(t) == 2 and all(x > 0 for x in t) else tuple(default)
    except (TypeError, ValueError):
        return tuple(default)


def stage_layers(partition=PP_PARTITION):
    """{PP stage: set of its decoder layer indices} for a 2-stage partition (25/23 -> 0-24, 25-47)."""
    a, b = partition
    return {0: set(range(0, a)), 1: set(range(a, a + b))}


MAP_PASS, MAP_CLASS_ONLY, MAP_FAIL = "GPU_METHOD_MAP_PASS", "METHOD_CLASS_ONLY", "GPU_METHOD_MAP_FAIL"


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def bits_state(vals):
    """READ (every row has int bits and int group) / UNREADABLE (every row has bits and group None) / PARTIAL / NONE (no rows)
    over the (method, bits, group) values of one family (routed or MTP)."""
    vals = list(vals)
    if not vals:
        return "NONE"
    rd = [_is_int(b) and _is_int(g) for _, b, g in vals]
    un = [b is None and g is None for _, b, g in vals]
    return "READ" if all(rd) else "UNREADABLE" if all(un) else "PARTIAL"


def method_map_check(results, need_mtp, expected=None, partition=PP_PARTITION, tp=2, n_ranks=4):
    """K-7 / V-1: the per-rank worker-extension rows (k100_method_map) -> GPU_METHOD_MAP_PASS. Exactly the ranks
    {0, 1, 2, 3}, each answering without an RPC error; every rank holds exactly the routed-expert layers of its PP stage
    (25/23: ranks 0-1 = layers 0-24, ranks 2-3 = 25-47) and each with a humming method and the checkpoint's bits / group;
    no rank disagreeing; a routed layer outside 0-47 (e.g. the MTP layer 48 counted as routed) -> UNEXPECTED_ROUTED_LAYERS;
    MTP experts (W4 g128 AutoGPTQMoEMethod) only on the last-stage ranks, required when MTP is on. Unquantized is not banned
    globally (only these layers).
    rev 3 (K3_ONLY GPU v1: the probe read bits None on every layer and failed a model that loaded): verdict
      GPU_METHOD_MAP_PASS   every check above, bits / group read on every row and equal to the checkpoint;
      METHOD_CLASS_ONLY     every check above, every class right, but bits / group unreadable (None) on EVERY row of a family
                            (routed and / or MTP; the other family READ and right): the probe could not read them, the model
                            is not called mismatched (gate_ok True, ok False);
      GPU_METHOD_MAP_FAIL   anything else: a wrong class, a read bit / group that differs, a row whose bit sources disagree
                            (BITS_SOURCES_DISAGREE), bits readable on only some rows of a family (BITS_PARTIALLY_READ), missing
                            ranks / layers / MTP, rank errors, wrong PP ownership, ranks disagreeing."""
    exp = expected or k100_expected_map()
    out = {"ok": False, "gate_ok": False, "verdict": MAP_FAIL, "layers_expected": len(exp)}
    if not isinstance(results, list) or not results:
        return dict(out, why=f"no rpc results: {repr(results)[:200]}", layers_matching=0, layers_class_only=0)
    stl = stage_layers(partition)
    seen, mtp, conflicts, errors, ranks, unexpected, wrong_pp, mtp_pp0 = {}, {}, [], [], [], [], [], []
    src_conflicts, sources, drafter_paths = [], {}, {}
    for w in results:
        if not isinstance(w, dict):
            errors.append("non-dict rank result: " + repr(w)[:160])
            continue
        r = w.get("rank")
        ranks.append(r)
        if "drafter_path" in w:
            drafter_paths[str(r)] = w.get("drafter_path")
        errors += [f"rank {r}: {str(e)[:160]}" for e in (w.get("errors") or [])]
        for k_ in ("rank_error", "error"):
            if w.get(k_):
                errors.append(f"rank {r}: {k_} {str(w[k_])[:160]}")
        st = (r // tp) if isinstance(r, int) and not isinstance(r, bool) else None
        mine = set()
        for e in w.get("experts") or []:
            if not isinstance(e, dict):
                continue
            v = (e.get("method"), e.get("bits"), e.get("group"))
            sk = str(e.get("source"))
            sources[sk] = sources.get(sk, 0) + 1
            if e.get("bits_conflict"):
                src_conflicts.append({"rank": r, "name": e.get("name"), "sources": e.get("sources")})
            if e.get("mtp"):
                if st != 1:
                    mtp_pp0.append({"rank": r, "name": e.get("name")})
                k = str(e.get("name"))
                if k in mtp and mtp[k] != v:
                    conflicts.append({"key": k, "a": list(mtp[k]), "b": list(v), "rank": r})
                mtp.setdefault(k, v)
                continue
            L_ = e.get("layer")
            if not isinstance(L_, int) or isinstance(L_, bool) or L_ not in exp:
                unexpected.append({"rank": r, "layer": L_, "name": e.get("name")})
                continue
            mine.add(L_)
            if L_ in seen and seen[L_] != v:
                conflicts.append({"key": L_, "a": list(seen[L_]), "b": list(v), "rank": r})
            seen.setdefault(L_, v)
        if st in stl and mine != stl[st]:
            wrong_pp.append({"rank": r, "stage": st, "missing": sorted(stl[st] - mine)[:6], "foreign": sorted(mine - stl[st])[:6]})
    rank_ok = sorted(x for x in ranks if isinstance(x, int) and not isinstance(x, bool)) == list(range(n_ranks)) and len(ranks) == n_ranks
    missing = sorted(L for L in exp if L not in seen)
    r_state, m_state = bits_state(seen[L] for L in exp if L in seen), bits_state(mtp.values())
    unread = lambda v: v[1] is None and v[2] is None          # noqa: E731
    # rev 3: a class mismatch is always bad; a read bit / group must equal the checkpoint; in an UNREADABLE family the class
    # alone is compared (METHOD_CLASS_ONLY); in a PARTIAL family an unreadable row is bad (BITS_PARTIALLY_READ)
    bad = [{"layer": L, "got": list(seen[L]), "want": [ROUTED_METHODS[0], *exp[L]]} for L in sorted(exp)
           if L in seen and (seen[L][0] not in ROUTED_METHODS
                             or (not (r_state == "UNREADABLE" and unread(seen[L])) and tuple(seen[L][1:]) != tuple(exp[L])))]
    mtp_bad = {k: list(v) for k, v in mtp.items() if v[0] not in MTP_METHODS
               or (not (m_state == "UNREADABLE" and unread(v)) and tuple(v[1:]) != MTP_BITS_GROUP)}
    mtp_ok = not mtp_bad and not mtp_pp0 and (bool(mtp) or not need_mtp)
    n_class = len([L for L in exp if L in seen and seen[L][0] in ROUTED_METHODS])
    n_ok = len(exp) - len(missing) - len(bad) if r_state != "UNREADABLE" else 0
    why = []
    if not rank_ok:
        why.append(f"RANKS: {ranks} (need exactly {list(range(n_ranks))})")
    if errors:
        why.append(f"RANK_ERRORS: {errors[:3]}")
    if unexpected:
        why.append(f"UNEXPECTED_ROUTED_LAYERS: {unexpected[:4]}")
    if wrong_pp:
        why.append(f"WRONG_PP_LAYERS: {wrong_pp[:3]}")
    if missing:
        why.append(f"{len(missing)} layers not seen: {missing[:8]}")
    if bad:
        why.append(f"{len(bad)} layers differ from the checkpoint map: {bad[:3]}")
    if conflicts:
        why.append(f"ranks disagree: {conflicts[:2]}")
    if mtp_pp0:
        why.append(f"MTP_ON_PP0: {mtp_pp0[:2]}")
    if not mtp_ok and not mtp_pp0:
        why.append(f"MTP experts {'missing' if not mtp else mtp_bad}")
    if src_conflicts:
        why.append(f"BITS_SOURCES_DISAGREE: {src_conflicts[:2]}")
    for fam, s_ in (("routed", r_state), ("mtp", m_state)):
        if s_ == "PARTIAL":
            why.append(f"BITS_PARTIALLY_READ ({fam}): bits / group read on some rows only")
    class_only = [fam for fam, s_ in (("routed", r_state), ("mtp", m_state)) if s_ == "UNREADABLE"]
    verdict = MAP_FAIL if why else MAP_CLASS_ONLY if class_only else MAP_PASS
    out.update(ok=verdict == MAP_PASS, gate_ok=verdict in (MAP_PASS, MAP_CLASS_ONLY), verdict=verdict,
               layers_matching=n_ok, layers_class_only=n_class if r_state == "UNREADABLE" else 0,
               bits_readable={"routed": r_state, "mtp": m_state}, class_only_families=class_only, bits_sources=sources,
               source_conflicts=src_conflicts[:8], drafter_paths=drafter_paths,
               missing=missing, bad=bad[:12], unexpected=unexpected[:12], wrong_pp=wrong_pp[:8],
               conflicts=conflicts[:8], mtp={k: list(v) for k, v in mtp.items()}, mtp_ok=mtp_ok, need_mtp=bool(need_mtp),
               ranks=ranks, ranks_ok=rank_ok, errors=errors[:6], partition=list(partition), why="; ".join(why)[:800] or None)
    if verdict == MAP_CLASS_ONLY:
        out["note"] = (f"METHOD_CLASS_ONLY: every class matches the checkpoint map; bits / group unreadable on every "
                       f"{' and '.join(class_only)} row (a probe limit, not a mismatch)")
    return out


SCALE_ACTIVE_RX = r"\((\w+) pid=(\d+)\)[^\n]*?QSA E4M3 KV active on (\S+): k_scale=([-\d.eE+]+) v_scale=([-\d.eE+]+)(\s+\(DEFAULT 1\.0)?"
SCALE_APPLIED_RX = r"\((\w+) pid=(\d+)\)[^\n]*applied static K/V scales from (.+?) to (\d+) of (\d+) layer\(s\)"


def _layer_of(name):
    m = re.search(r"layers\.(\d+)\.", str(name))
    return int(m.group(1)) if m else None


def scales_check(log_text, sidecar, partition=PP_PARTITION, tp=2, n_ranks=4, rtol=1e-5):
    """V-4: STATIC_SCALES_APPLIED per rank from the server log. Every 'QSA E4M3 KV active on <layer>' line is attributed to
    its worker (Worker_PP<p>_TP<t> -> rank p x TP + t; one pid per rank, else a duplicate); all 4 ranks must report; each
    rank's set of scaled layers == the sidecar's QSA layers of its PP stage (25/23: PP0 3, 7, ..., 23; PP1 27, ..., 47) with
    the sidecar's k / v values; the MTP layer (not in the sidecar) at DEFAULT 1.0 is recorded as expected, never an error."""
    t = log_text or ""
    stl = stage_layers(partition)
    want = {L: (float(v["k_scale"]), float(v["v_scale"])) for n, v in (sidecar or {}).items() for L in [_layer_of(n)] if L is not None}
    want_by_stage = {st: {L for L in want if L in lay} for st, lay in stl.items()}
    per, pids, bad_vals, untagged, mtp_default = {}, {}, [], [], {}
    for wname, pid, layer, k, v, dflt in re.findall(SCALE_ACTIVE_RX, t):
        m = re.search(r"PP(\d+)_TP(\d+)", wname)
        if not m:
            untagged.append(wname)
            continue
        r = int(m.group(1)) * tp + int(m.group(2))
        pids.setdefault(r, set()).add(pid)
        L = _layer_of(layer)
        if L is None or L not in want:
            if dflt:
                mtp_default.setdefault(r, []).append(layer)
            else:
                bad_vals.append({"rank": r, "layer": layer, "why": "not in the sidecar"})
            continue
        if dflt:
            bad_vals.append({"rank": r, "layer": layer, "why": "DEFAULT 1.0 on a QSA layer of the sidecar"})
            continue
        per.setdefault(r, set()).add(L)
        wk, wv = want[L]
        if abs(float(k) - wk) > rtol * max(1e-12, abs(wk)) or abs(float(v) - wv) > rtol * max(1e-12, abs(wv)):
            bad_vals.append({"rank": r, "layer": L, "k": float(k), "v": float(v), "want": [wk, wv]})
    applied = {}
    for wname, pid, path, n, m_ in re.findall(SCALE_APPLIED_RX, t):
        mm = re.search(r"PP(\d+)_TP(\d+)", wname)
        if mm:
            applied.setdefault(int(mm.group(1)) * tp + int(mm.group(2)), []).append([int(n), int(m_)])
    dup = sorted(r for r, ps in pids.items() if len(ps) > 1)
    missing_ranks = sorted(set(range(n_ranks)) - set(per))
    wrong_subset = {r: {"got": sorted(per[r]), "want": sorted(want_by_stage[r // tp])} for r in sorted(per)
                    if r // tp in want_by_stage and per[r] != want_by_stage[r // tp]}
    wrong_applied = {r: a for r, a in applied.items() if r // tp in want_by_stage and any(x[0] != len(want_by_stage[r // tp]) for x in a)}
    why = []
    if not want or any(not v for v in want_by_stage.values()):
        why.append("sidecar empty / a stage without QSA layers")
    if missing_ranks:
        why.append(f"ranks without scale lines: {missing_ranks}")
    if dup:
        why.append(f"duplicated ranks (two workers): {dup}")
    if wrong_subset:
        why.append(f"wrong QSA subset: {wrong_subset}")
    if bad_vals:
        why.append(f"values != the sidecar: {bad_vals[:3]}")
    if wrong_applied:
        why.append(f"'applied ... to N of M' != the stage's count: {wrong_applied}")
    return {"ok": not why, "per_rank": {r: sorted(v) for r, v in sorted(per.items())}, "expected_per_stage": {st: sorted(v) for st, v in want_by_stage.items()},
            "applied_lines": applied, "mtp_default_scale": mtp_default, "untagged_workers": untagged[:6], "duplicated_ranks": dup,
            "missing_ranks": missing_ranks, "bad_values": bad_vals[:8], "why": "; ".join(why)[:800] or None}


# ===================================================================================== arithmetic probe (K-12)
ARITH = ((4827, "+", 3594), (90071, "-", 4568), (736, "*", 48), (1234, "*", 5678), (999, "+", 1), (50000, "-", 1),
         (86, "*", 97), (123456, "+", 654321))


def arith_cases():
    """8 fixed exact-answer prompts (multi-digit add / sub / mul with carries) in the lab's chat_probes form."""
    f = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}
    return tuple((f"Reply with exactly one number: {a} {op} {b} =", str(f[op](a, b))) for a, op, b in ARITH)


def arith_score(res):
    """chat_probes result -> exact-match per case (content stripped, a trailing period dropped) + the lab's contains-match.
    Recorded only, never gated."""
    cases = list((res or {}).get("cases") or [])
    for c in cases:
        c["exact"] = (c.get("content") or "").strip().rstrip(".").strip() == c.get("want")
    return {"n": len(cases), "exact": sum(1 for c in cases if c["exact"]), "contains": sum(1 for c in cases if c.get("ok")),
            "errors": sum(1 for c in cases if c.get("error")), "cases": cases, "gated": False}


def blocks_estimate(pin_bytes, ref_pin_bytes, ref_blocks):
    """Linear pool estimate from a measured (pin, blocks) pair: blocks ~ pin / bytes_per_block."""
    if not ref_blocks or not ref_pin_bytes:
        return None
    return int(pin_bytes / (ref_pin_bytes / ref_blocks))


# ===================================================================================== labels
SCREEN = {"rank1_min": 0.90, "mean_max": 0.15}       # NB0's screening thresholds (external audit 2)
TRIAGE = {"mean_max": 1.0, "rank1_min": 0.5}         # NB0's severe-discrepancy triage (external audit 3)
TF_RANK = {"LOWBIT_TF_CLOSE": 0, "LOWBIT_TF_DIVERGENT": 1, "LOWBIT_TF_DIVERGENT_SEVERE": 2}


def _fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def tf_label(c):
    """One comparison (cmp_full / compare_tf record) -> LOWBIT_TF_CLOSE (inside NB0's screening limits: mean <= 0.15 and
    rank-1 >= 0.90) / LOWBIT_TF_DIVERGENT / LOWBIT_TF_DIVERGENT_SEVERE (NB0 triage: mean > 1.0 or rank-1 < 0.5) / INCONCLUSIVE_*."""
    if not isinstance(c, dict):
        return "INCONCLUSIVE_MISSING"
    if c.get("status") != "MEASURED":                    # K-5: PARTIAL (a subset of positions) is not a reading
        return "INCONCLUSIVE_" + str(c.get("status") or "MISSING").replace("INCONCLUSIVE_", "")
    if c.get("expected") is not None and c.get("n") != c.get("expected"):
        return "INCONCLUSIVE_INCOMPLETE_POSITIONS"
    m, r1 = c.get("mean_abs_dlogprob"), c.get("rank1_agreement")
    if not (_fin(m) and _fin(r1)):
        return "INCONCLUSIVE_NONFINITE"
    if m > TRIAGE["mean_max"] or r1 < TRIAGE["rank1_min"]:
        return "LOWBIT_TF_DIVERGENT_SEVERE"
    if m <= SCREEN["mean_max"] and r1 >= SCREEN["rank1_min"]:
        return "LOWBIT_TF_CLOSE"
    return "LOWBIT_TF_DIVERGENT"


def yardstick(c, y):
    """klee100 vs W4 against NB0's FP8-KV vs BF16-KV divergence at the same band and metric (the change NB0 rejected):
    BELOW (smaller mean and higher rank-1), ABOVE (larger mean and lower rank-1), MIXED, or INCONCLUSIVE."""
    if not isinstance(c, dict) or not isinstance(y, dict) or not all(_fin(d.get(k)) for d in (c, y)
                                                                      for k in ("mean_abs_dlogprob", "rank1_agreement")):
        return {"vs_nb0_fp8_kv": "INCONCLUSIVE"}
    dm, dr = c["mean_abs_dlogprob"] - y["mean_abs_dlogprob"], c["rank1_agreement"] - y["rank1_agreement"]
    lab = "BELOW_NB0_FP8_KV" if (dm < 0 and dr > 0) else "ABOVE_NB0_FP8_KV" if (dm > 0 and dr < 0) else "MIXED"
    return {"vs_nb0_fp8_kv": lab, "mean_ratio": round(c["mean_abs_dlogprob"] / y["mean_abs_dlogprob"], 3) if y["mean_abs_dlogprob"] else None,
            "rank1_delta": round(dr, 4), "nb0_fp8_kv": {k: y.get(k) for k in ("mean_abs_dlogprob", "p99_abs_dlogprob", "rank1_agreement")}}


TF_BANDS = ("2k", "8k", "40k", "90k")
BUDGET_SKIPS = ("SKIPPED_BUDGET", "SKIPPED_BOOT_BUDGET", "SKIPPED_BUDGET_PRIORITY", "SKIPPED_K2_RESERVE")


def tf_reading(per_band, skipped=None, bands=TF_BANDS):
    """per_band {band: {"label_full": label, ...}} -> the reading. K-5: the worst full-prompt label only when EVERY band
    (2k / 8k / 40k / 90k) is measured; otherwise TF_INCOMPLETE (+ worst_measured, missing), or TF_PARTIAL_BUDGET when the
    only missing band is 90k and it was dropped under budget (`skipped` {band: status} in BUDGET_SKIPS); TF_NOT_MEASURED
    when nothing is measured."""
    labs = {b: ((per_band or {}).get(b) or {}).get("label_full") for b in bands}
    meas = {b: l for b, l in labs.items() if l in TF_RANK}
    missing = [b for b in bands if b not in meas]
    out = {"per_band": labs, "missing": missing, "complete": not missing}
    if not meas:
        return dict(out, reading="TF_NOT_MEASURED")
    worst = max(meas, key=lambda b: (TF_RANK[meas[b]], bands.index(b)))
    out.update(worst_measured=meas[worst], worst_band=worst)
    if not missing:
        out["reading"] = meas[worst]
    elif missing == ["90k"] and (skipped or {}).get("90k") in BUDGET_SKIPS:
        out.update(reading="TF_PARTIAL_BUDGET", skipped_90k=(skipped or {}).get("90k"))
    else:
        out["reading"] = "TF_INCOMPLETE"
    return out


def boot_clean(rec):
    return isinstance(rec, dict) and rec.get("status") in ("DONE",) and rec.get("correctness") == "CLEAN" and rec.get("config_ok") is True


# ===================================================================================== K3: FP8 KV first boot (R3-1 + the user's order)
B3_PIN = 3553580524                                   # NB1 R2 B3: W4, FP8 KV, MTP K3, seqs 8, text-only, pin = Merlin FP8 + 1 GiB
B3_WORKER_FREE_MIB = {0: 639.0, 1: 639.0, 2: 205.0, 3: 191.0}   # B3's worker (torch) free memory per rank (R1-3 basis)
W4_LOAD_GIB = 16.89                                   # NB1 R2: W4 "Model loading took" 16.89-16.91 GiB per rank (the low end)
FP8_BASE_PIN = 2479838700                             # Merlin's own L4 FP8 pin (RC2 FP8, 150 blocks)


def k3_pin_steps(klee_gib, w4_gib=W4_LOAD_GIB, b3_free=None, b3_pin=B3_PIN, base_pin=FP8_BASE_PIN, safety_mib=1024,
                 graph_mib=0.1 * 1024, workspace_mib=768, fracs=(0.8, 0.5), tp=2, align=2 * MiB):
    """K3's pins with no probe boot before it, from measured data: per card, the extra headroom over B3 = B3's worker free
    + the W4-vs-klee100 weight difference of its stage (the measured W4 load vs the CPU kit's Gate D klee100 estimate)
    - 1 GiB safety - the seqs 16 / captures-to-64 margin (NB2 SEED2 +0.1 GiB graphs + 768 MiB workspace). est_extra =
    min over cards. Steps = [B3 + 0.8 x est, B3 + 0.5 x est, B3, Merlin's FP8 base], deduplicated, descending."""
    b3_free = b3_free or B3_WORKER_FREE_MIB
    head = {}
    for i, f in b3_free.items():
        st = "PP0" if stage_of_card(i, tp) == 0 else "PP1"
        freed = (float(w4_gib) - float(klee_gib[st])) * 1024
        head[i] = float(f) + freed - safety_mib - graph_mib - workspace_mib
    extra = min(head.values())
    raw = [int(b3_pin + fr * extra * MiB) // align * align for fr in fracs if extra > 0] + [int(b3_pin), int(base_pin)]
    steps = sorted({p for p in raw if p >= base_pin}, reverse=True)
    return {"steps": steps, "est_extra_mib": round(extra, 1), "headroom_mib_per_card": {str(k): round(v, 1) for k, v in head.items()},
            "binding_card": min(head, key=head.get), "inputs": {"klee100_gate_d_gib": dict(klee_gib), "w4_load_gib": w4_gib,
                                                              "b3_pin": b3_pin, "b3_worker_free_mib": {str(k): v for k, v in b3_free.items()},
                                                              "safety_mib": safety_mib, "graph_mib": graph_mib, "workspace_mib": workspace_mib}}


K3V1_PIN = 5270142976                                 # K3_ONLY GPU v1 (2026-11-01): the first rung, it booted (1,096,865 KV tokens)
K3V1_FREE_MIB = {0: 4583.0, 1: 4579.0, 2: 3257.0, 3: 3257.0}   # its free memory per card after ready, min(nvidia-smi, worker)
K3_LEAVE_MIB = (0.7 * 1024, 1.7 * 1024)               # free memory left on the tightest card at the two raised rungs


def k3_measured_rungs(free_mib=None, pin=K3V1_PIN, leave_mib=K3_LEAVE_MIB, align=2 * MiB):
    """rev 3 (user: raise the K3 pin): rungs above a MEASURED K3 boot. --kv-cache-memory-bytes is per worker GPU
    (CacheConfig.kv_cache_memory_bytes "Size of KV Cache per GPU in bytes", vllm/config/cache.py:260-267) and every card
    gets the same pin, so pin + X takes X MiB from every card's free memory: rung(leave) = pin + (min over cards of free -
    leave) MiB, rounded down to `align` (2 MiB, like every other rung). The v1 reading (4583 / 4579 / 3257 / 3257 MiB,
    binding cards 2 / 3 = PP1, which also hold the MTP drafter) with leave 0.7 / 1.7 GiB -> 7,933,526,016 B (7.39 GiB) and
    6,859,784,192 B (6.39 GiB). The reading was taken after ready (CUDA graphs to 64 captured, MTP loaded, no request yet):
    the 0.7 GiB left on the top rung is for request activations (prefill chunks of max_num_batched_tokens 1024, TF prompt
    logits); the 1.7 GiB of the second rung also covers a ~1 GiB TF transient + the K ladder's 1 GiB safety rule's
    margin; a launch OOM steps down (ladder_run), a body OOM before the core retries once one rung lower (V-3)."""
    free_mib = dict(free_mib or K3V1_FREE_MIB)
    fmin = min(float(v) for v in free_mib.values())
    rungs = [int((int(pin) + int((fmin - lv) * MiB)) // align * align) for lv in leave_mib if fmin - lv > 0]
    return {"rungs": sorted({r for r in rungs if r > int(pin)}, reverse=True), "from_pin": int(pin),
            "free_mib_per_card": {str(k): v for k, v in free_mib.items()}, "free_min_mib": fmin, "binding_card": min(free_mib, key=free_mib.get),
            "leave_mib": [round(x, 1) for x in leave_mib], "align": align,
            "basis": "K3_ONLY GPU v1 2026-11-01 k100_results.json free_mib_by_card (min of nvidia-smi and the worker) at its first rung"}


def k3_ladder(est, measured=None):
    """rev 3: K3's TRIED rungs, descending: the measured rungs (k3_measured_rungs), then the B3-based ladder from its step 1
    (step 0, the highest estimate, was never tried: V-3) = 5,270,142,976 (v1's booted rung), B3 3,553,580,524, Merlin's FP8
    base 2,479,838,700. Every rung is tried from index 0."""
    measured = measured if measured is not None else k3_measured_rungs()
    steps = sorted({int(p) for p in list(measured.get("rungs") or []) + list((est or {}).get("steps") or [])[1:]}, reverse=True)
    return {"steps": steps, "measured": measured, "estimate": est, "tried_from": 0,
            "note": "every step is tried in order (launch OOM -> next; one body-OOM retry before the core -> next)"}


def ladder_min(a, b):
    """K's ladder from K0's reading (a) and K3's (b, R1-3 rule): per frac the lower pin (both ok), else the one that is ok."""
    oks = [x for x in (a, b) if isinstance(x, dict) and x.get("ok") and x.get("steps")]
    if not oks:
        return dict(a or b or {}, sources=[])
    if len(oks) == 1:
        return dict(oks[0], sources=["K0" if oks[0] is a else "K3"])
    by = {}
    for src, lad in (("K0", a), ("K3", b)):
        for st in lad["steps"]:
            k = st["frac"]
            if k not in by or st["kv_bytes"] < by[k]["kv_bytes"]:
                by[k] = dict(st, source=src)
    steps = [by[k] for k in sorted(by, reverse=True)]
    return dict(a, steps=steps, sources=["K0", "K3"], k0=a, k3=b)


def map_verdict(rec):
    """rev 3: the boot's GPU method-map verdict (GPU_METHOD_MAP_PASS / METHOD_CLASS_ONLY / GPU_METHOD_MAP_FAIL); a record
    from before rev 3 (no verdict) -> PASS when ok is True, else None."""
    mm = (rec.get("method_map") if isinstance(rec, dict) else None) or {}
    return mm.get("verdict") or (MAP_PASS if mm.get("ok") is True else None)


def class_only_tag(*recs):
    """rev 3: ' + METHOD_CLASS_ONLY' when any of these boots passed its method map on the class alone."""
    return " + " + MAP_CLASS_ONLY if any(map_verdict(r) == MAP_CLASS_ONLY for r in recs) else ""


def k3_label(k3, k3_read):
    """R3-1: K3's own sub-label, at the head of the final label; it never changes the K / K2 verdict. rev 3: a K3 whose
    method map passed on the class alone carries ' + METHOD_CLASS_ONLY'."""
    if not isinstance(k3, dict) or k3.get("status") in (None, "NOT_RUN"):
        return "K3_NOT_RUN"
    st = str(k3.get("status"))
    if st.startswith("SKIPPED"):
        return "K3_" + st
    return _k3_label(k3, k3_read, st) + class_only_tag(k3)


def _k3_label(k3, k3_read, st):
    rd = (k3_read or {}).get("reading", "TF_NOT_MEASURED")
    full = rd in TF_RANK and (k3_read or {}).get("complete") is True      # V-2: over K3's designated bands (2k / 8k)
    if boot_clean(k3):
        return f"K3_FP8_LOAD_PASS + {rd}" if full else f"K3_FP8_INCOMPLETE_TF + {rd}"
    if "NO_GO_GARBAGE_UNDER_LOAD" in (k3.get("status"), k3.get("correctness")):
        return f"K3_FP8_GARBAGE_UNDER_LOAD + {rd}"
    if st == "TF_OOM" or st.startswith("BODY_OOM_"):                     # V-3: K3_FP8_BODY_OOM_<item>_AT_PIN<gib>
        return f"K3_FP8_{st}_AT_PIN{round(int(k3.get('kv_bytes') or 0) / GiB, 2)}"
    return "K3_FP8_" + (st if st != "DONE" else str(k3.get("correctness") or "NOT_CLEAN"))


SCREEN_OF = {"LOWBIT_TF_CLOSE": "CLOSE", "LOWBIT_TF_DIVERGENT": "DIVERGENT", "LOWBIT_TF_DIVERGENT_SEVERE": "SEVERE"}


def boot_predicates(rec, reading, fp8=False):
    """V-12: machine-readable predicates of one boot (downstream tools read these, never the label): boot_clean,
    tf_complete, numerical_screen (CLOSE / DIVERGENT / SEVERE / INCONCLUSIVE), method_map_pass, scales_applied_all_ranks
    (FP8 boots; None for BF16), accuracy_certified (always False: an ARC accuracy run is a separate step)."""
    rec = rec if isinstance(rec, dict) else {}
    rd = (reading or {}).get("reading")
    checks = ((rec.get("readback") or {}).get("checks") or {})
    return {"boot_clean": boot_clean(rec), "tf_complete": rd in TF_RANK and (reading or {}).get("complete", True) is True,
            "numerical_screen": SCREEN_OF.get(rd, "INCONCLUSIVE"),
            "method_map_pass": (rec.get("method_map") or {}).get("ok") is True, "method_map_verdict": map_verdict(rec),
            "scales_applied_all_ranks": ((rec.get("scales") or {}).get("ok") is True and checks.get("STATIC_SCALES_APPLIED") is True) if fp8 else None,
            "accuracy_certified": False}


def k2_tag(k2):
    """R2-12: K2's result tag for the label: K2_CLEAN, K2_GARBAGE_UNDER_LOAD (T 0.6 garbage in a load window), else None."""
    if boot_clean(k2):
        return "K2_CLEAN"
    if isinstance(k2, dict) and "NO_GO_GARBAGE_UNDER_LOAD" in (k2.get("status"), k2.get("correctness")):
        return "K2_GARBAGE_UNDER_LOAD"
    return None


def final_label(gates, k_rec, k2_rec, tf_read, k3_rec=None, k3_read=None, profile="ALL"):
    """K3 first (R3-1 + the user's order): `K3_<...> | <the BF16 label>` once K3 was reached (run or skipped); a gate
    that failed before K3 leaves FAIL_<gate> alone. The BF16 part is final_label_bf16."""
    out = final_label_bf16(gates, k_rec, k2_rec, tf_read)
    gate_failed = any(ok is not True for ok, _ in (gates or {}).values())
    if profile == "K3_ONLY" and not gate_failed:          # V-9: the BF16 boots belong to the other profile
        out = {"label": "BF16_NOT_IN_PROFILE", "accuracy": out["accuracy"]}
    elif not gate_failed and class_only_tag(k_rec, k2_rec):   # rev 3: K / K2 passed the method map on the class alone
        out = dict(out, label=out["label"] + class_only_tag(k_rec, k2_rec))
    if not isinstance(k3_rec, dict) or k3_rec.get("status") in (None, "NOT_RUN"):      # K3 never reached
        return out
    k3l = k3_label(k3_rec, k3_read)
    return dict(out, label=f"{k3l} | {out['label']}", k3=k3l, bf16=out["label"])


def final_label_bf16(gates, k_rec, k2_rec, tf_read):
    """The notebook's one label. gates: ordered {name: (ok, why)} of the pre-boot gates. GPU_LOAD_PASS needs every gate,
    boot K clean (probes / sampled engine faults, complete garbage-free oracle, config read-back, no NaN), boot K2 clean
    and a complete TF reading (K-5, else GPU_LOAD_INCOMPLETE_TF). R2-3: K dead of OOM after its core measurements with K2
    clean keeps K2's result (<head>_K_<status>_AT_PIN<gib> + K2_CLEAN). R2-12: K2 garbage under load ->
    <head>_K2_GARBAGE_UNDER_LOAD. R2-1: the 120k row's verdict is appended (+ LONG_<verdict>), never a gate. The TF
    reading is appended; never an accuracy verdict."""
    acc = "NOT_TESTED (an ARC accuracy run is a separate step)"
    for name, (ok, why) in (gates or {}).items():
        if ok is not True:                                # V-11: an ABORT_* gate is its own label (ABORT_DEVICE_FAULT)
            return {"label": name if name.startswith("ABORT_") else f"FAIL_{name}", "why": why, "accuracy": acc}
    tfr = (tf_read or {}).get("reading", "TF_NOT_MEASURED")
    full = tfr in TF_RANK                                 # K-5: PASS only with a complete TF reading (all four bands)
    head = "GPU_LOAD_PASS" if full else "GPU_LOAD_INCOMPLETE_TF"
    lv = ((k_rec.get("long_120k") or {}).get("verdict")) if isinstance(k_rec, dict) else None
    tail = f" + LONG_{lv}" if lv else ""
    tag = k2_tag(k2_rec)
    k2why = {"k2": (k2_rec or {}).get("status"), "k2_correctness": (k2_rec or {}).get("correctness"),
             "why": (k2_rec or {}).get("why") or (k2_rec or {}).get("early_stop") or (k2_rec or {}).get("config_mismatch")}
    if not boot_clean(k_rec):
        st = (k_rec or {}).get("status")
        if k_oom_after_core(k_rec) and tag:
            gib = round(int(k_rec.get("kv_bytes") or 0) / GiB, 2)
            return dict(k2why, label=f"{head}_K_{st}_AT_PIN{gib} + {tag} + {tfr}{tail}", tf_complete=full, accuracy=acc)
        return {"label": f"FAIL_BOOT_K_{st}" if st not in (None, "DONE") else f"FAIL_BOOT_K_{(k_rec or {}).get('correctness') or 'NOT_RUN'}",
                "why": (k_rec or {}).get("why") or (k_rec or {}).get("early_stop") or (k_rec or {}).get("config_mismatch"),
                "accuracy": acc, "k2": k2why["k2"], "k2_correctness": k2why["k2_correctness"]}
    if tag == "K2_CLEAN":
        return {"label": f"{head} + {tfr}{tail}", "tf_complete": full, "accuracy": acc}
    if tag == "K2_GARBAGE_UNDER_LOAD":
        return dict(k2why, label=f"{head}_K2_GARBAGE_UNDER_LOAD + {tfr}{tail}", tf_complete=full, accuracy=acc)
    return dict(k2why, label=f"{head}_MTP_OFF_ONLY + {tfr}{tail}", tf_complete=full, accuracy=acc)
