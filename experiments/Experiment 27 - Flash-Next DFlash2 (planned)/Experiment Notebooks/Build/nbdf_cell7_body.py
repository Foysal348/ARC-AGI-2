# ==================================================================== CELL 7 body (exec'd by the CELL 7 stub into the notebook globals)
KV_KIT_GATE = json.loads(r"""{"passed": false, "per_k": {}, "require_gpu_load_check": true, "why": ["no --kv-gate-report given at build time (the real CPU kit's report)"]}""")   # the real CPU kit's get_kv_cache_configs gate (build time); not passed -> D arms skipped
_EMB = [(NBDF_LIB_SRC, NBDF_LIB_SHA, "nbdf_lib"), (NB4T_LIB_SRC, NB4T_LIB_SHA, "nb4t_lib"), (TL_SRC, TL_SHA, "nb4_timeline"),
        (PPACK_SRC, PPACK_SHA, "nb4_ppack")] + ([(SEAM_SRC, SEAM_SHA, "dflash_seam")] if SEAM_SHA else [])
for _src, _sha, _n in _EMB:
    if hashlib.sha256(_src.encode("utf-8")).hexdigest() != _sha:
        raise RuntimeError(f"EMBED_SHA_MISMATCH: {_n}")
    (TMP / f"{_n}.py").write_text(_src, encoding="utf-8")
if str(TMP) not in sys.path:
    sys.path.insert(0, str(TMP))
T = importlib.import_module("nb4t_lib")
N = importlib.import_module("nbdf_lib")
TL = importlib.import_module("nb4_timeline")
P = importlib.import_module("nb4_ppack")
DS, SEAM_API, SEAM_LOAD = None, None, {"embedded": bool(SEAM_SHA), "sha256": SEAM_SHA or None}
if SEAM_SHA:
    try:
        DS = importlib.import_module("dflash_seam")
        SEAM_API = N.seam_api(DS)
        SEAM_LOAD["caps"] = SEAM_API["caps"]
        if not SEAM_API["ok"]:
            SEAM_LOAD["error"], DS, SEAM_API = "dflash_seam lacks " + ", ".join(SEAM_API["missing"]), None, None
    except Exception as e:
        SEAM_LOAD["error"], DS, SEAM_API = f"{type(e).__name__}: {e}"[:400], None, None
else:
    SEAM_LOAD["error"] = "SEAM_NOT_EMBEDDED: the builder had no dflash_seam.py"
NBDF = R.setdefault("nbdf", {})
# ---- E27 B2 (user rule 2026-11-01): only hard breakage stops an arm; every heuristic is recorded. The lab's degenerate
# check is reduced to hard garbage (EMPTY / REPEATED_CHAR / BANG_GARBAGE) for every gate (lab steady, smoke, oracle, long
# oracle, sampled, windows); nbdf_lib.garbage_review keeps the full check (.e27_full) and records LOW_VARIETY & co. as the
# window's labels. The cells' probe / sampled / oracle / quality gates are rebound to their hard-only nbdf_lib forms.
if not hasattr(L.degenerate, "e27_full"):
    L.degenerate = N.hard_only(L.degenerate)


def short_gate_failed(o):
    """B2: stop only on a probe REQUEST error or a hard sampled failure (error, dead server, no token, hard garbage)."""
    return N.short_gate_hard(o.get("probes"), o.get("sampled"), FAKE)


def quality_gate(o, ref):
    """B2: FAIL_* only for hard breakage; FAIL_HIGH_MARGIN / wrong probe answers / a short sampled request are recorded."""
    return N.quality_gate_hard(o, ref, FAKE, CLEAN)


def oracle_gate(o, need=(), extra=()):
    """B2: NO_GO only for request errors / hard garbage (oracle or a window) / a probe request error."""
    return N.oracle_gate_hard(o, need, extra, _windows(o), FAKE, CLEAN)


def evidence_gate(o, expected_rows, require_sampled=True):
    """B2 (fail-closed evidence kept): a missing piece is INCONCLUSIVE; NO_GO only for a probe request error, a hard sampled
    failure, oracle request errors or hard oracle garbage; wrong probe answers / a short sampled request are recorded."""
    pr = o.get("probes")
    if not isinstance(pr, dict):
        return "INCONCLUSIVE_PROBES_MISSING"
    if not FAKE and N._req_err(pr):
        return "NO_GO_SHORT_PROBE"
    if require_sampled:
        sm = o.get("sampled")
        if not isinstance(sm, dict):
            return "INCONCLUSIVE_NO_SAMPLED_TEST"
        if N.sampled_hard_fail(sm):
            return "NO_GO_SAMPLED_PATH"
    if expected_rows is None:
        return None
    orc = o.get("oracle")
    if not isinstance(orc, dict):
        return "INCONCLUSIVE_NO_ORACLE"
    if any(orc.get("errors") or []):
        return "NO_GO_ORACLE_REQUEST_ERROR"
    if any(g in N.HARD_GARBAGE for g in (orc.get("garbage") or [])):
        return "NO_GO_ORACLE_GARBAGE"
    if orc.get("row_count") != expected_rows or orc.get("complete") is not True:
        return "INCONCLUSIVE_ORACLE_INCOMPLETE"
    return None


NBDF.update(question="does DFlash2 (incoai drafter, K = 4 / 3 / 7) beat MTP K3 on the Q-defer base at 8k, 40k, 90k and 116k, "
                     "with greedy token ids EXACT / NEAR_TIE vs Q, no prefix-hit collapse and no concurrency contamination?",
            lib=N.LIB_VERSION, lib_sha256=NBDF_LIB_SHA, nb4t_lib_sha256=NB4T_LIB_SHA, timeline_sha256=TL_SHA, ppack_sha256=PPACK_SHA,
            seam=SEAM_LOAD, kv_kit_gate=KV_KIT_GATE, nb4t_notebook_sha256="675b1b9db76d49db1c71bcb271d9a7e9433abfb9c0536b93dbdb70543d073fd0", e18_r2_cell0_sha256="abe694be2a35b4dc00bfbf18fe2fdb0916bff15d74f502b5f5836db30cd46ec0",
            merlin_flags=json.loads(r"""{"Q": {"change": "none: NB4T's Q (nb4_ppack rev 2 + NB4_PPDEFER=1, Merlin's own spec + captures)", "merlin": ["tools/merlin/launch.py:441 (spec = {method mtp, num_speculative_tokens R.MTP_K, draft_sample_method probabilistic})", "tools/merlin/registry.py:49 (MTP_K = 3)", "tools/merlin/launch.py:459-460 (--compilation-config FULL_AND_PIECEWISE + capture_sizes)", "tools/merlin/registry.py:83-89 (capture_sizes_for(8, 3) = [1,2,3,4,6,8,12,16,24,32])"]}, "D_common": {"change": "seam env (DFLASH_SEAM=1 [+ DFLASH_DRAFT_EMBED_HOST=1 when the seam has it] [+ DFLASH_QSA_RING_WIDEN=1 at K >= 5]) + trailing --speculative-config {\"method\":\"dflash\",\"model\":<drafter>,\"num_speculative_tokens\":K,\"draft_sample_method\":\"probabilistic\",\"kv_cache_dtype\":\"auto\",\"attention_backend\":\"FLASH_ATTN\"} + trailing --compilation-config with the multiples of 1 + K up to 8 x (1 + K)", "override": ["tools/merlin/launch.py:110-111 (argv = prefix + vllm serve ckpt + args + EXTRA: extra args come last, the last occurrence wins; Merlin has no knob for the spec method / model: :441 hard-codes mtp)", "tools/merlin/launch.py:452 (Merlin's own --speculative-config), :459-460 (its --compilation-config)"], "spec_fields": ["vllm/config/speculative.py:395 num_speculative_tokens", ":398 model", ":401 method (explicit: the mount path detection at :1357-1373 needs 'dflash' in the path)", ":409 draft_tensor_parallel_size (unset -> the target TP 2, :1579-1583 + :1744-1773)", ":426 attention_backend (:1832-1839 parses the name)", ":430 kv_cache_dtype (None = inherit fp8_e4m3; 'auto' = BF16)", ":380-385 + :1820-1830 (the draft config overrides)", ":602 draft_sample_method (DFlash2: fp32 draft logits, dflash2/speculator.py:200-204; speculator.py:246-258)", ":1498-1499 parallel_drafting = True", ":1968-1974 DFlash: K new slots", ":2017-2018 use_dflash()"], "engine": ["vllm/config/vllm.py:661-664 (DFlash lookahead K + 1 slots)", ":697-715 (uniform_decode_query_len = 1 + K)", "vllm/v1/worker/gpu/spec_decode/__init__.py:27-39 (dflash + DFlash2DraftModel -> DFlash2Speculator)", "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:126-153 (draft graphs FULL_DECODE_ONLY, no PIECEWISE; eager warning when the backend lacks full graphs; 'Capturing model for DFlash2 speculator...'), :170 (progress bar)", "vllm/platforms/cuda.py:446 ('Using <selected> backend.' = the drafter's attention_backend), :507 (auto selection)", "vllm/v1/worker/gpu/cudagraph_utils.py:332-357 (sizes rounded up to multiples of 1 + K; above the max dropped)", "vllm/v1/worker/gpu/model_runner.py:386-393 + :524-533 (aux hidden states for dflash + PP relay check)", "vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py:100-111 (target_layer_ids + 1 -> (6, 16, 26, 36, 46))", "vllm/model_executor/models/config.py:1049-1058 (the Qwen4Exp allowlist: the seam adds dflash)", "vllm/models/qwen4_exp/common/qsa_cache.py:955-973 (QSA ring 4 x ceil((4 + K) / 4); the seam widens K >= 5)", "vllm/v1/core/kv_cache_utils.py:2063-2119 (the drafter group is not an eagle group: no trailing-block drop on hits)"], "draft_kv": "BF16 (auto) + FLASH_ATTN: FP8 drafter KV makes FA2 refuse on sm_89 (fa_utils.py:306-311) -> FlashInfer -> eager drafter (verify A B6); D4b = FP8 + TRITON_ATTN (triton_attn.py:302-308, 327-328), last + optional (upstream #54690)", "captures": "the multiples of 1 + K up to 8 x (1 + K): K3 4..32, K4 5..40, K7 8..64 (every decode batch an exact graph for the target AND the FULL_DECODE_ONLY drafter); Q keeps Merlin's ladder (b = 5 / 7 padded to 6 / 8)", "memory": "PP1 per card (verify A B9 / B section 3): drafter 0.727 GiB + embed 0.592 GiB (unless on the host) + fp32 draft_logits 8 x K x 248320 x 4 B (32 MB at K4) - the MTP module (~1.0-1.3 GiB); D pins on the ladder 3,285,145,068 / 3,016,709,612 (Q 3,553,580,524); 512 MiB free per rank after load / KV / graphs / first decode, else one rung down", "readback": ["config/model.py:735 'Resolved architecture: DFlash2DraftModel'", "seam events E27_DFLASH {json}: aux_layers (4 ranks, ids, local [6, 16 | 26, 36, 46], slot base 2), embed (PP1, sources, abs mean), embed_sidecar, qsa_ring", "v1/spec_decode/metrics.py:121-138 SpecDecoding lines (per-position acceptance)", "nb1_cache_info: kv_cache_specs (block sizes, CircularBufferSpec = the QSA ring capacity, the drafter's SlidingWindowSpec dtype), kv_cache_config groups", "kv_cache_utils 'Maximum concurrency for 131,072 tokens per request' -> F, pool blocks vs Q", "'Model loading took' / 'Graph capturing finished' per rank"]}, "D4s": {"k": 4, "sw_block": "DFLASH_DRAFT_SW_BLOCK=1024: kit v4 R9 205 blocks at 3,016,709,612 B (BF16 drafter page 10.5 MB < the target's 14.2 MB stride) vs 71 at block 4096 (page 41.9 MB); the seam lowers cache_config.block_size engine-wide (v1/engine/core.py:436-448): read back, SW_BLOCK_LOWERED_ENGINE_BLOCK recorded; pin: the D pin 3,016,709,612 first (the kit's 205 blocks), one step-down to 2,748,274,156 on OOM / the 512 MiB gate", "ring": "capacity 8 divides 4096 and 1024"}, "D3s": {"k": 3, "sw_block": "as D4s at K = 3 (kit R9 K3_swblock1024)"}, "D4": {"k": 4, "role": "the 71-block BF16 control (block 4096): smoke + oracle + 8k x 2 only"}, "D7": {"k": 7, "ring": "NEEDED: 4 x ceil(11 / 4) = 12 does not divide 4096 (qsa_cache.py:962); widened to 16 (DFLASH_QSA_RING_WIDEN=1); runs only when the seam's ring patch applied AND verified"}, "D4b": {"k": 4, "kv": "fp8_e4m3 + TRITON_ATTN (optional, last)"}, "ISO": "the first failing D arm reruns once with NB4_PPDEFER unset (same patched tree): QDEFER_INTERACTION / DFLASH_ITSELF", "long_bands": {"rule": "user 2026-11-01: every speed and correctness test up to ~120k context", "oracle": "greedy 256 tokens on PROMPTS['90k'][7] + PROMPTS['116k'][7] (Q = reference; D vs Q EXACT / NEAR_TIE)", "windows": "40k / 90k / 116k at the arm's resident level (pool + F of the boot; running == level confirmed, one step down on a KV-bound window); Q runs 90k and 116k twice (A/A noise)", "reset": "/reset_prefix_cache (VLLM_SERVER_DEV_MODE=1) before the first window of each long band"}}"""))
B3_PIN = KV_FP8 + (1 << 30)                 # NB1 R2 B3 = NB4S B = NB4T Q: 3,553,580,524 B (FP8 KV)
if B3_PIN != 3553580524:
    raise RuntimeError(f"B3_PIN {B3_PIN} != 3553580524")
Q_EXTRA = EXT_FLAG + ["--language-model-only", "--profiler-config", PROFILER_CFG]   # = NB4T Q_EXTRA
Q_BASE_ENV = dict(DIAG_ENV)                                                      # = NB4T Q_BASE_ENV
RESP_DIR = OUT / "responses"                # memory rule: the complete response of every request, gzipped jsonl per arm
RESP_DIR.mkdir(parents=True, exist_ok=True)
RESP = {"arm": "setup", "rows": None, "counts": {}}
_RESP_LOCK = threading.Lock()


def _resp_write(st):
    r = st.row()
    pr = st.body.get("prompt") or []
    rec = dict(r, arm=RESP["arm"], item=CUR.get("item"), port=st.port, prompt_len=len(pr),
               prompt_sha=hashlib.sha256(json.dumps(pr).encode()).hexdigest()[:16],
               tokens_sha=hashlib.sha256(json.dumps(r.get("tokens") or []).encode()).hexdigest()[:16],
               sampling={k: v for k, v in st.body.items() if k != "prompt"},
               ttft=round(st.t_first - st.t_start, 3) if st.t_first and st.t_start else None,
               sec=round(st.t_end - st.t_start, 3) if st.t_end and st.t_start else None, t=time.time())
    with _RESP_LOCK:
        with gzip.open(RESP_DIR / f"responses_{RESP['arm']}.jsonl.gz", "at", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        RESP["counts"][RESP["arm"]] = RESP["counts"].get(RESP["arm"], 0) + 1
        if RESP["rows"] is not None:
            RESP["rows"].append(rec)


ACTIVE_STREAMS = set()                      # every running L.Stream (the admission watchdog stops them on a wedge)


def _mk_nbdf_run(prev):
    """CELL 1's _traced_run (requests.jsonl) + the per-arm gz file; wrapped once (nbdf_lib.wrap_once)."""
    def _nbdf_run(self):
        ACTIVE_STREAMS.add(self)
        try:
            prev(self)
        finally:
            ACTIVE_STREAMS.discard(self)
            try:
                _resp_write(self)
            except Exception as e:
                R.setdefault("errors", []).append(f"response save: {e!r}"[:200])
    return _nbdf_run


N.wrap_once(L.Stream, "run", _mk_nbdf_run)
CUR = {"item": None, "kv_bytes": None}


# ---- the drafter Kaggle Model: one dir (glob, any framework case), pinned shas from the header + tensor bytes (no model load)
def drafter_resolve():
    """Exactly one dir under the Model mount (any framework-dir case) with config.json + model.safetensors on a bf16-embed path."""
    base = MD / N.DRAFTER_BASE
    cands = sorted({p.parent for p in base.rglob("config.json") if (p.parent / "model.safetensors").is_file()
                    and N.DRAFTER_VARIATION in p.parent.relative_to(base).parts}) if base.is_dir() else []
    if len(cands) != 1:
        return None, (f"DRAFTER_MISSING: no {N.DRAFTER_VARIATION} dir with config.json + model.safetensors under {base} (attach "
                      f"{N.DRAFTER_MODEL})" if not cands else f"DRAFTER_AMBIGUOUS: {[str(c) for c in cands]}")
    return cands[0], "ok"


def _sha_file(p, lo=0, n=None, rows=None):
    """sha256 of p[lo:lo+n] (whole file when n None); rows = row bytes: also counts all-zero rows."""
    h, zero, left_, buf = hashlib.sha256(), 0, n, b""
    zrow = b"\0" * rows if rows else None
    with open(p, "rb") as fh:
        fh.seek(lo)
        while left_ is None or left_ > 0:
            b = fh.read(1 << 24 if left_ is None else min(1 << 24, left_))
            if not b:
                break
            h.update(b)
            if left_ is not None:
                left_ -= len(b)
            if rows:
                buf += b
                k = len(buf) // rows * rows
                mv = memoryview(buf)
                zero += sum(1 for i in range(0, k, rows) if mv[i:i + rows] == zrow)
                buf = buf[k:]
    return h.hexdigest(), zero


def drafter_facts():
    d, why = drafter_resolve()
    if d is None:
        return {"ok": False, "why": why, "status": "DRAFTER_MISSING"}
    t0 = time.time()
    files, shas = {}, {}
    for f in sorted(p for p in d.rglob("*") if p.is_file()):
        rel = str(f.relative_to(d))
        files[rel] = f.stat().st_size
        shas[rel] = _sha_file(f)[0]
    st = d / "model.safetensors"
    keys = N.safetensors_keys(st)
    with open(st, "rb") as fh:
        hlen = int.from_bytes(fh.read(8), "little")
    emb = [k for k in keys if k.endswith("embed_tokens.weight")]
    facts = {"files": {k: [files[k], shas[k]] for k in files}, "tensors": len(keys), "embed": {}, "mask_row": {}, "zero_rows": None}
    if len(emb) == 1:
        e = keys[emb[0]]
        a, b = e["offsets"]
        row = int(e["shape"][1]) * 2
        es, zr = _sha_file(st, 8 + hlen + a, b - a, rows=row)
        facts["embed"] = {"name": emb[0], "shape": e["shape"], "dtype": e["dtype"], "nbytes": b - a, "sha256": es}
        facts["zero_rows"] = zr
        with open(st, "rb") as fh:
            fh.seek(8 + hlen + a + N.MASK_TOKEN_ID * row)
            mr = fh.read(row)
        facts["mask_row"] = {"sha256": hashlib.sha256(mr).hexdigest(), "norm": round(N.bf16_row_norm(mr), 8)}
    pins_ok, bad = N.drafter_pins_check(facts)
    cfg = json.loads((d / "config.json").read_text())
    man = None
    mf = next((d / n for n in ("dflash2_manifest.json", "E27_DRAFTER_MANIFEST.json") if (d / n).is_file()), None)
    if mf is not None:
        try:
            m = json.loads(mf.read_text())
            man = {k: v for k, v in (m.get("files") or {}).items() if k in files}
        except Exception:
            man = None
    chk = N.drafter_check(cfg, files, {"model.safetensors": keys}, shas, man)
    ok = pins_ok and chk["ok"]
    return {"ok": ok, "status": "OK" if ok else "DRAFTER_SHA_MISMATCH", "why": "ok" if ok else "; ".join(bad + ([chk["why"]] if not chk["ok"] else []))[:900],
            "dir": str(d), "pins_ok": pins_ok, "pin_mismatches": bad, "check": {k: chk.get(k) for k in ("ok", "why", "embed_in_checkpoint",
                                                                                                        "tensors", "selector_top_k", "block_size",
                                                                                                        "sliding_window", "manifest")},
            "facts": facts, "sec": round(time.time() - t0, 1),
            "config": {k: cfg.get(k) for k in ("architectures", "hidden_size", "num_hidden_layers", "num_key_value_heads", "head_dim",
                                               "vocab_size", "layer_types", "sliding_window", "dflash_config", "dtype")}}


DRAFTER = {"ok": False, "why": "not checked yet"}
try:
    DRAFTER = drafter_facts() if not FAKE else {"ok": True, "why": "fake", "dir": str(TMP / "fake_drafter"), "check": {"embed_in_checkpoint": True}}
except Exception as e:
    DRAFTER = {"ok": False, "status": "DRAFTER_CHECK_ERROR", "why": f"DRAFTER_CHECK_ERROR {type(e).__name__}: {e}"[:400]}
NBDF["drafter"] = DRAFTER
DRAFTER_DIR = DRAFTER.get("dir") or str(MD / N.DRAFTER_REL)
EMBED_IN_CKPT = bool((DRAFTER.get("check") or {}).get("embed_in_checkpoint"))
SEAM_ENVS, SEAM_NOTES = {}, {}
for _k in N.D_ARMS:
    SEAM_ENVS[_k], SEAM_NOTES[_k] = N.seam_env_for(SEAM_API, N.K_OF[_k], EMBED_IN_CKPT, sidecar_path=str(CKPT), host_embed=False,
                                                   mask_row_sha=N.DRAFTER_PINS["mask_row"]["sha256"],
                                                   conv_impl="torch" if _k in N.CONV_TORCH_ARMS else None,
                                                   draft_sw_block=N.SW_BLOCK if _k in N.SW_BLOCK_ARMS else None)
NBDF["seam_envs"], NBDF["seam_env_notes"] = SEAM_ENVS, SEAM_NOTES
ARMS = N.arm_plan(Q_EXTRA, Q_BASE_ENV, B3_PIN, DRAFTER_DIR, q_env=P.server_env("Q-defer"), seam_envs=SEAM_ENVS)
N.check_arm_plan(ARMS)                      # never-do list + "each D arm = Q + exactly the DFlash change" (raises: no boot)
ISO = {}                                    # "<K>_ISO": the Q-defer isolation rerun (built when a D arm fails)
_AKEYS = ("name", "port", "env", "change", "expect", "expect_cfg", "extra", "kv_bytes", "k", "spec", "captures", "coverage",
          "ring_needed", "ring_capacity_expected", "streams", "windows", "long_bands", "long_aa", "est_min", "cap_min", "title", "optional")
NBDF["arms"] = {k: {x: a.get(x) for x in _AKEYS} for k, a in ARMS.items()}
NBDF["order"] = list(N.ORDER)
NBDF["oom_policy"] = ("Q: NB4T's (a launch OOM -> one retry 0.25 GiB lower); D arms: start on the pin ladder below Q (pin_plan, at most "
                      "2 rungs), a launch OOM or a rank below 512 MiB free (after ready / after the first decode) -> one rung lower, "
                      f"up to {N.D_OOM_RETRIES} more; a later D arm starts at the pin that worked; a body OOM never retries")
NBDF["never_do"] = {"env": sorted(T.NEVER_DO_ENV), "argv": list(T.NEVER_DO_ARGS), "argv_values": T.NEVER_DO_ARG_VALUES}
NBDF["budget"] = {"hard_stop_min": BUDGET / 60, "start_reserve_sec": N.START_RESERVE_SEC, "body_reserve_sec": N.BODY_RESERVE_SEC,
                  "priority": "Q > D4 > D3 > D7 > D4b (an arm starts only with est_min x 60 + 300 s left; D7 / D4b dropped first)",
                  "planned_walk": N.budget_walk({}, BUDGET / 60, 8.0, ARMS)}
ARM_OF = {}                                 # launch name (retries included) -> arm
PIN = {"kv_bytes": B3_PIN, "source": "NB1 R2 B3 = NB4T Q"}
D_PIN = {"kv_bytes": None, "plan": None, "source": None}
SW_PIN = {"kv_bytes": None, "source": None}   # SW arms (D4s / D3s / D4t): the D pin (Q's pin - 2 rungs) first, then one rung down
NBDF_READY_SEC = 20 * 60                    # readiness gate per boot (NB4T: 627 s first compiling boot, 264-282 s later)
_HIST = {}
LONG_ORACLE_P = [PROMPTS["90k"][N.LONG_VARIANT], PROMPTS["116k"][N.LONG_VARIANT]]
PREFIX_PROMPTS = {v: prompt(N.PREFIX_P_TOKENS, v) for v in N.PREFIX_VARIANTS}   # 2 prompts x 3 turns (no window uses them)
PREFIX_FILLER = prompt(BANDS["8k"], 42)     # the turn-2 user message body, sized so turn 2's generation completes a block
Q_ACC = {}                                  # Q's greedy oracle acceptance (a diagnostic reference only, GPT audit B2)
Q_REC = {}                                  # Q's record (the 8k throughput rule of SHORT_EXIT_ARMS reads its windows)


def _tree_py():
    eng = merlin_install()
    return Path(eng["kv"]["MERLIN_TREE"]), eng["kv"]["MERLIN_VENV"] + "/bin/python"


def _ppack_apply():
    tree, py = _tree_py()
    return P.apply_ppack_patch(tree, py)


def _ppack_revert():
    tree, py = _tree_py()
    rec = P.revert_ppack_patch(tree, py)
    v = P.verify_ppack_patch(tree)
    if v["state"] != "not_applied":
        raise RuntimeError("REVERT_NOT_PRISTINE: " + json.dumps(v)[:400])
    rec["verify_after"] = v
    return rec


def _seam_apply(ring):
    tree, py = _tree_py()
    fn = SEAM_API["fns"]["apply"]
    rec = fn(tree, py, ring=ring) if SEAM_API["caps"]["ring_kw"] else fn(tree, py)
    return rec if isinstance(rec, dict) else {"returned": repr(rec)[:200]}


def _seam_verify(ring):
    tree, _ = _tree_py()
    fn = SEAM_API["fns"]["verify"]
    v = fn(tree, ring=ring) if SEAM_API["caps"]["ring_kw"] else fn(tree)
    return v if isinstance(v, dict) else {"state": str(v)}


def _seam_revert(ring):
    tree, py = _tree_py()
    fn = SEAM_API["fns"]["revert"]
    rec = fn(tree, py, ring=ring) if SEAM_API["caps"]["ring_kw"] else fn(tree, py)
    rec = rec if isinstance(rec, dict) else {"returned": repr(rec)[:200]}
    v = _seam_verify(ring)
    if N.tree_state(v) != "ORIGINAL":
        raise RuntimeError("SEAM_REVERT_NOT_PRISTINE: " + json.dumps(v, default=str)[:400])
    rec["verify_after"] = v
    rec["ppack_after"] = P.verify_ppack_patch(tree)["state"]       # nb4_ppack must be intact (still applied) after the seam revert
    return rec


PSEQ = T.PatchSeq(_ppack_apply, _ppack_revert)
SSEQ = N.SeamSeq(_seam_apply, _seam_verify, _seam_revert)


def tree_check(arm):
    """Three-state check before every boot (GPT analysis 2): nb4_ppack FULLY_PATCHED, dflash_seam FULLY_PATCHED (or ORIGINAL when
    it never applied: Q only); INVALID_MIXED anywhere aborts the D arms."""
    if FAKE:
        return {"ok": True, "fake": True}
    tree, _ = _tree_py()
    pp = N.tree_state(P.verify_ppack_patch(tree))
    if SEAM_API is not None and SEAM_API["caps"].get("tree_state") and SSEQ.state == "applied":
        try:
            NBDF.setdefault("combined_tree_state", {})[arm["key"]] = DS.combined_tree_state(tree, P, ring=SSEQ.ring)
        except Exception as e:
            NBDF.setdefault("combined_tree_state", {})[arm["key"]] = repr(e)[:200]
    sm = None
    if SEAM_API is not None and SSEQ.state == "applied":
        try:
            sm = N.tree_state(_seam_verify(SSEQ.ring))
        except Exception as e:
            sm = "INVALID_MIXED"
            R["errors"].append(f"seam verify: {e!r}"[:200])
    ok = pp == "FULLY_PATCHED" and (sm in (None, "FULLY_PATCHED")) and (arm["key"] == "Q" or sm == "FULLY_PATCHED")
    return {"ok": ok, "ppack": pp, "seam": sm, "seam_seq": SSEQ.state}


def _prom_nbdf(port):
    """e23_lab.prom (same dict, one HTTP read) + the iteration-token histogram buckets with their read time (NB4T)."""
    txt = L.U.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=15).read().decode("utf-8", "replace")
    agg, buckets = T.prom_parse(txt)
    h = _HIST.setdefault(port, [])
    h.append((time.perf_counter(), buckets))
    del h[:-600]
    return agg


L.prom = _prom_nbdf


def _hist_at(port, t):
    best = None
    for ts, b in _HIST.get(port, []):
        if ts <= t + 0.01:
            best = (ts, b)
    return best if best and t - best[0] < 1.0 else None


def _prefix_counters(port):
    try:
        m = L.prom(port)
        return {"queries": m.get("vllm:prefix_cache_queries_total"), "hits": m.get("vllm:prefix_cache_hits_total"),
                "prompt": m.get(L.PROMPT)}
    except Exception:
        return {}


def _prefix_delta(c0, c1):
    if not c0 or not c1 or None in (c0.get("queries"), c1.get("queries"), c0.get("hits"), c1.get("hits")):
        return {"rate": None}
    q, h = c1["queries"] - c0["queries"], c1["hits"] - c0["hits"]
    return {"queries": q, "hits": h, "rate": round(h / q, 4) if q > 0 else None}


def _mk_merlin_fn_server(prev):
    """prev = RC2 CELL 2 + NB1's per-boot hooks (+ NB4T's build-knob line, unused here); wrapped ONCE (nbdf_lib.wrap_once:
    a second execution of this cell must not wrap the wrapper)."""
    def merlin_fn_server(name, kv, port, kv_bytes):
        """E27 NBDF: NB1's Merlin plan, then (fail closed: no server is started) nbdf_lib.launch_check on the FULL final env /
        argv of THIS launch: never-do, watched env (DFLASH_* only on D arms), the effective spec + captures, seqs, pin."""
        return _nbdf_launch_check(prev(name, kv, port, kv_bytes), name, port, kv_bytes)
    return merlin_fn_server


def _nbdf_launch_check(s, name, port, kv_bytes):
    arm = ARM_OF.get(name)
    popped = {}
    for w, want in ((arm or {}).get("expect") or {}).items():
        if want is None and w in s.env:
            popped[w] = s.env.pop(w)
    bad = N.launch_check(arm, s.env, s.argv, kv_bytes) if arm else [f"launch {name} has no arm"]
    NBDF.setdefault("launch", {})[name] = {
        "arm": (arm or {}).get("key"), "port": port, "kv_bytes": kv_bytes, "never_do_and_plan": bad,
        "spec": T.spec_from_argv(s.argv), "captures": N.captures_from_argv(s.argv), "max_num_seqs": T.last_value(s.argv, "--max-num-seqs"),
        "env": {k: s.env[k] for k in sorted(s.env) if k.startswith(("CUDA_", "NCCL_", "NB4_", "VLLM_SERVER_DEV", "VLLM_USE_V2",
                                                                    "VLLM_PP_LAYER_PARTITION", "DFLASH", "P2P"))},
        "argv_tail": s.argv[-16:], "help": HELP.get(name), "popped_unplanned_env": popped}
    save()
    if bad:
        raise RuntimeError("NBDF_LAUNCH_CHECK: " + "; ".join(bad))
    return s


merlin_fn_server = N.wrap_once(globals(), "merlin_fn_server", _mk_merlin_fn_server)


def free_min(port):
    smi = L.gpu_mem() or []
    w = rpc_cache_info(port)
    wf = w.get("free_mib_per_rank") if isinstance(w, dict) else None
    sf = [t_ - u for _, u, t_ in smi] if smi else None
    mins = None
    if sf and wf and len(sf) == len(wf):
        mins = [min(a, b) if isinstance(b, (int, float)) else a for a, b in zip(sf, wf)]
    return {"nvidia_smi": sf, "worker": wf, "min": mins or sf or wf}


def proc_environ():
    blobs = {}
    for pid, _ in (engine_procs() or [])[:24]:
        try:
            blobs[pid] = Path(f"/proc/{pid}/environ").read_bytes()
        except Exception:
            blobs[pid] = None
    return blobs


def _live(o, key, val):
    o[key] = val
    rec = LIVE.get("rec")
    if rec is not None:
        rec.setdefault("live", {})[key] = val
    save()
    return val


def q_readback(s, arm):
    """nb4_ppack's 4 'NB4_PPQ init' lines + the Q-defer exec / fallback share per rank (every arm but the isolation rerun)."""
    if not arm.get("q_label"):
        return {"label": "NOT_APPLICABLE", "detail": {"why": "Q-defer off (isolation rerun)"}}
    try:
        lab, det = P.readback_label(s.log() or "", arm["q_label"])
    except Exception as e:
        lab, det = "NOT_INTEGRATED", {"error": f"{type(e).__name__}: {e}"[:300]}
    shares = {pid: {"exec_share": v.get("exec_share"), "fallback_share": v.get("fallback_share")}
              for pid, v in ((det or {}).get("defer_per_rank") or {}).items()}
    return {"label": lab, "detail": det, "shares": shares}


def _wall(perf_t):
    """perf_counter time -> the server log's 'MM-DD HH:MM:SS' clock (same container, local time)."""
    return time.strftime("%m-%d %H:%M:%S", time.localtime(time.time() - (time.perf_counter() - perf_t)))


def hist_window(port, s, band, n, arm):
    """NB1's window() + the micro-batch split (iteration-token histogram, the arm's K), the SpecDecoding lines inside the
    window, the prefix-hit rate, and the garbage review of the window's COMPLETE rows (windowed check gates)."""
    c0 = _prefix_counters(port)
    RESP["rows"] = []
    try:
        r = window(port, s, band, n)
    finally:
        rows, RESP["rows"] = RESP["rows"] or [], None
    c1 = _prefix_counters(port)
    h0, h1 = (_hist_at(port, r["t0"]) if r.get("t0") else None), (_hist_at(port, r["t1"]) if r.get("t1") else None)
    d = T.hist_delta(h0[1], h1[1]) if h0 and h1 else None
    r["parity"] = T.parity_from_hist(d, n, PP, arm["k"])
    r["level"] = r.get("level") or n
    pd = _prefix_delta(c0, c1)
    r["prefix_hits"], r["prefix_hit_rate"] = pd, pd.get("rate")
    if r.get("t0") and r.get("t1"):
        r["wall_t0"], r["wall_t1"] = _wall(r["t0"]), _wall(r["t1"])
        r["spec_log"] = N.spec_summary(N.spec_decoding_lines(s.log() or ""), arm["k"], r["wall_t0"], r["wall_t1"])
    N.apply_windowed_gate(r, N.garbage_review(rows, TOK.decode, L.degenerate), n)
    r["responses_saved"] = len(rows)
    return r


def _trace_spans(d):
    """Graph node-count modes + median span per rank (the drafter's FULL_DECODE_ONLY replay vs the target's verify graph)."""
    by = {}
    for k_, f in enumerate(sorted(Path(d).rglob("*.pt.trace.json*"))):
        tr = T._open(f)
        rank, _, _ = T._rank_of(f.name, tr, k_)
        by[rank] = T.rank_graphs(tr)[0]
    return N.graph_spans(by)


def profile_window(port, arm, band="8k", n=None):
    """A profiled window (NB4T): lab profiled_load, bounded POST /start_profile, traces kept per arm and band."""
    n = n or arm["streams"]
    shutil.rmtree(PROF_DIR, ignore_errors=True)
    PROF_DIR.mkdir(parents=True, exist_ok=True)
    rec = L.profiled_load(port, PROMPTS[band], n, PROF_DIR, secs=6.0, max_new=MAX_NEW[band], spec=True, prefill_tok_s=PREFILL["tok_s"])
    rec["profile_request"] = {"delay_iterations": 20, "max_iterations": 40, "secs": 6.0, "level": n, "band": band}
    if "trace" in rec:
        rec["lab_trace_summary"] = rec.pop("trace")
    kept = TRACES / f"profile_{arm['key'].lower()}_{band}"
    shutil.rmtree(kept, ignore_errors=True)
    kept.mkdir(parents=True, exist_ok=True)
    rec["trace_files"] = []
    for f in sorted(PROF_DIR.rglob("*.pt.trace.json*")):
        if f.stat().st_size < 120e6:
            shutil.copy2(f, kept / f.name)
            rec["trace_files"].append([f.name, f.stat().st_size])
    rec["trace_dir"] = str(kept)
    shutil.rmtree(PROF_DIR, ignore_errors=True)
    return rec


def win_need(band, n):
    if band.startswith("profile"):
        b = band.replace("profile_", "")
        return 360 + 10 * max(0, n - 8) + n * BANDS.get(b, 8192) / max(500.0, PREFILL["tok_s"])
    return WIN + 120 + n * BANDS[band] / max(500.0, PREFILL["tok_s"])


def reset_prefix(port):
    """POST /reset_prefix_cache (VLLM_SERVER_DEV_MODE=1) after the engine is idle."""
    out = {"idle": L.wait_idle(port, 90)}
    try:
        req = L.U.Request(f"http://127.0.0.1:{port}/reset_prefix_cache", data=b"", method="POST")
        r = L.U.urlopen(req, timeout=120)
        out.update(ok=200 <= r.status < 300, status=r.status)
    except Exception as e:
        out.update(ok=False, error=f"{type(e).__name__}: {e}"[:200])
    try:
        out["usage_after"] = kvu(port)
    except Exception:
        pass
    return out


def _one(port, ids, n_tok, temperature=0.0, logprobs=2, seed=None, timeout=900):
    g = L.Stream(port, ids, n_tok, temperature=temperature, seed=seed, logprobs=logprobs, ignore_eos=True)
    g.start()
    g.join(timeout)
    if g.is_alive():
        g.stop()
        g.join(10)
        r = g.row()
        r["error"] = r["error"] or "TIMEOUT"
        return r
    return g.row()


def smoke(port, arm):
    """Fail-fast smoke right after the boot: 2 sampled streams x 2,048 tokens (acceptance > ~0, windowed garbage) + the
    concurrency contamination ladder: one greedy probe alone, then next to 2 / 4 / 7 other sampled requests (7 + the probe = 8
    = max_num_seqs): identical token ids, else NEAR_TIE (recorded) or CONCURRENCY_CONTAMINATION. Per step: the probe's
    token sha, the drafted / accepted deltas and the prefix-hit delta."""
    k = arm["k"]
    out = {"streams": N.SMOKE_STREAMS, "max_new": N.SMOKE_TOKENS}
    a0, c0 = accept_pos(port), _prefix_counters(port)
    RESP["rows"] = []
    ss = [L.Stream(port, PROMPTS["2k"][2 + i], N.SMOKE_TOKENS, temperature=TEMP, seed=600 + i, ignore_eos=True) for i in range(N.SMOKE_STREAMS)]
    for g in ss:
        g.start()
    for g in ss:
        g.join(600)
    for g in ss:
        if g.is_alive():
            g.stop()
            g.join(10)
    time.sleep(1.0)
    rows, RESP["rows"] = RESP["rows"] or [], None
    out["acceptance"] = N.acceptance_delta(a0, accept_pos(port), k)
    out["prefix"] = _prefix_delta(c0, _prefix_counters(port))
    out["tokens"] = [len(g.tokens) for g in ss]
    out["errors"] = sorted({g.err for g in ss if g.err})[:3]
    out["garbage"] = N.garbage_review(rows, TOK.decode, L.degenerate)
    lad = {}
    probe = PROMPTS["2k"][0]
    a0, c0 = accept_pos(port), _prefix_counters(port)
    alone = _one(port, probe, N.CONC_TOKENS)
    lad["alone"] = {"error": alone.get("error"), "tokens_sha": hashlib.sha256(json.dumps(alone.get("tokens")).encode()).hexdigest()[:16],
                    "acceptance": N.acceptance_delta(a0, accept_pos(port), k), "prefix": _prefix_delta(c0, _prefix_counters(port))}
    for c in N.CONC_LADDER:
        others = [L.Stream(port, PROMPTS["2k"][1 + (i % 7)], 1024, temperature=TEMP, seed=700 + 10 * c + i, ignore_eos=True) for i in range(c)]
        for g in others:
            g.start()
        t_ = time.time()
        while time.time() - t_ < 60 and sum(g.t_first is not None for g in others) < c:
            time.sleep(0.5)
        a0, c0 = accept_pos(port), _prefix_counters(port)
        row = _one(port, probe, N.CONC_TOKENS)
        a1, c1 = accept_pos(port), _prefix_counters(port)
        for g in others:
            g.stop()
        for g in others:
            g.join(10)
        L.wait_idle(port, 60)
        lad[str(c)] = {"others": c, "others_decoding": sum(g.t_first is not None for g in others), "error": row.get("error"),
                       "tokens_sha": hashlib.sha256(json.dumps(row.get("tokens")).encode()).hexdigest()[:16],
                       "vs_alone": L.compare([alone], [row]), "acceptance": N.acceptance_delta(a0, a1, k), "prefix": _prefix_delta(c0, c1)}
    out["ladder"] = lad
    out["ladder_gate"] = N.contamination_gate(lad)
    out["ladder_brief"] = {kk: {"sha": v.get("tokens_sha"), "status": (v.get("vs_alone") or {}).get("status")} for kk, v in lad.items()}
    return out


def _measured(port, ids, n_tok, k):
    a0, c0 = accept_pos(port), _prefix_counters(port)
    r = _one(port, ids, n_tok)
    a1, c1 = accept_pos(port), _prefix_counters(port)
    pd = _prefix_delta(c0, c1)
    return r, {"acceptance": N.acceptance_delta(a0, a1, k), "prefix": pd, "prefix_hits": (pd.get("hits") or 0) > 0,
               "error": r.get("error"), "tokens": len(r.get("tokens") or []),
               "tokens_sha": hashlib.sha256(json.dumps(r.get("tokens")).encode()).hexdigest()[:16]}


def prefix_test(port, arm, block=4096):
    """Cold vs warm acceptance, 2 prompts x 3 turns (GPT analyses 2 / 3; upstream #47930 / #47926 / #58894). Per prompt (6,144
    tokens): WARM = the cache reset, turn 1 (greedy 2,048: block 2 completes with KV the speculative steps wrote), turn 2 =
    prompt + turn 1 + the commit suffix + a filler user message sized so its 512 greedy tokens complete the next block,
    turn 3 = turn 2 + its output + the suffix (greedy 256: reuses that block). COLD = the cache reset, then turn 2 alone;
    the cache reset, turn 3 alone. Per turn: AL / per position, drafted / accepted, prefix hits, token ids (cold vs warm)."""
    k = arm["k"]
    out = {"block": block, "prompts": []}
    for v, P0 in PREFIX_PROMPTS.items():
        pr = {"variant": v, "turns": {}}
        try:
            pr["reset_warm"] = reset_prefix(port)
            t1, m1 = _measured(port, P0, N.PREFIX_T1, k)
            if t1.get("error") or not t1.get("tokens"):
                pr["error"] = f"turn 1: {t1.get('error')}"
                out["prompts"].append(pr)
                continue
            head = list(P0) + list(t1["tokens"]) + list(SUFFIX_IDS)
            need = max(0, (-(-len(head) // block) + 1) * block - N.PREFIX_T2 - len(head))
            turn2 = head + list(PREFIX_FILLER[:need]) + list(SUFFIX_IDS)
            t2, m2 = _measured(port, turn2, N.PREFIX_T2, k)
            turn3 = turn2 + list(t2.get("tokens") or []) + list(SUFFIX_IDS)
            t3, m3 = _measured(port, turn3, N.PREFIX_T3, k)
            pr.update(turn1=m1, prompt_tokens={"1": len(P0), "2": len(turn2), "3": len(turn3)})
            pr["turns"]["2"], pr["turns"]["3"] = {"warm": m2}, {"warm": m3}
            for t, ids, n_tok, wr in (("2", turn2, N.PREFIX_T2, t2), ("3", turn3, N.PREFIX_T3, t3)):
                pr["turns"][t]["reset_cold"] = reset_prefix(port)
                cr, mc = _measured(port, ids, n_tok, k)
                pr["turns"][t]["cold"] = mc
                pr["turns"][t]["warm_vs_cold"] = L.compare([cr], [wr])
        except Exception as e:
            pr["error"] = f"{type(e).__name__}: {e}"[:300]
        out["prompts"].append(pr)
    return out


def long_oracle(port, key=None, ref=None, k=3):
    """Greedy LONG_NTOK tokens (ignore_eos) on the 90k + 116k prompts (variant 7). Q stores the reference (complete +
    garbage-free only); a D arm is compared with it by token ids (L.compare: EXACT / NEAR_TIE / FAIL_HIGH_MARGIN)."""
    a0 = accept_pos(port)
    rows = L.greedy(port, LONG_ORACLE_P, n_tok=N.LONG_NTOK, timeout=max(300, min(1500, boot_left() - 120)))
    garbage = [L.degenerate(r.get("text")) for r in rows]
    complete = len(rows) == len(LONG_ORACLE_P) and all(not r.get("error") and len(r.get("tokens") or []) == N.LONG_NTOK for r in rows)
    if key and complete and not any(garbage):
        ROWS[key] = rows
    elif key:
        R["errors"].append(f"long oracle {key}: reference not stored (incomplete / error / garbage)")
    res = {"bands": ["90k", "116k"], "prompt_tokens": [len(p) for p in LONG_ORACLE_P], "n_tok": N.LONG_NTOK, "garbage": garbage,
           "errors": [r.get("error") for r in rows], "tokens": [len(r.get("tokens") or []) for r in rows], "complete": complete,
           "text_head": [(r.get("text") or "")[:160] for r in rows], "acceptance": N.acceptance_delta(a0, accept_pos(port), k)}
    if ref:
        res["vs_" + ref] = L.compare(ROWS.get(ref), rows) if ROWS.get(ref) else {"status": "INCONCLUSIVE_NO_ORACLE"}
    return res


RPC_KEYS = ",".join(N.WATCHED) + ",NCCL_P2P_DISABLE,P2P"


def band_level(o, band):
    """nbdf_lib.level_for_band on THIS boot's pool (num_gpu_blocks), max concurrency (its log) and block size."""
    cap = o.get("capacity_after_ready") or {}
    pool = cap["num_gpu_blocks"][0] if len(cap.get("num_gpu_blocks") or []) == 1 else None
    blk = o.get("engine_block") or (((o.get("kv_layout") or {}).get("block_sizes") or {}).get("FullAttentionSpec") or [N.BLOCK_TOKENS])[0]
    return N.level_for_band(band, BANDS[band], pool, cap.get("max_concurrency_131072"), int(blk or N.BLOCK_TOKENS), o.get("kv_layout"),
                            PREFILL["tok_s"], WIN)


class _watched:
    """The admission watchdog around one item of a D arm: a daemon thread polls /metrics every 5 s; running == 0 with
    waiting > 0 for > 120 s -> ADMISSION_DEADLOCK on the arm record (running / waiting / KV usage) and every live stream
    of the item is stopped (a bounded stop); the arm's remaining long items are then skipped."""

    def __init__(self, o, port, label, active=True):
        self.o, self.port, self.label, self.active = o, port, label, active
        self.watch, self.stop_ev, self.th = None, threading.Event(), None

    def _fire(self, info):
        self.o["admission_deadlock"] = dict(info, arm=self.o.get("arm"))
        for g in list(ACTIVE_STREAMS):
            try:
                g.stop()
            except Exception:
                pass

    def _loop(self):
        t0 = time.perf_counter()
        while not self.stop_ev.wait(5.0):
            try:
                m = L.prom(self.port)
            except Exception:
                continue
            if self.watch.feed(time.perf_counter() - t0, int(m.get(L.RUN, 0)), int(m.get(L.WAIT, 0)), m.get(L.KVU)):
                break

    def __enter__(self):
        if self.active:
            self.watch = N.AdmissionWatch(on_fire=self._fire, label=self.label)
            self.th = threading.Thread(target=self._loop, daemon=True, name=f"nbdf_admission_{self.label}")
            self.th.start()
        return self

    def __exit__(self, *a):
        self.stop_ev.set()
        if self.th is not None:
            self.th.join(10)
        return False


def admission_probes(o, port):
    """D4s smoke: one single request each at 16k and 40k prompt tokens (64 greedy tokens), each under the watchdog."""
    out = {}
    for n_tok in N.ADMISSION_PROBE_TOKENS:
        if o.get("admission_deadlock"):
            out[str(n_tok)] = {"status": "SKIPPED_ADMISSION_DEADLOCK"}
            continue
        ids = PROMPTS["40k"][6] if n_tok == 40000 else prompt(n_tok, 51)
        t0 = time.perf_counter()
        with _watched(o, port, f"admission_probe_{n_tok}"):
            r = _one(port, ids, 64, timeout=900)
        out[str(n_tok)] = {"prompt_tokens": len(ids), "tokens": len(r.get("tokens") or []), "error": r.get("error"),
                           "sec": round(time.perf_counter() - t0, 1),
                           "status": "WEDGE" if (o.get("admission_deadlock") or {}).get("item") == f"admission_probe_{n_tok}" else
                           ("PASS" if len(r.get("tokens") or []) == 64 and not r.get("error") else "FAIL")}
    return out


def run_window(o, s, arm, band, n, tag=""):
    """One steady window (gated by due()); a KV-bound window steps the level down once (residency confirmed, recorded)."""
    key = f"{band}@{n}{tag}"
    need = win_need(band, n)
    if not due(need):
        o["steady"][key] = {"status": "SKIPPED_BOOT_BUDGET", "need_sec": round(need), "boot_left_sec": round(boot_left())}
        o["items_skipped"].append(key)
        return None
    if not s.alive():
        o["steady"][key] = {"status": "SKIPPED_SERVER_DEAD"}
        return None
    if o.get("admission_deadlock"):
        o["steady"][f"{band}@{n}{tag}"] = {"status": "SKIPPED_ADMISSION_DEADLOCK", "deadlock": o["admission_deadlock"]}
        o["items_skipped"].append(f"{band}@{n}{tag}")
        return None
    if n <= 0:                                    # a valid F with no room for one stream: recorded, never a fallback level
        o["steady"][f"{band}@0{tag}"] = {"status": "SKIPPED_CAPACITY", "level": 0, "long_level": (o.get("long_levels") or {}).get(band)}
        o["items_skipped"].append(f"{band}@0{tag}")
        return None
    CUR["item"] = key
    try:
        with _watched(o, s.port, f"window_{band}@{n}{tag}", active=arm.get("method") == "dflash"):
            r = hist_window(s.port, s, band, n, arm)
        if (o.get("admission_deadlock") or {}).get("item") == f"window_{band}@{n}{tag}":
            r = dict(r, status="ADMISSION_DEADLOCK", deadlock=o["admission_deadlock"])
    except Exception as e:
        r = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"[:400], "tb": traceback.format_exc()[-1200:]}
    N.capacity_window_status(r, ((o.get("long_levels") or {}).get(band) or {}).get("source") == "CAPACITY_UNVERIFIED")
    o["steady"][key] = r
    _live(o, "steady", o["steady"])
    if N.kv_bound(r) and n > 1 and band in ("40k",) + N.LONG_BANDS and due(win_need(band, n - 1)):
        r["residency"] = f"KV-bound at {n}: one step down to {n - 1}"
        o.setdefault("residency_retry", {})[band] = [n, n - 1]
        return run_window(o, s, arm, band, n - 1, tag)
    return key


def dflash_rb(arm, t):
    """nbdf_lib.dflash_readback + dflash_seam.readback_label (ENGAGED required: aux 4 ranks, embed own + mask row sha, load
    check lines, ring line at K >= 5)."""
    rb = N.dflash_readback(arm, t)
    if SEAM_API and SEAM_API["caps"].get("readback_label"):
        try:
            sig = str(inspect.signature(DS.readback_label))
            kw = {"no_ingraph_meta": arm["env"].get("DFLASH_NO_INGRAPH_META") == "1"} if "no_ingraph_meta" in sig else {}
            if "conv_impl" in sig:
                kw["conv_impl"] = arm["env"].get("DFLASH_CONV_IMPL") or "fused"
            lab = DS.readback_label(t, ring_k=arm["k"] if arm["ring_needed"] else None,
                                    mask_row_sha=arm["env"].get("DFLASH_MASK_ROW_SHA"),
                                    embed_host=arm["env"].get("DFLASH_DRAFT_EMBED_HOST") == "1", **kw)
        except Exception as e:
            lab = ("NOT_INTEGRATED", {"error": repr(e)[:200]})
        rb = N.merge_seam_label(rb, lab)
    return rb


MEM_RX = re.compile(r"Model loading took|Initial free memory|Available KV cache memory|peak|Memory profiling|CUDA graph pool|"
                    r"Graph capturing finished|non-torch|GPU KV cache size|Maximum concurrency")


def _can_step(arm):
    """USER RULE 2026-11-01: the 512 MiB memory gate steps a D arm one rung down only while another launch try remains and
    the budget holds it; otherwise the low reading is recorded and the arm measures on (only a real OOM / crash stops it)."""
    return (arm.get("method") == "dflash" and (CUR.get("tries_left") or 0) > 0
            and left() >= NBDF_READY_SEC + arm["cap_min"] * 60 + 180)


def _mem_gate(p, o, key):
    """nbdf_lib.memory_gate (fail-closed, 4 ranks); a None reading is retried once after 5 s, then recorded as None
    (no step-down; recorded: MEMORY_GATE_LOW_OR_MISSING flag, never a verdict override)."""
    mg = N.memory_gate((o.get(key) or {}).get("min"))
    if mg[0] is None:
        time.sleep(5)
        o[key] = free_min(p)
        mg = N.memory_gate((o[key] or {}).get("min"))
        o[key + "_retried"] = True
    return mg


def arm_body(arm, ref, store):
    """Every arm, the same items in the same order: facts + read-backs -> memory gate -> smoke (+ ladder) -> memory gate (first
    decode) -> oracle (+ AL, diagnostic) -> long oracle -> prefix cold / warm -> 8k x 2 -> 40k -> 90k -> 116k (Q x 2) ->
    profiled 8k (+ 116k when due); every item gated by due() on the boot deadline."""
    def body(ss):
        s = ss[0]
        p = s.port
        n = arm["streams"]
        is_q, is_d = arm["key"] == "Q", arm.get("method") == "dflash"
        o = {"arm": arm["key"], "k": arm["k"], "items_skipped": [], "streams": n}
        CUR["item"] = "facts"
        o = boot_facts(s, dict(o, census=kv_census(s), probes=L.chat_probes(p), sampled=sampled_check(s)), s.name)
        LIVE["out"] = o
        CUR["item"] = "readback"
        t = s.log() or ""
        _live(o, "env_readback_raw", rpc(p, "nb4s_env", RPC_KEYS))
        _live(o, "env_readback", T.parse_env_readback(o["env_readback_raw"], arm["expect"]))
        _live(o, "proc_environ", T.parse_proc_environ(proc_environ(), arm["expect"]))
        _live(o, "config_readback", N.config_readback(arm, t, CUR["kv_bytes"]))
        _live(o, "q_readback", q_readback(s, arm))
        _live(o, "kv_layout", N.kv_layout_readback(o.get("workers"), arm))
        _live(o, "graph_counts", N.graph_counts(t))
        if is_q:
            _live(o, "seam_inert", N.q_seam_inert(t))
        else:
            _live(o, "dflash_readback", dflash_rb(arm, t))
            _live(o, "ingraph_meta", N.ingraph_meta(t))
            o["kv_kit_summary"] = (KV_KIT_GATE.get("kv_summary") or {}).get(arm.get("parent") or arm["key"])
        o["fine_grained_disabled"] = N.FINE_GRAINED_RX in t
        o["engine_block"] = N.engine_block(t, o.get("kv_layout"))
        o["kv_draft_group_warning"] = N.DRAFT_GROUP_WARN_RX in t
        o["stage_graphs"] = N.stage_graphs(t)
        _live(o, "free_after_ready_min", free_min(p))
        _live(o, "capacity_after_ready", T.capacity_facts(dict(o, pin=CUR["kv_bytes"])))
        o["memory_log_lines"] = [l[:260] for l in t.splitlines() if MEM_RX.search(l)][:40]
        _live(o, "memory_stages", N.memory_stages(o))
        mg = _mem_gate(p, o, "free_after_ready_min")
        o["memory_gate_ready"] = {"ok": mg[0], "why": mg[1]}
        if is_d and mg[0] is False:
            if _can_step(arm):
                o.update(status="MEMORY_GATE_STEP_DOWN", early_stop="after ready: " + mg[1])
                return o
            o.setdefault("recorded_gates", []).append("memory gate after ready (no rung / budget left, arm continues): " + mg[1])
        if short_gate_failed(o):
            o.update(early_stop="SHORT_CORRECTNESS_GATE_FAILED", census_end=kv_census(s),
                     quality_gate="FAIL_SAMPLED" if (o["sampled"] or {}).get("ok") is not True else "FAIL_PROBES")
            return o
        sm_ = N.spec_mismatch(is_d, o.get("config_readback"), o.get("dflash_readback"))
        if sm_[0]:                                # spec method / K / drafter not the plan: this arm only (the others run)
            o.update(status="NOT_INTEGRATED", not_integrated=True, early_stop="SPEC_MISMATCH: " + sm_[1][:400])
            return o
        for nm in ("env_readback", "config_readback", "dflash_readback"):
            if (o.get(nm) or {}).get("definitive"):      # recorded (USER RULE 2026-11-01): the arm still measures
                o.setdefault("recorded_gates", []).append(f"{nm} contradicts the plan: " + str(o[nm].get("why"))[:300])
                log("NBDF", arm["key"], nm, "contradiction recorded (arm continues):", str(o[nm].get("why"))[:200])
        if arm.get("q_label") and (o["q_readback"] or {}).get("label") == "NOT_INTEGRATED":
            o.update(status="NOT_INTEGRATED", not_integrated=True, early_stop="nb4_ppack init lines: " + json.dumps(o["q_readback"])[:300])
            return o
        if is_d and KV_KIT_GATE.get("require_gpu_load_check", True) and \
                not ((o.get("dflash_readback") or {}).get("checks") or {}).get("draft_load_check"):
            o.update(status="NO_GO_DRAFT_LOAD_CHECK", early_stop="no draft_load_check line on both PP1 workers (the seam's "
                     "loaded-weights check) before any window: " + json.dumps(((o.get("dflash_readback") or {}).get("rec") or {}).get("draft_load_check"))[:300])
            return o
        if due(240):
            CUR["item"] = "smoke"
            _live(o, "smoke", smoke(p, arm))
            o["smoke_gate"] = N.smoke_gate(o["smoke"], is_q)
            _live(o, "free_after_smoke_min", free_min(p))
            mg = _mem_gate(p, o, "free_after_smoke_min")
            o["memory_gate_first_decode"] = {"ok": mg[0], "why": mg[1]}
            if is_d and o["smoke_gate"].startswith("FAIL"):       # errors / garbage / short / zero accept / contamination
                o.update(status="NO_GO_SMOKE", early_stop=f"smoke {o['smoke_gate']}", census_end=kv_census(s))
                return o
            if o["smoke_gate"] != "PASS":
                o.setdefault("recorded_gates", []).append(f"smoke {o['smoke_gate']} (recorded, arm continues)")
            if is_d and mg[0] is False:
                if _can_step(arm):
                    o.update(status="MEMORY_GATE_STEP_DOWN", early_stop="after the first decode: " + mg[1])
                    return o
                o.setdefault("recorded_gates", []).append("memory gate after the first decode (no rung / budget left, arm continues): " + mg[1])
        else:
            o["items_skipped"].append("smoke")
        if due(300):
            CUR["item"] = "prefill_probe"
            _live(o, "prefill", prefill_probe(p))
        if not due(240):
            o["items_skipped"].append("oracle")
            o["status"] = "SKIPPED_BOOT_BUDGET_ORACLE"
            return o
        CUR["item"] = "oracle"
        a0 = accept_pos(p)
        _live(o, "oracle", oracle(p, key=store, refs=(ref,)))
        o["oracle_acceptance"] = N.acceptance_delta(a0, accept_pos(p), arm["k"])
        if is_q:
            Q_ACC.update(o["oracle_acceptance"] or {})
        else:
            o["al_vs_q"] = N.al_vs_q(o["oracle_acceptance"], Q_ACC)          # diagnostic only (GPT audit B2)
        o["oracle_vs"] = (o["oracle"].get("vs_" + ref) or {}).get("status")
        o["quality_gate"] = quality_gate(o, ref)
        if o["quality_gate"].startswith("FAIL"):              # oracle errors / garbage / FAIL_HIGH_MARGIN / probes / sampled
            o["load_tests_skipped"] = f"oracle gate vs {ref}: {o['quality_gate']}"
            o["status"] = "NO_GO_QUALITY"
            o["census_end"] = kv_census(s)
            return o
        if o["quality_gate"] != "PASS":
            o.setdefault("recorded_gates", []).append(f"oracle gate vs {ref}: {o['quality_gate']} (recorded, arm continues)")
        if arm.get("short_only"):                 # D4t: smoke + greedy oracle + the two 8k@8 windows (vs D4), nothing else
            o["steady"], o["long_levels"], o["long_gate"], o["prefix_gate"] = {}, {}, "NOT_MEASURED_SHORT_ARM", "NOT_MEASURED_SHORT_ARM"
            if due(150):
                CUR["item"] = "warmup"
                _live(o, "warmup", warmup(p, n))
            run_window(o, s, arm, "8k", n, "#1")
            run_window(o, s, arm, "8k", n, "#2")
            CUR["item"] = "end"
            t = s.log() or ""
            o.update(census_end=kv_census(s), free_end_min=free_min(p), q_readback_end=q_readback(s, arm),
                     spec_log_all=N.spec_summary(N.spec_decoding_lines(t), arm["k"]), dflash_readback=dflash_rb(arm, t),
                     ingraph_meta=N.ingraph_meta(t), stage_graphs=N.stage_graphs(t), responses_saved=RESP["counts"].get(RESP["arm"]))
            o["memory_stages"] = N.memory_stages(o)
            return o
        if is_d and (arm.get("parent") or arm["key"]) in N.ADMISSION_PROBE_ARMS and due(240):
            CUR["item"] = "admission_probes"
            _live(o, "admission_probes", admission_probes(o, p))
        if due(300) and not o.get("admission_deadlock"):
            CUR["item"] = "long_oracle"
            with _watched(o, p, "long_oracle", active=is_d):
                _live(o, "long_oracle", long_oracle(p, key="q_long" if is_q else None, ref=None if is_q else "q_long", k=arm["k"]))
            o["long_gate"] = N.long_oracle_gate(o, is_q)
            if o["long_gate"].startswith(("NO_GO", "FAIL")):
                o.update(status="NO_GO_LONG_ORACLE", load_tests_skipped=f"long oracle {o['long_gate']}", census_end=kv_census(s))
                return o
        else:
            o["items_skipped"].append("long_oracle")
            o["long_gate"] = ("INCONCLUSIVE_ADMISSION_DEADLOCK" if o.get("admission_deadlock") else "INCONCLUSIVE_LONG_ORACLE_SKIPPED_BUDGET")
        if due(240) and not o.get("admission_deadlock"):
            CUR["item"] = "prefix_cache"
            try:
                with _watched(o, p, "prefix_cache", active=is_d):
                    _live(o, "prefix_cache", prefix_test(p, arm, block=o.get("engine_block") or N.BLOCK_TOKENS))
            except Exception as e:
                o["prefix_cache"] = {"error": f"{type(e).__name__}: {e}"[:300]}
            o["prefix_gate"] = N.prefix_gate(o["prefix_cache"])
        else:
            o["items_skipped"].append("prefix_cache")
            o["prefix_gate"] = "INCONCLUSIVE_PREFIX_SKIPPED_BUDGET"
        _live(o, "q_readback", q_readback(s, arm))
        qf_ = N.qdefer_real_failure(o["q_readback"]) if arm.get("q_label") else (False, "Q-defer off")
        o["qdefer_check"] = {"real_failure": qf_[0], "why": qf_[1]}
        if qf_[0]:                                # only a REAL Q-defer failure stops (USER RULE 2026-11-01); labels are recorded
            lab_ = (o["q_readback"] or {}).get("label")
            o.update(status="NOT_INTEGRATED" if lab_ == "NOT_INTEGRATED" else "QDEFER_FAILED", not_integrated=lab_ == "NOT_INTEGRATED",
                     early_stop="nb4_ppack read-back: " + qf_[1] + " | " + json.dumps(o["q_readback"])[:300])
            return o
        if due(150):
            CUR["item"] = "warmup"
            _live(o, "warmup", warmup(p, n))
        else:
            o["items_skipped"].append("warmup")
        o["steady"] = {}                      # RC2's _windows() gate reads "steady": garbage under load (windowed) is NO_GO here too
        o["long_levels"] = {}
        run_window(o, s, arm, "8k", n, "#1")
        run_window(o, s, arm, "8k", n, "#2")
        long_ok = True
        if (arm.get("parent") or arm["key"]) in N.SHORT_EXIT_ARMS and Q_REC:
            ex, why, det = N.short_exit(T._win(Q_REC), o["steady"])
            o["short_exit"] = {"exit": ex, "why": why, "detail": det}
            if ex:                                # GPT audit B2: only THIS arm's 40k / 90k / 116k windows are skipped
                long_ok = False
                for b in ("40k",) + N.LONG_BANDS:
                    o["steady"][f"{b}@skipped"] = {"status": "SKIPPED_SHORT_EXIT", "why": why}
                o["items_skipped"].append("long_windows_short_exit")
        bands = (("40k",) + N.LONG_BANDS) if long_ok else ()
        if "40k" in bands:
            lv, why = band_level(o, "40k")
            o["long_levels"]["40k"] = dict(why, level=lv)
            run_window(o, s, arm, "40k", lv)
        for band in [b for b in N.LONG_BANDS if b in bands]:
            lv, why = band_level(o, band)
            CUR["item"] = f"reset_{band}"
            o["long_levels"][band] = dict(why, level=lv, reset=reset_prefix(p) if lv > 0 and due(win_need(band, lv) + 60) else "SKIPPED")
            for tag in (("#1", "#2") if arm["long_aa"] else ("",)):
                run_window(o, s, arm, band, lv, tag)
        lv116 = (o["long_levels"].get("116k") or {}).get("level") or 0
        for band, lvb in (("8k", n),) + ((("116k", lv116),) if long_ok and lv116 > 0 else ()):
            pk = f"profile_{band}@{lvb}"
            if due(win_need(pk.split("@")[0], lvb)) and s.alive() and not o.get("admission_deadlock"):
                CUR["item"] = pk
                try:
                    o["steady"][pk] = profile_window(p, arm, band, lvb)
                except Exception as e:
                    o["steady"][pk] = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"[:400]}
            else:
                o["steady"][pk] = {"status": "SKIPPED_BOOT_BUDGET"}
                o["items_skipped"].append(pk)
        _live(o, "steady", o["steady"])
        CUR["item"] = "end"
        t = s.log() or ""
        o["census_end"] = kv_census(s)
        o["workers_end"] = rpc_cache_info(p)
        o["free_end_min"] = free_min(p)
        o["memory_stages"] = N.memory_stages(o)
        o["q_readback_end"] = q_readback(s, arm)
        o["ppq_stats_last"] = [e for e in P.parse_log(t) if e.get("event") == "stats"][-8:]
        o["jit_after_warmup"] = jit_kernels(t)[-12:]
        o["config_readback_end"] = N.config_readback(arm, t, CUR["kv_bytes"])
        o["spec_log_all"] = N.spec_summary(N.spec_decoding_lines(t), arm["k"])
        if not is_q:
            o["dflash_readback"] = dflash_rb(arm, t)               # the whole log: the SpecDecoding lines now exist
        else:
            o["seam_inert"] = N.q_seam_inert(t)
        o["responses_saved"] = RESP["counts"].get(RESP["arm"])
        o["prefix_reuse"] = N.prefix_reuse(o, t)
        o["stage_graphs"] = N.stage_graphs(t)
        o["kv_draft_group_warning"] = N.DRAFT_GROUP_WARN_RX in t
        sl_ = N.spec_decoding_lines(t)
        o["preempt_accept"] = {wk: N.preempt_accept(w, arm["k"], sl_) for wk, w in (o.get("steady") or {}).items()
                               if isinstance(w, dict) and (w.get("preemptions") or 0) > 0}
        if not is_d:
            o["ingraph_meta"] = None
        else:
            o["ingraph_meta"] = N.ingraph_meta(t)
        return o
    return body


def arm_watched(body, cap_min, name):
    def b(ss):
        cap = T.body_cap_sec(cap_min, left())
        BOOT.update(deadline=time.time() + cap, fired=None)

        def fire():
            BOOT["fired"] = name
            for x in ss:
                try:
                    x.kill()
                except Exception:
                    pass
        tm = threading.Timer(cap + 300, fire)
        tm.daemon = True
        tm.start()
        try:
            o = body(ss)
            LIVE["out"] = o if isinstance(o, dict) else None
        finally:
            tm.cancel()
            BOOT["deadline"] = None
        o["body_cap_sec"] = round(cap)
        if BOOT["fired"]:
            o["boot_watchdog"] = f"FIRED after {cap + 300:.0f} s: server killed"
        return o
    return b


def arm_boot(arm, body):
    """NB4T's boot (the gate, one Merlin launch, the /dev/shm cleanup) + the three-state tree check; Q: one OOM retry 0.25 GiB
    lower; D arms: start on D_PIN, one rung lower on a launch OOM or the 512 MiB memory gate, up to D_OOM_RETRIES more."""
    is_d = arm.get("method") == "dflash"
    is_sw = (arm.get("parent") or arm["key"]) in N.SW_BLOCK_ARMS
    kvb = ((SW_PIN["kv_bytes"] or PIN["kv_bytes"] - N.SW_START_RUNGS * OOM_STEP) if is_sw else (D_PIN["kv_bytes"] or arm["kv_bytes"])) if is_d else PIN["kv_bytes"]
    port, tries, o = arm["port"], [], {}
    ref = "rc2_fp8" if arm["key"] == "Q" else "q"
    max_tries = 1 + ((N.SW_OOM_RETRIES if is_sw else N.D_OOM_RETRIES) if is_d else 1)
    while True:
        lname = arm["name"] if not tries else f"{arm['name']}_t{len(tries)}"
        tc = tree_check(arm)
        if not tc["ok"]:
            return {"status": "TREE_INVALID_MIXED", "tree_check": tc, "tries": tries, "launch_name": lname, "pin": kvb}
        NB1_EXTRA[lname], NB1_ENV[lname], NB1_KV[lname] = list(arm["extra"]), dict(arm["env"]), kvb
        ARM_OF[lname] = arm
        CUR["kv_bytes"] = kvb
        CUR["tries_left"] = max_tries - len(tries) - 1
        BOOT["tag"] = lname
        RESP["arm"] = lname
        gate = boot_gate()
        if gate["ok"] is not True:
            return {"status": "GPU_NOT_EMPTY", "boot_gate": gate, "tries": tries, "launch_name": lname, "pin": kvb}
        shm0 = shm_names()
        mon = LoadMonitor(lname)
        mon.start()
        srv, CUR["item"] = None, "launch"
        try:
            srv = merlin_fn_server(lname, "fp8", port, kvb)
            o = with_servers([srv], NBDF_READY_SEC, arm_watched(body, arm["cap_min"], lname))
        finally:
            mon.stop_ev.set()
        try:
            full = srv.log() or ""
        except Exception:
            full = ""
        lines = full.splitlines()
        o.update(kv_bytes=kvb, pin=kvb, launch_name=lname, port=port, boot_gate=gate, tree_check=tc, shm_cleanup=shm_cleanup(shm0),
                 ready_sec=getattr(srv, "ready_sec", None), load_monitor_last=mon.last, oom_in_log=bool(OOM_ERR.search(full)),
                 xid_in_log=bool(XID.search(full)), oom_lines=[l[:240] for l in lines if OOM_ERR.search(l)][:6],
                 xid_lines=[l[:240] for l in lines if XID.search(l)][:6], died_in=CUR["item"])
        if srv is not None:
            o["q_readback_final"] = q_readback(srv, arm)
            o.setdefault("config_readback", N.config_readback(arm, full, kvb))
            if is_d:
                o["dflash_readback_final"] = dflash_rb(arm, full)
                o.setdefault("dflash_readback", o["dflash_readback_final"])
            o["launch_error_lines"] = [l[:300] for l in lines if re.search(r"Error|Traceback|NotImplementedError|assert", l)][-12:]
        oom = o.get("status") == "LAUNCH_FAILED" and o["oom_in_log"]
        memstep = o.get("status") == "MEMORY_GATE_STEP_DOWN"
        tries.append({"launch_name": lname, "port": port, "kv_bytes": kvb, "status": o.get("status"), "oom": oom, "memory_step": memstep,
                      "device_fault": o["xid_in_log"], "ready_sec": o["ready_sec"]})
        if not (oom or (is_d and memstep)) or o["xid_in_log"] or len(tries) >= max_tries or left() < NBDF_READY_SEC + arm["cap_min"] * 60:
            break
        kvb -= OOM_STEP
        port += 1
        log("boot", "OOM" if oom else "memory gate", lname, "-> one rung lower, KV_BYTES", kvb, "port", port)
    o["tries"] = tries
    if o.get("status") == "MEMORY_GATE_STEP_DOWN":
        o["status"] = "MEMORY_GATE_FAILED"
    if o.get("status") == "SERVER_DIED":
        o["death"] = ("BOOT_CAP_KILL" if o.get("boot_watchdog") else
                      f"BODY_OOM_{o.get('died_in')}" if o["oom_in_log"] and not o["xid_in_log"] else
                      f"DEVICE_FAULT_{o.get('died_in')}" if o["xid_in_log"] else f"SERVER_DIED_IN_{o.get('died_in')}")
        o["status"] = o["death"]
    o["capacity"] = T.capacity_facts(o)
    return gated(o, need=(ref,), expected_rows=4)


def analyze_arm(o, key):
    """After the boot (GPUs idle): graph stalls + nb4_timeline on the 8k profile; graph spans (drafter vs target replay) of
    every profile (8k, 116k)."""
    st_ = o.get("steady") or {}
    o["graph_spans"] = {}
    for pk, w in st_.items():
        if pk.startswith("profile") and isinstance(w, dict) and w.get("trace_dir") and list(Path(w["trace_dir"]).glob("*.pt.trace.json*")):
            try:
                o["graph_spans"][pk] = _trace_spans(w["trace_dir"])
            except Exception as e:
                o["graph_spans"][pk] = {"error": repr(e)[:200]}
    pk = next((k_ for k_ in st_ if k_.startswith("profile_8k")), None)
    d = (st_.get(pk) or {}).get("trace_dir") if pk else None
    if not d or not list(Path(d).glob("*.pt.trace.json*")):
        o["stall"] = {"error": "NO_TRACE_FILE"}
        o["timeline"] = {"error": "NO_TRACE_FILE"}
        return o
    t = time.time()
    try:
        st = T.graph_stalls(d)
        (OUT / f"stall_{key.lower()}.json").write_text(json.dumps(st, indent=1, default=str))
        o["stall"] = {k: v for k, v in st.items() if k != "by_rank"}
        o["stall"]["by_rank"] = {r: {x: y for x, y in v.items() if x != "rows"} for r, v in (st.get("by_rank") or {}).items()}
    except Exception as e:
        o["stall"] = {"error": f"{type(e).__name__}: {e}"[:400]}
    try:
        rep = TL.strip_private(TL.analyze(d))
        (OUT / f"timeline_{key.lower()}.json").write_text(json.dumps(rep, indent=1, default=str))
        o["timeline"] = T.summarize_timeline(rep)
        o["stage_compute"] = T.stage_compute(o["timeline"])
        try:
            (OUT / f"timeline_{key.lower()}.md").write_text(TL.format_report(rep))
        except Exception as e:
            o["timeline"]["format_report_error"] = repr(e)[:200]
    except Exception as e:
        o["timeline"] = {"error": f"{type(e).__name__}: {e}"[:400]}
    o["parity_record"] = T.parity_of(o)
    o["analysis_sec"] = round(time.time() - t, 1)
    return o


def _arm(key):
    return ARMS.get(key) or ISO.get(key)


def run_arm(key):
    arm = _arm(key)
    o = arm_boot(arm, arm_body(arm, "rc2_fp8" if key == "Q" else "q", "q" if key == "Q" else None))
    LIVE["out"] = o
    try:
        o["arm"], o["planned"] = key, {x: arm.get(x) for x in _AKEYS}
        o["free_mib_per_card"] = free_per_card(o, o.get("launch_name"))
        if left() > 120:
            analyze_arm(o, key)
    except Exception as e:
        keep_after_error(o, e)
    return o


def _arm_rec(key):
    a = _arm(key)
    r = (R["sections"].get(key) or {}).get(a["name"]) if a else None
    return r if isinstance(r, dict) else None


def arm_clean(o):
    return isinstance(o, dict) and o.get("status") == "DONE" and o.get("correctness") == "CLEAN" and o.get("quality_gate") == "PASS"


def skip_arm(key, why, status="SKIPPED_DEPENDENCY", **kw):
    R["sections"].setdefault(key, {})[_arm(key)["name"]] = dict({"status": status, "why": why}, **kw)
    save()


def finalize():
    try:
        recs = {}
        for k in list(ARMS) + list(ISO):
            r = _arm_rec(k)
            if r is not None and "stall" not in r and any(str(x).startswith("profile") and (v or {}).get("trace_dir")
                                                          for x, v in (r.get("steady") or {}).items() if isinstance(v, dict)):
                try:
                    analyze_arm(r, k)
                except Exception as e:
                    r["stall"] = {"error": repr(e)[:300]}
            recs[k] = r
        v = N.verdict({k: r for k, r in recs.items() if r is not None}, ARMS)
    except Exception as e:
        v = {"verdict": "ERROR_IN_VERDICT", "why": f"{type(e).__name__}: {e}"[:400], "tb": traceback.format_exc()[-1500:]}
    v["arms_status"] = {k: (_arm_rec(k) or {}).get("status") for k in list(ARMS) + list(ISO)}
    v["patch"], v["seam"], v["drafter_ok"] = NBDF.get("patch"), NBDF.get("seam_seq"), (NBDF.get("drafter") or {}).get("ok")
    v["d_pin"], v["kv_kit_gate"], v["responses_saved"] = D_PIN, KV_KIT_GATE.get("passed"), RESP["counts"]
    v["minutes"] = round((time.time() - T_START) / 60, 1)
    NBDF["verdict"] = v
    try:
        (OUT / "nbdf_verdict.json").write_text(json.dumps(v, indent=1, default=str))
    except Exception:
        pass
    save()
    log("NBDF VERDICT", v.get("verdict"), "|", str(v.get("why"))[:400])
    return v


def _patch_rec(pseq):
    return {"state": pseq.state, "log": pseq.log, "error": pseq.error, "apply": pseq.apply_rec, "revert": pseq.revert_rec}


def _seam_rec(sseq):
    return {"state": sseq.state, "ring": sseq.ring, "log": sseq.log, "error": sseq.error, "apply": sseq.apply_rec,
            "verify": sseq.verify_rec, "revert": sseq.revert_rec}


def nbdf_main(runner=None, pseq=None, sseq=None):
    """install -> #60490 + extension -> sha gate -> nb4_ppack -> dflash_seam (after nb4_ppack; a failure = every D arm
    NOT_INTEGRATED, Q still runs) -> Q -> D4 (AL early exit) -> D3 -> D7 (ring verified) -> D4b; the first failing D arm reruns
    once with Q-defer off; every D arm needs the drafter pins, the KV kit gate, a clean engaged Q and its full minimum of time.
    The seam is reverted first, then nb4_ppack; the verdict is written in the finally block whatever happens."""
    runner = runner or run_arm
    pseq = pseq or PSEQ
    sseq = sseq or SSEQ
    try:
        m = R["sections"]["A0"]["merge"]
        if not m.get("ok"):
            NBDF["stop"] = "MERGE_FAILED: no boot started"
            for k in ARMS:
                skip_arm(k, NBDF["stop"])
            return
        step("A0", "merlin_install", 10, lambda: {"install": (lambda e: {"ok": True, "tree": e["kv"].get("MERLIN_TREE")} if not FAKE else {"fake": True})(merlin_install())})
        pt = step("A0", "patch_60490", 3, apply_60490)
        if pt.get("status") != "DONE" or not pt.get("ok"):
            for k in ARMS:
                skip_arm(k, f"#60490 patch / worker extension not verified: {pt.get('error') or pt.get('status')}")
            return
        if "rc2_fp8" not in ROWS and not FAKE:     # recorded (USER RULE 2026-11-01): Q's oracle vs RC2 is then INCONCLUSIVE_NO_ORACLE
            NBDF["rc2_rows"] = "MISSING: RC2's recorded FP8 oracle rows do not match this notebook's prompts (Q runs; its gate vs RC2 is recorded)"
            log("NBDF", NBDF["rc2_rows"])
        ok, why = sha_gate(min(1500, left() - 900))
        R["setup"]["sha_gate"] = {"ok": ok, "why": why}
        if not ok:
            for k in ARMS:
                skip_arm(k, "sha gate: " + why, "SKIPPED_SHA_GATE")
            return
        ok_p, why_p = pseq.before_arm(ARMS["Q"])
        NBDF["patch"] = _patch_rec(pseq)
        q_tree_ok = ok_p
        if not ok_p:                              # only Q (coordinator 2026-11-01): each D arm runs its own tree check below
            skip_arm("Q", why_p, "NOT_INTEGRATED", not_integrated=True)
        if SEAM_API is None and sseq is SSEQ:
            seam_ok, seam_why = False, "NOT_INTEGRATED: " + str(SEAM_LOAD.get("error"))
        else:
            seam_ok, seam_why = sseq.apply()
        ring = N.ring_status(sseq) if seam_ok else (False, False, "no seam")
        NBDF["seam_seq"], NBDF["seam_ok"], NBDF["seam_why"] = _seam_rec(sseq), seam_ok, seam_why
        NBDF["ring"] = {"available": ring[0], "verified": ring[1], "why": ring[2]}
        try:
            NBDF["ppack_after_seam"] = P.verify_ppack_patch(_tree_py()[0])["state"] if not FAKE else "fake"
        except Exception as e:
            NBDF["ppack_after_seam"] = repr(e)[:200]
        save()
        log("NBDF seam:", seam_ok, seam_why, "| ring:", NBDF["ring"], "| drafter:", DRAFTER.get("status"), DRAFTER.get("why"))
        q = (step("Q", ARMS["Q"]["name"], ARMS["Q"]["est_min"], lambda: runner("Q")) if q_tree_ok
             else R["sections"]["Q"][ARMS["Q"]["name"]])
        bq = N.base_usable(q)                      # USER RULE 2026-11-01: only a REAL Q failure skips the D arms
        bs = N.base_check(q)
        NBDF["base"] = {"usable": bq[0], "label": bq[1], "why": bq[2], "strict_nb4t_base_check": {"ok": bs[0], "label": bs[1], "why": bs[2]}}
        save()
        if not bq[0]:
            for k in N.D_ARMS:
                skip_arm(k, f"Q not a usable base (status {q.get('status')}, correctness {q.get('correctness')}, gate "
                            f"{q.get('quality_gate')}, base {bq[1]}: {bq[2]})")
            return
        PIN.update(kv_bytes=q.get("pin") or PIN["kv_bytes"], source="Q's final pin")
        Q_REC.clear()
        Q_REC.update(q)
        fr = (q.get("free_after_ready_min") or {}).get("min") or []
        pp1 = min(fr[2:4]) if len(fr) >= 4 and all(isinstance(x, (int, float)) for x in fr[2:4]) else None
        plan = N.pin_plan(PIN["kv_bytes"], pp1, 4, N.host_embed_on(SEAM_ENVS.get("D4") or {}))   # host embed off: embed counted
        D_PIN.update(kv_bytes=plan["pin"], plan=plan, source="pin_plan from Q's PP1 free memory")
        NBDF["d_pin"] = D_PIN
        stop_d, isos = None, 0                   # stop_d only for a broken tree (never for AL / speed: GPT audit B2)
        for k in N.D_ARMS:
            a = ARMS[k]
            if stop_d:
                skip_arm(k, stop_d, "TREE_INVALID_MIXED", not_integrated=True)
                continue
            if not DRAFTER.get("ok"):
                skip_arm(k, f"{DRAFTER.get('status') or 'DRAFTER'}: {DRAFTER.get('why')}", "NOT_INTEGRATED", not_integrated=True)
                continue
            if not KV_KIT_GATE.get("passed"):
                skip_arm(k, "the real CPU kit's get_kv_cache_configs gate did not pass: " + str(KV_KIT_GATE.get("why")), "SKIPPED_KV_KIT_GATE")
                continue
            if not seam_ok:
                skip_arm(k, seam_why, "NOT_INTEGRATED", not_integrated=True)
                continue
            if not (KV_KIT_GATE.get("per_k") or {}).get(k):   # recorded (USER RULE 2026-11-01): the boot's own KV read-back plans it
                NBDF.setdefault("recorded_gates", {})[k] = f"no per-K KV summary from the real kit for {k} ({N.KV_RUN_OF[k]}): runs on its boot read-back"
            if a["ring_needed"] and not ring[1]:
                skip_arm(k, f"K {a['k']} needs the QSA ring widening: {ring[2]}", "SKIPPED_NO_RING")
                continue
            if not N.arm_fits(left(), a):
                skip_arm(k, f"left {left() / 60:.1f} min < its full minimum {a['est_min']} + {N.START_RESERVE_SEC // 60} min "
                            "(priority Q > D4 > D3 > D7 > D4b)", "SKIPPED_BUDGET")
                continue
            ok_s, why_s = sseq.check(k)
            ok_p, why_p = pseq.before_arm(a)
            NBDF["patch"], NBDF["seam_seq"] = _patch_rec(pseq), _seam_rec(sseq)
            if not ok_s or not ok_p:
                skip_arm(k, why_s if not ok_s else why_p, "TREE_INVALID_MIXED" if "INVALID" in (why_s + why_p) else "NOT_INTEGRATED",
                         not_integrated=True)
                stop_d = "dflash_seam / nb4_ppack tree not FULLY_PATCHED: " + (why_s if not ok_s else why_p)
                continue
            r = step(k, a["name"], a["est_min"], lambda k=k: runner(k))
            if r.get("status") == "DONE" and r.get("pin"):
                if k in N.SW_BLOCK_ARMS:
                    SW_PIN.update(kv_bytes=r["pin"], source=f"{k}'s final pin")
                else:
                    D_PIN.update(kv_bytes=min(D_PIN["kv_bytes"], r["pin"]), source=f"{k}'s final pin")
            if r.get("status") == "TREE_INVALID_MIXED":
                stop_d = "TREE_INVALID_MIXED before " + k
            if N.needs_nometa(r):                 # vLLM #63587: ONE relaunch with DFLASH_NO_INGRAPH_META=1 (budget permitting)
                na = N.nometa_arm(a)
                ISO[na["key"]] = na
                NBDF.setdefault("variants", {})[na["key"]] = {x: na.get(x) for x in _AKEYS}
                if N.arm_fits(left(), na):
                    r = step(na["key"], na["name"], na["est_min"], lambda kk=na["key"]: runner(kk))
                else:
                    skip_arm(na["key"], f"left {left() / 60:.1f} min < {na['est_min']} + 5 min", "SKIPPED_BUDGET")
            if isos < N.ISO_MAX and N.needs_isolation(r) and not str(r.get("smoke_gate", "")).endswith("ZERO_ACCEPT"):
                ia = N.iso_arm(a)
                ISO[ia["key"]] = ia
                isos += 1
                NBDF.setdefault("iso", {})[ia["key"]] = {x: ia.get(x) for x in _AKEYS}
                if N.arm_fits(left(), ia):
                    step(ia["key"], ia["name"], ia["est_min"], lambda kk=ia["key"]: runner(kk))
                else:
                    skip_arm(ia["key"], f"left {left() / 60:.1f} min < {ia['est_min']} + 5 min", "SKIPPED_BUDGET")
    finally:
        try:
            sseq.finish()
        except Exception as e:
            sseq.error = f"finish: {type(e).__name__}: {e}"[:300]
        try:
            pseq.finish()
        except Exception as e:
            pseq.error = f"finish: {type(e).__name__}: {e}"[:300]
        NBDF["patch"], NBDF["seam_seq"] = _patch_rec(pseq), _seam_rec(sseq)
        finalize()


def _gpu_util():
    while not _gpu_util_stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=20)
            if r.returncode == 0:
                with open(OUT / "nbdf_gpu_util.jsonl", "a") as fh:
                    fh.write(json.dumps({"t": round(time.time() - T_START, 1), "gpus": r.stdout.strip().splitlines()}) + "\n")
        except Exception:
            pass
        _gpu_util_stop.wait(5)


_gpu_util_stop = threading.Event()
if not RERUN and not os.environ.get("RC2_UNIT"):
    if NVSMI_OK:
        threading.Thread(target=_gpu_util, daemon=True).start()
    nbdf_main()
    _gpu_util_stop.set()
    R["host_phase"] = HOST_PHASE
    R["prefill"] = PREFILL
    R["minutes"] = round((time.time() - T_START) / 60, 1)
    save()
save()
