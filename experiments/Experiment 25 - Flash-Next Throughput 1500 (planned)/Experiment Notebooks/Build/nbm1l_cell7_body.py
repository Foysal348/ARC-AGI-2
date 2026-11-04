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
            merlin_flags=json.loads(r"""{"Q": {"change": "none: NB4T's Q (nb4_ppack rev 2 + NB4_PPDEFER=1, Merlin's own spec + captures)", "merlin": ["tools/merlin/launch.py:441 (spec = {method mtp, num_speculative_tokens R.MTP_K, draft_sample_method probabilistic})", "tools/merlin/registry.py:49 (MTP_K = 3)", "tools/merlin/launch.py:459-460 (--compilation-config FULL_AND_PIECEWISE + capture_sizes)", "tools/merlin/registry.py:83-89 (capture_sizes_for(8, 3) = [1,2,3,4,6,8,12,16,24,32])"]}, "D_common": {"change": "seam env (DFLASH_SEAM=1 [+ DFLASH_DRAFT_EMBED_HOST=1 when the seam has it] [+ DFLASH_QSA_RING_WIDEN=1 at K >= 5]) + trailing --speculative-config {\"method\":\"dflash\",\"model\":<drafter>,\"num_speculative_tokens\":K,\"draft_sample_method\":\"probabilistic\",\"kv_cache_dtype\":\"auto\",\"attention_backend\":\"FLASH_ATTN\"} + trailing --compilation-config with the multiples of 1 + K up to 8 x (1 + K)", "override": ["tools/merlin/launch.py:110-111 (argv = prefix + vllm serve ckpt + args + EXTRA: extra args come last, the last occurrence wins; Merlin has no knob for the spec method / model: :441 hard-codes mtp)", "tools/merlin/launch.py:452 (Merlin's own --speculative-config), :459-460 (its --compilation-config)"], "spec_fields": ["vllm/config/speculative.py:395 num_speculative_tokens", ":398 model", ":401 method (explicit: the mount path detection at :1357-1373 needs 'dflash' in the path)", ":409 draft_tensor_parallel_size (unset -> the target TP 2, :1579-1583 + :1744-1773)", ":426 attention_backend (:1832-1839 parses the name)", ":430 kv_cache_dtype (None = inherit fp8_e4m3; 'auto' = BF16)", ":380-385 + :1820-1830 (the draft config overrides)", ":602 draft_sample_method (DFlash2: fp32 draft logits, dflash2/speculator.py:200-204; speculator.py:246-258)", ":1498-1499 parallel_drafting = True", ":1968-1974 DFlash: K new slots", ":2017-2018 use_dflash()"], "engine": ["vllm/config/vllm.py:661-664 (DFlash lookahead K + 1 slots)", ":697-715 (uniform_decode_query_len = 1 + K)", "vllm/v1/worker/gpu/spec_decode/__init__.py:27-39 (dflash + DFlash2DraftModel -> DFlash2Speculator)", "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:126-153 (draft graphs FULL_DECODE_ONLY, no PIECEWISE; eager warning when the backend lacks full graphs; 'Capturing model for DFlash2 speculator...'), :170 (progress bar)", "vllm/platforms/cuda.py:446 ('Using <selected> backend.' = the drafter's attention_backend), :507 (auto selection)", "vllm/v1/worker/gpu/cudagraph_utils.py:332-357 (sizes rounded up to multiples of 1 + K; above the max dropped)", "vllm/v1/worker/gpu/model_runner.py:386-393 + :524-533 (aux hidden states for dflash + PP relay check)", "vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py:100-111 (target_layer_ids + 1 -> (6, 16, 26, 36, 46))", "vllm/model_executor/models/config.py:1049-1058 (the Qwen4Exp allowlist: the seam adds dflash)", "vllm/models/qwen4_exp/common/qsa_cache.py:955-973 (QSA ring 4 x ceil((4 + K) / 4); the seam widens K >= 5)", "vllm/v1/core/kv_cache_utils.py:2063-2119 (the drafter group is not an eagle group: no trailing-block drop on hits)"], "draft_kv": "BF16 (auto) + FLASH_ATTN: FP8 drafter KV makes FA2 refuse on sm_89 (fa_utils.py:306-311) -> FlashInfer -> eager drafter (verify A B6); D4b = FP8 + TRITON_ATTN (triton_attn.py:302-308, 327-328), last + optional (upstream #54690)", "captures": "the multiples of 1 + K up to 8 x (1 + K): K3 4..32, K4 5..40, K7 8..64 (every decode batch an exact graph for the target AND the FULL_DECODE_ONLY drafter); Q keeps Merlin's ladder (b = 5 / 7 padded to 6 / 8)", "memory": "PP1 per card (verify A B9 / B section 3): drafter 0.727 GiB + embed 0.592 GiB (unless on the host) + fp32 draft_logits 8 x K x 248320 x 4 B (32 MB at K4) - the MTP module (~1.0-1.3 GiB); D pins on the ladder 3,285,145,068 / 3,016,709,612 (Q 3,553,580,524); 512 MiB free per rank after load / KV / graphs / first decode, else one rung down", "readback": ["config/model.py:735 'Resolved architecture: DFlash2DraftModel'", "seam events E27_DFLASH {json}: aux_layers (4 ranks, ids, local [6, 16 | 26, 36, 46], slot base 2), embed (PP1, sources, abs mean), embed_sidecar, qsa_ring", "v1/spec_decode/metrics.py:121-138 SpecDecoding lines (per-position acceptance)", "nb1_cache_info: kv_cache_specs (block sizes, CircularBufferSpec = the QSA ring capacity, the drafter's SlidingWindowSpec dtype), kv_cache_config groups", "kv_cache_utils 'Maximum concurrency for 131,072 tokens per request' -> F, pool blocks vs Q", "'Model loading took' / 'Graph capturing finished' per rank"]}, "D4s": {"k": 4, "sw_block": "DFLASH_DRAFT_SW_BLOCK=1024: kit v4 R9 205 blocks at 3,016,709,612 B (BF16 drafter page 10.5 MB < the target's 14.2 MB stride) vs 71 at block 4096 (page 41.9 MB); the seam lowers cache_config.block_size engine-wide (v1/engine/core.py:436-448): read back, SW_BLOCK_LOWERED_ENGINE_BLOCK recorded; pin: the D pin 3,016,709,612 first (the kit's 205 blocks), one step-down to 2,748,274,156 on OOM / the 512 MiB gate", "ring": "capacity 8 divides 4096 and 1024"}, "D3s": {"k": 3, "sw_block": "as D4s at K = 3 (kit R9 K3_swblock1024)"}, "D4": {"k": 4, "role": "the 71-block BF16 control (block 4096): smoke + oracle + 8k x 2 only"}, "D7": {"k": 7, "ring": "NEEDED: 4 x ceil(11 / 4) = 12 does not divide 4096 (qsa_cache.py:962); widened to 16 (DFLASH_QSA_RING_WIDEN=1); runs only when the seam's ring patch applied AND verified"}, "D4b": {"k": 4, "kv": "fp8_e4m3 + TRITON_ATTN (optional, last)"}, "ISO": "the first failing D arm reruns once with NB4_PPDEFER unset (same patched tree): QDEFER_INTERACTION / DFLASH_ITSELF", "long_bands": {"rule": "user 2026-11-01: every speed and correctness test up to ~120k context", "oracle": "greedy 256 tokens on PROMPTS['90k'][7] + PROMPTS['116k'][7] (Q = reference; D vs Q EXACT / NEAR_TIE)", "windows": "40k / 90k / 116k at the arm's resident level (pool + F of the boot; running == level confirmed, one step down on a KV-bound window); Q runs 90k and 116k twice (A/A noise)", "reset": "/reset_prefix_cache (VLLM_SERVER_DEV_MODE=1) before the first window of each long band"}, "K100": {"boot": "klee100 LOAD rev 3 (K3_ONLY 7773e63f) / K100Q / E28 K100", "cfg": {"arms": ["Q", "D3s", "D4s"], "seqs": 24, "conc_ladder": [2, 8, 23], "q_pin": 7933526016, "q_ladder": [7933526016, 7665090560], "d_retry": [[7933526016, {"DFLASH_DRAFT_EMBED_HOST": "1"}], [7665090560, {"DFLASH_DRAFT_EMBED_HOST": "1"}], [7396655104, {"DFLASH_DRAFT_EMBED_HOST": "1"}], [6859784192, {"DFLASH_DRAFT_EMBED_HOST": "1"}]], "pin_ladder": [7933526016, 7665090560, 7396655104, 6859784192], "memory_gate": "flag", "q_ref": "k100_none", "method_map_need_mtp": "q_only", "q_i8": true, "q_captures": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96], "wkeys": ["8k@16#1", "8k@16#2", "profile_8k@16", "40k@16"], "windows": [["8k", 16, "#1"], ["8k", 16, "#2"], ["20k", 24, ""], ["40k", 16, ""], ["90k", 8, ""], ["116k", 6, ""]], "profile_level": 16, "tf_bands": ["2k", "8k", "40k"], "tf_mean_margin": 0.05, "tf_rank1_margin": 0.03, "long_thr_floor": {"90k": 0.037, "116k": 0.054}, "shorten_arm": "D4s", "shorten_min_left_min": 25, "tap_mode_alt": "pre_combine", "tap_est_min": 15, "tap_cap_min": 10, "prefill_admit_floor_tok_s": 3000}, "method_map": "need_mtp only on MTP arms: the DFlash2 drafter is dense (5 layers, no MoE) -> no MTP / drafter expert rows", "scales": "the sidecar has the 12 target QSA layers only; the MTP attention layer is DEFAULT 1.0 and recorded (never an error)"}, "NBM1": {"lib": "nbm1-lib 2026-11-02 r1 (MTP levers on Q = NBDF-K100 Q / E28 K100) + nbm1l-overlay 2026-11-02 r1 (LIGHT)", "lib_sha256": "e64482c456b860b87176836e6fdfc0cb10bbe2a10797eb11169aa69ff32400b8", "source_refs": "nbm1l_build_facts.json (size: the notebook stays < 400 kB)", "overlay_sha256": "2ce100d1f70d831bc5f840558780093989458e95ba76956dce1eff0b7eeb1563"}}"""))
B3_PIN = int(K100_CFG["q_pin"])             # E27 K100: every arm starts at the K100Q / E28 top K3 rung (Q pool 479, F 6)
if B3_PIN != 7933526016:
    raise RuntimeError(f"K100 Q pin {B3_PIN} != 7933526016")
Q_EXTRA = EXT_FLAG + ["--language-model-only", "--profiler-config", PROFILER_CFG]   # = NB4T Q_EXTRA
Q_BASE_ENV = dict(DIAG_ENV)                                                      # = NB4T Q_BASE_ENV
N.configure(SEQS=int(K100_CFG["seqs"]), CONC_LADDER=tuple(K100_CFG["conc_ladder"]), Q_CAPTURES=list(K100_CFG["q_captures"]),
            Q_TAIL=["--compilation-config", N.compilation_json(K100_CFG["q_captures"])], WKEYS=tuple(K100_CFG["wkeys"]),
            PREFIX_RECORD_ONLY=True, SHORT_EXIT_ARMS=(), LONG_THR_FLOOR=dict(K100_CFG["long_thr_floor"]),
            TF_MEAN_MARGIN=float(K100_CFG["tf_mean_margin"]), TF_RANK1_MARGIN=float(K100_CFG["tf_rank1_margin"]), TF_REQUIRED=True)   # E27 K100
Q_EXTRA = Q_EXTRA + ["--max-num-seqs", str(K100_CFG["seqs"])]               # E27 K100: every arm at 24 sequences (E28)
Q_BASE_ENV["MERLIN_SHARE_DRAFT_WORKSPACE"] = "1"                              # E27 K100: = E28 / K100Q (inert with DFlash)
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
N.restrict_arms(K100_CFG["arms"])                # E27 K100: Q > D3s > D4s
ARMS = {k: ARMS[k] for k in N.ORDER}
N.check_arm_plan(ARMS)                      # never-do list + "each D arm = Q + exactly the DFlash change" (raises: no boot)
# ---- E25 NB-M1: MTP levers on Q = NBDF-K100's Q verbatim (nbm1_lib, embedded); CELL 1's NBM1_ON switches the arms
NBM1_CFG = json.loads(r"""{"lib_sha256": "e64482c456b860b87176836e6fdfc0cb10bbe2a10797eb11169aa69ff32400b8", "budget_min": 155, "deadman_min": 160, "copies": {"b": {"Q": true, "V65": false, "V32": false, "I4": false, "IDX": true, "S28": true, "S32": true, "LP": true, "G1": false, "G2": false, "K4": false, "DYN": false, "PROF": false}, "c": {"Q": true, "V65": false, "V32": false, "I4": false, "IDX": false, "S28": false, "S32": false, "LP": false, "G1": true, "G2": true, "K4": true, "DYN": true, "PROF": true}}, "prof_windows": [["8k", 16], ["20k", 24]], "prof_secs": 30.0, "light": true, "overlay_sha256": "2ce100d1f70d831bc5f840558780093989458e95ba76956dce1eff0b7eeb1563", "owners": {"b": "hosen42", "c": "foysalemonshanto"}}""")
NBM1_LIB_SRC = r'''"""E25 NB-M1 library (2026-11-02): MTP lever arms on the PROVEN production base Q = NBDF-K100's Q arm verbatim = E28 K100
production (klee100 3bpw-MTP on Merlin v3.0.0 ad42e37, TP2 x PP2, MTP K3 probabilistic, Q-defer, I8 int8 draft head,
seqs 24, pin 7,933,526,016, its Q pin ladder, A/A, windows 8k@16 x 2 / 20k@24 / 40k@16 / 90k@8 / 116k@6, TF, oracle,
profiles). Pure logic, stdlib only; nbdf_lib (N) and nb4t_lib (T) are read through the module objects handed in.

Arms (each = Q + exactly ONE change, run in this order; no cross-arm skip):
  V65  VLLM_DRAFT_VOCAB_FILE (+ VLLM_DRAFT_VOCAB_SHA256) = the 65,536-token ARC table (draft_head.py / mtp.py:525-600)
  V32  the same with the 32,768-token table
  I4   draft_lm_head_quant int4 (DRAFT_HEAD=int4 into Merlin's build(): launch.py:319-322; speculative.py:610-622)
  IDX  --attention-config {"indexer_kv_dtype":"fp8"} (JSON form: the dotted form fails the --help gate)
  S28  --max-num-seqs 28 + captures = Merlin's capture_sizes_for(28, 3) (Q's rule extended)
  S32  --max-num-seqs 32 + capture_sizes_for(32, 3)
  LP   --max-num-batched-tokens 2048 --long-prefill-token-threshold 1024 --long-prefill-token-threshold-adaptive
  G1   VLLM_GDN_DECODE_FASTPATH=1          G2   VLLM_GDN_SHARED_ALIGN_INDEX=1
  K4   MTP K4 (trailing --speculative-config) + captures = every multiple of 5 up to 24 x 5
  DYN  K4 + num_speculative_tokens_per_batch_size [[1, 8, 4], [9, 24, 3]] + captures covering both tiers
  PROF no change: 30-s torch.profiler windows at 8k@16 and 20k@24 inside Q's own boot (NB-S1's profile machinery)
The draft head of a lever = Q's (int8 while Q_I8 is on, else BF16), I4's = int4. Vocab: Merlin quantizes the REDUCED
head (QuantizedDraftLMHead wraps the N/TP-row ParallelLMHead; N/TP % 64 == 0), so a vocab arm runs vocab + int8 = Q + one
change (draft_head.py, speculator.py:547-603, mtp.py:548-566 / 697-752). Real failures (that arm only, NOT_INTEGRATED):
the arm's own change not read back (env in all 4 workers, the REDUCED DRAFT VOCAB lines, the draft head lines, the indexer
cache dtype, the logged seqs / batched tokens / long-prefill threshold / spec K / dynamic schedule) or present on another
arm."""
import copy
import hashlib
import json
import math
import re

LIB_VERSION = "nbm1-lib 2026-11-02 r1 (MTP levers on Q = NBDF-K100 Q / E28 K100)"
LEVERS = ("V65", "V32", "I4", "IDX", "S28", "S32", "LP", "G1", "G2", "K4", "DYN")
ARM_KEYS = ("Q",) + LEVERS
SWITCH_KEYS = ARM_KEYS + ("PROF",)
SWITCH_DEFAULT = {k: True for k in SWITCH_KEYS}
COPIES = {"a": dict({k: False for k in SWITCH_KEYS}, Q=True, V65=True, V32=True, I4=True),
          "b": dict({k: False for k in SWITCH_KEYS}, Q=True, IDX=True, S28=True, S32=True, LP=True),
          "c": dict({k: False for k in SWITCH_KEYS}, Q=True, G1=True, G2=True, K4=True, DYN=True, PROF=True)}
OWNERS = {"a": "foysalemonshanto", "b": "esfersami50", "c": "foysalemonshanto"}
PORTS = {"Q": 8011, "V65": 8101, "V32": 8111, "I4": 8121, "IDX": 8131, "S28": 8141, "S32": 8151, "LP": 8161,
         "G1": 8171, "G2": 8181, "K4": 8191, "DYN": 8201}
GDN_ENV = {"G1": "VLLM_GDN_DECODE_FASTPATH", "G2": "VLLM_GDN_SHARED_ALIGN_INDEX"}
ENV_VOCAB, ENV_VOCAB_SHA = "VLLM_DRAFT_VOCAB_FILE", "VLLM_DRAFT_VOCAB_SHA256"     # draft_head.py:32-33 (Merlin's names)
M1_ENV_KEYS = (GDN_ENV["G1"], GDN_ENV["G2"], ENV_VOCAB, ENV_VOCAB_SHA)
VOCAB_DS = "foysalemonshanto/q27-e29-final-payload"
VOCAB_TABLES = {"V65": ("draft_vocab_k100_arc_n65536.pt", "a915fa6d5ad711817976de5c988a3950f43e7b2fbc058b04325e75c8501cf022", 65536),
                "V32": ("draft_vocab_k100_arc_n32768.pt", "a164d1a71d27dab600c014673c34accfdeb8695141a63be8aa4bfb4026695248", 32768)}
VOCAB_SIZE, TP, N_WORKERS = 248320, 2, 4
IDX_ARGS = ["--attention-config", '{"indexer_kv_dtype":"fp8"}']
LP_ARGS = ["--max-num-batched-tokens", "2048", "--long-prefill-token-threshold", "1024", "--long-prefill-token-threshold-adaptive"]
SEQ_ARMS = {"S28": 28, "S32": 32}
K4 = 4
DYN_SCHEDULE = [[1, 8, 4], [9, 24, 3]]
DYN_FAIL_RX = ("Overriding cudagraph_mode", "Disabling num_speculative_tokens_per_batch_size")
CAPTURE_LADDER = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128]   # Merlin registry.py:53 _LADDER
PROF_WINDOWS = (("8k", 16), ("20k", 24))
PROF_SECS = 30.0
TITLES = {"Q": "Q = NBDF-K100 Q verbatim = E28 K100 production (MTP K3, I8, Q-defer, seqs 24, pin 7,933,526,016), A/A",
          "V65": "Q + VLLM_DRAFT_VOCAB_FILE 65,536-token ARC table (+ its sha256), reduced head under Q's int8 quant",
          "V32": "Q + VLLM_DRAFT_VOCAB_FILE 32,768-token ARC table (+ its sha256), reduced head under Q's int8 quant",
          "I4": "Q + draft_lm_head_quant int4 (DRAFT_HEAD=int4) instead of int8",
          "IDX": "Q + --attention-config {\"indexer_kv_dtype\":\"fp8\"}",
          "S28": "Q + --max-num-seqs 28 (captures capture_sizes_for(28, 3))",
          "S32": "Q + --max-num-seqs 32 (captures capture_sizes_for(32, 3))",
          "LP": "Q + --max-num-batched-tokens 2048 --long-prefill-token-threshold 1024 --long-prefill-token-threshold-adaptive",
          "G1": "Q + VLLM_GDN_DECODE_FASTPATH=1", "G2": "Q + VLLM_GDN_SHARED_ALIGN_INDEX=1",
          "K4": "Q + MTP K4 (captures every multiple of 5 up to 120)",
          "DYN": "Q + MTP K4 at <= 8 running, K3 at 9-24 (num_speculative_tokens_per_batch_size)"}

# ------------------------------------------------------------------ the time model (minutes; GPU data 2026-11-01)
# NBDF-K100 GPU (q27-xhigh-e27-nbdf-k100): setup 5.6, Q ready 543 s, Q body hit its 41-min cap after the profiles (TF not
# run), Q post 6.4; window spacing 8k@16 110 s, 20k@24 215 s, 40k@16 183 s, 90k@8 275 s, 116k@6 272 s. NB-S1 GPU v1: first
# boot ready 732 s, later boots 285 / 255 s. A changed compile key (vocab table, attention / scheduler config, spec K)
# recompiles: "recompile" in between.
SETUP_MIN = 6.0
READY_MIN = {"first": 11.0, "recompile": 7.0, "warm": 4.8, "warm_capture": 5.5}
ITEM_MIN = {"facts": 1.5, "smoke_ladder": 3.0, "smoke": 1.3, "prefill_probe": 0.5, "oracle": 1.5, "oracle_aa": 1.5,
            "long_oracle": 2.5, "prefix": 4.0, "warmup": 1.0, "profiles_q": 7.0, "tf_q": 3.0, "tf": 1.5, "end": 1.2,
            "prof": 7.3}
WINDOW_MIN = {"8k@16": 1.85, "20k@24": 3.6, "40k@16": 3.05, "90k@8": 4.6, "116k@6": 4.55, "20k@28": 4.2, "20k@32": 4.6}
POST_MIN = {"q": 3.8, "lever": 0.8}
Q_WINDOWS = [["8k", 16, "#1"], ["8k", 16, "#2"], ["20k", 24, ""], ["40k", 16, ""], ["90k", 8, ""], ["116k", 6, ""]]
LEVER_ITEMS_OFF = {"ladder": False, "prefill_probe": False, "long_oracle": False, "prefix": False, "profiles": False, "tf": False}
PLAN = {   # per lever: ready class, window set (A/A-free; 90k + 116k always), the TF (target-numerics levers only)
    "V65": ("recompile", [["8k", 16, "#1"], ["20k", 24, ""], ["40k", 16, ""], ["90k", 8, ""], ["116k", 6, ""]], False),
    "V32": ("recompile", [["8k", 16, "#1"], ["20k", 24, ""], ["40k", 16, ""], ["90k", 8, ""], ["116k", 6, ""]], False),
    "I4": ("recompile", [["8k", 16, "#1"], ["20k", 24, ""], ["40k", 16, ""], ["90k", 8, ""], ["116k", 6, ""]], False),
    "IDX": ("recompile", [["8k", 16, "#1"], ["90k", 8, ""], ["116k", 6, ""]], True),
    "S28": ("warm_capture", [["20k", 28, ""], ["90k", 8, ""], ["116k", 6, ""]], False),   # 20k at the arm's seqs (Q: 20k@24)
    "S32": ("warm_capture", [["20k", 32, ""], ["90k", 8, ""], ["116k", 6, ""]], False),
    "LP": ("recompile", [["8k", 16, "#1"], ["90k", 8, ""], ["116k", 6, ""]], True),
    "G1": ("warm", [["8k", 16, "#1"], ["90k", 8, ""], ["116k", 6, ""]], False),
    "G2": ("warm", [["8k", 16, "#1"], ["90k", 8, ""], ["116k", 6, ""]], False),
    "K4": ("recompile", [["8k", 16, "#1"], ["90k", 8, ""], ["116k", 6, ""]], False),
    "DYN": ("recompile", [["8k", 16, "#1"], ["90k", 8, ""], ["116k", 6, ""]], False)}
BUDGET_MIN = 180            # the hard stop of every copy (the copies differ only in NBM1_ON): max over the copies' walks
DEADMAN_MIN = BUDGET_MIN + 5
COPY_RUNTIME_MAX = 175      # build-time assertion: each copy's planned runtime (setup + its arms) <= this (C: Q + PROF = 69)
START_RESERVE_MIN = 5.0     # nb4t_lib START_RESERVE_SEC: an arm starts only with est_min + 5 min left


def _wkey(w):
    return f"{w[0]}@{int(w[1])}"


def q_time(prof_on):
    """(est_min, cap_min, breakdown) of Q (+ PROF's two 30-s profiles inside its boot when on)."""
    body = {k: ITEM_MIN[k] for k in ("facts", "smoke_ladder", "prefill_probe", "oracle", "oracle_aa", "long_oracle", "prefix",
                                       "warmup", "profiles_q", "tf_q", "end")}
    body["windows"] = round(sum(WINDOW_MIN[_wkey(w)] for w in Q_WINDOWS), 2)
    if prof_on:
        body["prof"] = ITEM_MIN["prof"]
    b = sum(body.values())
    est = READY_MIN["first"] + b + POST_MIN["q"]
    return int(math.ceil(est)), int(math.ceil(b + 1.0)), {"ready": READY_MIN["first"], "body": body, "post": POST_MIN["q"]}


def lever_time(key):
    rc, wins, tf = PLAN[key]
    body = {k: ITEM_MIN[k] for k in ("facts", "smoke", "oracle", "warmup", "end")}
    body["windows"] = round(sum(WINDOW_MIN[_wkey(w)] for w in wins), 2)
    if tf:
        body["tf"] = ITEM_MIN["tf"]
    b = sum(body.values())
    est = READY_MIN[rc] + b + POST_MIN["lever"]
    return int(math.ceil(est)), int(math.ceil(b + 1.0)), {"ready": READY_MIN[rc], "ready_class": rc, "body": body, "post": POST_MIN["lever"]}


# ------------------------------------------------------------------ switches (CELL 1's NBM1_ON)
def switch_map(on, keys=SWITCH_KEYS):
    """NBM1_ON -> {key: bool}: absent (None) or not a dict -> every arm on; a dict -> bool(on[k]), a missing key -> off."""
    if on is None or not isinstance(on, dict):
        return {k: True for k in keys}
    return {k: bool(on.get(k, False)) for k in keys}


def switch_record(on, keys=SWITCH_KEYS):
    m = switch_map(on, keys)
    return {"NBM1_ON": on if isinstance(on, (dict, type(None))) else repr(on)[:200], "effective": m, "absent": on is None,
            "invalid_not_a_dict": on is not None and not isinstance(on, dict),
            "unknown_keys": sorted(str(k) for k in set(on) - set(keys)) if isinstance(on, dict) else [],
            "off": [k for k, v in m.items() if not v]}


# ------------------------------------------------------------------ captures / argv helpers
def capture_sizes_for(seqs, k=3):
    """= Merlin tools/merlin/registry.py:83-89 (the shipped ladder, the top appended)."""
    top = int(seqs) * (int(k) + 1)
    s = [x for x in CAPTURE_LADDER if x <= top]
    if top not in s:
        s.append(top)
    return s


def k4_captures(seqs=24):
    return [b * (K4 + 1) for b in range(1, int(seqs) + 1)]


def dense_schedule(schedule, max_seqs, k_max):
    """= Merlin v1/spec_decode/dynamic/utils.py build_dynamic_sd_schedule_lookup (nbs2_lib.dense_schedule)."""
    parsed = sorted((int(a), int(b), int(k)) for a, b, k in schedule)
    dense, nxt, last = [0] * (max_seqs + 1), 1, None
    for a, b, k in parsed:
        if a > nxt and last is not None:
            for x in range(nxt, min(a, max_seqs + 1)):
                dense[x] = min(k_max, last)
        for x in range(max(a, nxt), min(b, max_seqs) + 1):
            dense[x] = min(k_max, k)
        nxt, last = max(nxt, b + 1), k
        if nxt > max_seqs:
            break
    for x in range(nxt, max_seqs + 1):
        dense[x] = min(k_max, last)
    return dense


def dyn_captures(seqs=24):
    d = dense_schedule(DYN_SCHEDULE, int(seqs), K4)
    return sorted({b * (1 + d[b]) for b in range(1, int(seqs) + 1)})


def dyn_coverage(captures, seqs=24):
    """Every b = 1 .. seqs an exact graph at its own tier's query length 1 + K(b) (cudagraph_utils.py rounding)."""
    d = dense_schedule(DYN_SCHEDULE, int(seqs), K4)
    caps = sorted(set(int(c) for c in captures))
    bad = []
    for b in range(1, int(seqs) + 1):
        q = 1 + d[b]
        g = {-(-c // q) * q for c in caps if -(-c // q) * q <= caps[-1]}
        if b * q not in g:
            bad.append(b)
    return {"covers_all": not bad, "uncovered_or_padded_b": bad, "padded_b": [], "uncovered_b": bad,
            "tiers": sorted(set(d[1:]))}


def set_last(extra, flag, value):
    """The LAST occurrence of `flag` gets `value` (vLLM's argparse: last wins)."""
    out = list(extra)
    idx = [i for i, x in enumerate(out) if x == flag]
    if not idx or idx[-1] + 1 >= len(out):
        raise ValueError(f"{flag} not in the base argv")
    out[idx[-1] + 1] = value
    return out


def spec_with_head(spec, head):
    sp = dict(spec)
    sp.pop("draft_lm_head_quant", None)
    if head:
        sp["draft_lm_head_quant"] = head
    return sp


# ------------------------------------------------------------------ vocab tables (the E29 payload dataset)
def vocab_candidates(inp, env):
    import os.path as _p
    d = (env or {}).get("NBM1_VOCAB_DIR")
    out = [_p.join(str(inp), "datasets", VOCAB_DS), _p.join(str(inp), VOCAB_DS.split("/")[1])]
    return out + ([d] if d else [])


def resolve_vocab(inp, env, reader=None):
    """{arm: {"file", "path", "sha256", "n", "found", "why"}}: the first candidate dir whose file has the pinned sha256."""
    import os.path as _p

    def rd(p):
        with open(p, "rb") as fh:
            return fh.read()
    reader = reader or rd
    out = {}
    for k, (name, sh, n) in VOCAB_TABLES.items():
        rec = {"file": name, "sha256": sh, "n": n, "path": None, "found": False, "tried": []}
        for d in vocab_candidates(inp, env):
            p = _p.join(d, name)
            try:
                b = reader(p)
            except Exception as e:
                rec["tried"].append([p, type(e).__name__])
                continue
            got = hashlib.sha256(b).hexdigest()
            rec["tried"].append([p, got[:12]])
            if got == sh:
                rec.update(path=p, found=True)
                break
        rec["why"] = "ok" if rec["found"] else f"VOCAB_TABLE_MISSING: {name} (sha {sh[:12]}) not under {vocab_candidates(inp, env)} (dataset {VOCAB_DS})"
        out[k] = rec
    return out


# ------------------------------------------------------------------ the arm plan
def m1_arms(N, aq, switch=None, vocab=None):
    """aq = NBDF-K100's Q arm (nbdf_lib.arm_plan + configure(SEQS=24, ...) + the K100 Q tail). Returns {key: arm}: Q =
    the same arm (budget fields and the M1 env keys in its expect only), every lever = a deep copy + exactly one change."""
    sw = switch_map(switch) if not isinstance(switch, dict) or set(switch) != set(SWITCH_KEYS) else dict(switch)
    vocab = vocab or {}

    def expect_of(a):
        e = {w: a["env"].get(w, N.WATCHED_DEFAULT.get(w)) for w in N.WATCHED}
        for k in M1_ENV_KEYS:
            e[k] = a["env"].get(k)
        return e

    q = copy.deepcopy(aq)
    est, cap, tb = q_time(sw.get("PROF", True) and sw.get("Q", True))
    q.update(key="Q", name="nbm1_q", title=TITLES["Q"], est_min=est, cap_min=cap, time_model=tb, m1_lever=False,
             m1_change=None, m1_items=None, m1_head="q")
    q["expect"] = expect_of(q)
    arms = {"Q": q}
    for k in LEVERS:
        a = copy.deepcopy(q)
        env_add, argv_ops, head = {}, [], None
        if k in GDN_ENV:
            env_add[GDN_ENV[k]] = "1"
        elif k in VOCAB_TABLES:
            v = vocab.get(k) or {}
            env_add[ENV_VOCAB] = v.get("path") or f"MISSING:{VOCAB_TABLES[k][0]}"
            env_add[ENV_VOCAB_SHA] = VOCAB_TABLES[k][1]
        elif k == "I4":
            head = "int4"
        elif k == "IDX":
            argv_ops = [["append", list(IDX_ARGS)]]
        elif k == "LP":
            argv_ops = [["append", list(LP_ARGS)]]
        elif k in SEQ_ARMS:
            s = SEQ_ARMS[k]
            caps = capture_sizes_for(s, 3)
            argv_ops = [["set", "--max-num-seqs", str(s)], ["set", "--compilation-config", N.compilation_json(caps)]]
            a["captures"], a["streams"] = caps, s
            a["expect_cfg"].update(max_num_seqs=s, captures=caps)
            a["coverage"] = N.coverage(caps, s, 3)
        elif k in ("K4", "DYN"):
            sp = {"method": "mtp", "num_speculative_tokens": K4, "draft_sample_method": "probabilistic"}
            caps = k4_captures(24)
            if k == "DYN":
                sp["num_speculative_tokens_per_batch_size"] = copy.deepcopy(DYN_SCHEDULE)
                caps = dyn_captures(24)
            a["spec"], a["k"], a["captures"] = sp, K4, caps
            argv_ops = [["append", ["--speculative-config", N.spec_json(spec_with_head(sp, "int8"))]],
                        ["set", "--compilation-config", N.compilation_json(caps)]]
            a["expect_cfg"].update(num_speculative_tokens=K4, captures=caps)
            a["coverage"] = N.coverage(caps, 24, K4) if k == "K4" else dyn_coverage(caps, 24)
            ok_r, cap_r = N.qsa_ring(K4)
            a["ring_needed"], a["ring_capacity_expected"] = not ok_r, cap_r
            a["m1_trailing_spec"] = True
        a["env"].update(env_add)
        a["extra"] = apply_argv_ops(q["extra"], argv_ops)
        a["expect"] = expect_of(a)
        e_, c_, tb_ = lever_time(k)
        a.update(key=k, name=f"nbm1_{k.lower()}", port=PORTS[k], title=TITLES[k], long_aa=False, lever=True, m1_lever=True,
                 m1_change={"env": env_add, "argv": argv_ops, "head": head}, m1_head=head or "q",
                 m1_items=dict(LEVER_ITEMS_OFF, tf=PLAN[k][2]), m1_no_ladder=True, windows_fixed=copy.deepcopy(PLAN[k][1]),
                 est_min=e_, cap_min=c_, time_model=tb_, optional=False, short_only=False, iso=False)
        if k in VOCAB_TABLES:
            a["m1_vocab"] = {"file": VOCAB_TABLES[k][0], "sha256": VOCAB_TABLES[k][1], "n": VOCAB_TABLES[k][2],
                             "path": (vocab.get(k) or {}).get("path"), "found": bool((vocab.get(k) or {}).get("found"))}
        arms[k] = a
    return arms


def apply_argv_ops(extra, ops):
    x = list(extra)
    for op in ops or []:
        if op[0] == "append":
            x = x + list(op[1])
        elif op[0] == "set":
            x = set_last(x, op[1], op[2])
        else:
            raise ValueError(f"unknown argv op {op}")
    return x


def apply_head(N, arm, q_i8_on):
    """At boot (arm_boot's first lines): the lever's draft head = int4 (I4) or Q's (int8 while Q_I8 is on, else BF16).
    K4 / DYN carry a trailing spec: its draft_lm_head_quant follows the head. Returns the head."""
    head = "int4" if arm.get("m1_head") == "int4" else ("int8" if q_i8_on else None)
    arm["draft_head"] = head
    arm["expect_cfg"]["draft_head"] = head
    if arm.get("m1_trailing_spec"):
        js = N.spec_json(spec_with_head(arm["spec"], head))
        x = list(arm["extra"])
        idx = [i for i, v in enumerate(x) if v == "--speculative-config"]
        x[idx[-1] + 1] = js
        arm["extra"] = x
        for op in (arm.get("m1_change") or {}).get("argv") or []:     # the plan record follows (check_arms stays exact)
            if op[0] == "append" and op[1][:1] == ["--speculative-config"]:
                op[1][1] = js
    return head


def install(N, arms):
    """Rebind nbdf_lib's arm-set constants to the NB-M1 arms (every arm MTP: no D / SW / short / probe arm); the verdict
    then compares every lever with Q. Idempotent. Returns the new ORDER."""
    N.ORDER, N.D_ARMS = tuple(arms), tuple(k for k in arms if k != "Q")
    N.OPTIONAL = N.SHORT_ARMS = N.CONV_TORCH_ARMS = N.SHORT_EXIT_ARMS = N.ADMISSION_PROBE_ARMS = N.SW_BLOCK_ARMS = ()
    N.KV_RUN_OF = {}
    N.K_OF.update({k: a["k"] for k, a in arms.items()})
    return N.ORDER


def check_arms(N, T, arms, aq=None):
    """Raises unless: Q first and the listed order; unique names / ports (Merlin's 8000..8355 scheme); every arm MTP with
    Q-defer on the patched tree, no never-do item, coverage of every decode batch; Q = aq (when given) except the budget /
    name / expect fields; every lever = Q's env + exactly its planned env change, Q's argv with exactly its argv ops, Q's
    spec / head except the planned K / head change."""
    if tuple(arms) != ARM_KEYS:
        raise ValueError(f"NB-M1 arms {tuple(arms)} != {ARM_KEYS}")
    q = arms["Q"]
    if len({a["name"] for a in arms.values()}) != len(arms) or len({a["port"] for a in arms.values()}) != len(arms):
        raise ValueError("NB-M1: duplicate arm name / port")
    if max(a["port"] for a in arms.values()) + 8 > 8355:
        raise ValueError("NB-M1: ports outside Merlin's 8000..8355 scheme (with retry / iso offsets)")
    if aq is not None:
        for f in ("env", "extra", "spec", "captures", "k", "method", "kv_bytes", "q_label", "needs_patch", "expect_cfg", "port"):
            qa, qb = q.get(f), aq.get(f)
            if f == "expect_cfg":                # arm_boot sets draft_head at boot (Q_I8): not a plan difference
                qa, qb = ({x: y for x, y in (v or {}).items() if x != "draft_head"} for v in (qa, qb))
            if qa != qb:
                raise ValueError(f"NB-M1 Q.{f} differs from NBDF-K100's Q")
    for k, a in arms.items():
        if a.get("method") != "mtp" or (a.get("spec") or {}).get("method") != "mtp":
            raise ValueError(f"NB-M1 {k}: not an MTP arm")
        bad = T.never_do_violations(a["env"], a["extra"])
        if bad:
            raise ValueError(f"NB-M1 {k}: never-do {bad}")
        if not a.get("needs_patch") or a.get("q_label") != N.Q_LABEL or a["env"].get(T.ENV_DEFER) != "1":
            raise ValueError(f"NB-M1 {k}: every arm runs Q-defer on the patched tree")
        if any(x.startswith("DFLASH_") for x in a["env"]):
            raise ValueError(f"NB-M1 {k}: a DFLASH_* variable on an MTP arm")
        cov = a.get("coverage") or {}
        if not cov.get("covers_all"):
            raise ValueError(f"NB-M1 {k}: decode batches without a FULL graph {cov}")
        if N.captures_from_argv(a["extra"]) != a["captures"] or T.last_value(a["extra"], "--max-num-seqs") != str(a["expect_cfg"]["max_num_seqs"]):
            raise ValueError(f"NB-M1 {k}: argv captures / seqs != plan")
        if set(a["expect"]) != set(N.WATCHED) | set(M1_ENV_KEYS):
            raise ValueError(f"NB-M1 {k}: expect keys")
        if k == "Q":
            if any(q["env"].get(x) for x in M1_ENV_KEYS) or "--speculative-config" in q["extra"]:
                raise ValueError("NB-M1 Q: an M1 change on the base")
            continue
        ch = a["m1_change"]
        diff = {e: a["env"].get(e) for e in set(a["env"]) | set(q["env"]) if a["env"].get(e) != q["env"].get(e)}
        if diff != ch["env"]:
            raise ValueError(f"NB-M1 {k}: env diff {diff} != the planned change {ch['env']}")
        if a["extra"] != apply_argv_ops(q["extra"], ch["argv"]):
            raise ValueError(f"NB-M1 {k}: argv differs from Q beyond the planned change")
        n_changes = bool(ch["env"]) + bool(ch["head"]) + (1 if ch["argv"] else 0)
        if n_changes != 1:
            raise ValueError(f"NB-M1 {k}: {n_changes} change kinds (exactly one)")
        if k in ("K4", "DYN"):
            want = {"method": "mtp", "num_speculative_tokens": K4, "draft_sample_method": "probabilistic"}
            if k == "DYN":
                want["num_speculative_tokens_per_batch_size"] = DYN_SCHEDULE
                d = dense_schedule(DYN_SCHEDULE, 24, K4)
                if d[1:9] != [4] * 8 or d[9:25] != [3] * 16:
                    raise ValueError(f"NB-M1 DYN dense schedule {d[1:]}")
            if a["spec"] != want or a["k"] != K4:
                raise ValueError(f"NB-M1 {k}: spec {a['spec']}")
        elif a["spec"] != q["spec"] or a["k"] != q["k"]:
            raise ValueError(f"NB-M1 {k}: spec / K differ from Q")
        if (ch["head"] is not None) != (k == "I4") or (k == "I4" and ch["head"] != "int4"):
            raise ValueError(f"NB-M1 {k}: draft head change")
        if not a.get("windows_fixed") or not any(w[0] == "90k" for w in a["windows_fixed"]) or not any(w[0] == "116k" for w in a["windows_fixed"]):
            raise ValueError(f"NB-M1 {k}: the window set must include 90k and 116k (user rule)")
    return True


def pre_gate(arm, kv_gate):
    """Before a lever's boot: None = launch, else (status, why). The real CPU kit gate (closed build: every lever
    SKIPPED_KV_KIT_GATE, Q still runs); a vocab arm needs its table found with the pinned sha."""
    if not (kv_gate or {}).get("passed"):
        return "SKIPPED_KV_KIT_GATE", "the NB-M1 real CPU kit gate did not pass: " + str((kv_gate or {}).get("why"))[:400]
    v = arm.get("m1_vocab")
    if v is not None and not v.get("found"):
        return "NOT_INTEGRATED", f"VOCAB_TABLE_MISSING: {v.get('file')} (sha {str(v.get('sha256'))[:12]}) not found in {VOCAB_DS}"
    return None


def item_on(arm, item):
    """A body item of this arm (Q: every item; a lever: LEVER_ITEMS_OFF + its plan's TF)."""
    return bool((arm.get("m1_items") or {}).get(item, True))


# ------------------------------------------------------------------ read-backs (real failures)
_VOCAB_RX = re.compile(r"REDUCED DRAFT VOCAB ACTIVE: tp_rank=(\d+) N=(\d+) of (\d+) \(file (\S+), table")
_WTAG = re.compile(r"\(Worker_PP(\d+)_TP(\d+)")


def env_rows(env_raw):
    """[(rank, {key: value})] of the nb4s_env RPC (every requested key, RPC_KEYS)."""
    res = (env_raw or {}).get("results") if isinstance(env_raw, dict) else None
    out = []
    for w in res if isinstance(res, list) else []:
        w = w if isinstance(w, dict) else {}
        out.append((w.get("rank"), w.get("env") if isinstance(w.get("env"), dict) else {}))
    return out


def env_values(env_raw, key):
    return [e.get(key) for _, e in env_rows(env_raw)]


def vocab_lines(log_text):
    out = []
    for l in (log_text or "").splitlines():
        m = _VOCAB_RX.search(l)
        if m:
            w = _WTAG.search(l)
            out.append({"tp_rank": int(m.group(1)), "n": int(m.group(2)), "of": int(m.group(3)), "file": m.group(4),
                        "worker": f"PP{w.group(1)}_TP{w.group(2)}" if w else None})
    return out


def _nondefault(log_text):
    return next((l for l in (log_text or "").splitlines() if "non-default args:" in l), "")


def logged_int(log_text, key):
    m = re.search(r"'" + re.escape(key) + r"': (\d+)", _nondefault(log_text))
    return int(m.group(1)) if m else None


def logged_bool(log_text, key):
    m = re.search(r"'" + re.escape(key) + r"': (True|False)", _nondefault(log_text))
    return (m.group(1) == "True") if m else None


def logged_indexer_fp8(log_text):
    t = (log_text or "")
    lines = [l for l in t.splitlines() if "non-default args:" in l or "Initializing a V1 LLM engine" in l]
    return any(re.search(r"indexer_kv_dtype['\"]?\s*[:=]\s*['\"]fp8", l) for l in lines)


def idx_readback(workers):
    """The QSA indexer compressed-key cache spec (nb1_cache_info kv_cache_specs) dtype per rank (nbs2_lib.idx_readback)."""
    seen = []
    for w in ((workers or {}).get("ranks") or []):
        if not isinstance(w, dict):
            continue
        for s in w.get("kv_cache_specs") or []:
            if isinstance(s, dict) and "compressed_key_cache" in str(s.get("example") or ""):
                seen.append({"rank": w.get("rank"), "dtype": str(s.get("dtype")), "page_size_bytes": s.get("page_size_bytes")})
    if not seen:
        return {"engaged": None, "specs": []}
    fp8 = [x for x in seen if "float8" in x["dtype"] or "fp8" in x["dtype"].lower()]
    return {"engaged": len(fp8) == len(seen), "specs": seen[:8]}


def readback(arm, log_text, env_raw, crb, workers, i8rb):
    """The lever's own change read back (real failure -> NOT_INTEGRATED, this arm only) and every OTHER lever's change
    absent. crb = nbdf_lib.config_readback, i8rb = the E28 I8 read-back with want = the arm's head."""
    k, why = arm["key"], []
    ch = arm.get("m1_change") or {"env": {}, "argv": [], "head": None}
    t = log_text or ""
    rec = {"arm": k}
    for key in M1_ENV_KEYS:                      # the env in all 4 workers (planned value) / absent everywhere
        vals = env_values(env_raw, key)
        rec[key] = vals
        want = (arm.get("env") or {}).get(key)
        if want is not None:
            if len(vals) != N_WORKERS or any(v != want for v in vals):
                why.append(f"{key} in the workers {vals}, expected {want!r} on {N_WORKERS}")
        elif any(v is not None for v in vals):
            why.append(f"{key} set in a worker of {k}: {vals}")
    vl = vocab_lines(t)
    rec["vocab_lines"] = vl[:4]
    if k in VOCAB_TABLES:
        n = VOCAB_TABLES[k][2]
        good = [x for x in vl if x["n"] == n and x["of"] == VOCAB_SIZE and x["file"] == arm["env"].get(ENV_VOCAB)]
        if len({x["tp_rank"] for x in good}) != TP:
            why.append(f"REDUCED DRAFT VOCAB ACTIVE lines with N={n} and the planned file on {TP} ranks: {vl[:4]}")
    elif vl:
        why.append(f"a reduced draft vocab on {k}: {vl[:2]}")
    lab = (i8rb or {}).get("label")
    rec["head_label"], rec["head"] = lab, arm.get("draft_head")
    want_lab = "I8_ENGAGED" if arm.get("draft_head") else "BF16_HEAD"
    if lab != want_lab:
        why.append(f"draft head read-back {lab} (want {want_lab} for head {arm.get('draft_head')!r})")
    heads = (i8rb or {}).get("draft_head_lines") or []
    if arm.get("draft_head") and len(heads) < TP:
        why.append(f"{len(heads)} draft head build lines (< {TP} PP1 ranks)")
    if k in VOCAB_TABLES and arm.get("draft_head") and heads and any(h.get("n") != VOCAB_TABLES[k][2] // TP for h in heads):
        why.append(f"the quantized head is not the reduced one: {heads[:2]}")
    ir = idx_readback(workers)
    rec["idx"] = ir
    rec["idx_logged_fp8"] = logged_indexer_fp8(t)
    if k == "IDX":
        if ir["engaged"] is False or (ir["engaged"] is None and not rec["idx_logged_fp8"]):
            why.append(f"indexer_kv_dtype fp8 not applied (worker specs {ir.get('specs')[:2]}, logged {rec['idx_logged_fp8']})")
    elif ir["engaged"] is True or rec["idx_logged_fp8"]:
        why.append("the fp8 indexer cache on a non-IDX arm")
    log_ = (crb or {}).get("log") or {}
    seqs = log_.get("max_num_seqs")
    rec["logged_max_num_seqs"] = seqs
    if seqs is not None and seqs != arm["expect_cfg"]["max_num_seqs"]:
        why.append(f"logged max_num_seqs {seqs} != {arm['expect_cfg']['max_num_seqs']}")
    if k in SEQ_ARMS and seqs != SEQ_ARMS[k]:
        why.append(f"logged max_num_seqs {seqs} != {SEQ_ARMS[k]}")
    mbt, lpt, lpa = (logged_int(t, "max_num_batched_tokens"), logged_int(t, "long_prefill_token_threshold"),
                     logged_bool(t, "long_prefill_token_threshold_adaptive"))
    rec.update(logged_max_num_batched_tokens=mbt, logged_long_prefill_token_threshold=lpt, logged_long_prefill_adaptive=lpa)
    if k == "LP":
        if (mbt, lpt, lpa) != (2048, 1024, True):
            why.append(f"logged batched tokens / long-prefill threshold / adaptive {(mbt, lpt, lpa)} != (2048, 1024, True)")
    elif mbt not in (None, 1024) or lpt not in (None, 0):
        why.append(f"long-prefill / batched-token change on {k}: {(mbt, lpt, lpa)}")
    sp = log_.get("speculative_config") if isinstance(log_.get("speculative_config"), dict) else None
    rec["logged_spec"] = sp
    want_k = arm["k"]
    if sp is not None and int(sp.get("num_speculative_tokens") or 0) != want_k:
        why.append(f"logged num_speculative_tokens {sp.get('num_speculative_tokens')} != {want_k}")
    if k in ("K4", "DYN") and sp is None:
        why.append("no logged speculative_config (K 4 not read back)")
    if k == "DYN" and sp is not None and sp.get("num_speculative_tokens_per_batch_size") in (None, []):
        why.append("the dynamic schedule is not in the logged speculative_config")
    if k != "DYN" and sp is not None and sp.get("num_speculative_tokens_per_batch_size"):
        why.append(f"a dynamic schedule on {k}")
    hits = [l.strip()[:200] for l in t.splitlines() if any(x in l for x in DYN_FAIL_RX)]
    rec["dyn_fallback_lines"] = hits[:4]
    if hits:
        why.append(f"dynamic speculative decoding fell back (FULL graphs lost / schedule disabled): {hits[:2]}")
    rec["why"] = "; ".join(why) or None
    rec["real_failure"] = bool(why)
    rec["ok"] = not why
    return rec


# ------------------------------------------------------------------ PROF (inside Q's boot)
def prof_record(q_rec, switch):
    """PROF's own record from Q's boot: DONE when both 30-s profiled windows kept trace files, else NOT_INTEGRATED."""
    sw = switch_map(switch) if not isinstance(switch, dict) else switch
    if not sw.get("PROF", True):
        return {"status": "SKIPPED_SWITCH_OFF", "why": "switched off in CELL 1 (NBM1_ON)"}
    if not sw.get("Q", True):
        return {"status": "SKIPPED_DEPENDENCY", "why": "PROF runs inside Q's boot; Q is switched off"}
    st = (q_rec or {}).get("steady") or {}
    keys = [f"profile_prof_{b}@{n}" for b, n in PROF_WINDOWS]
    wins = {kk: st.get(kk) for kk in keys}
    if not any(isinstance(w, dict) for w in wins.values()):
        return {"status": "NOT_RUN", "why": f"Q status {(q_rec or {}).get('status')}: no PROF window ran", "windows": {}}
    got = {kk: [f[0] for f in ((w or {}).get("trace_files") or [])] for kk, w in wins.items()}
    ok = all(got[kk] for kk in keys)
    return {"status": "DONE" if ok else "NOT_INTEGRATED", "not_integrated": not ok, "secs": PROF_SECS,
            "windows": {kk: {"status": (w or {}).get("status"), "trace_files": got[kk][:8], "trace_dir": (w or {}).get("trace_dir"),
                             "profile_request": (w or {}).get("profile_request")} for kk, w in wins.items()},
            "why": "ok" if ok else f"PROF windows without trace files: {[kk for kk in keys if not got[kk]]}"}


# ------------------------------------------------------------------ budget / copies
def copy_estimate(arms, switch):
    """The planned walk of one copy (setup, then every switched-on arm in order with est_min + 5 min left)."""
    sw = switch_map(switch)
    left = BUDGET_MIN - SETUP_MIN
    walk, runtime = [], SETUP_MIN
    for k, a in arms.items():
        if not sw.get(k, True):
            continue
        est = q_time(sw.get("PROF") and sw.get("Q"))[0] if k == "Q" else a["est_min"]
        fits = left >= est + START_RESERVE_MIN
        walk.append({"arm": k, "est_min": est, "left_min": round(left, 1), "runs": fits})
        if fits:
            left -= est
            runtime += est
    return {"walk": walk, "runtime_min": round(runtime, 1), "left_end_min": round(left, 1),
            "all_run": all(w["runs"] for w in walk), "budget_min": BUDGET_MIN}


# ------------------------------------------------------------------ records (record only)
def _valid(T, w):
    try:
        return bool(T.window_ok(w)[0])
    except Exception:
        return False


def band_table(T, Q, X, thr_floor=None, speed_min=0.02):
    """Per band: Q's and X's valid windows (any level), mean tok/s, gain, the threshold max(Q's 8k A/A spread, 2 %, the
    long floor). Recorded only (labels / verdicts never stop an arm)."""
    wq, wx = T._win(Q or {}) or {}, T._win(X or {}) or {}
    aa = None
    a8 = [wq.get(k) for k in ("8k@16#1", "8k@16#2")]
    if all(isinstance(w, dict) and _valid(T, w) for w in a8):
        aa = abs(a8[0]["tok_s"] - a8[1]["tok_s"]) / ((a8[0]["tok_s"] + a8[1]["tok_s"]) / 2)
    out = {"q_aa_spread_8k": None if aa is None else round(aa, 4), "bands": {}}
    for band in ("8k", "20k", "40k", "90k", "116k"):
        qv = [v["tok_s"] for kk, v in wq.items() if isinstance(v, dict) and kk.split("@")[0] == band and _valid(T, v)]
        xv = [v["tok_s"] for kk, v in wx.items() if isinstance(v, dict) and kk.split("@")[0] == band and _valid(T, v)]
        r = {"q_tok_s": [round(x, 1) for x in qv], "x_tok_s": [round(x, 1) for x in xv]}
        if qv and xv:
            qm, xm = sum(qv) / len(qv), sum(xv) / len(xv)
            thr = max(aa or 0.0, speed_min, float((thr_floor or {}).get(band) or 0.0))
            g = xm / qm - 1.0
            r.update(gain=round(g, 4), threshold=round(thr, 4), label="GAIN" if g > thr else ("REGRESSION" if g < -thr else "WITHIN_NOISE"))
        else:
            r["label"] = "INCONCLUSIVE"
        out["bands"][band] = r
    return out


def arm_records(T, rec, key, q_rec=None, thr_floor=None):
    rec = rec or {}
    acc = rec.get("oracle_acceptance") or {}
    return {"arm": key, "status": rec.get("status"), "al": acc.get("al"), "per_pos": acc.get("per_pos"),
            "windows": {k: {f: (w or {}).get(f) for f in ("status", "tok_s", "tick_ms", "mean_acceptance_length", "preemptions",
                                                          "running_eq_level", "level")}
                        for k, w in (rec.get("steady") or {}).items() if isinstance(w, dict)},
            "pool_blocks": (rec.get("workers") or {}).get("num_gpu_blocks"), "residents": rec.get("k100_residents"),
            "pin": rec.get("pin"), "tries": rec.get("tries"), "m1_readback": {k: (rec.get("m1_readback") or {}).get(k) for k in ("ok", "why")},
            "i8": (rec.get("i8_readback") or {}).get("label"), "tf_gate": (rec.get("tf_gate") or {}).get("status"),
            "oracle_vs": rec.get("oracle_vs"), "bands_vs_q": band_table(T, q_rec, rec, thr_floor) if key != "Q" and q_rec else None}
'''
if hashlib.sha256(NBM1_LIB_SRC.encode("utf-8")).hexdigest() != NBM1_CFG["lib_sha256"]:
    raise RuntimeError("NB-M1: the embedded nbm1_lib does not match its build sha")
import types as _nbm1_types
M1 = _nbm1_types.ModuleType("nbm1_lib")
exec(compile(NBM1_LIB_SRC, "nbm1_lib.py", "exec"), M1.__dict__)
NBM1L_OVL_SRC = r'''# E25 NB-M1L overlay (2026-11-02, user decision: the LIGHT build). exec'd into nbm1_lib's namespace right after nbm1_lib
# (notebook: embedded + sha-checked; builder / tests: the same). Only these changes:
#  1. Q (copies B / C): facts, the smoke / first request (no ladder), the prefill probe (the windows' admission rate),
#     warmup, its windows 8k@16 x2 / 20k@24 / 40k@16 / 90k@8 / 116k@6. NO ladder, oracle, oracle A/A, long oracle, prefix
#     test, TF, profiles. PROF (copy C) keeps its two 30-s windows inside Q's boot.
#  2. Levers: NO TF, NO oracle vs Q; facts, the smoke hard gate, the read-back of the one change, warmup, the windows.
#  3. Real-failure rules, the OOM ladder, NBM1_ON, the hard-only gates: unchanged; a label that needed a removed item is
#     INCONCLUSIVE / NOT_IN_PLAN (record only).
#  4. Time: body caps sized on the body's own due() rule (win_need); copies B (hosen42) / C (foysalemonshanto).
assert not globals().get("LIGHT"), "nbm1l overlay applied twice"
LIGHT = True
LIB_VERSION = LIB_VERSION + " + nbm1l-overlay 2026-11-02 r1 (LIGHT)"
COPIES = {"b": dict(COPIES["b"]), "c": dict(COPIES["c"])}
OWNERS = {"b": "hosen42", "c": "foysalemonshanto"}
TITLES = dict(TITLES, Q="Q = NBDF-K100 Q verbatim (MTP K3, I8, Q-defer, seqs 24, pin 7,933,526,016), LIGHT: facts + smoke + windows")
Q_ITEMS = {"ladder": False, "prefill_probe": True, "oracle": False, "oracle_aa": False, "long_oracle": False, "prefix": False,
           "profiles": False, "tf": False}
LEVER_ITEMS = dict(LEVER_ITEMS_OFF, oracle=False, oracle_aa=False, tf=False)
PLAN = {k: (v[0], v[1], False) for k, v in PLAN.items()}      # no TF on any lever (IDX / LP included)
BAND_TOK = {"8k": 8192, "20k": 20000, "40k": 40000, "90k": 90000, "116k": 116000}
WIN_SEC, ADMIT_TOK_S, LONG_RESET_SEC = 90.0, 3000.0, 60.0    # body: WIN, K100 prefill_admit_floor_tok_s, the long reset's +60 s
COPY_RUNTIME_MAX = 150      # build-time assertion: each light copy's planned runtime <= this
BUDGET_MIN = 155            # the light hard stop: max over B / C of (runtime, every arm's full cap) + 5 min, rounded up to 5
DEADMAN_MIN = BUDGET_MIN + 5


def _c(x):
    return int(math.ceil(round(x, 6)))


def win_need_sec(band, n, prof=False):
    """= the body's win_need() at the admission rate min(prefill, 3000 tok/s): a gated item starts only with this left."""
    if prof:
        return 360 + 10 * max(0, n - 8) + n * BAND_TOK[band] / ADMIT_TOK_S
    return WIN_SEC + 120 + n * BAND_TOK[band] / ADMIT_TOK_S


def body_plan(items):
    """[(name, min, need_sec | None)] in body order -> (body_min, cap_min, rows): the cap holds every gated item's due() need
    from its planned start, + 1 min."""
    t, need, rows = 0.0, 0.0, []
    for nm, m, nd in items:
        if nd is not None:
            need = max(need, t + nd / 60.0)
        rows.append([nm, m, None if nd is None else round(nd / 60.0, 2), round(t, 2)])
        t += m
    return round(t, 2), _c(max(t, need) + 1.0), rows


def body_items(windows, q, prof):
    it = [("facts", ITEM_MIN["facts"], None), ("smoke", ITEM_MIN["smoke"], 240)]
    if q:
        it.append(("prefill_probe", ITEM_MIN["prefill_probe"], 300))
    it.append(("warmup", ITEM_MIN["warmup"], 150))
    for b, n, tag in windows:
        nd = win_need_sec(b, int(n)) + (LONG_RESET_SEC if b in ("90k", "116k") else 0.0)
        it.append((f"{b}@{int(n)}{tag}", WINDOW_MIN[f"{b}@{int(n)}"], nd))
    if prof:
        for b, n in PROF_WINDOWS:
            it.append((f"prof_{b}@{n}", ITEM_MIN["prof"] / len(PROF_WINDOWS), win_need_sec(b, n, prof=True)))
    it.append(("end", ITEM_MIN["end"], None))
    return it


def q_time(prof_on):
    body, cap, rows = body_plan(body_items(Q_WINDOWS, True, prof_on))
    post = POST_MIN["q"] if prof_on else POST_MIN["lever"]
    return _c(READY_MIN["first"] + body + post), cap, {"ready": READY_MIN["first"], "body": body, "post": post, "items": rows, "light": True}


def lever_time(key):
    rc, wins, _ = PLAN[key]
    body, cap, rows = body_plan(body_items(wins, False, False))
    return (_c(READY_MIN[rc] + body + POST_MIN["lever"]), cap,
            {"ready": READY_MIN[rc], "ready_class": rc, "body": body, "post": POST_MIN["lever"], "items": rows, "light": True})


_m1_arms_full = m1_arms


def m1_arms(N, aq, switch=None, vocab=None):
    """nbm1_lib.m1_arms (Q + exactly one change per lever, unchanged) + the light item plans (Q_ITEMS / LEVER_ITEMS)."""
    arms = _m1_arms_full(N, aq, switch, vocab)
    arms["Q"].update(m1_items=dict(Q_ITEMS), m1_no_ladder=True)
    for k in LEVERS:
        arms[k]["m1_items"] = dict(LEVER_ITEMS)
    return arms


def copy_need_min(arms, switch):
    """The latest planned end of a copy with every arm's FULL body cap (start + ready + cap + post): the hard stop holds it
    + 5 min, so no switched-on arm's cap is clipped by the budget."""
    sw = switch_map(switch)
    t, end = SETUP_MIN, SETUP_MIN
    for k, a in arms.items():
        if not sw.get(k, True):
            continue
        est, cap, tb = q_time(sw.get("PROF") and sw.get("Q")) if k == "Q" else (a["est_min"], a["cap_min"], a["time_model"])
        end = max(end, t + tb["ready"] + cap + tb["post"])
        t += est
    return round(end, 1)
'''
if hashlib.sha256(NBM1L_OVL_SRC.encode("utf-8")).hexdigest() != NBM1_CFG["overlay_sha256"]:
    raise RuntimeError("NB-M1L: the embedded nbm1l overlay does not match its build sha")
exec(compile(NBM1L_OVL_SRC, "nbm1l_overlay.py", "exec"), M1.__dict__)   # E25 NB-M1L (light): item plans + time
_nbm1_on = globals().get("NBM1_ON")          # CELL 1 arm switches (E25 NB-M1); absent -> every arm on
M1_SWITCH = M1.switch_map(_nbm1_on)
M1_VOCAB = M1.resolve_vocab(INP, dict(os.environ))   # the E29 payload dataset (sha-pinned tables); missing -> V65 / V32 NOT_INTEGRATED
_nbm1_q0 = ARMS["Q"]                          # NBDF-K100's Q (m1_arms deep-copies it)
ARMS = M1.m1_arms(N, _nbm1_q0, M1_SWITCH, M1_VOCAB)
M1.install(N, ARMS)
M1.check_arms(N, T, ARMS, _nbm1_q0)          # raises: no boot
NBDF["nbm1"] = {"lib": M1.LIB_VERSION, "cfg": NBM1_CFG, "arms": {k: a["title"] for k, a in ARMS.items()},
                "switch": M1.switch_record(_nbm1_on), "vocab": M1_VOCAB, "walk": M1.copy_estimate(ARMS, M1_SWITCH),
                "time": {k: {"est_min": a["est_min"], "cap_min": a["cap_min"], "model": a.get("time_model")} for k, a in ARMS.items()}}
print("E25 NB-M1 arm switches:", M1_SWITCH, "| walk:", json.dumps(NBDF["nbm1"]["walk"])[:600], flush=True)
NBDF["nbm1"]["light"] = {"version": M1.LIB_VERSION, "q_items": M1.Q_ITEMS, "lever_items": M1.LEVER_ITEMS,
                         "need_min": M1.copy_need_min(ARMS, M1_SWITCH), "budget_min": M1.BUDGET_MIN}
ISO = {}                                    # "<K>_ISO": the Q-defer isolation rerun (built when a D arm fails)
_AKEYS = ("name", "port", "env", "change", "expect", "expect_cfg", "extra", "kv_bytes", "k", "spec", "captures", "coverage",
          "ring_needed", "ring_capacity_expected", "streams", "windows", "long_bands", "long_aa", "est_min", "cap_min", "title", "optional")

def _k100_next_pin(kvb):
    """The next lower rung of the K100 pin ladder (None below the last one)."""
    low = [x for x in [int(x) for x in K100_CFG["pin_ladder"]] if x < int(kvb)]
    return max(low) if low else None


def _k100_retry(arm, is_d, n_oom):
    """A's B3 (coordinator decision 4): the n-th launch OOM of an arm -> (pin, env added) or None. D: DFLASH_DRAFT_EMBED_HOST=1
    at the SAME pin first, then 7,665,090,560 -> 7,396,655,104 -> 6,859,784,192 (never the -1 GiB step first); Q: one rung."""
    if is_d:
        plan = K100_CFG["d_retry"]
        return (int(plan[n_oom - 1][0]), dict(plan[n_oom - 1][1])) if 1 <= n_oom <= len(plan) else None
    ql = K100_CFG["q_ladder"]
    return (int(ql[n_oom]), {}) if 1 <= n_oom < len(ql) else None


# ---- E27 K100 (coordinator 2026-11-01, final design): Q = E28 production (I8), Q's A/A, the TF envelope, deferred reruns
Q_I8 = {"want": bool(K100_CFG["q_i8"]), "on": bool(K100_CFG["q_i8"]), "dropped": None}
NB4T_BUILD_ENV = {}                         # per launch name: DRAFT_HEAD=int8 into Merlin's build() (Q only, while Q_I8 is on)
Q_AA = {}                                   # Q's own A/A: greedy oracle twice (cache reset between) + its ladder
Q_TF_AA = {}                                # Q's teacher-forced A/A per band (fresh vs repeat on the same boot)
PENDING = []                                # (kind, arm) reruns deferred to the end: nometa / iso / tap (D4s always runs first)
NBDF.update(k100_cfg=K100_CFG, q_i8=Q_I8, q_aa=Q_AA, q_tf_aa=Q_TF_AA, pending=PENDING)
if "20k" not in BANDS:                      # the production-level window 20k@24 (K3_ONLY baseline 562 tok/s)
    BANDS["20k"] = 20000
    PROMPTS["20k"] = [prompt(BANDS["20k"], v) for v in range(int(K100_CFG["seqs"]))]
    MAX_NEW["20k"] = MAX_LEN - BANDS["20k"] - 512
TF_P = {b: PROMPTS[b][5] for b in K100_CFG["tf_bands"]}     # teacher-forced 2k / 8k / 40k (RC2's TF variant 5)


def _k100_residents(o):
    """Capacity facts (coordinator decision 8): the boot's resident bound at 40k / 64k / 90k / 120k with the target block 4096
    (the MIXED path when the engine block is 1024), F at the boot's own block, the pool; Q's reference 29 / 21 / 17 / 13."""
    cap = o.get("capacity_after_ready") or {}
    pool = cap["num_gpu_blocks"][0] if len(cap.get("num_gpu_blocks") or []) == 1 else None
    blk = o.get("engine_block") or (((o.get("kv_layout") or {}).get("block_sizes") or {}).get("FullAttentionSpec") or [N.BLOCK_TOKENS])[0]
    out = {"pool": pool, "engine_block": blk, "mc_131072": cap.get("max_concurrency_131072"), "q_reference": {"40k": 29, "64k": 21, "90k": 17, "120k": 13}}
    for nm, tok, band in (("40k", 40000, "40k"), ("64k", 65536, "90k"), ("90k", 90000, "90k"), ("120k", 120000, "116k")):
        try:
            lv, inf = N.level_for_band(band, tok, pool, cap.get("max_concurrency_131072"), int(blk or N.BLOCK_TOKENS), o.get("kv_layout"),
                                       1e9, 1.0, seqs=64)
            out[nm] = {"level": lv, "F": inf.get("F"), "block": inf.get("block"), "source": inf.get("source"), "mixed": inf.get("mixed")}
        except Exception as e:
            out[nm] = {"error": f"{type(e).__name__}: {e}"[:200]}
    return out


def _k100_tap_arm(a):
    """The tap-mode diagnostic (research B): the D arm with DFLASH_TAP_MODE=<alt>, smoke + oracle + 8k only (end of run)."""
    return N.variant_arm(a, "_TAP", "_tap", 9, env_add={"DFLASH_TAP_MODE": K100_CFG["tap_mode_alt"]},
                         title_add=f"DFLASH_TAP_MODE={K100_CFG['tap_mode_alt']} (diagnostic smoke, budget permitting)",
                         short_only=True, est_min=int(K100_CFG["tap_est_min"]), cap_min=int(K100_CFG["tap_cap_min"]))


def _k100_run_pending(runner):
    """Coordinator decision 1: no rerun may starve D4s; the nometa / Q-defer-off / tap-mode reruns run after every arm, each
    only when its full minimum still fits (else SKIPPED_BUDGET, recorded)."""
    isos = 0
    for kind, k in list(PENDING):
        a = ARMS[k]
        if kind == "iso":
            if isos >= N.ISO_MAX:
                continue
            isos += 1
            na = N.iso_arm(a)
        elif kind == "nometa":
            na = N.nometa_arm(a)
        else:
            na = _k100_tap_arm(a)
        if na["key"] in ISO:
            continue
        ISO[na["key"]] = na
        NBDF.setdefault("variants", {})[na["key"]] = {x: na.get(x) for x in _AKEYS}
        if N.arm_fits(left(), na):
            step(na["key"], na["name"], na["est_min"], lambda kk=na["key"]: runner(kk))
        else:
            skip_arm(na["key"], f"left {left() / 60:.1f} min < {na['est_min']} + 5 min", "SKIPPED_BUDGET")
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
PIN = {"kv_bytes": B3_PIN, "source": "E27 K100: K100Q / E28 top K3 rung"}
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
    ok = pp == "FULLY_PATCHED" and (sm in (None, "FULLY_PATCHED")) and (arm.get("method") != "dflash" or sm == "FULLY_PATCHED")   # E25 NB-M1
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


def profile_window(port, arm, band="8k", n=None, secs=6.0):   # E25 NB-M1: secs (Q's own 6 s; PROF 30 s)
    """A profiled window (NB4T): lab profiled_load, bounded POST /start_profile, traces kept per arm and band."""
    n = n or arm["streams"]
    shutil.rmtree(PROF_DIR, ignore_errors=True)
    PROF_DIR.mkdir(parents=True, exist_ok=True)
    rec = L.profiled_load(port, PROMPTS[band], n, PROF_DIR, secs=float(secs), max_new=MAX_NEW[band], spec=True, prefill_tok_s=PREFILL["tok_s"])
    rec["profile_request"] = {"delay_iterations": 20, "max_iterations": 40, "secs": float(secs), "level": n, "band": band}
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
    for c in (() if arm.get("m1_no_ladder") else N.CONC_LADDER):   # E25 NB-M1: a lever skips the ladder (Q's A/A keeps it)
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


RPC_KEYS = ",".join(N.WATCHED) + ",NCCL_P2P_DISABLE,P2P," + ",".join(M1.M1_ENV_KEYS)   # E25 NB-M1: + the GDN / vocab keys


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
    if N.kv_bound(r) and n > 1 and (band in ("40k",) + N.LONG_BANDS or arm.get("m1_lever")) and due(win_need(band, n - 1)):   # E25 NB-M1: a lever steps down on any band
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
    return False                              # E27 K100: the memory gate is a flag (coordinator decision 4)


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
        is_lv = bool(arm.get("m1_lever"))           # E25 NB-M1: an MTP lever arm (Q + one change)
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
        if not is_d:                              # E25 NB-M1: every MTP arm (Q and the levers) -> the seam-inert check
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
        o["k100_residents"] = _k100_residents(o)              # E27 K100: residents at 40k / 64k / 90k / 120k (block 4096)
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
        if is_q:                                  # E28 K100 I8: INT8 requested -> I8_ENGAGED required, else the same pin + BF16 head
            o["i8_readback"] = (i8_readback(t, list(getattr(s, "argv", None) or []), arm.get("draft_head")) if not FAKE
                                else {"label": "FAKE_SERVER"})
            if arm.get("draft_head") == "int8" and o["i8_readback"].get("label") not in ("I8_ENGAGED", "FAKE_SERVER"):
                o.update(status="I8_READBACK_UNVERIFIED", early_stop="I8: " + json.dumps(o["i8_readback"], default=str)[:300])
                return o
        if is_lv and not FAKE:                    # E25 NB-M1: the lever's own change, read back (a real failure stops this arm only)
            o["i8_readback"] = i8_readback(t, list(getattr(s, "argv", None) or []), arm.get("draft_head"))
            _live(o, "m1_readback", M1.readback(arm, t, o.get("env_readback_raw"), o.get("config_readback"), o.get("workers"), o["i8_readback"]))
            if o["m1_readback"].get("real_failure"):
                o.update(status="NOT_INTEGRATED", not_integrated=True, early_stop="NB-M1 read-back: " + str(o["m1_readback"].get("why"))[:400])
                return o
        _live(o, "k100_gates", k100_boot_gates(p, s, arm, t) if not FAKE else {"real_failure": False, "fake": True})
        if o["k100_gates"].get("real_failure"):    # E27 K100: wrong quant kernels / FP8 KV without its static scales
            o.update(status="NO_GO_K100_BOOT_GATES", early_stop=str(o["k100_gates"].get("why"))[:400])
            return o
        if due(240):
            CUR["item"] = "smoke"
            _live(o, "smoke", smoke(p, arm))
            o["smoke_gate"] = N.smoke_gate(o["smoke"], is_q, q_nondet=Q_AA.get("clean") is not True)   # E27 K100: ladder vs Q's A/A
            o["smoke_pos0"] = N.pos0_label((o["smoke"] or {}).get("acceptance"))                        # STRUCTURAL / TAP_SUSPECT (recorded)
            _live(o, "free_after_smoke_min", free_min(p))
            mg = _mem_gate(p, o, "free_after_smoke_min")
            o["memory_gate_first_decode"] = {"ok": mg[0], "why": mg[1]}
            if (is_d or is_lv) and o["smoke_gate"].startswith("FAIL"):   # E25 NB-M1: levers too (FAIL_* = hard labels only)
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
        if due(300) and M1.item_on(arm, "prefill_probe"):   # E25 NB-M1: Q only (a lever reuses Q's PREFILL)
            CUR["item"] = "prefill_probe"
            _live(o, "prefill", prefill_probe(p))
        if M1.item_on(arm, "oracle"):           # E25 NB-M1L: the oracle only when the arm's plan has it (light: never)
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
            if is_q:                                  # E27 K100: Q's A/A = the greedy oracle again after a cache reset + Q's own ladder
                CUR["item"] = "oracle_aa"
                o["oracle_aa_reset"] = reset_prefix(p)
                _live(o, "oracle_aa", oracle(p, key=None, refs=("q",)))
                Q_AA.update(oracle=(o["oracle_aa"].get("vs_q") or {}).get("status"),
                            ladder=N.contamination_gate((o.get("smoke") or {}).get("ladder")))
                Q_AA["clean"] = N.q_aa_clean(Q_AA)
                o["q_aa"] = dict(Q_AA)
            elif isinstance(o["oracle"].get("vs_" + ref), dict):   # D: token ids judged against Q's A/A (k100_rel)
                _vs = o["oracle"]["vs_" + ref]
                _vs["status_raw"] = _vs.get("status")
                _vs["status"] = N.k100_rel(_vs.get("status"), Q_AA.get("clean") is True)
            o["oracle_vs"] = (o["oracle"].get("vs_" + ref) or {}).get("status")
            o["quality_gate"] = quality_gate(o, ref)
            if o["quality_gate"].startswith("FAIL"):              # oracle errors / garbage / FAIL_HIGH_MARGIN / probes / sampled
                o["load_tests_skipped"] = f"oracle gate vs {ref}: {o['quality_gate']}"
                o["status"] = "NO_GO_QUALITY"
                o["census_end"] = kv_census(s)
                return o
            if o["quality_gate"] != "PASS":
                o.setdefault("recorded_gates", []).append(f"oracle gate vs {ref}: {o['quality_gate']} (recorded, arm continues)")
        else:
            o["items_skipped"].append("oracle")
            o["oracle_vs"], o["quality_gate"] = None, "NOT_IN_PLAN_NB_M1L"   # recorded only (verdict: INCONCLUSIVE_ORACLE)
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
        if due(300) and not o.get("admission_deadlock") and M1.item_on(arm, "long_oracle"):   # E25 NB-M1: Q only
            CUR["item"] = "long_oracle"
            with _watched(o, p, "long_oracle", active=is_d):
                _live(o, "long_oracle", long_oracle(p, key="q_long" if is_q else None, ref=None if is_q else "q_long", k=arm["k"]))
            if not is_q and isinstance((o.get("long_oracle") or {}).get("vs_q_long"), dict):   # E27 K100: vs Q's A/A
                _lv = o["long_oracle"]["vs_q_long"]
                _lv["status_raw"] = _lv.get("status")
                _lv["status"] = N.k100_rel(_lv.get("status"), Q_AA.get("clean") is True)
            o["long_gate"] = N.long_oracle_gate(o, is_q)
            if o["long_gate"].startswith(("NO_GO", "FAIL")):
                o.update(status="NO_GO_LONG_ORACLE", load_tests_skipped=f"long oracle {o['long_gate']}", census_end=kv_census(s))
                return o
        else:
            o["items_skipped"].append("long_oracle")
            o["long_gate"] = ("INCONCLUSIVE_ADMISSION_DEADLOCK" if o.get("admission_deadlock") else
                              "NOT_IN_PLAN_NB_M1" if not M1.item_on(arm, "long_oracle") else "INCONCLUSIVE_LONG_ORACLE_SKIPPED_BUDGET")
        if due(240) and not o.get("admission_deadlock") and M1.item_on(arm, "prefix"):   # E25 NB-M1: Q only (record-only)
            CUR["item"] = "prefix_cache"
            try:
                with _watched(o, p, "prefix_cache", active=is_d):
                    _live(o, "prefix_cache", prefix_test(p, arm, block=o.get("engine_block") or N.BLOCK_TOKENS))
            except Exception as e:
                o["prefix_cache"] = {"error": f"{type(e).__name__}: {e}"[:300]}
            o["prefix_gate"] = N.prefix_gate(o["prefix_cache"])
        else:
            o["items_skipped"].append("prefix_cache")
            o["prefix_gate"] = "NOT_IN_PLAN_NB_M1" if not M1.item_on(arm, "prefix") else "INCONCLUSIVE_PREFIX_SKIPPED_BUDGET"
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
        long_ok = True                            # E27 K100: the fixed window list (no short exit)
        for band, lvw, tag in (arm.get("windows_fixed") or K100_CFG["windows"]):   # E25 NB-M1: a lever's A/A-free set (90k + 116k always)
            lvw = int(lvw)
            if band in ("40k",) + N.LONG_BANDS:
                lv, why = band_level(o, band)
                use = min(lvw, lv) if isinstance(lv, int) and lv > 0 else lvw
                o["long_levels"][band] = dict(why, level=use, fixed=lvw, resident=lv)
                if band in N.LONG_BANDS:
                    CUR["item"] = f"reset_{band}"
                    o["long_levels"][band]["reset"] = reset_prefix(p) if due(win_need(band, use) + 60) else "SKIPPED"
                lvw = use
            _pf = PREFILL.get("tok_s")
            PREFILL["tok_s"] = min(float(_pf or 3000.0), float(K100_CFG["prefill_admit_floor_tok_s"]))   # the admit limit scales up
            try:
                run_window(o, s, arm, band, lvw, tag)
            finally:
                PREFILL["tok_s"] = _pf
        lv116 = (o["long_levels"].get("116k") or {}).get("level") or 0
        for band, lvb in (((("8k", int(K100_CFG["profile_level"])),) + ((("116k", lv116),) if long_ok and lv116 > 0 else ()))
                          if M1.item_on(arm, "profiles") else ()):   # E25 NB-M1: Q's own profiles (verbatim); none on a lever
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
        o["measured_before_tf"] = True            # E27 K100: a server death in the TF below keeps the measured verdict (flag)
        if due(240) and s.alive() and M1.item_on(arm, "tf"):   # E25 NB-M1: Q; a lever only when its plan has TF (IDX / LP)
            CUR["item"] = "tf_k100"
            try:
                if is_q:
                    _live(o, "tf_ref", tforced(p, key="q_tf"))
                    _live(o, "tf_aa", tforced(p, ref="q_tf") if s.alive() else {"error": "server died"})
                    Q_TF_AA.clear()
                    Q_TF_AA.update({b_: v_ for b_, v_ in (o["tf_aa"] or {}).items() if isinstance(v_, dict)})
                else:
                    _live(o, "tf_vs_q", tforced(p, ref="q_tf"))
                    o["tf_gate"] = N.tf_gate(o["tf_vs_q"], Q_TF_AA, tuple(K100_CFG["tf_bands"]))
            except Exception as e:
                o["tf_error"] = f"{type(e).__name__}: {e}"[:300]
            o["tf_server_died"] = not s.alive()
        else:
            o["items_skipped"].append("tf_k100")
            if not M1.item_on(arm, "tf"):
                o["tf_plan"] = "NOT_IN_PLAN_NB_M1"
        if is_q and M1_SWITCH.get("PROF", True):   # E25 NB-M1 PROF: the stage-1 profiler on Q (no change), NB-S1's 30-s windows
            for band, lvb in M1.PROF_WINDOWS:
                pk = f"profile_prof_{band}@{lvb}"
                if due(win_need("profile_" + band, lvb)) and s.alive() and not o.get("admission_deadlock"):
                    CUR["item"] = pk
                    try:
                        o["steady"][pk] = profile_window(p, dict(arm, key="PROF"), band, lvb, secs=float(M1.PROF_SECS))
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
        if is_d:                                  # E25 NB-M1: every MTP arm -> the seam-inert check
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
    kvb = int(K100_CFG["q_pin"])              # E27 K100: every arm starts at 7,933,526,016 B
    port, tries, o = arm["port"], [], {}
    ref = K100_CFG["q_ref"] if arm["key"] == "Q" else "q"   # E27 K100
    max_tries = 1 + (len(K100_CFG["d_retry"]) if is_d else len(K100_CFG["q_ladder"]))   # E27 K100: D 4 OOM retries; Q 1 OOM + 1 I8 drop
    n_oom = 0
    while True:
        lname = arm["name"] if not tries else f"{arm['name']}_t{len(tries)}"
        tc = tree_check(arm)
        if not tc["ok"]:
            return {"status": "TREE_INVALID_MIXED", "tree_check": tc, "tries": tries, "launch_name": lname, "pin": kvb}
        if arm["key"] == "Q":                     # E27 K100: Q = E28 production, the INT8 draft head while Q_I8 is on
            arm["draft_head"] = "int8" if Q_I8["on"] else None
            arm["expect_cfg"]["draft_head"] = arm["draft_head"]
            NB4T_BUILD_ENV[lname] = {"DRAFT_HEAD": "int8"} if Q_I8["on"] else {}
        elif arm.get("m1_lever"):                 # E25 NB-M1: the lever's draft head (Q's int8 / BF16 per Q_I8; I4 int4)
            M1.apply_head(N, arm, Q_I8["on"])
            NB4T_BUILD_ENV[lname] = {"DRAFT_HEAD": arm["draft_head"]} if arm["draft_head"] else {}
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
        oom = bool(o["oom_in_log"]) and not o["xid_in_log"]   # E25 NB-M1 (NB-S1 GPU v1: OOM at the first request after ready) -> the ladder
        memstep = o.get("status") == "MEMORY_GATE_STEP_DOWN"
        tries.append({"launch_name": lname, "port": port, "kv_bytes": kvb, "status": o.get("status"), "oom": oom, "memory_step": memstep,
                      "device_fault": o["xid_in_log"], "ready_sec": o["ready_sec"]})
        i8_drop = arm["key"] == "Q" and Q_I8["on"] and (o.get("status") == "I8_READBACK_UNVERIFIED"
                                                        or (o.get("status") == "LAUNCH_FAILED" and not o["xid_in_log"]))
        if i8_drop:                               # E28 K100 I8: the SAME pin with the BF16 head (I8_UNSUPPORTED_ON_K100)
            Q_I8.update(on=False, dropped={"try": lname, "status": o.get("status"), "readback": (o.get("i8_readback") or {}).get("label")})
            tries[-1]["i8_drop"] = True
        n_oom += 1 if (oom and not i8_drop) else 0
        if not (oom or i8_drop) or o["xid_in_log"] or len(tries) >= max_tries or left() < NBDF_READY_SEC + arm["cap_min"] * 60:
            break
        if not i8_drop:                           # A's B3: D -> EMBED_HOST at the same pin, then 0.25-GiB rungs; Q -> one rung
            nxt = _k100_retry(arm, is_d, n_oom)
            if nxt is None:
                break
            kvb = nxt[0]
            if nxt[1]:
                arm["env"].update(nxt[1])
                arm["expect"] = {w: arm["env"].get(w, N.WATCHED_DEFAULT.get(w)) for w in N.WATCHED}
                arm.setdefault("retry_env", {}).update(nxt[1])
        port += 1
        log("boot", "I8 drop" if i8_drop else "OOM", lname, "-> KV_BYTES", kvb, "env", arm.get("retry_env"), "port", port)
    o["tries"] = tries
    if o.get("status") == "MEMORY_GATE_STEP_DOWN":
        o["status"] = "MEMORY_GATE_FAILED"
    if o.get("status") == "SERVER_DIED":
        o["death"] = ("BOOT_CAP_KILL" if o.get("boot_watchdog") else
                      f"BODY_OOM_{o.get('died_in')}" if o["oom_in_log"] and not o["xid_in_log"] else
                      f"DEVICE_FAULT_{o.get('died_in')}" if o["xid_in_log"] else f"SERVER_DIED_IN_{o.get('died_in')}")
        o["status"] = o["death"]
    o["capacity"] = T.capacity_facts(o)
    if not M1.item_on(arm, "oracle"):         # E25 NB-M1L: no oracle in the plan -> the hard-only evidence (probes, sampled, windows)
        return gated(o, need=(), expected_rows=None)
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
    o = arm_boot(arm, arm_body(arm, K100_CFG["q_ref"] if key == "Q" else "q", "q" if key == "Q" else None))   # E27 K100
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
    try:                                      # E25 NB-M1 records (raw numbers per arm, every lever vs Q per band; record only)
        v["nbm1_records"] = {k: M1.arm_records(T, _arm_rec(k), k, _arm_rec("Q"), N.LONG_THR_FLOOR) for k in list(ARMS) + list(ISO)}
    except Exception as e:
        v["nbm1_records"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    v["nbm1_prof"] = (R["sections"].get("PROF") or {}).get("nbm1_prof")
    v["nbm1_switch"] = M1_SWITCH
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
        ks = step("A0", "k100_setup", 3, k100_setup)          # E27 K100: overlay merge + K6, the QSA static scales
        if ks.get("status") != "DONE" or ks.get("ok") is not True:
            for k in ARMS:
                skip_arm(k, f"K100 setup: {ks.get('why') or ks.get('error') or ks.get('status')}", "NOT_INTEGRATED", not_integrated=True)
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
        if q_tree_ok and not M1_SWITCH.get("Q", True):   # E25 NB-M1: CELL 1 switch (Q off: the levers still run)
            skip_arm("Q", "switched off in CELL 1 (NBM1_ON)", "SKIPPED_SWITCH_OFF")
        q = (step("Q", ARMS["Q"]["name"], ARMS["Q"]["est_min"], lambda: runner("Q")) if q_tree_ok and M1_SWITCH.get("Q", True)
             else R["sections"]["Q"][ARMS["Q"]["name"]])
        R["sections"].setdefault("PROF", {})["nbm1_prof"] = M1.prof_record(q, M1_SWITCH)   # E25 NB-M1 PROF (inside Q's boot)
        save()
        bq = N.base_usable(q)                      # USER RULE 2026-11-01: only a REAL Q failure skips the D arms
        bs = N.base_check(q)
        NBDF["base"] = {"usable": bq[0], "label": bq[1], "why": bq[2], "strict_nb4t_base_check": {"ok": bs[0], "label": bs[1], "why": bs[2]}}
        save()
        if not bq[0]:                              # E27 K100: recorded only; the D arms run anyway (no cross-arm skips)
            NBDF["base_not_usable"] = f"{bq[1]}: {bq[2]}"
            log("NBDF K100: Q not a usable base (recorded, the D arms run):", NBDF["base_not_usable"])
        PIN.update(kv_bytes=q.get("pin") or PIN["kv_bytes"], source="Q's final pin")
        Q_REC.clear()
        Q_REC.update(q)
        D_PIN.update(kv_bytes=int(K100_CFG["q_pin"]), plan=None, source="E27 K100: every arm starts at 7,933,526,016 B")
        NBDF["d_pin"] = D_PIN
        for k in M1.LEVERS:                        # E25 NB-M1: every lever = Q + one change, in order (no cross-arm skip)
            a = ARMS[k]
            if not M1_SWITCH.get(k, True):          # CELL 1 arm switch off -> not booted (recorded, no other arm affected)
                skip_arm(k, "switched off in CELL 1 (NBM1_ON)", "SKIPPED_SWITCH_OFF")
                continue
            g_ = M1.pre_gate(a, KV_KIT_GATE)        # the real-kit gate (closed build: SKIPPED_KV_KIT_GATE) + the vocab table
            if g_:
                skip_arm(k, g_[1], g_[0], not_integrated=g_[0] == "NOT_INTEGRATED")
                continue
            if not N.arm_fits(left(), a):
                skip_arm(k, f"left {left() / 60:.1f} min < its full minimum {a['est_min']} + {N.START_RESERVE_SEC // 60} min "
                            "(priority = the NB-M1 order)", "SKIPPED_BUDGET")
                continue
            ok_p, why_p = pseq.before_arm(a)
            NBDF["patch"] = _patch_rec(pseq)
            if not ok_p:
                skip_arm(k, why_p, "TREE_INVALID_MIXED" if "INVALID" in why_p else "NOT_INTEGRATED", not_integrated=True)
                continue
            r = step(k, a["name"], a["est_min"], lambda k=k: runner(k))
            if N.needs_isolation(r) and not str(r.get("smoke_gate", "")).endswith("ZERO_ACCEPT"):
                PENDING.append(("iso", k))              # the K100 deferred Q-defer-off rerun of a died arm (budget permitting)
        _k100_run_pending(runner)
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
