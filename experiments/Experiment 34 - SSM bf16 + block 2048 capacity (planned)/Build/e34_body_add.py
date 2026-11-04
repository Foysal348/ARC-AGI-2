"""E34: the code blocks the E34 builder (build_e34.py) inserts into NB-D1's CELL 7 body (text constants only; nothing here
runs at import). Each block is placed by an exact-once anchor in build_e34.py."""

# ---- after NB-D1's setup (its D1.arm_plan + D1.check_arm_plan on NB-D1's own arms): E34's arms on NB-D1's B
E34_SETUP = r'''
# ==================================================================== E34: SSM bf16 + block 2048 (B / S16 / S16B2 / S16B2W on NB-D1's B)
E34_LIB_SHA = "__E34LIBSHA__"
E34_LIB_SRC = _nbd1_ds("e34_lib.py", E34_LIB_SHA)      # the payload dataset (sha-pinned)
if hashlib.sha256(E34_LIB_SRC.encode("utf-8")).hexdigest() != E34_LIB_SHA:
    raise RuntimeError("EMBED_SHA_MISMATCH: e34_lib")
(TMP / "e34_lib.py").write_text(E34_LIB_SRC, encoding="utf-8")
E34 = importlib.import_module("e34_lib")
E34_CFG = json.loads(r"""__E34CFG__""")
_e34_on = globals().get("E34_ON")            # CELL 1 arm switches (E34): absent -> E34.SWITCH_DEFAULT, a missing key -> off
NBD1_SWITCH = E34.switch_map(_e34_on)        # the name nbd1_main / prof_plan / the verdict read
ARMS = E34.arm_plan(D1, N, NBDF_ARMS["D3s"], seam_env_k4=SEAM_ENVS.get("D4s"), switch=NBD1_SWITCH, pin=int(K100_CFG["q_pin"]),
                    seqs=int(K100_CFG["seqs"]))
E34.check_arm_plan(D1, N, T, ARMS, NBDF_ARMS["D3s"], NBD1_SWITCH)   # B = NB-D1's B; each arm = B + its change (raises: no boot)
E34R = R.setdefault("e34", {})
E34R.update(lib=E34.LIB_VERSION, lib_sha256=E34_LIB_SHA, cfg=E34_CFG, switch=E34.switch_record(_e34_on),
            walk=E34.copy_walk(ARMS, NBD1_SWITCH, BUDGET / 60), need_min=E34.copy_need_min(ARMS, NBD1_SWITCH),
            never_do_lifted={"entry": E34.LIFTED, "arms": list(E34.SSM_ARMS), "nb4t_lib_value": T.NEVER_DO_ARG_VALUES.get(E34.SSM_FLAG)},
            cpu_planner=E34.CPU_PLANNER, w_pred=E34.W_PRED,
            question="does the GDN SSM state in bf16 (S16), with block 2048 + drafter SW 512 (S16B2) and more sequences (S16B2W), keep "
                     "B's accuracy (teacher-forced logprobs on 69k-113k recorded thinking traces, greedy oracle, long oracle) and add "
                     "resident streams / throughput at 8k-116k (raw numbers, labels record-only)?",
            arms={k: {x: a.get(x) for x in ("name", "port", "d1_change", "e34", "spec", "captures", "streams", "windows_fixed", "est_min",
                                             "cap_min", "time_model", "title", "d1_items", "env", "extra", "e34_smoke")} for k, a in ARMS.items()})
NBD1.update(switch=E34R["switch"], walk=E34R["walk"], need_min=E34R["need_min"], arms=E34R["arms"], e34=True)


def _e34_tf_long():
    """The recorded thinking traces (dataset foysalemonshanto/q27-e34-tf-long, sha-pinned); None when absent (recorded)."""
    for d in (INP / "datasets/foysalemonshanto/q27-e34-tf-long", INP / "q27-e34-tf-long", Path(os.environ.get("E34_TF_LONG_DIR") or "/nonexistent")):
        f = d / E34.TF_LONG_FILE
        if f.is_file() and hashlib.sha256(f.read_bytes()).hexdigest() == E34_CFG["tf_long_sha256"]:
            return json.loads(f.read_bytes().decode("utf-8"))
    return None


print("E34 arm switches:", NBD1_SWITCH, "| walk:", json.dumps(E34R["walk"])[:700], flush=True)
'''

# ---- after NB-D1's 20k prompt extension: distinct prompts for the max-resident windows + the TF set
E34_PROMPTS = r'''
for _b, _n in (("20k", E34.W_MAX), ("40k", E34.W_MAX), ("90k", 24), ("116k", 24)):   # E34: distinct prompts for the W windows
    if len(PROMPTS[_b]) < _n:
        PROMPTS[_b] = list(PROMPTS[_b]) + [prompt(BANDS[_b], v) for v in range(len(PROMPTS[_b]), _n)]
E34_TF_LONG = _e34_tf_long()
TF_P = E34.tf_set(PROMPTS, E34_TF_LONG, smoke=bool(NBD1_SWITCH.get("SMOKE")))   # E34: NBDF's TF code on this set
K100_CFG["tf_bands"] = list(TF_P)            # N.tf_gate reads the band names (recorded only)
E34R["tf"] = {"bands": {b: len(ids) for b, ids in TF_P.items()}, "long_dataset": E34_TF_LONG is not None,
              "long_traces": {k: {x: v.get(x) for x in ("task", "prompt_tokens", "p1_tokens", "len")}
                              for k, v in ((E34_TF_LONG or {}).get("traces") or {}).items()},
              "why_missing": None if E34_TF_LONG is not None else "TF_LONG_DATASET_ABSENT: dataset " + E34.TF_LONG_DS + " not mounted / sha mismatch"}
print("E34 TF set:", E34R["tf"]["bands"], "| long traces mounted:", E34R["tf"]["long_dataset"], flush=True)
'''

# ---- the E34 functions (before _gpu_util, after NB-D1's block)
E34_MAIN = r'''
# ==================================================================== E34: boot capacity / read-back records, [LIVE] lines, W re-plan, records
def e34_capacity_rec(o, arm, t):
    """Records only (never stops): the boot capacity (pool, block, F + token-equivalent, mc, KV tokens, group sizes, padding),
    the dtype / block / SW read-back per worker, printed explicitly ([E34 READBACK] + the [LIVE] boot line)."""
    try:
        o["e34_capacity"] = E34.capacity(o, t, o.get("workers"))
        rb = E34.readback(arm, t, o.get("workers"), o.get("dflash_readback"))
        o["e34_readback_boot"] = rb
        print("[E34 READBACK]", arm["key"], json.dumps({"cache_config.mamba_ssm_cache_dtype per rank": rb.get("ssm_dtype_per_rank"),
                                                       "cache_config.block_size per rank": rb.get("block_per_rank"),
                                                       "drafter SW block per rank": rb.get("sw_block_per_rank"),
                                                       "GDN state dtypes": [w.get("gdn_state_dtypes") for w in rb.get("workers") or []],
                                                       "GDN unpadded / padded page": [(w.get("gdn_unpadded_page"), w.get("gdn_page")) for w in rb.get("workers") or []],
                                                       "log": {k: v for k, v in (rb.get("log") or {}).items() if k != "lines"},
                                                       "unreadable": rb.get("unreadable"), "real_failure": rb.get("why")}, default=str)[:3000], flush=True)
        for ln in ((rb.get("log") or {}).get("lines") or [])[:12]:
            print("[E34 BOOTLOG]", arm["key"], ln, flush=True)
        if arm["key"] in E34.W_ARMS:
            o["e34_w"] = E34R.get("w")
        print(E34.boot_line((time.time() - T_START) / 60.0, arm["key"], o["e34_capacity"], rb), flush=True)
    except Exception as e:
        o["e34_capacity"] = {"error": f"{type(e).__name__}: {e}"[:300]}


def e34_item_live(o, key, val):
    """A [LIVE] line after every test (print only; never raises)."""
    try:
        if key in E34.LIVE_ITEMS:
            print(E34.item_line((time.time() - T_START) / 60.0, (o or {}).get("arm"), key, val), flush=True)
    except Exception as e:
        print(f"[LIVE] {key}: line error {type(e).__name__}: {e}"[:200], flush=True)


def e34_w_replan(k, a):
    """S16B2W's max_num_seqs from S16B2's measured boot (num_gpu_blocks + the max-concurrency line, the static admission bound
    at 40k, 24 .. 40); S16B2 not readable -> the CPU planner's W (recorded). The re-planned arm is checked (E34.check_arm_plan)
    before it replaces the planned one; a failed check keeps the planned arm (recorded)."""
    r = _arm_rec("S16B2") or {}
    w, info = E34.w_from_capacity(r.get("e34_capacity"))
    src = info.get("source")
    if w is None:
        w, src = int(a["streams"]), "CPU planner W (S16B2's boot capacity unreadable or S16B2 not run)"
    rec = dict(info, w=w, planned=int(a["streams"]), source=src)
    if w != int(a["streams"]):
        try:
            na = E34.lever_arm(D1, N, ARMS[D1.BASE_KEY], k, NBD1_SWITCH, w)
            tmp = dict(ARMS)
            tmp[k] = na
            E34.check_arm_plan(D1, N, T, tmp, NBDF_ARMS["D3s"], NBD1_SWITCH)
            ARMS[k] = na
            rec["replanned"] = True
        except Exception as e:
            rec.update(replanned=False, error=f"{type(e).__name__}: {e}"[:300], w=int(a["streams"]))
    E34R["w"] = rec
    NBD1.setdefault("arms", {})[k] = {x: ARMS[k].get(x) for x in ("name", "port", "streams", "captures", "windows_fixed", "est_min", "cap_min", "extra")}
    log("E34 W:", json.dumps(rec, default=str)[:600])
    print(f"[LIVE {(time.time() - T_START) / 60.0:.1f}] {k} W = {ARMS[k]['streams']} ({src}; S16B2 pool {info.get('pool')}, F {info.get('F')}, "
          f"block {info.get('attn_block')})", flush=True)
    save()
    return ARMS[k]


'''
