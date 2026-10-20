## Code cell 1

```diff
--- original
+++ exp7
@@ -37,6 +37,26 @@
 SAMPLING = {"temperature": float(os.environ.get("Q27X_TEMPERATURE", 0.6)), "top_p": 0.95, "top_k": 20}   # v1.4: russcore LB-10.14 sampling
 
 # ---- engine (PIV/V3.1 C1 settings; context, concurrency and MTP changed)
+# === EXPERIMENT 7: all experimental paths default OFF ===
+E7_ORDERING = os.environ.get("Q27X_E7_ORDERING", "original").strip().lower()  # original/codex
+E7_ADMISSION = os.environ.get("Q27X_E7_ADMISSION", "baseline").strip().lower()  # baseline/fixed/adaptive
+E7_EXTRA_CAP = int(os.environ.get("Q27X_E7_EXTRA_CAP", "16"))
+# Hard user rule: never silently downgrade 49,152-token thinking allowance.
+# A prompt that cannot fit full budget uses the existing valid input fallback.
+E7_STRICT_FULL_BUDGET = os.environ.get("Q27X_E7_STRICT_FULL_BUDGET", "1") == "1"
+E7_POLL_SEC = float(os.environ.get("Q27X_E7_POLL_SEC", "15"))
+E7_KV_SOFT = float(os.environ.get("Q27X_E7_KV_SOFT", "0.62"))  # provisional, not calibrated
+E7_KV_PROJECTED_MAX = float(os.environ.get("Q27X_E7_KV_PROJECTED_MAX", "0.78"))
+E7_KV_HARD = float(os.environ.get("Q27X_E7_KV_HARD", "0.90"))
+E7_COOLDOWN_SEC = float(os.environ.get("Q27X_E7_COOLDOWN_SEC", "120"))
+assert E7_ORDERING in ("original", "codex")
+assert E7_ADMISSION in ("baseline", "fixed", "adaptive")
+assert 16 <= E7_EXTRA_CAP <= 20
+assert 1 <= E7_POLL_SEC <= 60
+assert 0.20 < E7_KV_SOFT < E7_KV_PROJECTED_MAX < E7_KV_HARD < 1.0
+if E7_ADMISSION != "baseline":
+    assert E7_EXTRA_CAP > 16, "Experiment admission requires Q27X_E7_EXTRA_CAP=18 (or 20 after GPU gate)"
+# Frozen baseline is always reproducible by choosing original/baseline/max_num_seqs=16.
 MAX_NUM_SEQS = int(os.environ.get("Q27X_MAX_NUM_SEQS", 16))   # upper bound; actual slots = KV tokens // EXPECTED_LIVE_TOKENS_PER_SEQ
 MAX_NUM_BATCHED_TOKENS = 4096   # v1.3.5: = discussion max-prefill-tokens 4096; 32768 ran out of memory in run 2 (MTP drafter)
 GPU_MEMORY_UTILIZATION = float(os.environ.get("Q27X_GPU_MEM", 0.90))   # v1.4.1: kept at 0.90 so prefix caching is the only change
@@ -75,6 +95,7 @@
           "DRAIN_ANSWER_MAX_TOKENS", "MAX_MODEL_LEN", "PROMPT_TOKEN_LIMIT", "SAMPLING", "MAX_NUM_SEQS",
           "MAX_NUM_BATCHED_TOKENS", "GPU_MEMORY_UTILIZATION", "ENABLE_PREFIX_CACHING", "SPECULATIVE_CONFIG", "KV_CACHE_DTYPE", "VERIFY_TIMEOUT_SEC",
           "TOTAL_BUDGET_SEC", "MIN_THINK_BUDGET"]}
+CONFIG["experiment7"] = {k: globals()[k] for k in ("E7_ORDERING", "E7_ADMISSION", "E7_EXTRA_CAP", "E7_STRICT_FULL_BUDGET", "E7_POLL_SEC", "E7_KV_SOFT", "E7_KV_PROJECTED_MAX", "E7_KV_HARD", "E7_COOLDOWN_SEC")}
 CONFIG["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"]
 CONFIG["profile_row"] = P
 print(json.dumps(CONFIG, indent=1, default=str))

```
## Code cell 3

```diff
--- original
+++ exp7
@@ -33,12 +33,98 @@
                                          enable_thinking=True, reasoning_effort=REASONING_EFFORT)
 
 # a unit = one test input of one task; request ids, seeds, records and the scheduler all work on units
+
+# === EXPERIMENT 7A: EXACT CODEX PRIORITY FUNCTION (source preserved verbatim) ===
+import numpy as np
+def get_puzzle_priority(key, data):
+    train = data[key]['train']
+    n_train = len(train)
+
+    avg_input_size = np.mean([
+        np.prod(np.shape(x['input'])) for x in train
+    ])
+    size_ratios = [
+        max(np.prod(np.shape(x['output'])),
+            np.prod(np.shape(x['input']))) /
+        max(min(np.prod(np.shape(x['output'])),
+                np.prod(np.shape(x['input']))), 1)
+        for x in train
+    ]
+    max_ratio = max(size_ratios)
+    avg_ratio = np.mean(size_ratios)
+
+    n_colors = len(set(
+        int(c)
+        for x in train
+        for arr in [x['input'], x['output']]
+        for c in np.unique(arr).tolist()
+    ))
+    is_dynamic = any(
+        np.shape(x['input']) != np.shape(x['output'])
+        for x in train
+    )
+
+    if is_dynamic and max_ratio > 30:
+        tier = 3
+    elif is_dynamic and max_ratio > 5:
+        tier = 2
+    elif avg_input_size > 400:
+        tier = 2
+    else:
+        tier = 1
+
+    fine = (
+        avg_input_size / 900 * 0.30 +
+        (avg_ratio - 1) / 10 * 0.25 +
+        n_colors / 10 * 0.20 +
+        (0.25 if is_dynamic else 0.0)
+    )
+
+    if n_train >= 6 and tier < 3:
+        fine *= 0.7
+
+    return (tier, fine)
+
+# Sort on the ORIGINAL full JSON dictionary, then filter EVAL subset.
+# Python sort is stable: equal (tier, fine) preserves original file order.
+_E7_FULL_CHALLENGES = _ch if P["source"] == "eval" else test_challenges
+# OFF mode should not even calculate donor priority (historical baseline isolation).
+if E7_ORDERING == "codex":
+    _E7_FULL_SORTED = sorted(
+        _E7_FULL_CHALLENGES.keys(),
+        key=lambda k: get_puzzle_priority(k, _E7_FULL_CHALLENGES)
+    )
+    _E7_SELECTED_IDS = set(TASK_IDS)
+    E7_ORDERED_TASKS = [t for t in _E7_FULL_SORTED if t in _E7_SELECTED_IDS]
+    E7_PRIORITY = {t: get_puzzle_priority(t, _E7_FULL_CHALLENGES) for t in E7_ORDERED_TASKS}
+else:
+    E7_ORDERED_TASKS = list(TASK_IDS)
+    E7_PRIORITY = {}
+E7_ORDERED_UNITS = [
+    f"{tid}#{qi}"
+    for tid in E7_ORDERED_TASKS
+    for qi in range(len(CHALLENGES[tid]["test"]))
+]
+
+def e7_schedule_units():
+    return E7_ORDERED_UNITS if E7_ORDERING == "codex" else UNITS
+
 UNITS, UNIT_TASK, UNIT_Q = [], {}, {}
 for _tid in TASK_IDS:
     for _qi in range(len(CHALLENGES[_tid]["test"])):
         _uid = f"{_tid}#{_qi}"
         UNITS.append(_uid); UNIT_TASK[_uid] = _tid; UNIT_Q[_uid] = _qi
 UNIT_INDEX = {u: i for i, u in enumerate(UNITS)}             # v1.1 (F13): seeds do not depend on timing
+from collections import Counter as _E7Counter
+assert _E7Counter(E7_ORDERED_UNITS) == _E7Counter(UNITS), "Codex ordering must permute units"
+E7_ORDERING_REPORT = {
+    "selected": E7_ORDERING, "donor_function_sha256": "3f6536e80d445389da50ed61b3c4b80edc05ff25989aa1ecf7fdef0626196230",
+    "n_tasks": len(E7_ORDERED_TASKS), "n_units": len(E7_ORDERED_UNITS),
+    "tiers": {str(i): sum(p[0] == i for p in E7_PRIORITY.values()) for i in (1, 2, 3)},
+    "first_5": E7_ORDERED_TASKS[:5], "last_5": E7_ORDERED_TASKS[-5:],
+    "ordered_task_ids": E7_ORDERED_TASKS,
+}
+print("Experiment7 ordering:", E7_ORDERING_REPORT["selected"], E7_ORDERING_REPORT["tiers"])
 N_HIDDEN_UNITS = sum(len(t["test"]) for t in test_challenges.values())   # 259 on Kaggle; the projection is per query
 
 PROMPTS, PROMPT_IDS, SKIPPED, P1_CAPS = {}, {}, {}, {}
@@ -47,6 +133,10 @@
     text = render_prompt(_task["train"], _task["test"][UNIT_Q[uid]]["input"])
     ids = tokenizer.encode(text, add_special_tokens=False)
     cap = phase1_cap(len(ids))
+    if E7_STRICT_FULL_BUDGET and cap != THINK_BUDGET:
+        SKIPPED[uid] = {"prompt_tokens": len(ids), "reason": "experiment7_full_thinking_budget_required",
+                        "would_have_been_cap": int(cap)}
+        continue
     if cap < min(MIN_THINK_BUDGET, THINK_BUDGET):  # v1.2: only prompts that leave < 4k thinking tokens are skipped
         SKIPPED[uid] = {"prompt_tokens": len(ids), "reason": "insufficient_context"}; continue
     PROMPTS[uid] = text; PROMPT_IDS[uid] = ids; P1_CAPS[uid] = cap

```
## Code cell 5

```diff
--- original
+++ exp7
@@ -139,24 +139,33 @@
     _LADDER.append((KV_CACHE_DTYPE, SPECULATIVE_CONFIG, _first, False))
     _LADDER.append((KV_CACHE_DTYPE, SPECULATIVE_CONFIG, "TRITON_ATTN", False))
 _LADDER += [("auto", SPECULATIVE_CONFIG, "FLASH_ATTN", False), ("auto", None, "FLASH_ATTN", False)]
+# === EXPERIMENT 7B: engine capacity is independent of application admission ===
+# Try preferred recipe at requested capacity, then SAME recipe at proven cap 16,
+# before trying degraded cache/backend/speculative recipes.
+_E7_TRIALS = []
+for _idx, _recipe in enumerate(_LADDER):
+    _caps = [MAX_NUM_SEQS, 16] if _idx == 0 and MAX_NUM_SEQS > 16 else [min(16, MAX_NUM_SEQS)]
+    for _cap in _caps:
+        _E7_TRIALS.append((_cap, *_recipe))
 _seen = []
 llm = None
-for _kv, _spec, _backend, _pc in _LADDER:
-    if (_kv, json.dumps(_spec), _backend, _pc) in _seen:
+for _cap, _kv, _spec, _backend, _pc in _E7_TRIALS:
+    _key = (_cap, _kv, json.dumps(_spec), _backend, _pc)
+    if _key in _seen:
         continue
-    _seen.append((_kv, json.dumps(_spec), _backend, _pc))
+    _seen.append(_key)
     _t0 = time.perf_counter()
-    _kw = dict(ENGINE_CONFIG, attention_config={"backend": _backend}, enable_prefix_caching=bool(_pc))
+    _kw = dict(ENGINE_CONFIG, max_num_seqs=int(_cap), attention_config={"backend": _backend}, enable_prefix_caching=bool(_pc))
     if _kv != "auto": _kw["kv_cache_dtype"] = _kv
     if _spec is not None: _kw["speculative_config"] = dict(_spec)
-    _att = {"kv_cache_dtype": _kv, "speculative": _spec, "attention_backend": _backend, "prefix_caching": bool(_pc)}
+    _att = {"requested_engine_cap": int(_cap), "kv_cache_dtype": _kv, "speculative": _spec, "attention_backend": _backend, "prefix_caching": bool(_pc)}
     print(f"ENGINE ATTEMPT: kv_cache_dtype={_kv} speculative={_spec} attention_backend={_backend} prefix_caching={_pc}", flush=True)
     try:
         llm = LLM(**_kw)
         _att["smoke_tokens"] = _engine_smoke(llm)
         _att.update(ok=True, sec=round(time.perf_counter() - _t0, 1))
         ENGINE_INFO["attempts"].append(_att)
-        ENGINE_INFO.update(speculative=_spec, kv_cache_dtype=_kv, attention_backend=_backend, prefix_caching=bool(_pc))
+        ENGINE_INFO.update(selected_max_num_seqs=int(_cap), speculative=_spec, kv_cache_dtype=_kv, attention_backend=_backend, prefix_caching=bool(_pc))
         break
     except Exception as e:
         err = f"{type(e).__name__}: {e}"[:2000]
@@ -199,10 +208,13 @@
         ENGINE_INFO["attention_backend_resolved"] = str(getattr(engine.vllm_config.attention_config, "backend", None))
     except Exception as e:
         ENGINE_INFO["attention_backend_resolved"] = None; ENGINE_INFO["attention_backend_error"] = repr(e)[:300]
-    CONCURRENCY = MAX_NUM_SEQS
+    # In baseline mode this is identical to v1.4.4 when max_num_seqs=16.
+    _e7_real_cap = int(ENGINE_INFO["selected_max_num_seqs"])
+    CONCURRENCY = min(16, _e7_real_cap)
     if ENGINE_INFO["kv_tokens"]:
-        CONCURRENCY = max(4, min(MAX_NUM_SEQS, ENGINE_INFO["kv_tokens"] // EXPECTED_LIVE_TOKENS_PER_SEQ))
+        CONCURRENCY = max(4, min(CONCURRENCY, ENGINE_INFO["kv_tokens"] // EXPECTED_LIVE_TOKENS_PER_SEQ))
 ENGINE_INFO["concurrency"] = int(CONCURRENCY); ENGINE_INFO["vllm"] = vllm.__version__
+ENGINE_INFO["experiment7"] = {"application_baseline_cap": int(CONCURRENCY), "engine_cap": int(ENGINE_INFO.get("selected_max_num_seqs", 0)), "admission_mode": E7_ADMISSION}
 print("engine:", json.dumps(ENGINE_INFO, default=str)[:3000])
 
 def make_params(max_tokens, seed):

```
## Code cell 6

```diff
--- original
+++ exp7
@@ -1,4 +1,165 @@
 # ==================================================================== CELL 6 — continuous xhigh scheduler (vLLM step API), one unit per test input
+# Experiment 7 KVAdmission controller — stdlib only, embedded verbatim in notebook.
+import math
+from collections import deque
+
+KV_METRIC_NAMES = {
+    "kv": "vllm:kv_cache_usage_perc",
+    "running": "vllm:num_requests_running",
+    "waiting": "vllm:num_requests_waiting",
+    "preempt": "vllm:num_preemptions",
+}
+
+def e7_parse_metrics(rows):
+    """One-engine numeric metrics; reject duplicate/malformed/other-engine values."""
+    values = {}
+    try:
+        for row in rows:
+            name = row.get("name")
+            if name not in KV_METRIC_NAMES.values():
+                continue
+            labels = row.get("labels") or {}
+            if str(labels.get("engine", "0")) != "0":
+                return None
+            if name in values:
+                return None
+            val = float(row["value"])
+            if not math.isfinite(val):
+                return None
+            values[name] = val
+        if set(values) != set(KV_METRIC_NAMES.values()):
+            return None
+        kv, run, waiting, pre = (values[KV_METRIC_NAMES[k]] for k in ("kv", "running", "waiting", "preempt"))
+        if not (0 <= kv <= 1 and 0 <= run and run.is_integer() and
+                0 <= waiting and waiting.is_integer() and 0 <= pre and pre.is_integer()):
+            return None
+        return {"kv": kv, "running": int(run), "waiting": int(waiting), "preempt": int(pre)}
+    except (TypeError, ValueError, KeyError, AttributeError):
+        return None
+
+class E7KVAdmission:
+    """Decides admissions only; never modifies engine requests or their token limits."""
+    def __init__(self, *, mode, engine_cap, base_cap, extra_cap, kv_tokens,
+                 poll_sec=15., soft=0.62, projected=0.78, hard=0.90, cooldown_sec=120.):
+        self.mode = str(mode)
+        self.engine_cap = int(engine_cap)
+        self.base_cap = int(base_cap)
+        self.extra_cap = min(int(extra_cap), self.engine_cap)
+        self.kv_tokens = int(kv_tokens or 0)
+        self.poll_sec = float(poll_sec)
+        self.soft = float(soft)
+        self.projected = float(projected)
+        self.hard = float(hard)
+        self.cooldown_sec = float(cooldown_sec)
+        self.prev_sample = None
+        self.sample = None
+        self.sample_id = 0
+        self.extra_on_sample_id = -1
+        self.next_poll = 0.0
+        self.cooldown_until = 0.0
+        self.transition_sample_id = -1
+        self.transition_at = -float("inf")
+        self.growth = deque(maxlen=5)
+        self.events = []
+        self.counters = {"baseline":0,"fixed":0,"adaptive_extra":0,"blocked":0,"metric_invalid":0,"preempt_alerts":0}
+
+    def due(self, now):
+        return self.mode == "adaptive" and now >= self.next_poll
+
+    def observe(self, rows, now):
+        self.next_poll = now + self.poll_sec
+        val = e7_parse_metrics(rows)
+        if val is None:
+            self.sample = None
+            self.growth.clear()
+            self.counters["metric_invalid"] += 1
+            self.log(now,"metric_invalid")
+            return
+        val["t"] = float(now)
+        if self.sample is not None:
+            prev = self.sample
+            if val["preempt"] != prev["preempt"]:
+                self.cooldown_until = max(self.cooldown_until, now + self.cooldown_sec)
+                self.counters["preempt_alerts"] += 1
+                self.growth.clear()
+                self.log(now, "preemption_or_counter_reset", old=prev["preempt"], new=val["preempt"])
+            dt = now - prev["t"]
+            if 0 < dt <= max(3*self.poll_sec, 60.):
+                self.growth.append(max(0.0,(val["kv"]-prev["kv"])/dt))
+        self.prev_sample = self.sample
+        self.sample = val
+        self.sample_id += 1
+        self.log(now,"metric",**{k:val[k] for k in ("kv","running","waiting","preempt")})
+
+    def mark_phase_transition(self, now):
+        # One fresh completed snapshot, plus a short hold, needed before extra admission.
+        self.transition_at = float(now)
+        self.transition_sample_id = self.sample_id
+
+    def decide(self, now, host_active, next_prompt_tokens):
+        if host_active >= self.engine_cap:
+            return False,"engine_cap"
+        # The normal v1.4.4 scheduler fills to baseline. Only a genuine emergency
+        # snapshot can override it; when all requests are idle, eviction may be needed.
+        fresh = self.sample is not None and 0 <= now-self.sample["t"] <= max(2*self.poll_sec,30.0)
+        if host_active < self.base_cap:
+            if fresh and self.sample["kv"] >= self.hard and host_active > 0:
+                return False,"emergency_kv"
+            return True,"baseline"
+        if self.mode == "baseline":
+            return False,"baseline_cap"
+        if host_active >= self.extra_cap:
+            return False,"extra_cap"
+        if self.mode == "fixed":
+            return True,"fixed"
+        # Adaptive extra requests require valid, fresh, settled engine telemetry.
+        if not fresh:
+            return False,"stale_metrics"
+        m = self.sample
+        if m["running"] != host_active or m["waiting"] != 0:
+            return False,"engine_mismatch_or_waiting"
+        if now < self.cooldown_until:
+            return False,"preemption_cooldown"
+        if self.sample_id <= self.transition_sample_id or now-self.transition_at < self.poll_sec:
+            return False,"phase_transition"
+        if self.sample_id == self.extra_on_sample_id:
+            return False,"one_extra_per_snapshot"
+        if self.prev_sample is None or not self.growth:
+            return False,"growth_not_ready"
+        if self.kv_tokens <= 0:
+            return False,"kv_pool_unknown"
+        if m["kv"] >= self.soft:
+            return False,"kv_soft"
+        # PROVISIONAL pressure guard, not a physical-memory guarantee.
+        # 60s max recent growth plus prompt block reserve and 8%-pt margin.
+        slope=max(self.growth)
+        prompt_reserve=max(0.05, 1.5*int(next_prompt_tokens)/self.kv_tokens + 0.015)
+        estimate=m["kv"] + 60*slope + prompt_reserve + 0.08
+        if estimate > self.projected:
+            return False,"kv_projected"
+        return True,"adaptive_extra"
+
+    def record(self, now, uid, reason, accepted):
+        if accepted:
+            self.counters[reason] = self.counters.get(reason,0)+1
+            if reason == "adaptive_extra":
+                self.extra_on_sample_id = self.sample_id
+        else:
+            self.counters["blocked"] += 1
+        self.log(now,"admission" if accepted else "blocked",uid=uid,reason=reason)
+
+    def log(self, now, kind, **kw):
+        # Keep bounded event memory; report has aggregate counters too.
+        self.events.append({"t":round(float(now),3),"kind":kind,**kw})
+        if len(self.events)>10000:
+            self.events=self.events[-9000:]
+
+    def report(self):
+        return {"mode":self.mode,"engine_cap":self.engine_cap,"baseline_cap":self.base_cap,
+                "extra_cap":self.extra_cap,"kv_pool":self.kv_tokens,"last_sample":self.sample,
+                "counters":self.counters,"events":self.events,
+                "note":"Experimental KV guard, not a proof of memory safety"}
+
 RECORDS_PATH = OUT / "records.jsonl"
 STATE = {}          # uid -> dict
 FINAL = {}          # uid -> attempts for that test input
@@ -7,6 +168,32 @@
 GEN_TOKENS_TOTAL = 0
 SCHED = {"started": time.perf_counter(), "admitted": 0, "phase2_continue": 0, "phase2_forced": 0, "drained": 0,
          "aborted_at_hard_stop": 0, "not_admitted": 0, "max_active": 0, "steps": 0}
+# Engine recipe must match the preferred pinned configuration before any adaptive admission.
+_E7_ENGINE_CAP = int(ENGINE_INFO.get("selected_max_num_seqs", 0))
+_E7_PREFERRED = bool(
+    ENGINE_READY and ENGINE_INFO.get("kv_cache_dtype") == KV_CACHE_DTYPE
+    and ENGINE_INFO.get("speculative") == SPECULATIVE_CONFIG
+    and ENGINE_INFO.get("prefix_caching") == ENABLE_PREFIX_CACHING
+    and ENGINE_INFO.get("attention_backend") == "FLASHINFER"
+    and str(ENGINE_INFO.get("attention_backend_resolved", "")).upper() in ("FLASHINFER", "NONE")
+)
+_E7_ADMISSION_EFFECTIVE = E7_ADMISSION
+_E7_DISABLED_REASON = None
+if E7_ADMISSION == "adaptive" and not (_E7_PREFERRED and _E7_ENGINE_CAP > CONCURRENCY and ENGINE_INFO.get("kv_tokens")):
+    _E7_ADMISSION_EFFECTIVE = "baseline"
+    _E7_DISABLED_REASON = "preferred_engine_or_kv_unavailable"
+if E7_ADMISSION == "fixed" and _E7_ENGINE_CAP <= CONCURRENCY:
+    _E7_ADMISSION_EFFECTIVE = "baseline"
+    _E7_DISABLED_REASON = "engine_did_not_expand"
+KV_GATE = E7KVAdmission(
+    mode=_E7_ADMISSION_EFFECTIVE, engine_cap=_E7_ENGINE_CAP,
+    base_cap=CONCURRENCY, extra_cap=E7_EXTRA_CAP,
+    kv_tokens=ENGINE_INFO.get("kv_tokens"), poll_sec=E7_POLL_SEC,
+    soft=E7_KV_SOFT, projected=E7_KV_PROJECTED_MAX,
+    hard=E7_KV_HARD, cooldown_sec=E7_COOLDOWN_SEC)
+SCHED["experiment7"] = {"ordering":E7_ORDERING,"requested_admission":E7_ADMISSION,
+                          "effective_admission":_E7_ADMISSION_EFFECTIVE,
+                          "disabled_reason":_E7_DISABLED_REASON}
 
 def write_submission():
     sub = identity_submission(test_challenges)
@@ -24,25 +211,43 @@
 def admit(uid):
     st = {"uid": uid, "phase": 1, "prompt_ids": PROMPT_IDS[uid], "p1_ids": [], "p2_ids": [], "p2_prefix": [],
           "p1_finish": None, "p2_finish": None, "t_admit": time.perf_counter(), "t_done": None, "mode": None}
+    rid = _req(uid, 1)
+    try:
+        engine.add_request(rid, {"prompt_token_ids": list(st["prompt_ids"])},
+                           make_params(P1_CAPS[uid], SEED + 10 * UNIT_INDEX[uid] + 1))
+    except Exception:
+        # Unknown partial engine registration -> best-effort cleanup; scheduler fails safely.
+        try: engine.abort_request([rid])
+        except Exception: pass
+        raise
     STATE[uid] = st
-    engine.add_request(_req(uid, 1), {"prompt_token_ids": list(st["prompt_ids"])},
-                       make_params(P1_CAPS[uid], SEED + 10 * UNIT_INDEX[uid] + 1))
     SCHED["admitted"] += 1
 
 def start_phase2(st, answer_cap):
     """think closed but answer cut -> continue; think never closed -> append stop text (token-exact) and answer"""
     closed = THINK_CLOSE_ID in st["p1_ids"]
-    if not closed:                                  # v1.1 (F8): an EOS before </think> must not precede the stop text
-        while st["p1_ids"] and st["p1_ids"][-1] in EOS_IDS:
-            st["p1_ids"].pop()
-    prefix = list(st["prompt_ids"]) + list(st["p1_ids"]) + ([] if closed else list(STOP_IDS))
+    p1_clean = list(st["p1_ids"])
+    if not closed:  # Do not mutate original stream before successful engine admission.
+        while p1_clean and p1_clean[-1] in EOS_IDS:
+            p1_clean.pop()
+    prefix = list(st["prompt_ids"]) + p1_clean + ([] if closed else list(STOP_IDS))
     room = MAX_MODEL_LEN - len(prefix) - 8
     cap = min(int(answer_cap), room)
     if cap < 64:
         return False
+    rid = _req(st["uid"], 2)
+    try:
+        engine.add_request(rid, {"prompt_token_ids": prefix},
+                           make_params(cap, SEED + 10 * UNIT_INDEX[st["uid"]] + 2))
+    except Exception:
+        try: engine.abort_request([rid])
+        except Exception: pass
+        raise
+    st["p1_ids"] = p1_clean
     st["phase"] = 2; st["p2_prefix"] = [] if closed else list(STOP_IDS); st["mode"] = "continue" if closed else "forced_close"
+    st["p2_cap"] = cap
     SCHED["phase2_continue" if closed else "phase2_forced"] += 1
-    engine.add_request(_req(st["uid"], 2), {"prompt_token_ids": prefix}, make_params(cap, SEED + 10 * UNIT_INDEX[st["uid"]] + 2))
+    KV_GATE.mark_phase_transition(time.perf_counter())
     return True
 
 def answer_text(st):
@@ -91,7 +296,7 @@
 
 def run_scheduler():
     global GEN_TOKENS_TOTAL
-    queue = [u for u in UNITS if u in PROMPT_IDS]
+    queue = [u for u in e7_schedule_units() if u in PROMPT_IDS]
     last_log = time.perf_counter(); drained = False
     while True:
         now = time.perf_counter()
@@ -110,8 +315,27 @@
                 SCHED["drained"] += 1; s["p1_finish"] = "drained"
                 if not start_phase2(s, DRAIN_ANSWER_MAX_TOKENS): finish(s, "drain_no_room")
             active = [s for s in STATE.values() if s["phase"] in (1, 2)]
-        while queue and len(active) < CONCURRENCY and now < ADMIT_DEADLINE and not drained:
-            admit(queue.pop(0)); active = [s for s in STATE.values() if s["phase"] in (1, 2)]
+        # Poll only in adaptive mode; sampling has a measured CPU cost.
+        if KV_GATE.due(now):
+            KV_GATE.observe(metrics_snapshot(), time.perf_counter())
+        while queue and now < ADMIT_DEADLINE and not drained:
+            now = time.perf_counter()
+            uid = queue[0]
+            allowed, reason = KV_GATE.decide(now, len(active), len(PROMPT_IDS[uid]))
+            if not allowed:
+                KV_GATE.record(now, uid, reason, False)
+                break
+            # Queue pop only AFTER successful registration; exceptions keep fallback valid.
+            try:
+                admit(uid)
+            except Exception:
+                SCHED["not_admitted"] = len(queue)
+                raise
+            queue.pop(0)
+            KV_GATE.record(time.perf_counter(), uid, reason, True)
+            active = [s for s in STATE.values() if s["phase"] in (1, 2)]
+            if reason == "adaptive_extra":
+                break  # never add two extras using one metrics sample
         SCHED["max_active"] = max(SCHED["max_active"], len(active))
         if not active:
             if not queue or now >= ADMIT_DEADLINE or drained:
@@ -141,6 +365,7 @@
             print(f"  .. {el / 60:6.1f} min | active {len(active)} | queued {len(queue)} | done {len(FINAL)} | "
                   f"gen {GEN_TOKENS_TOTAL} tok | {GEN_TOKENS_TOTAL / max(el, 1e-6):.0f} tok/s", flush=True)
             METRICS_SNAPSHOTS.append({"t_min": round(el / 60, 1), "active": len(active), "metrics": metrics_snapshot()})
+    SCHED["kv_controller_counters"] = KV_GATE.counters
     SCHED["not_admitted"] = len(queue)
     SCHED["wall_sec"] = round(time.perf_counter() - SCHED["started"], 1)
     SCHED["gen_tokens"] = GEN_TOKENS_TOTAL
@@ -164,6 +389,7 @@
     except BaseException:
         pass
 finally:
+    SCHED.setdefault("kv_controller_counters", KV_GATE.counters)
     SCHED.setdefault("wall_sec", round(time.perf_counter() - SCHED["started"], 1))
     SCHED.setdefault("gen_tokens", GEN_TOKENS_TOTAL)
     SCHED.setdefault("tok_per_sec", round(GEN_TOKENS_TOTAL / max(SCHED["wall_sec"], 1e-6), 1))
@@ -174,5 +400,10 @@
         write_json_atomic(OUT / "vllm_metrics.json", METRICS_SNAPSHOTS)
     except Exception:
         pass
+    try:
+        write_json_atomic(OUT / "experiment7_kv_report.json", KV_GATE.report())
+        write_json_atomic(OUT / "experiment7_ordering.json", E7_ORDERING_REPORT)
+    except Exception as _e:
+        print("Experiment7 diagnostic save error:", repr(_e)[:200], flush=True)
     write_submission()
 print("scheduler done:", json.dumps({k: v for k, v in SCHED.items() if k != "error"}), "| error:", bool(SCHED_ERROR))

```
