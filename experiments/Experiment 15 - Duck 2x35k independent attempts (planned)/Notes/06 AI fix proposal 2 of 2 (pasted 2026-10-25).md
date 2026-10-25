# 06 — AI fix proposal 2 of 2 for E15B (pasted by the user, 2026-10-25), verbatim

The user: "these are the two AIs' fixes, not three; now say finally what you want to take." Comparison: Notes/07.

---

ARC-AGI-2 — Config B: সমস্যা ১–৫৫, সমাধান ও কোড বাস্তবায়ন
ভিত্তি: 02 — Prioritized fix list for config B, rev 2b (২৪ অক্টোবর ২০২৬), আগের ৩৫টি fix-এর তালিকা এবং পূর্ববর্তী E14S/EVAL120 source।

আপনার চূড়ান্ত পরিকল্পনা অনুযায়ী:
- প্রতিটি task-এ দুটি স্বাধীন Duck attempt, সর্বোচ্চ 35,000 generated tokens করে।
- সব 240 task-এর প্রথম attempt আগে, তারপর দ্বিতীয় attempt।
- xhigh reasoning; পুরোনো 28k pause/resume বাদ।
- DFlash2 width 6, SGLang 0.5.21, 4×L4।
- 16 streams পরীক্ষা, ব্যর্থ হলে 14, তারপর 12-তে fallback।
- Turn rollback, submit dispatch, context trimming ও deadline problems সমাধান।
- পুরোনো working notebooks এবং LB 34.58 fallback অক্ষত।
আমি পুরোনো ZIP থেকে বাস্তব arc2_duck.py, tool_agent.py, python_tool_sandbox.py ও E14S notebook-cell source-ও দেখেছি। নিচের কোডগুলো তাই সংশ্লিষ্ট function ও পুরোনো logic ধরে তৈরি। তবে এগুলো প্রস্তাবিত integration patches, নতুন E15B notebook-এ প্রয়োগ করে tests পাস করেছে—এমন দাবি নয়।
ধাপ ০ — পুরোনো working version নিরাপদ রাখা
CPU development: GO
প্রথমে নতুন build path খুলুন। পুরোনো build_e14.py বা pushed notebook overwrite করবেন না।
Experiment_15/
├── build/
│   └── build_e15b.py
├── src/
│   └── e15b/
│       ├── config.py
│       ├── scheduling.py
│       ├── safety.py
│       ├── telemetry.py
│       └── submission.py
├── tests/
├── notebooks/
├── reports/
└── README.md


প্রথমে build_e15b.py হবে পুরোনো builder-এর copy। E15B-specific সব পরিবর্তন feature flag-এ থাকবে।

src/e15b/config.py:
from dataclasses import dataclass@dataclass(frozen=True)class BConfig:    enabled: bool = True    attempts: int = 2    tokens_per_attempt: int = 35_000    commit_reserve: int = 3_000    duck_context_window: int = 100_000    min_admit_sec: int = 300    request_timeout_sec: int = 3_000    workers: int = 16    server_max_running: int = 17    mamba_cache_slots: int = 68    reasoning_effort: str = "xhigh"    e16_enabled: bool = False    allow_resume: bool = False    fair_share: bool = False    kv_admit_limit: float = 0.95    def validate(self):        assert self.attempts == 2        assert self.tokens_per_attempt == 35_000        assert 0 < self.commit_reserve < self.tokens_per_attempt        assert self.server_max_running >= self.workers        assert self.mamba_cache_slots >= 4 * self.server_max_running        assert self.duck_context_window < 155_648        assert not self.allow_resume        assert not self.fair_share



এই config এক জায়গায় রাখলে notebook, scheduler, tests এবং report-এ ভিন্ন সংখ্যা চলে যাওয়ার ঝুঁকি কমবে।

Test:
def test_b_config():    cfg = BConfig()    cfg.validate()    assert cfg.workers == 16    assert cfg.server_max_running == 17    assert cfg.tokens_per_attempt == 35000



ধাপ ১ — Fix #1, #2, #3, #45: স্বাধীন attempt, stage ordering ও resume বাদ
এখানে চারটি সংশোধন একই সঙ্গে করতে হবে।
১.১ Fix #1 — প্রতিটি attempt-এর নিজস্ব agent
File: arc2_duck.py, BranchSession ও solve_all_r6()।

বর্তমান source-এ প্রতিটি (task_id, branch)-এর জন্য আলাদা BranchSession তৈরি হয়। এটি ভালো; নতুন scheduler-এ সেই isolation বজায় রাখতে হবে।

BranchSession._ensure()-এ নিচের invariant রাখুন:
def attempt_key(task_id: str, attempt_no: int):    assert attempt_no in (0, 1)    return (task_id, attempt_no)# Each attempt must construct its own BranchSession.# Never set attempt_2.agent = attempt_1.agent.def assert_independent(a0, a1):    assert a0 is not a1    assert a0.agent is not a1.agent    assert a0.run_dir != a1.run_dir    assert a0.view == "facts"    assert a1.view == "facts"



Seed-এর বিষয়ে একটি সংশোধন: পুরোনো code-এ LOCAL_ANALYZER_SEED=-1 হলে request-এ seed পাঠানো হয় না। দুটি fresh request তবুও stochastic হতে পারে। তাই আলাদা seed পাঠানো independence-এর জন্য অপরিহার্য নয়; তবে reproducible paired experiments-এর জন্য ভালো।

Stable, task-specific seed:
import hashlibdef seed_for_attempt(base_seed, task_id, attempt_no):    raw = f"{base_seed}:{task_id}:{attempt_no}".encode()    seed = int.from_bytes(        hashlib.sha256(raw).digest()[:4], "big"    ) & 0x7fffffff    return seed



tool_agent.py-এর _chat_completion()-এ বর্তমান:
seed=_LOCAL_ANALYZER_SEED,



প্রতিস্থাপন করুন:
seed=getattr(    self, "_arc2_sampling_seed",    _LOCAL_ANALYZER_SEED),



নতুন agent তৈরির পরে:
agent._arc2_sampling_seed = seed_for_attempt(    base_seed=20261024,    task_id=self.task_id,    attempt_no=self.branch,)



এটি _ensure()-এ agent তৈরির সঙ্গে বসবে; self.agent = ... হওয়ার পরে, কোনো প্রথম request পাঠানোর আগে।

Tests: একই task-এর attempt 0/1 seed আলাদা, কিন্তু একই experiment rerun-এ deterministic। দুটি attempt-এর agent/history/run directory আলাদা।
১.২ Fix #2 — সব attempt 1 শেষ হওয়ার আগে attempt 2 নয়
বর্তমান source-এ stage1_left আসলে কতটি task শুরু হওয়া বাকি তা গণনা করে। এটি কতটি শেষ হয়নি, তা নয়।

solve_all_r6()-এর state-এ যুক্ত করুন:
state["stage1_finished"] = 0state["stage1_completed_ids"] = set()



বর্তমান stage-2 gate:
if prio[0] == 1 and state["stage1_left"] > 0:



B mode-এ প্রতিস্থাপন:
if prio[0] == 1 and b_mode:    if state["stage1_finished"] < len(order):        push(prio, tid, b, kind)        cond.wait(timeout=2.0)        continue



একটি attempt 0 terminal হওয়ার পরে, worker-এর with cond: completion section-এ:
if b_mode and b == 0 and s.status == "done":    if tid not in state["stage1_completed_ids"]:        state["stage1_completed_ids"].add(tid)        state["stage1_finished"] += 1



stage1_finished update এবং stage2_items() একই condition lock-এর মধ্যে থাকবে।

Test: শেষ attempt 1 artificial delay-তে আটকে থাকলে attempt 2 শুরু হবে না। শেষটি terminal হওয়ার পরেই দ্বিতীয় phase খুলবে।
১.৩ Fix #3 ও #45 — আর কোনো pause/resume নয়
বর্তমান source-এ branch 1-এর জন্য এখনও fair-share chunk তৈরি হয়। শুধু first_exposure_tokens=None দিলে সেটা বন্ধ হয় না।

solve_all_r6()-এ নতুন argument যোগ করুন:
b_mode: bool = False,



Worker-এর pause selection:
if b_mode:    pause = Noneelif kind == "start1" and b == 0:    pause = first_exposure_tokens or Noneelse:    with cond:        sh = share_tokens()    cur = s.agent.generated_tokens if s.agent else 0    cap = max_generated_tokens or 0    pause = (        cur + sh        if sh and (not cap or cur + sh < cap)        else None    )s.run(pause_at_tokens=pause, tag=tag)



কিন্তু এটুকু যথেষ্ট নয়। stage2_items() এখনও resume0 এবং resume1 তৈরি করতে পারে।

B mode-এর জন্য শুরুতেই return করা আলাদা scheduling branch বসান:
def stage2_items(tid):    if b_mode:        s0 = sessions.get((tid, 0))        if s0 is None or s0.status != "done":            return        if branches != 2:            return        if (tid, 1) in sessions or (tid, 1) in queued:            return        # D2: accepted tasks become optional, lowest priority.        cls = stage2_class([s0])        push(rank(tid, 1, 0), tid, 1, "start1")        return    # Existing rev6 stage2_items implementation stays here.    ...



এখানে ...-এর জায়গায় বর্তমান legacy implementation অপরিবর্তিত রাখতে হবে; এটি পুরো function replacement নয়।

বাধ্যতামূলক test:
assert all(    event["kind"] not in {"resume0", "resume1", "resume"}    for event in b_scheduler_events)assert max(attempts_started_per_task.values()) <= 2assert all(    r["runs"][0]["pause_at"] is None    for r in completed_b_records    if r.get("runs"))



এগুলো PASS না করলে B experiment GO নয়।
ধাপ ২ — Fix #4, #5, #46, #47, #48: token budget, context ও final commit
২.১ Fix #4 — পুরোনো request caps বাদ
বর্তমান _arc2_apply_cap() 12,288 / 24,576 / 32,768 cap ব্যবহার করে।

B mode-এ request cap হবে remaining attempt budget। তবে Rev 2b-এর #48 অনুসারে শেষ 3,000 tokens একটি commit request-এর জন্য রাখা হবে।

arc2_duck.py, _arc2_apply_cap()-এ B branch যোগ করুন:
def _arc2_apply_cap(self) -> int:    if getattr(self, "_arc2_mode_b", False):        budget = 35_000        reserve = 3_000        used = int(self.generated_tokens)        remaining = max(0, budget - used)        if not getattr(self, "_arc2_commit_started", False):            cap = max(1, remaining - reserve)        else:            cap = max(1, remaining)        # The context accounting must use the same cap.        self._max_output_tokens = cap        self._reply_reserve_tokens = cap        self._context_budget_tokens = max(            1024,            self._arc2_window            - cap            - self._request_safety_margin_tokens,        )        return cap    # Existing rev6 cap selection stays unchanged.    ...



গুরুত্বপূর্ণ: generated_tokens-এ provider usage-এর accepted generated tokens গণনা হচ্ছে কি না যাচাই করতে হবে। Server এক request-এ cap slightly overshoot বা final tool response দিতে পারে; তাই host-level clamp রাখা প্রয়োজন।

আর remaining == 0 হলে নতুন model request পাঠানো যাবে না। সেই guard analyze()-এর আগে রাখতে হবে।
২.২ Fix #48 — 32k শেষে একটি final commit turn
মূল ঝুঁকি হলো, 35k-এ request শেষ হয়ে গেল কিন্তু model তখনও <think>-এর ভেতরে; কোনো executable transform() লিখল না।

B-এর জন্য একবারের commit state:
def b_commit_needed(agent, budget=35000, reserve=3000):    used = int(agent.generated_tokens)    return (        not getattr(agent, "_arc2_commit_started", False)        and used >= budget - reserve        and used < budget        and agent.arc2_accepted is None    )B_COMMIT_PROMPT = (    "The attempt is almost out of tokens. "    "Do not restart the analysis. Using your best current "    "hypothesis, call the `python` tool now: define a pure "    "`transform(grid)` and execute `check(transform)`. "    "If a retained function already exists, check it. "    "Keep the response concise.")



এই control message আগের 32k request সম্পূর্ণ হওয়ার পর, পরবর্তী analyze() request-এর আগে append করতে হবে।
if b_commit_needed(agent):    agent._arc2_commit_started = True    agent._arc2_pending_commit_prompt = B_COMMIT_PROMPT



_arc2_pending_commit_prompt-কে বর্তমান control-message append mechanism-এর সঙ্গে integrate করতে হবে। শুধু attribute set করলে message পাঠানো হবে না।

Test matrix: 31k-এ successful check, 32k-এ thinking-only response, 33k-এ no tool এবং 35k-এ budget exhaustion। প্রতিটি ক্ষেত্রে final request budget অতিক্রম করবে না; commit সর্বোচ্চ একবার পাঠাবে।

এটি #48-এর একটি নতুন hypothesis। উপকার হবে কি না GPU evaluation-এ মাপতে হবে।
২.৩ Fix #5 ও #46 — Context window fixed 100k
E14S notebook CELL 6-এ বর্তমানে:
E12_RUN["rev6_window"] = min(    131072,    int(        E12_R6["E12_R6_KV_FRACTION"]        * _kv / max(1, E12_RUN["streams"])    ))



16 streams-এ এই formula window প্রায় 54k-তে নামিয়ে দেয়। B-এর জন্য এটা ভুল configuration।

প্রতিস্থাপন:
if E15B_CONFIG.enabled:    E12_RUN["rev6_window"] = min(        100_000,        SGL_CONTEXT_LEN - 4_096,    )else:    E12_RUN["rev6_window"] = min(        131_072,        int(E12_R6["E12_R6_KV_FRACTION"]            * _kv / max(1, E12_RUN["streams"]))    )os.environ["E12_DUCK_CONTEXT_WINDOW"] = str(    E12_RUN["rev6_window"])



এটি “never trim” guarantee নয়। প্রতি request-এ prompt length + response reserve 100k-এর ভেতরে রাখতে হবে। বেশি হলে event log-এ context_trim_emergency=1 এবং removed tokens লিখুন।

16 × 100k KV pool-এ একসঙ্গে রাখা যাবে না। তাই 100k হলো per-attempt logical context ceiling, memory residency guarantee নয়।
২.৪ Fix #47 — 35k request timeout
Notebook-এ এখন 1,800 seconds। নতুন পরিকল্পনায় conservative 3,000 seconds ব্যবহার করুন।
E12_DUCK_REQUEST_TIMEOUT = 3000



কিন্তু BranchSession.run()-এ থাকা absolute deadline clamp অপরিবর্তিত থাকবে:
request_timeout_seconds = max(    0.1,    min(        self.analyzer_timeout,        deadline - time.monotonic()    ))



Test: Fake response 1,900 সেকেন্ডে success হয়; 1,800 সেকেন্ডে ভুল timeout হবে না। Deadline-এ 120 সেকেন্ড বাকি থাকলে timeout 120 সেকেন্ডের বেশি হবে না।
২.৫ Fix #49 — min_admit_sec 1,300 করবেন না
পুরোনো worker() সরাসরি return করে:
if left < min_admit_sec or state["stop"]:    return



সুতরাং 1,300 দিলে শেষ প্রায় 22 মিনিটে আর নতুন task শুরু হবে না।

প্রাথমিক setting:
min_admit_sec = 300



এবং log:
state["admission_stopped_at_sec"] = (    deadline - time.monotonic())



তবে min_admit_sec=300 কোনো নতুন 35k attempt সম্পূর্ণ হওয়ার নিশ্চয়তা দেয় না। এটি শুধু late attempts-কে short useful work করার সুযোগ দেয়। Late requests-এর result যেন final snapshot-এর আগে harvest হয়, সেটি #54-তে যাচাই করতে হবে।
ধাপ ৩ — Fix #10–14: Duck-এর সবচেয়ে গুরুত্বপূর্ণ bugs
৩.১ Fix #10 — দুটি tool call-এর প্রথমটি execute হওয়ার পর পুরো turn rollback
File: vendor/inference/agent/tool_agent.py, প্রায় L7050–L7110।

বর্তমান source-এ দুটি tool call থাকলে প্রথমটি execute হওয়ার পর:
if tool_index < len(tool_calls) - 1:    preserve_history = Falsebreak



শেষে preserve_history=False হলে আগের পুরো history ফিরিয়ে দেওয়া হয়। ফলে প্রথম tool call-এর ফলও হারিয়ে যেতে পারে।

সমাধান হলো executed tool call ও matching result রাখা, শুধু unexecuted calls বাদ দেওয়া।

প্রথমে helper:
def retain_executed_tool_prefix(    messages: list[dict],    original_calls: list[dict],    executed_count: int,):    assert 0 <= executed_count <= len(original_calls)    kept_calls = original_calls[:executed_count]    kept_ids = {c["id"] for c in kept_calls}    # Locate the assistant message containing this tool-call batch.    index = None    for i in range(len(messages) - 1, -1, -1):        m = messages[i]        if m.get("role") != "assistant":            continue        calls = m.get("tool_calls") or []        if [c.get("id") for c in calls] == [            c.get("id") for c in original_calls        ]:            index = i            break    if index is None:        raise RuntimeError("Assistant tool-call turn not found")    messages[index]["tool_calls"] = kept_calls    # Remove only results belonging to dropped calls    # within this assistant turn's response span.    end = len(messages)    kept = messages[:index + 1]    for m in messages[index + 1:end]:        if m.get("role") == "tool":            if m.get("tool_call_id") in kept_ids:



তারপর পুরোনো দুইটি preserve_history=False assignment-এর বদলে ব্যবহার করুন:
if tool_index < len(tool_calls) - 1:    retain_executed_tool_prefix(        messages,        tool_calls,        executed_count=tool_index + 1,    )    preserve_history = True



একই executed tool call-এর result ইতোমধ্যে append হওয়ার পরে এই কাজ করতে হবে।

Validation tests:
1. দুটি tool call; প্রথমটি execute → প্রথম call ও result থাকে, দ্বিতীয় call বাদ।
2. দুটি call-ই execute → দুটোই থাকে।
3. প্রথম call error result দেয়, দ্বিতীয়টি unexecuted → error result থাকে।
4. Executed call ID এবং tool-result ID matching।
5. History reload করলেও previous accepted candidates অক্ষত।
এই fix-এর পরে আরেকটি বিষয় দেখবেন: assistant turn-এর tool-call count আর retained message-এর count সমান কি না। শুধু preserve_history=True করে সব tool-call message রেখে দিলে invalid conversation history তৈরি হতে পারে।
৩.২ Fix #11 — submit-কে আলাদা tool হিসেবে ডাকলে হারিয়ে না ফেলা
বর্তমান tool_agent.py-তে _dispatch_tool() শুধু python গ্রহণ করে:
if name == "python":    return self._run_python_tool(state_path, arguments)



B mode-এ submit-এর জন্য bounded alias যোগ করুন। এটি user-provided arbitrary Python expression চালাবে না—শুধু বৈধ function identifier।
import reimport jsondef _dispatch_tool(self, state_path, name, arguments):    self._ensure_session(state_path)    if name == "python":        return self._run_python_tool(state_path, arguments)    if (        getattr(self, "_arc2_mode_b", False)        and name == "submit"    ):        fn_name = str(            arguments.get("function")            or arguments.get("name")            or "transform"        )        if not re.fullmatch(            r"[A-Za-z_][A-Za-z0-9_]*", fn_name        ):            return _ToolDispatchResult(                json.dumps({                    "error": "submit requires a function identifier"                })            )        return self._run_python_tool(            state_path,            {"code": f"submit({fn_name})"},        )    return _ToolDispatchResult(        json.dumps({"error": f"Unknown tool: {name}"})



এখানে submit()-এর বিদ্যমান host verification অপরিবর্তিত থাকবে। Alias নিজে কোনো candidate accepted ঘোষণা করবে না।

Tests: submit(transform) success, undefined transform → controlled NameError, malformed function argument rejected, host train-verification failure rejected।
৩.৩ Fix #12 — Prompt এবং exposed tool একমত
File: arc2_duck.py, arc2_system_prompt_r6()।

বর্তমান prompt-এ one-tool rule আছে। সেটি আরও স্পষ্ট করুন:
TOOL_CONTRACT = (    "There is exactly one external tool: `python`. "    "`check(transform)` and `submit(transform)` are "    "Python functions inside that tool, not separate tools. "    "Never call a tool named `check` or `submit`. "    "Instead call `python` with code such as "    "`check(transform)` or `submit(transform)`.")



এটি tool-session rules-এ append করুন।

Test:
assert {tool["function"]["name"] for tool in tools} == {"python"}assert "not separate tools" in system_prompt.lower()



#11-এর alias হলো defensive recovery; #12 হলো model-কে সঠিক API ব্যবহার শেখানো। দুটোই রাখবেন।
৩.৪ Fix #13 — candidate ও trans auto-check
বর্তমান:
_R6_AUTOCHECK_RE = re.compile(    r"^(transform|solve|solution|predict)\w*$"    r"|^\w+_(alt\w*|v\d+)$")



প্রতিস্থাপন:
_R6_AUTOCHECK_RE = re.compile(    r"^(?:transform|solve|solution|predict|candidate|trans)\w*$"    r"|^\w+_(?:alt\w*|v\d+)$")



কিন্তু regex match করলেই কোনো function validate হবে না। বিদ্যমান _r6_single_grid_arg(src) check এবং host verification একই রাখতে হবে।

Tests:
assert _R6_AUTOCHECK_RE.fullmatch("trans")assert _R6_AUTOCHECK_RE.fullmatch("candidate")assert _R6_AUTOCHECK_RE.fullmatch("candidate_v2")assert not _R6_AUTOCHECK_RE.fullmatch("random_helper")



এর সঙ্গে AST-level test: def trans(grid) গ্রহণ, def trans(x, y) বাদ, non-grid output বাদ, wrong-train retained নয়।
৩.৫ Fix #14 — Sandbox sys, time ও helper-import messages
File: python_tool_sandbox.py।

পুরো sys module expose না করে সীমিত shim দিন:
import sys as _host_sysimport types_safe_sys = types.ModuleType("sys")def _bounded_setrecursionlimit(n):    n = int(n)    if not 100 <= n <= 10_000:        raise ValueError(            "recursion limit must be between 100 and 10000"        )    _host_sys.setrecursionlimit(n)_safe_sys.setrecursionlimit = _bounded_setrecursionlimit_safe_sys.maxsize = _host_sys.maxsizedef _safe_import(name, globals=None, locals=None,                 fromlist=(), level=0):    root = str(name or "").split(".", 1)[0]    if root == "sys" and name == "sys":        return _safe_sys    if root in {"inspect_pair", "arc_overview", "comp_sets"}:        raise ImportError(            f"`{root}` is already defined in the Python "            "tool; call it directly instead of importing it."        )    if root not in SAFE_MODULES:        raise ImportError(            f"Module '{name}' is not allowed. "            "numpy is not available."        )



SAFE_MODULES-এ time যুক্ত করুন; কিন্তু os, subprocess, socket, ctypes, importlib বা unrestricted sys নয়।

sys shim-কে পূর্ণ security boundary ধরে নেওয়া যাবে না; বিদ্যমান subprocess isolation, restricted builtins ও resource limits বজায় রাখতে হবে।

Tests: sys.setrecursionlimit(2000) success; sys.modules unavailable; sys.exit unavailable; import time success; import numpy blocked; helper-import instructional error; subprocess/OS access blocked।

ধাপ ৪ — Fix #15–18: Crash, restart, preflight ও worker safety
Fix	কোডে কী পরিবর্তন	বাধ্যতামূলক test
#15 Server watchdog	Server process, /health, decode-progress timestamp ও request failures আলাদা monitor	Healthy 1900s generation ভুল করে kill হয় না
#16 Launch fallback	16-stream → 14-stream → verified E14S 12-stream → W1 launch config	প্রতিটি tier fail করলে পরের tier, শেষে safe fallback
#17 Self-test	Synthetic fixed ARC grid দিয়ে sandbox/tokenizer preflight	Hidden task data preflight-এ ব্যবহৃত হয় না
#18 Worker exception	পুরো worker job lifecycle-এ exception capture; task failure record ও final snapshot	একটি task fail করলে অন্য workers চলতে থাকে

#15 — Watchdog example
def server_needs_recovery(    process_alive: bool,    health_ok: bool,    decode_progress_recent: bool,    health_failures: int,):    if not process_alive:        return True    # A long xhigh response is not a dead server.    if decode_progress_recent:        return False    return not health_ok and health_failures >= 3



একটি request 30 মিনিট কোনো response না দিলেই server restart করবেন না। Non-streaming long reasoning-এ এমনটা স্বাভাবিক হতে পারে। Decode log/progress দেখা জরুরি।

আর live engine restart-এর সময় pending requests automatic replay করবেন না, যতক্ষণ না request cancellation ও stale-result rejection সঠিকভাবে কাজ করছে।
#16 — Launch fallback
Notebook engine cell-এ conceptual control flow:
ENGINE_TIERS = (    ("B16", 16, 17, 68),    ("B14", 14, 15, 60),    ("E14S12", 12, 13, 56),    ("W1", 12, 13, 56),)chosen = Nonefor label, workers, mrr, mamba in ENGINE_TIERS:    flags = flags_for_tier(label, workers, mrr, mamba)    result = launch_and_validate(flags)    if result["ready"] and result["capacity_ok"]:        chosen = {            "label": label,            "workers": workers,            "mrr": mrr,            "mamba": mamba,            "result": result,        }        breakif chosen is None:    ENGINE_READY = False    raise RuntimeError(        "No serving tier passed launch and capacity gates"    )ENGINE_READY = TrueE12_STREAMS = chosen["workers"]



flags_for_tier() ও launch_and_validate()-কে বর্তমান e11_flags(), server launcher এবং capacity gate-এর সঙ্গে bind করতে হবে। এখানে নতুন function নামগুলো adapter boundaries; copy-paste করলেই সম্পূর্ণ notebook চলবে না।

গুরুত্বপূর্ণ: E14S12 ও W1 একই জিনিস নয়। E14S-এ width6/graphs13 আছে, W1-এ width8 ও পুরোনো graph coverage। Fallback tier-এর নাম ও বাস্তব flags মিলতে হবে।
#17 — Synthetic self-test
SYNTHETIC_TASK = {    "train": [        {"input": [[1, 0], [0, 1]],         "output": [[1, 0], [0, 1]]}    ],    "test": [{"input": [[2, 0], [0, 2]]}],}



এটি sandbox-এর check(), retained function, tool call এবং output validation পরীক্ষা করবে। Hidden task ID hardcode করা যাবে না।
#18 — Worker safety
Existing solve_all_r6() ইতোমধ্যে s.run() ঘিরে try/except আছে। সেটি পুরো job lifecycle পর্যন্ত প্রসারিত করতে হবে, বিশেষ করে sess(), publish() এবং stage2_items() failure-এর জন্য।
try:    run_one_scheduled_job(job)except Exception as exc:    record_worker_failure(job, exc)finally:    release_active_slot(job)    notify_scheduler()



release_active_slot() idempotent হওয়া চাই; একই worker-এ double decrement হলে scheduler-এর running counter নষ্ট হবে।

Test: একটি publish callback exception দিলেও অন্য 239 task admission বন্ধ হয় না; error task-এর final fallback থাকে।
ধাপ ৫ — Fix #19–28, #50–52: Candidate, sandbox feedback ও submission safety
৫.১ Fix #19 ও #52 — Attempt 2 কোন task আগে পাবে?
আগের source-এর rank() এবং stage2_class() ইতোমধ্যে progress-based ordering করে। নতুন scheduler লেখার দরকার নেই; B-এর resume round বাদ দিয়ে existing class priority বজায় রাখুন।
def b_attempt2_class(records):    # 0: partial train progress, no accepted program    # 1: no meaningful progress    # 2: one accepted/verified program    # 3: two or more verified, distinct predictions    # Reuse existing stage2_class() when possible.    return stage2_class(records)def b_attempt2_priority(task_id, records, order_index):    return (        b_attempt2_class(records),        order_index[task_id],    )



rank()-এ এই order আগে থেকেই থাকলে শুধু unit test যোগ করুন।

Test: partial-progress task প্রথম, no-progress দ্বিতীয়, accepted task শেষ। কিন্তু deadline-এ partial tasks শেষ না হওয়ায় অন্য classes কখনো শুরু না হলে report-এ starvation count দিতে হবে।
৫.২ Fix #20 — CPU rule-library fallback
বর্তমানে program না পেলে test input-ই predicted output হয়। নতুন fallback rule library ব্যবহার করা যাবে, কিন্তু train data-তে কোনো rule যাচাই না করে test grid তৈরি করা ঠিক নয়।

প্রাথমিক deterministic library:
def identity(g):    return [row[:] for row in g]def rotate_180(g):    return [row[::-1] for row in g[::-1]]def transpose_grid(g):    return [list(row) for row in zip(*g)]def flip_horizontal(g):    return [row[::-1] for row in g]SAFE_RULES = [    identity,    rotate_180,    transpose_grid,    flip_horizontal,]def rule_fallback(task):    for fn in SAFE_RULES:        try:            if all(                fn(pair["input"]) == pair["output"]                for pair in task["train"]            ):                outs = [fn(q["input"]) for q in task["test"]]                if all(valid_grid(g) for g in outs):                    return outs, fn.__name__        except Exception:            continue    return [        q["input"] for q in task["test"]    ], "identity_fallback"



এটি optional P1। আগে historical EVAL120-এ মাপুন কতগুলো no-program task-এ rule library বাস্তব correct output দেয়। Zero gain হলে production-এ যুক্ত করার দরকার নেই।
৫.৩ Fix #21 — Retained-state inventory
প্রতিটি tool result-এর শেষে সব functions-এর code না পাঠিয়ে শুধু নাম ও ছোট summary দিন।
def retained_function_summary(retained):    names = sorted(        k for k, v in retained.items()        if callable(v) and not k.startswith("_")    )    shown = names[:20]    return (        "Retained functions: "        + (", ".join(shown) if shown else "(none)")    )



এটি tool formatter-এ সংযুক্ত করুন। বিদ্যমান persistent-function registry-র API অনুযায়ী retained সংগ্রহ করতে হবে; plain Python globals blindly enumerate করবেন না।

Test: transform retained হলে পরের turn-এর summary-তে থাকবে; overwritten/deleted function থাকবে না।
৫.৪ Fix #22 — Bounded search
check()-এর 30-second tool limit বজায় রাখুন। Model-কে সীমাহীন DFS চালাতে দেবেন না।
class SearchBudgetExceeded(Exception):    passclass SearchBudget:    def __init__(self, max_nodes=50_000):        self.max_nodes = max_nodes        self.visited = 0    def tick(self):        self.visited += 1        if self.visited > self.max_nodes:            raise SearchBudgetExceeded(                "Search node budget exceeded"            )



প্রতিটি recursive expansion-এ budget.tick()।

Host-side verification-এর আগে একটি train input-এ function output valid grid হয় কি না পরীক্ষা করুন। তবে train pair-এর exactness-এর সিদ্ধান্ত অবশ্যই existing clean-sandbox verifier নেবে।

Test: 50,001তম node-এ controlled stop; worker আটকে থাকে না; অন্য candidate হারায় না।
৫.৫ Fix #23 — overfit_flags দিয়ে candidate reject নয়
বর্তমান select_attempts_r5()-এর ভিত্তি retained verified outputs। সেটি রাখুন।
# Reporting only:candidate["overfit_flags"] = detect_overfit_flags(candidate)# Never:# if candidate["overfit_flags"]:#     reject_candidate()



কারণ এই heuristic-এর measured precision যথেষ্ট নয়। এটিকে hard gate করলে সঠিক program বাদ পড়তে পারে।
৫.৬ Fix #24, #25 ও #27 — Setup এবং staging safety
Python package setup-এর subprocess-এ timeout:
result = subprocess.run(    command,    check=True,    capture_output=True,    text=True,    timeout=300,)



Staging copy যদি পৃথক process-এ চলে, timeout হলে process terminate এবং partial destination মুছে ফেলতে হবে। শুধু future.cancel() দিয়ে চলমান thread থামানো নিশ্চিত হয় না।
def stop_copy_process(proc):    if proc.poll() is not None:        return    proc.terminate()    try:        proc.wait(timeout=10)    except subprocess.TimeoutExpired:        proc.kill()        proc.wait(timeout=5)



.bin payload দুটির জন্য:
assert all(p.is_file() for p in required_payloads)assert all(p.stat().st_size > 0 for p in required_payloads)



Build-এ hash manifest লিখে Kaggle runtime-এ SHA compare করুন। Model load শেষ হয়েছে বলেই payload files আগেভাগে delete করবেন না; hidden run শেষ না হওয়া পর্যন্ত রাখুন।
৫.৭ Fix #26 — submission.json একবার নয়, atomic snapshots
এখানে আপনার নতুন নথির প্রস্তাব একটু বদলাব।

Kaggle-এর official format অনুযায়ী প্রতিটি task-এর প্রতিটি test input-এ attempt_1, attempt_2 থাকতে হবে; সম্পূর্ণ file-এর সব task IDs রাখতে হবে। 

Kaggle




শুধু run-এর শেষে লেখার বদলে:
1. Initial valid fallback submission।
2. Completed task-এর পরে periodic atomic snapshot।
3. Final atomic write।
import jsonimport osfrom pathlib import Pathdef write_submission_atomic(path, submission):    path = Path(path)    tmp = path.with_name(path.name + ".tmp")    payload = json.dumps(        submission,        ensure_ascii=False,        separators=(",", ":"),    )    with open(tmp, "w", encoding="utf-8") as f:        f.write(payload)        f.flush()        os.fsync(f.fileno())    os.replace(tmp, path)def validate_submission(tasks, submission, valid_grid):    assert set(submission) == set(tasks)    for tid, task in tasks.items():        rows = submission[tid]        assert len(rows) == len(task["test"])        for row in rows:            assert set(row) == {"attempt_1", "attempt_2"}            assert valid_grid(row["attempt_1"])            assert valid_grid(row["attempt_2"])



একটি serialized writer lock ব্যবহার করুন। Save করার সময় live mutable records না নিয়ে immutable snapshot থেকে build_submission() চালান।

Crash test: 240-এর মধ্যে 60 task শেষ হওয়ার পরে process বন্ধ করুন। আগের submission.json সম্পূর্ণ parse হবে এবং সব 240 task ID থাকবে।
৫.৮ Fix #28 — Stale names, SHA ও labels
CONFIG-এ actual runtime setting লিখুন:
CONFIG["experiment"] = "E15B_2x35k"CONFIG["engine_arm"] = selected_engine_labelCONFIG["workers_requested"] = requested_workersCONFIG["workers_effective"] = effective_workersCONFIG["draft_width"] = 6CONFIG["reasoning_effort_effective"] = "xhigh"CONFIG["model_sha"] = model_shaCONFIG["notebook_sha"] = notebook_sha



E14S, W1 অথবা LOW লেখা থাকলেও বাস্তবে B16/XHIGH চললে report ভুল হবে। Static title নয়, effective launch/config থেকে label তৈরি করুন।
৫.৯ Fix #50 — Reasoning text-এ লেখা transform উদ্ধার
কখনো model reasoning-এ def transform লিখেছে, কিন্তু Python tool-এ execute করেনি।

এটি উদ্ধার করা সম্ভব, তবে model-produced code সরাসরি notebook process-এ exec() করবেন না।

প্রথমে AST দিয়ে complete function source বের করুন:
import astdef extract_transform_sources(text):    candidates = []    # Process code-fenced blocks in the real implementation.    blocks = extract_python_code_blocks(text)    for block in blocks:        try:            tree = ast.parse(block)        except SyntaxError:            continue        for node in tree.body:            if not isinstance(node, ast.FunctionDef):                continue            if not node.name.startswith(                ("transform", "trans", "candidate", "solve")            ):                continue            if len(node.args.args) != 1:                continue            src = ast.get_source_segment(block, node)            if src:                candidates.append(src)    return candidates



extract_python_code_blocks() হলো নতুন helper—complete Markdown fenced code অথবা syntax-verified code span নিতে হবে; এটি এখনও দেওয়া base source-এর function নয়।

প্রতিটি candidate existing restricted Python sandbox-এ execute, তারপর fresh host verifier-এ train-exact check করুন। Reasoning text-এর কোনো code host verification ছাড়া accepted নয়।

Test: complete def transform উদ্ধার হয়; incomplete function, unsafe import, 2-argument function এবং wrong-train function reject হয়।
৫.১০ Fix #51 — yield_tokens=2048 দরকার কি?
Rev 2b অনুসারে এটি rollback trigger করে। Fix #10 পাস করার পরে দুটি CPU test করুন:
# Option ALOCAL_ANALYZER_YIELD_TOKENS = 2048# Option BLOCAL_ANALYZER_YIELD_TOKENS = 0



B-তে attempt-level pause নেই, কিন্তু turn-level yield-এর অন্য কাজ থাকতে পারে। তাই source flow পরীক্ষা ছাড়া একে blind disable করবেন না।

Promotion: No-yield mode-এ history consistency, token budget, tool responsiveness ও deadline tests পাস করলে তবেই disable। অন্যথায় 2048 রেখে rollback fix ব্যবহার করুন।

ধাপ ৬ — Fix #29–35: প্রতিটি attempt-এর যথাযথ metrics
একটি structured record রাখলে পরের GPU run-এ শুধু score নয়, B কেন ভালো/খারাপ হলো তা বোঝা যাবে।

src/e15b/telemetry.py:
from dataclasses import dataclass, field@dataclassclass AttemptTelemetry:    task_id: str    attempt_no: int    seed: int | None    tokens_generated: int = 0    end_reason: str = "unknown"    first_transform_at: int | None = None    first_check_at: int | None = None    best_train_correct: int = 0    total_train_pairs: int = 0    code_written: int = 0    code_retained: int = 0    checks_attempted: int = 0    checks_executed: int = 0    checks_timed_out: int = 0    host_verified: int = 0    accepted: int = 0    harvested: int = 0    deadline_cut_tokens: int = 0    blocked_imports: int = 0    context_trim_events: int = 0    retract_events: int = 0    effective_effort: str = "xhigh"    events: list[dict] = field(default_factory=list)



সব status একই boolean-এ ফেলবেন না। code_written, code_executed, check_attempted, train_exact ও host_verified আলাদা ঘটনা।
Fix	কী মাপবেন	Validation
#29	Per-attempt seed, token, end reason, first transform/check	সব completed attempt-এর terminal record আছে
#30	35k-এ partial progress, attempt-2 recovery	Cutoff-risk denominator নির্দিষ্ট
#31	Written/retained/executed/verified candidate	Reasoning-only code accepted নয়
#32	Deadline-cut tokens	Client-generated আর server-reported usage আলাদা
#33	Actual retraction/preemption count	retraction_policy শব্দকে event হিসেবে গণনা নয়
#34	Accepted বনাম harvested	Harvested candidate double count নয়
#35	Effective effort	Request payload-এ xhigh, report-এও xhigh

#30-এর বিশেষ হিসাব
cutoff_risk = [    a for a in attempts    if a.tokens_generated >= 35000    and a.best_train_correct > 0    and a.host_verified == 0]recovered = sum(    second_attempt_new_correct(task_id=a.task_id)    for a in cutoff_risk)



second_attempt_new_correct() শুধু offline evaluation-এ ground truth দিয়ে হিসাব করবে। Hidden runtime selection-এ কখনো ground truth ব্যবহার করা যাবে না।

Report করুন: at_cutoff_with_partial, recovered_by_attempt2, late_correct_lost_estimate ও no_transform_at_cutoff। আলাদা করে দিলে 35k সিদ্ধান্তের causal trade-off অনেক পরিষ্কার হবে।
ধাপ ৭ — Fix #36–41: 11-running সমস্যা এবং 16-stream engine
এটি B-এর সবচেয়ে ঝুঁকিপূর্ণ নতুন অংশ। আগের nodelay experiment-এ 13-running অবস্থায় dimension mismatch হয়েছিল। তাই শুধু --min-free-slots-delay 1 যোগ করলেই 16-stream নিরাপদ হবে না।

Server graph sizes, DFlash buffers, Mamba slots, client capacity gate—সব একসঙ্গে মিলতে হবে।
৭.১ Fix #36 — DFlash min-free-slots delayer বন্ধ
আপনার সংযুক্ত SGLang 0.5.21 source analysis অনুযায়ী:
--min-free-slots-delay 1


এই setting min-free-slots admission delay বন্ধ করে।

কিন্তু আগের পরীক্ষায় এটি concurrency বাড়িয়ে crash ঘটিয়েছিল। তাই এই flag শুধু 16/14-stream smoke arm-এ, baseline E14S-এ নয়।

প্রাথমিক configuration:
Parameter	B16	B14	E14S fallback
Client streams	16	14	12
--max-running-requests	17	15	13
CUDA graph batch sizes	1..17	1..15	1..13
--max-mamba-cache-size	68	60	56
Draft width	6	6	6
Min-free-slots delay	1	1	Baseline setting

৭.২ Fix #37 — Engine flags একই function থেকে তৈরি
E14S-এর e11_flags() থেকে নতুন flags তৈরি করুন; hardcoded command-এর জায়গায় একটি function ব্যবহার করুন।
def replace_cli_flags(args, replacements):    """Replace supported CLI options and their value spans."""    keys = set(replacements)    out = []    i = 0    while i < len(args):        token = str(args[i])        if token in keys:            i += 1            while (                i < len(args)                and not str(args[i]).startswith("--")            ):                i += 1            continue        out.append(token)        i += 1    for flag, values in replacements.items():        if values is None:            continue        out.append(flag)        out.extend(str(v) for v in values)    return outdef b_engine_flags(base_flags, streams):    assert streams in (12, 14, 16)    if streams == 12:



এই function CLI-তে duplicate flags দূর করার জন্য। Notebook-এর আগের Python-level SGL_MAX_RUNNING, SGL_SERVER_MAX_RUNNING, E12_STREAMS এবং CONFIG["e11"]-ও একই selected tier অনুযায়ী বদলাতে হবে।

Test:
def flag_values(args, flag):    assert args.count(flag) == 1    i = args.index(flag) + 1    values = []    while i < len(args) and not str(args[i]).startswith("--"):        values.append(str(args[i]))        i += 1    return valuesdef test_b16_flags(base_flags):    flags = b_engine_flags(base_flags, 16)    assert flag_values(        flags, "--max-running-requests"    ) == ["17"]    assert flag_values(        flags, "--max-mamba-cache-size"    ) == ["68"]    assert flag_values(        flags, "--min-free-slots-delay"    ) == ["1"]    assert flag_values(        flags, "--cuda-graph-bs-decode"    ) == [str(i) for i in range(1, 18)]



৭.৩ Fix #38 — Client capacity gate এখনো 12-তে আটকে থাকতে পারে
E14S source-এ গুরুত্বপূর্ণ parameter:
SGL_KV_PER_REQUEST = 70_000



যদি KV pool 865k হয়:
\[
865000 / 70000 \approx 12.36
\]

অর্থাৎ 16-stream server উঠলেও client gate 12 stream-এর বেশি অনুমতি নাও দিতে পারে।

প্রাথমিক B sizing-এ 50_000 tokens/request ব্যবহার করা যায়, কিন্তু এটি শুধু capacity estimate; actual admission gate-কে runtime KV usage দেখতে হবে।
def capacity_allows(    streams,    kv_tokens,    mamba_slots,    kv_per_request,    usable_fraction=0.95,):    slots_per_request = 4    kv_capacity = int(        kv_tokens * usable_fraction    ) // kv_per_request    mamba_capacity = (        mamba_slots // slots_per_request    )    effective = min(        streams,        kv_capacity,        mamba_capacity,    )    return {        "effective_running": effective,        "kv_capacity": kv_capacity,        "mamba_capacity": mamba_capacity,        "allows_target": effective >= streams,    }



উদাহরণ:
cap = capacity_allows(    streams=16,    kv_tokens=865000,    mamba_slots=68,    kv_per_request=50000,)assert cap["allows_target"]



কিন্তু মনে রাখবেন:
- 16 × 50k = 800k।
- Usable KV প্রায় 0.95 × 865k = 821.75k।
- অর্থাৎ margin খুব বেশি নয়।
- 16টি request যদি 55k context-এ পৌঁছায়, প্রয়োজন 880k, যা pool-এর চেয়ে বেশি।
তাই স্থির 50k guard-এর পাশাপাশি actual KV occupancy-ভিত্তিক admission hold অবশ্যই চালু থাকবে।

আর isolation test যদি 16-stream-এ INCONCLUSIVE হয়, আগের অনুমোদিত 6-worker safety fallback বজায় থাকবে।
৭.৪ Fix #39 — 16-stream সত্যিই চলছে কি না মাপা
শুধু decode-log line count দিয়ে time-weighted concurrency পাওয়া যাবে না। প্রতি sampled interval-এ কতক্ষণ running level বজায় ছিল, সেটা ধরতে হবে।
def weighted_mean_running(samples, max_gap_sec=10.0):    # samples: [(timestamp_sec, running, offered), ...]    samples = sorted(samples)    weighted = 0.0    total = 0.0    for a, b in zip(samples, samples[1:]):        t0, running, offered = a        t1, _, _ = b        dt = t1 - t0        if dt <= 0 or dt > max_gap_sec:            continue        # Only count sustained-load intervals.        if offered < 16:            continue        weighted += running * dt        total += dt    return weighted / total if total else None



Smoke gate:
mean = weighted_mean_running(samples)assert mean is not Noneassert mean >= 15.5



এখানে offered মানে server-এর কাছে running + waiting requests-এর বাস্তব সংখ্যা। শুধু client worker count নয়।
৭.৫ Fix #40 — Long-context KV pressure
B16-এর expected 40–45k context যুক্তিসঙ্গত estimate, কিন্তু hard upper bound নয়।

Required KV tokensEstimated KV pool

0300,000600,000900,0001,200,00020000300004000045000500005500060000









16 concurrent streams এবং আনুমানিক 865k KV pool ধরে। শুধু context-memory sizing; SSM ও temporary allocations আলাদা।





সিদ্ধান্ত: 16-stream পুরো workload-এ চলবে কি না নির্ভর করছে actual context distribution-এর ওপর। E15B-তে server-side kv_usage_p50/p90/max, Mamba usage, retract/preempt events অবশ্যই record করতে হবে।

Capacity gate-এর test শুধু startup-এ নয়; requests যখন 40–60k context-এ পৌঁছায় তখনও চালাবেন।
৭.৬ Fix #41 — CUDA graph startup সময়
Graph 1..17 capture আগের graph 1..13-এর চেয়ে বেশি memory ও startup time নিতে পারে।

Notebook-এ আলাদা timings:
startup_metrics = {    "model_load_sec": None,    "jit_compile_sec": None,    "graph_capture_sec": None,    "server_ready_sec": None,    "kv_tokens": None,    "gpu_free_gb": None,}



Smoke-এ server_ready_sec, CUDA OOM, resolved graph batch sizes, final KV pool এবং memory headroom সব record করবেন।

NO-GO condition: Graph capture incomplete, launch timeout, OOM অথবা DFlash dimension mismatch। এগুলোর কোনোটি থাকলে B16 production-এ যাবে না।

ধাপ ৮ — Fix #42–44: GPU run-এর আগে এবং পরে validation
৮.১ Fix #42 — Engine report
reports/e15b_engine.json-এর জন্য:
engine_report = {    "requested_streams": 16,    "effective_streams": None,    "running_time_weighted_mean": None,    "time_at_target_fraction": None,    "queue_positive_seconds": None,    "decode_tok_s_by_running": {},    "decode_step_ms_by_running": {},    "kv_usage_max": None,    "mamba_usage_max": None,    "retractions": None,    "preemptions": None,    "ready_sec": None,    "graph_capture_sec": None,    "cuda_oom": False,    "dimension_mismatch": False,}



এতে 16-stream দাবি সংখ্যায় যাচাই করা যাবে।
৮.২ Fix #43 — CPU suite + exact SHA validation
GPU-এর আগে test order:
1. pytest — সব B unit tests।
2. Fake-server harness — two-attempt scheduling।
3. Fake-server fault injection — rollback, 35k cap, timeout, submission।
4. Notebook compilation।
5. Embedded source hash verification।
6. Kaggle CPU kit smoke।
7. Final exact notebook SHA।
Notebook hash:
from pathlib import Pathimport hashlibdef sha256_file(path):    return hashlib.sha256(        Path(path).read_bytes()    ).hexdigest()print(sha256_file(    "Q27-XHIGH-E15B_DUCK_SGLANG.ipynb"))



GO: সব BLOCKER/HIGH regression tests PASS। একটি test fail করলে শুধু test flakiness ধরে বাদ নয়—reproduce ও root cause report করতে হবে।
৮.৩ Fix #44 — B16 → B14 → B12 GPU smoke
এটি এখনো শুধু পরিকল্পনা। GPU run শুরু হয়নি।
GPU smoke acceptance criteria
- Model READY, actual KV/Mamba capacities পাওয়া গেছে।
- 16-stream sustained-load interval-এ time-weighted mean running ≥15.5।
- কমপক্ষে 10 মিনিট representative Duck workload।
- CUDA OOM বা DFlash dimension mismatch শূন্য।
- Unexpected retraction/preemption শূন্য রাখার চেষ্টা; ঘটলে কারণ যাচাই ছাড়া promote নয়।
- Baseline ও candidate correctness probe-এ কোনো নতুন material error নয়।
- Startup, decode tok/s, agent-effective tok/s ও KV pressure recorded।
Failure sequence: B16 ব্যর্থ → B14 আলাদা smoke → B14 ব্যর্থ → verified B12 fallback।

আগের nodelay arm 13-running-এ crash করেছিল। তাই শুধু 12-request parity যথেষ্ট নয়; 16-request concurrent stress, এবং max-running-requests=17 থাকার কারণে 17th admission/buffer safety-ও পরীক্ষা করতে হবে।

একটি সতর্কতা: GPU smoke PASS হলেও B16-এর +12% estimated throughput automatically validated হবে না। Agent-effective tok/s এবং correctness-এ লাভ দেখতে হবে।

ধাপ ৯ — Fix #53–55: preemption, deadline harvest এবং build constraints
৯.১ Fix #53 — 16-stream preemption
D66/L66 historical run-এ 16-slot workload-এ preemption ছিল—আপনার নতুন নথি সেটি উল্লেখ করেছে।

Smoke-এ explicit counters:
assert metrics["cuda_oom"] is Falseassert metrics["dimension_mismatch"] is Falseif metrics["preemptions"] > 0:    verdict = "INVESTIGATE"elif metrics["retractions"] > 0:    verdict = "INVESTIGATE"else:    verdict = "PASS_CONCURRENCY_STABILITY"



Preemption শব্দটি log-এ পাওয়া আর actual retraction event এক নয়। Parsed structured events বা নির্দিষ্ট server log signature ব্যবহার করুন।
৯.২ Fix #54 — Deadline-এ in-flight candidate harvest
এটি গুরুত্বপূর্ণ কারণ আপনার আগের run-এ deadline-এ কয়েকটি request কাটা পড়েছিল।

Current scheduler join_until = deadline + join_grace_sec ব্যবহার করে এবং শেষ snapshot freeze করে।

সমস্যা: Model তখনও request-এর মাঝখানে থাকলে তার unexecuted reasoning পরে check() করার সময় নাও থাকতে পারে।

সমাধান: final output margin-এর আগে একটি controlled drain/harvest phase।
def deadline_phase(now, solve_deadline,                   harvest_reserve_sec=120):    remaining = solve_deadline - now    if remaining <= 0:        return "STOP"    if remaining <= harvest_reserve_sec:        return "HARVEST_ONLY"    return "SOLVE"



HARVEST_ONLY-তে নতুন attempt admission বন্ধ থাকবে। Existing retained functions, completed candidate verifications এবং pending host checks bounded সময়ের মধ্যে শেষ হবে।

কোনো in-flight native call deadline পার করলে সেটিকে force করে main process-এ execute করবেন না; আগের verified snapshot বজায় রাখুন।

Test: Deadline-এ 7টি request in-flight; তাদের মধ্যে 3টির verified candidate আছে। Final submission-এ ওই 3টি candidate থাকবে এবং বাকিগুলোর valid fallback থাকবে।
৯.৩ Fix #55 — Notebook size, payload এবং prompt visibility
Build-time checks:
from pathlib import Pathnotebook = Path("Q27-XHIGH-E15B_DUCK_SGLANG.ipynb")assert notebook.stat().st_size < 400_000for payload in required_payloads:    assert payload.exists()    assert payload.stat().st_size > 0



এই 400 kB সীমা আপনার project-specific build target, Kaggle-এর universal notebook-size rule নয়।

Facts cap 3,600 chars / 800 tokens বড় test grid-এর দরকারি তথ্য বাদ দিচ্ছে কি না পরীক্ষা করুন। Prompt ছোট করা ভালো, কিন্তু omitted grid যেন Python tool-এ সম্পূর্ণ পাওয়া যায়।

Tests: 30×30 grid, multiple train/test pairs, hidden-like large task। Facts truncation হলেও train, test_inputs, show() এবং inspect_pair() দিয়ে original data হারাবে না।

ধাপ ১০ — Final integration test: একসঙ্গে সব fix
আগের ৫৫টি item আলাদা unit test-এ PASS করলেও integrated failure হতে পারে। তাই একটি fake-server end-to-end scenario দরকার।
def test_e15b_end_to_end(fake_server, tasks):    result = run_e15b(        tasks=tasks,        workers=16,        attempt_budget=35000,        resume=False,        fair_share=False,        fake_server=fake_server,    )    # Stage ordering    assert result.first_stage_all_terminal_before_second    # Independent attempts    assert result.unique_attempt_sessions    assert result.distinct_attempt_seeds    # No resume    assert result.resume_events == 0    # Budget    assert result.max_generated_per_attempt <= 35000    # Tool history    assert result.unmatched_tool_call_ids == 0    # Submission    assert result.submission_valid    assert result.all_task_ids_present    # Fail-safe    assert result.failed_workers_reported    assert result.deadline_bounded



এখানে run_e15b() এবং result fields নতুন integration-harness interface-এর উদাহরণ; এগুলো বর্তমান ZIP-এর existing APIs নয়। Build-এর সময় এগুলোকে বাস্তব solve_all_r6() ও notebook entrypoint-এর সঙ্গে সংযুক্ত করতে হবে।

অন্তত চারটি separate scenario চালান:
Scenario	Expected
Normal 2×35k	All first attempts before second; no resumes
One server crash	B16 recovery অথবা verified fallback; output valid
One worker/tool failure	অন্য tasks চলবে; failure record preserved
Deadline mid-request	Valid last snapshot ও bounded stop


ধাপ ১১ — পুরো ১–৫৫ নম্বরের status ও final priority
এই তালিকাটি implementation tracker হিসেবে ব্যবহার করুন।
Fix IDs	Priority	কাজের অবস্থা
1–5	BLOCKER	Fresh attempts, 35k budget, no resume/trim logic বাস্তবায়ন
6, 49	HIGH	min_admit_sec=300, premature exit এড়ানো
7–9	HIGH	Candidate assembly, harvest, xhigh
10–11	BLOCKER	Turn rollback এবং submit dispatch
12–14	HIGH	Prompt contract, autocheck, sandbox
15–18	HIGH	Watchdog, fallback, self-test, worker safety
19–23	P1	Attempt ordering, rules, inventory, bounded search
24–28	HIGH	Subprocess, payload, atomic submission, staging
29–35	Required metrics	Per-attempt/verification/cutoff reporting
36–38	BLOCKER for B16	Delayer, real stream count, client capacity
39–42	HIGH	True concurrency, KV, startup, metrics
43	Mandatory	Exact-SHA CPU test suite
44	Mandatory GPU gate	16→14→12 smoke; separate approval
45–49	BLOCKER	Legacy chunking, context calculation, timeout, commit, admission
50–52	P1	Reasoning-code harvest, yield, existing rank
53–54	HIGH	Preemption ও deadline harvest
55	Build requirement	Notebook size, payload, facts cap

যেগুলোকে একসঙ্গে fix করতে হবে
Batch A — Scheduler correctness

1, 2, 3, 4, 5, 6, 45, 46, 47, 48, 49

Batch B — Duck correctness

7, 8, 10, 11, 12, 13, 14, 18, 50, 51, 54

Batch C — Engine 16-stream

15, 16, 36, 37, 38, 39, 40, 41, 42, 44, 53

Batch D — Safe submission ও evidence

9, 17, 19–35, 43, 52, 55

ধাপ ১২ — বাস্তবে কোন ক্রমে কাজ করব?
1. E15B builder copy তৈরি। Exact previous source SHA freeze; নতুন variant আলাদা।
2. Scheduler Batch A। 2×35k fresh attempt, no-resume, request timeout, context ও commit tests।
3. Duck Batch B। Rollback, submit alias, autocheck, sandbox, host verification, deadline harvest।
4. Submission এবং telemetry। Initial fallback, atomic snapshots, correct two-attempt ordering ও metrics।
5. Engine Batch C। B16 flags ও fallback tiers build, কিন্তু GPU launch নয়।
6. CPU tests এবং Kaggle CPU kit। Exact notebook SHA, no crashes, valid submission.json।
7. B16 GPU smoke। আলাদা অনুমতি নিয়ে, নথিভুক্ত quota ও stop rule সহ। Fail হলে B14, তারপর B12।
8. EVAL120 G4B। Engine freeze করে B-এর program-production rate, 35k loss, attempt-2 recovery এবং score পরীক্ষা।
9. Final submission decision। EVAL120 score, hidden-scale runtime ও correctness পাস করলেই competition rerun। অন্যথায় LB 34.58 fallback অক্ষত।
চূড়ান্ত GO/NO-GO
Scope	Verdict
১–৫৫ অনুযায়ী E15B CPU implementation	GO
Existing E14S/W1 পরিবর্তন বা overwrite	NO-GO
B16 directly hidden competition run	NO-GO
CPU-tested B16 GPU smoke	Separate approval
B14/B12 fallback	Keep mandatory
B-এর 35k design guaranteed accuracy gain	এখনো প্রমাণিত নয়

আমার অগ্রাধিকার: প্রথমে B-এর মূল algorithm সঠিকভাবে implement করুন। তারপর 16-stream speed increase মাপুন। কারণ 16 streams যদি 12-এর চেয়ে দ্রুতও হয়, কিন্তু B-এর 35k cutoff বা incomplete tool history-এর জন্য executable candidates হারিয়ে যায়, তাহলে leaderboard score বাড়বে না।

সবচেয়ে গুরুত্বপূর্ণ acceptance gate হলো: পুরোনো verified candidates হারাবে না, দুটি attempt সত্যিই independent হবে, 35k budget ও deadline মানবে, এবং 16-stream arm বাস্তবে দ্রুত ও স্থিতিশীল হবে।

অফিশিয়াল competition rules-এ ২৬ অক্টোবর entry deadline এবং ২ নভেম্বর final submission deadline দেখানো আছে; বাকি সময়ে Kaggle-এর ১২ ঘণ্টার GPU run cap মেনে চলতে হবে। 

Kaggle
+1




কোনো code পরিবর্তন, commit, push বা GPU execution করিনি।
