# 04 — AI fix proposal 1 of 2 for E15B (pasted by the user, 2026-10-25), verbatim

The user's message: "I am giving 3 AIs' fixes; compare them with your implementation and say if you want to adapt." The comparison is in Notes/05.

---

Experiment 15B — ১ থেকে ৫৫: প্রতিটি সমস্যার Code Fix ও Validation Plan
আপনার সর্বশেষ Config B Rev 2b তালিকাকে ভিত্তি ধরছি। এখানে আগের ৩৫টি fix-এর সঙ্গে ৩৬–৫৫ নম্বরের engine, scheduling ও reliability changes-ও থাকবে।

আমি পূর্বের ZIP থেকে arc2_duck.py, tool_agent.py, python_tool_sandbox.py, openai_compat.py এবং E14S notebook cells মিলিয়ে implementation path পরীক্ষা করেছি।

লক্ষ্য: 2×35k independent attempts, প্রথমে সব task-এর attempt 1, তারপর attempt 2; XHIGH reasoning; ১৬-stream পরীক্ষাযোগ্য engine, ব্যর্থ হলে ১৪/১২-stream fallback; পুরোনো 34.58 LB submission অক্ষত।

এগুলো proposed patches, বাস্তবে প্রয়োগ করা finished build নয়। নিচের ছোট code fragments সঠিক location ও logic দেখাচ্ছে; বিশেষ করে vendor, scheduler এবং notebook-এর ক্ষেত্রে surrounding state/locks বজায় রেখে বসাতে হবে। প্রতিটি ধাপে test থাকবে।
Phase A — Attempt architecture
১. একই task-এ দুটি স্বাধীন attempt
ফাইল: arc2_duck.py, solve_all_r6()।

দুটি attempt-ই facts view ব্যবহার করবে, কিন্তু পৃথক agent ও history থাকবে।
BRANCHES = 2B1_VIEW = "facts"result = solve_all_r6(    tasks=tasks,    order=order,    branches=BRANCHES,    b1_view=B1_VIEW,    # Other existing arguments...)



BranchSession ইতিমধ্যে (task_id, branch) দিয়ে আলাদা instance তৈরি করে। তাই একই history reuse না করার কাঠামো আছে।

Seed বিষয়ে সংশোধন: বর্তমান Duck-এ LOCAL_ANALYZER_SEED=-1; openai_compat.py-তে negative seed request-এ পাঠানো হয় না। ফলে প্রতিবার নতুন sampling হতে পারে। Independent attempt-এর জন্য fixed distinct seed বাধ্যতামূলক নয়; তবে reproducible experiments চাইলে task+attempt-specific seed দিতে হবে।
import hashlibdef stable_seed(task_id, branch, experiment_seed=20261024):    data = f"{experiment_seed}:{task_id}:{branch}".encode()    return int.from_bytes(        hashlib.sha256(data).digest()[:4], "big"    ) & 0x7FFFFFFF



এটি actual API payload-এ per-agent seed হিসেবে যুক্ত করতে হবে; module-global variable বদলানো যাবে না, কারণ multiple threads একই সঙ্গে request করে।

Test: একই task-এর দুই session-এর object identity, initial history এবং captured request payload আলাদা। Explicit seed ব্যবহৃত হলে সেগুলিও পৃথক।
২. সব attempt 1 সম্পূর্ণ হওয়ার পর attempt 2
গুরুত্বপূর্ণ: বর্তমান stage1_left admission হওয়ার সময় কমে। তাই stage1_left == 0 মানে সব b0 শুরু হয়েছে, শেষ হয়েছে নয়।

আপনার চাহিদা যদি সত্যিই প্রথম ২৪০টি attempt সম্পূর্ণ হওয়ার পরে দ্বিতীয় পর্যায় হয়, তাহলে completion barrier লাগবে।
# Scheduler statestate["stage1_completed"] = 0# After branch 0 completes, under scheduler condition lock:if b == 0 and s.status == "done":    state["stage1_completed"] += 1# Before admitting stage 2, under the same lock:if prio[0] == 1 and state["stage1_completed"] < len(order):    push(prio, tid, b, kind)    cond.wait(timeout=2.0)    continue



এটি original stage1_left coverage check-এর অতিরিক্ত guard। Paused branch0 যেন completion হিসেবে গণনা না হয়।

Test: ২৪০ synthetic tasks-এর মধ্যে শেষ b0 আটকে রাখুন। তার আগে কোনো b1 request যাবে না। b0 completed 240/240 হলে stage 2 শুরু হবে।
৩. প্রতি attempt 35,000 generated tokens
E15B_MAX_GENERATED = 35_000E15B_FIRST_EXPOSURE = None



solve_all_r6()-এ:
max_generated_tokens=E15B_MAX_GENERATED,first_exposure_tokens=None,



BranchSession.run()-এর done_now() আগেই total generated token limit পরীক্ষা করে। তবে request-boundary stop হওয়ায় সর্বশেষ response-এর cap সঠিকভাবে clamp করতে হবে—নইলে overshoot হতে পারে।

Test: 34,900 tokens খরচ হলে নতুন ordinary request সর্বোচ্চ 100 tokens; মোট 35,000-এর বেশি নয়। Tool output ও generated output আলাদা গুনবেন।
৪. 12k/24k/32k request cap সরানো
ফাইল: arc2_duck.py, _arc2_apply_cap()।

এখন পুরোনো fixed caps-এর বদলে remaining attempt budget থেকে cap নির্ণয় করুন।
def _arc2_apply_cap(self) -> int:    remaining = max(        0,        35_000 - self.generated_tokens    )    cap = max(1, remaining)    self._max_output_tokens = cap    self._reply_reserve_tokens = cap    self._context_budget_tokens = max(        1024,        self._arc2_window        - cap        - self._request_safety_margin_tokens    )    return cap



এটি initial simplified version। শেষের 3k-token commit-turn policy (#48) যুক্ত করে চূড়ান্ত করতে হবে, তাই উপরের code সরাসরি final করবেন না।

Test: 0, 10k, 30k generated-এর পরে cap যথাক্রমে 35k, 25k, 5k। পুরোনো 12,288/24,576 limit সক্রিয় নেই।
৫. Attempt-এর মধ্যে context trimming কমানো
Notebook CELL 6-এ:
E15B_CONTEXT_WINDOW = 100_000os.environ["E12_DUCK_CONTEXT_WINDOW"] = str(    E15B_CONTEXT_WINDOW)



_arc2_apply_cap()-এ:
self._reply_reserve_tokens = capself._context_budget_tokens = (    self._arc2_window    - cap    - self._request_safety_margin_tokens)



কিন্তু 100k window মানে 16×100k context GPU memory-তে fit করবে—এমন নয়। তাই normal history trim বন্ধ রাখার লক্ষ্য থাকলেও emergency context reduction ও server rejection handling বন্ধ করবেন না।

Server context limit, live KV headroom ও actual per-request prompt length যাচাই ছাড়া no-trim guarantee দেওয়া যাবে না।

Test: 35k-generated attempt-এ normal 40k–60k prompt footprint, কোনো history block বাদ যাচ্ছে না; unsafe 100k+ prompt হলে graceful exception/emergency handling, CUDA OOM নয়।
৬. Admission guard 1,300 seconds নয়
আপনার Rev 2b-তে #49 অনুযায়ী এটি সংশোধিত।

বর্তমান worker:
if left < min_admit_sec:    return



তাই 1,300 seconds রাখলে worker-রা deadline-এর প্রায় ২২ মিনিট আগে নতুন কাজ নেওয়া বন্ধ করবে।

প্রাথমিকভাবে:
E15B_MIN_ADMIT_SEC = 300



ব্যবহার করুন।
min_admit_sec=E15B_MIN_ADMIT_SEC



এটি 35k সম্পূর্ণ করার guarantee নয়; late attempt-কে partial candidate ও harvest-এর সুযোগ দেওয়ার policy।

Test: 1,000s remaining-এ কাজ admit হয়; 250s remaining-এ নতুন কাজ শুরু হয় না। Already running branch-এর verified candidate হারাবে না।
৭. attempt_1 এবং attempt_2 সাজানো
আগের select_attempts_r5() host-verified candidates হারায়নি। সেটি অপরিবর্তিত রাখুন, কিন্তু E15B-specific selection layer তৈরি করুন।
def select_attempts_b(task, records):    b0 = [r for r in records if r["branch"] == 0]    b1 = [r for r in records if r["branch"] == 1]    # Reuse the existing host-verified candidate ordering    # within each branch.    # Assemble b0 preferred output, then a distinct b1 output.



এখানে শুধু placeholder নয়—নির্দিষ্ট algorithm হবে:
1. b0-এর latest accepted submission, না থাকলে latest accepted check।
2. b1-এর best host-verified distinct prediction।
3. অন্য branch-এর accepted alternate।
4. Train-partial candidate-এর valid test prediction।
5. Test input fallback।
একটি task-এ একাধিক test input থাকলে প্রত্যেকটির output আলাদাভাবে নির্বাচন করতে হবে। valid_grid() অবশ্যই প্রয়োগ করতে হবে।

Test: b0 ও b1 উভয় accepted কিন্তু আলাদা prediction → দুটিই থাকে; b0 missing → b1 becomes attempt_1; duplicate outputs → duplicate দ্বিতীয় candidate নয়; invalid partial output → fallback।
৮. 35k শেষে retained program harvest
এটি আরও একটি model request নয়। Existing clean host verifier ব্যবহার করতে হবে।

BranchSession.run()-এর terminal path-এ, record freeze হওয়ার আগে:
def harvest_at_end(agent):    kept = agent._kept_functions or {}    names = [        name for name, source in kept.items()        if _R6_AUTOCHECK_RE.match(name)        and _r6_single_grid_arg(source)    ]    # Delegate each name to the existing bounded,    # clean host-verification mechanism.    return names



প্রকৃত integration-এ existing _arc2_harvest_r6() এবং _verify_submission() ব্যবহার করতে হবে। নতুন verifier বানিয়ে acceptance logic duplicate করবেন না।

Test: Accepted candidate পাওয়া যায়, wrong-train reject হয়, timeout bounded থাকে এবং harvest-এর কারণে generated model tokens বাড়ে না।
৯. শুধু XHIGH effort
E12_REASONING_EFFORT = "xhigh"



Request payload-এ effective value log করুন; top-level config-এ লেখা থাকলেই যথেষ্ট নয়।

Test: Captured model requests-এ XHIGH active, LOW fallback নয়। API-এর effective reasoning configuration ও server log মিলতে হবে।
Phase B — Tool, sandbox ও failure recovery (#10–18)
১০. Parallel tool-call হলে পুরো turn হারিয়ে যাওয়ার bug
ফাইল: vendor/inference/agent/tool_agent.py, আনুমানিক lines 7094–7101।

বর্তমানে প্রথম tool call-এর পরে yield/stop হলে এবং আরও tool call বাকি থাকলে:
preserve_history = False



হতে পারে। এতে প্রথম সফল tool call-সহ পুরো turn rollback হয়।

এটি সরলভাবে True করাও ঠিক নয়; assistant message-এ দুইটি tool call থাকলে history-তে দুটি corresponding tool result থাকতে হবে।

Fix: প্রথম call execute হলে তার প্রকৃত result রাখুন। বাকি calls-এর জন্য explicit skipped tool result দিন।
def close_unexecuted_tool_calls(    messages, tool_calls, executed_count):    for call in tool_calls[executed_count:]:        messages.append({            "role": "tool",            "tool_call_id": call["id"],            "content": json.dumps({                "executed": False,                "reason": "yield_or_stop_before_execution"            })        })



Actual tool-loop integration:
# After executing the current tool:if yielded_control_reason or step_executed:    if tool_index + 1 < len(tool_calls):        close_unexecuted_tool_calls(            messages,            tool_calls,            tool_index + 1,        )    preserve_history = True    break



এখানে step_executed হলে underlying environment state বদলেছে কি না এবং history persist হবে কি না—দুটি invariant একসঙ্গে বজায় রাখতে হবে। কোনো unexecuted call পরে automatic replay করবেন না।

Tests: দুই tool call-এর প্রথমটি সফল, দ্বিতীয়টি pending → history-তে প্রথমটির real result ও দ্বিতীয়টির skipped result থাকবে। প্রথমটি দ্বিতীয়বার execute হবে না। Prompt replay-তে tool-call IDs valid থাকবে।
১১. submit ভুল করে external tool হিসেবে এলে
ফাইল: tool_agent.py, _dispatch_tool()।

বর্তমানে python ছাড়া অন্য tool name এলে:
return _ToolDispatchResult(    json.dumps({"error": f"Unknown tool: {name}"}))



E15B-তে submit-কে narrowly recognized alias হিসেবে গ্রহণ করা যায়।
def _dispatch_tool(self, state_path, name, arguments):    if name == "submit":        candidate = arguments.get("name")        if not isinstance(candidate, str):            return _tool_error("submit requires a name")        if not re.fullmatch(            r"[A-Za-z_][A-Za-z0-9_]*", candidate        ):            return _tool_error("invalid candidate name")        return self._run_python_tool(            state_path,            {"code": f"submit({candidate})"}        )    return super()._dispatch_tool(        state_path, name, arguments    )



এটি adapter-level example—আসল _ToolDispatchResult ও _run_python_tool return contract অনুযায়ী বসাতে হবে।

নিরাপত্তার জন্য model-supplied arbitrary Python string interpolate করবেন না। Candidate name শুধু valid identifier হবে, এবং আসল host verifier acceptance নির্ধারণ করবে।

Tests: submit(transform) accepted when train-exact; submit(missing) rejects; injected name reject; arbitrary unknown tool আগের মতো reject।
১২. Prompt-এ tool contract স্পষ্ট করা
ফাইল: arc2_system_prompt_r6()।
TOOL_CONTRACT = """You have exactly one external tool: python.Inside the python tool, you can call:- check(transform_function)- submit(transform_function)check and submit are Python helper functions,not external tools.Never emit function calls named check or submitas separate external tools."""



Existing system prompt-এর tool instructions-এর কাছে এটি যুক্ত করুন।
prompt = existing_prompt + "\n" + TOOL_CONTRACT



Test: Fake agent tool schema-তে শুধু python। Prompt-এ Python helper contract থাকবে; submit-কে independent external tool হিসেবে advertise করা হবে না।
১৩. trans এবং candidate autocheck
ফাইল: arc2_duck.py, _R6_AUTOCHECK_RE।
_R6_AUTOCHECK_RE = re.compile(    r"^(?:transform|solve|solution|predict|"    r"candidate|trans)(?:\w*)$"    r"|^\w+_(?:alt\w*|v\d+)$")



আরও নিরাপদভাবে very broad trans* match না করে named function variants সীমিত করতে পারেন:
_R6_AUTOCHECK_RE = re.compile(    r"^(?:transform\w*|solve\w*|solution\w*|"    r"predict\w*|candidate(?:_\w+)?|"    r"trans(?:_\w+)?)$")



এরপর existing _r6_single_grid_arg(source) check অপরিবর্তিত।

Tests: trans(grid) eligible; candidate(grid) eligible; trans(x,y) ineligible; helper returning a scalar → valid-grid verification fails; wrong train prediction never becomes accepted।
১৪. Sandbox-এর safe sys ও time
ফাইল: python_tool_sandbox.py।

এখানে পুরো sys module untrusted model code-এ expose করবেন না। Sandbox child-এর মধ্যে সীমিত shim দিন।
from types import SimpleNamespacedef safe_recursion_limit(n):    n = int(n)    if not (100 <= n <= 3000):        raise ValueError("recursion limit out of bounds")    sys.setrecursionlimit(n)SAFE_SYS = SimpleNamespace(    maxsize=sys.maxsize,    setrecursionlimit=safe_recursion_limit,)



Import interception:
def _safe_import(name, globals=None,                 locals=None, fromlist=(), level=0):    if level != 0:        raise ImportError("relative imports are blocked")    if name == "sys":        return SAFE_SYS    if name == "time":        return time    root = name.split(".", 1)[0]    if root in {"numpy", "subprocess", "socket"}:        raise ImportError(            f"{name} is unavailable in this sandbox"        )    if root in PRELUDE_HELPER_NAMES:        raise ImportError(            f"{root} is already defined; call it directly"        )    if root not in SAFE_MODULES:        raise ImportError(f"Module '{name}' is not allowed")    return builtins.__import__(        name, globals, locals, fromlist, level    )



PRELUDE_HELPER_NAMES-কে existing prelude names থেকে সীমিতভাবে সরবরাহ করতে হবে।

গুরুত্বপূর্ণ: Recursion limit বাড়ালেও sandbox-এর 30-second timeout, process isolation ও memory bounds অক্ষত থাকবে।

Tests: sys.maxsize works; sys.setrecursionlimit(500) works; enormous recursion limit rejects; sys.modules unavailable; time.monotonic() works; numpy/subprocess blocked।
১৫. Server watchdog
ফাইল: Notebook engine management cell।

Server process মারা গেলে endless request retries নয়। Bounded recovery state machine রাখুন।
def watchdog_tick(engine):    if engine.is_ready():        return "healthy"    if engine.restarts >= 1:        return "failed"    with engine.restart_lock:        if engine.is_ready():            return "healthy"        engine.restarts += 1        engine.stop_old_processes()        return (            "recovered"            if engine.launch_last_good()            else "failed"        )



engine.* এখানে existing notebook launch/kill/health-check functions-এর জন্য interface sketch; নতুন subprocess manager অন্ধভাবে বসাবেন না।

একটি অতিরিক্ত শর্ত: Server restart হলে in-flight request ও recurrent cache নষ্ট হয়। তাই আগের incomplete request-কে successful বলে report করা যাবে না। Recovered server থেকে fresh work শুরু করা যেতে পারে, কিন্তু পুরোনো partial attempt continuation কেবল history ও checkpoint integrity যাচাই করে।

Tests: Mid-run process killed; bounded restart once; no duplicate host submission; previous accepted candidates retained; second restart failure → fallback path।
১৬. Engine launch fallback: 16 → 14 → 12
এখন Rev 2 অনুযায়ী priority:
ENGINE_LEVELS = (16, 14, 12)



প্রতিটি level-এ সম্পূর্ণ consistent flags ও capacity configuration তৈরি করতে হবে।
def launch_with_fallback(launch, healthcheck):    failures = []    for n in ENGINE_LEVELS:        flags = engine_flags_for_streams(n)        ok = launch(flags)        if ok and healthcheck():            return {                "ok": True,                "streams": n,                "flags": flags,                "failures": failures,            }        failures.append(n)    return {        "ok": False,        "streams": 0,        "failures": failures,    }



এখানে ১২-stream fallback-এর মধ্যে known-good E14S/W1 configuration থাকতে হবে। একাধিক cold launch-এর সময় ও ১২ ঘণ্টার limit কঠোরভাবে মেনে চলতে হবে।

Tests: 16 fails → 14 works; 16/14 fail → 12 works; all fail → valid fallback artifact; actual stream count report accurate।
১৭. Sandbox/tokenizer self-test hidden task দিয়ে নয়
একটি স্থির synthetic task রাখুন।
SELF_TEST_TASK = {    "train": [{        "input": [[1, 0], [0, 2]],        "output": [[1, 0], [0, 2]],    }],    "test": [{        "input": [[3, 0], [0, 4]],    }],}



এই task-এ identity transform, helper availability, tokenizer loading, sandbox execution ও host verifier পরীক্ষা করা যায়।
def transform(grid):    return [row[:] for row in grid]



Test: Synthetic transform exact; tokenizer reasoning_tokens() positive; no hidden task read; smoke শেষে কোনো artificial candidate production submission-এ ঢোকে না।
১৮. Worker exception handling
বর্তমান worker-এ s.run() ঘিরে try/except আছে। কিন্তু session setup, publish, stage2_items, বা scheduler bookkeeping error হলে thread মারা যেতে পারে।

এখানে per-job cleanup অবশ্যই finally-এ হবে।
job_error = Nonetry:    session = sess(tid, branch)    session.run(pause_at_tokens=None)except Exception as exc:    job_error = f"{type(exc).__name__}: {exc}"    record_worker_failure(tid, branch, job_error)finally:    with cond:        state["running"] -= 1        active.discard((tid, branch))        cond.notify_all()



এটি scheduler-এর existing completion block-এর সঙ্গে merge করতে হবে; running দুইবার decrement করা যাবে না।

Test: Session constructor exception, run() exception, callback exception ও serialization exception—সব ক্ষেত্রেই worker slot release, terminal error record এবং scheduler exit হবে।
Phase C — Smaller fixes ও candidate selection (#19–28)
১৯. Attempt 2-এর priority: partial → no-progress → accepted
Existing stage2_class() ব্যবহার করুন। নতুন complex scorer দরকার নেই।
def attempt2_priority(task_records):    cls = stage2_class(task_records)    return cls  # lower class first



Class mapping:
Class	Meaning	Priority
0	Partial train progress	1
1	No progress	2
2	One verified candidate	3
3	Multiple verified candidates	4

Scheduler-এ b1 jobs rank করার সময় rounds নতুন B-তে zero থাকবে, কারণ b1 একবারই চালানো হবে।

Test: Four-task fake queue-এ priority 0→1→2→3। Attempt 1 incomplete থাকা অবস্থায় কোনো b1 শুরু নয়।
২০. CPU rule-library fallback
সুপারিশ: প্রথম build-এ optional রাখুন। Verified second attempt-এর বিকল্প হিসেবে অযাচাইকৃত rule-library output বসাবেন না।
def cpu_fallback(task, rule_library):    for rule in rule_library:        try:            if all(                rule(p["input"]) == p["output"]                for p in task["train"]            ):                predictions = [                    rule(q["input"])                    for q in task["test"]                ]                if all(valid_grid(g) for g in predictions):                    return predictions        except Exception:            continue    return None



এটি minimal sketch। Actual run-এ CPU rules-এর execution deadline এবং memory bound থাকতে হবে; validation-এ hidden test gold ব্যবহার করা যাবে না।

Test: Rule train-exact হলে candidate পাওয়া যায়; wrong-train rule reject; malformed grid reject; no rule → identity fallback।
২১. Tool result-এ retained function names দেখানো
Model বারবার NameError পাচ্ছে, কারণ আগে কোন function retained ছিল তা জানে না।
def retained_inventory(kept):    names = sorted(kept)    displayed = names[:20]    suffix = (        f" (+{len(names)-20} more)"        if len(names) > 20 else ""    )    return "Available functions: " + ", ".join(displayed) + suffix



Tool payload-এ existing stdout/error-এর পরে ছোট metadata যুক্ত করুন।
payload["available_functions"] = retained_inventory(    self._kept_functions or {})



Test: Current function set সঠিক; rejected function তালিকায় নেই; response character/token cap অতিক্রম করে না।
২২. Search/backtracking bounded করা
এটি primarily model-facing guidance এবং reusable helper।
def limited_dfs(start, expand, is_goal, max_nodes=20_000):    stack = [start]    visited = 0    while stack:        visited += 1        if visited > max_nodes:            return None        state = stack.pop()        if is_goal(state):            return state        stack.extend(expand(state))    return None



Prompt-এ বলুন candidate-এর একটি training input-এ runtime/shape smoke চালানোর পরে check() করতে।
sample = transform(train[0]["input"])assert valid_grid(sample)check(transform)



এটি sandbox prelude-তে train নাম থাকলে তার উপযোগী; actual helper name অনুযায়ী ব্যবহার করুন।

Test: Impossible search bounded; easy target solved; 30-second sandbox timeout অক্ষত; no infinite recursive search।
২৩. overfit_flags দিয়ে accepted candidate reject নয়
Existing flags diagnostics হিসেবে রাখুন।
candidate["overfit_flags"] = overfit_flags(    task, candidate_source, predictions)candidate["host_verified"] = (    candidate["train_exact"] is True    and candidate["safe_execution"] is True)



overfit_flags selection score থেকে সরিয়ে দিন। Fixed coordinates-ও ARC-এর সঠিক rule হতে পারে।

Test: Valid fixed-coordinate transform-এ warning থাকলেও accepted থাকে; wrong-train transform warning না থাকলেও rejected।
২৪. pip/venv timeout
def run_setup(command, *, timeout_sec=600):    cp = subprocess.run(        command,        text=True,        capture_output=True,        timeout=timeout_sec,        check=False,    )    if cp.returncode != 0:        raise RuntimeError(            cp.stderr[-2000:]        )    return cp



Notebook-level deadline কম থাকলে effective timeout-কে remaining time অনুযায়ী clamp করুন।

Test: Hanging subprocess kill, stderr record, no endless setup retry।
২৫. .bin payload hidden run শেষ না হওয়া পর্যন্ত রাখা
Original dataset immutable থাকবে।
payload_paths = [    MODEL_PAYLOAD,    DUCK_PAYLOAD,]for path in payload_paths:    assert path.is_file(), f"Missing payload: {path}"



Cleanup শুধু temporary extracted copies-এর ওপর হবে।
# Never delete source payloads.shutil.rmtree(temp_extract_dir, ignore_errors=True)



Test: Failed launch/restart-এর পর payload পুনরায় extract করা যায়; original SHA অপরিবর্তিত।
২৬. submission.json বারবার rewrite বন্ধ
সংশোধিত design: একবার initial fallback, একবার final atomic replacement। সম্পূর্ণ শেষে প্রথমবার লেখা নয়।
def atomic_submission(path, obj):    path = Path(path)    tmp = path.with_suffix(".json.tmp")    with tmp.open("w", encoding="utf-8") as f:        json.dump(obj, f)        f.flush()        os.fsync(f.fileno())    os.replace(tmp, path)



Notebook শুরুতে:
atomic_submission(    "/kaggle/working/submission.json",    complete_identity_fallback)



Solver শেষে:
atomic_submission(    "/kaggle/working/submission.json",    final_verified_submission)



Test: Process crash, invalid branch, deadline—সব অবস্থায় file exists এবং JSON schema valid। Final write interrupted হলেও আগের fallback intact।
২৭. Staging copy timeout হলে cancel
stop_copy = threading.Event()



Worker copy loop-এ:
for chunk in chunks:    if stop_copy.is_set():        return    copy_chunk(chunk)



Timeout:
stop_copy.set()copy_thread.join(timeout=5)if copy_thread.is_alive():    # Use process termination when supported;    # Python threads cannot be forcibly killed safely.    staging_status = "abandoned"



Original source-এ multiple subprocess readers থাকলে process group terminate করার ব্যবস্থা প্রয়োজন। শুধু thread join(timeout) দিলেই worker বন্ধ হয়েছে ধরে নেওয়া যাবে না।

Test: Slow copy timeout → no late rename/publish; original model path fallback কাজ করে; no background disk contention।
২৮. Report-এর stale label সংশোধন
CONFIG_REPORT = {    "variant": "E15B",    "attempts": 2,    "tokens_per_attempt": 35000,    "effort": "xhigh",    "engine_flags": effective_launch_argv,    "notebook_sha256": notebook_sha,}



effective_launch_argv নির্বাচিত fallback থেকে নিতে হবে, intended config থেকে নয়।

Test: 16 fails →14 fallback হলে report 14 বলবে; W1 fallback হলে w6-g13 label থাকবে না।

Phase D — Reporting, যাতে run-এর ফল সঠিকভাবে বুঝি (#29–35)
২৯. প্রতি attempt-এর সম্পূর্ণ record
attempt_record = {    "task_id": tid,    "attempt": branch + 1,    "seed": effective_seed,    "view": "facts",    "tokens_generated": agent.generated_tokens,    "stop_reason": stop_reason,    "first_transform_tokens": first_transform_tokens,    "first_check_tokens": first_check_tokens,    "best_train_k": best_k,    "best_train_n": train_count,    "harvested": harvested,}



Test: ১২০ tasks × ২ attempts → সর্বোচ্চ ২৪০ records; never-started branch-এর record না বানিয়ে not_admitted task summary-তে রাখুন।
৩০. 35k cutoff-এ late-solve risk মাপা
cutoff_snapshot = {    "reached_35k": generated >= 35000,    "has_retained_transform": bool(retained_candidates),    "best_partial_k": best_k,    "accepted_by_cutoff": accepted,}



Task-level summary:
metrics["cutoff_with_partial"] += int(    reached_35k and best_k > 0 and not accepted)



Attempt 2 একই task উদ্ধার করলে rescued_by_attempt2=True।

Test: 35k-এ partial 2/3, second accepted 3/3 → rescue। Gold label দেখে নয়, host-verification event দিয়ে হিসাব।
৩১. Candidate lifecycle আলাদা করা
candidate_metrics = {    "source_written": False,    "source_executed": False,    "retained": False,    "check_attempted": False,    "check_executed": False,    "check_timeout": False,    "train_pairs_exact": 0,    "host_verified": False,}



AST parse-এ function লেখা ধরা যাবে; তবে source reasoning-only হলে execution flag false।
tree = ast.parse(code)candidate_metrics["source_written"] = any(    isinstance(node, ast.FunctionDef)    for node in tree.body)



Test: Reasoning text-এ function কিন্তু tool-call নেই → written-only। Tool timeout → attempted but not executed। Host acceptance → verified।
৩২. Deadline-এ কাটা token হিসাব
usage_record = {    "tokens_completed": usage_output_tokens,    "request_started": started,    "request_finished": finished,    "deadline_cancelled": cancelled,    "unreported_tokens": None,}



Important: Server usage না পাঠালে generated token count অনুমান করে measured হিসেবে লিখবেন না। unknown/unreported রাখুন।

Test: Completed, cancelled before usage, timeout with partial usage—সব আলাদা।
৩৩. Retract counter-এর ভুল
Config string retraction_policy এবং actual runtime events এক নয়।
retract_count = sum(    1 for event in server_events    if event.get("kind") == "request_retracted")



যদি logs structured না হয়, version-specific SGLang log pattern দিয়ে parser করুন এবং raw matching line সংরক্ষণ করুন।

Test: retraction_policy লেখা আছে কিন্তু কোনো retraction নেই → zero। একটি real retract event → one।
৩৪. Accepted বনাম harvested
metrics = {    "explicit_accepted": 0,    "check_only_accepted": 0,    "end_harvest_accepted": 0,    "selected_final": 0,}



Candidate source identity দিয়ে deduplicate করতে হবে, যাতে একই function check() ও submit()-এ গুনলে double-count না হয়।

Test: Same source/output explicit submit ও harvest → one distinct candidate, two events।
৩৫. Effective reasoning effort
request_record["requested_effort"] = requested_effortrequest_record["effective_effort"] = resolved_effort



সাধারণভাবে server response effective effort ফেরত না দিলে effective_effort-কে নিশ্চিত observed value বলা যাবে না। Chat-template arguments ও request generation path inspect করে resolved value report করুন।

Test: LOW profile-এ top-level XHIGH থাকলেও individual request LOW হলে report LOW; E15B XHIGH tests actual payload/template দিয়ে validate।
Phase E — ১১-running সমস্যা ও ১৬-stream engine (#36–44)
এটি আপনার Rev 2-এর নতুন অংশ। এখানে CPU patch আর GPU-measured gain পৃথক রাখব।
৩৬. 11-running bottleneck: delayer disable
ফাইল: SGLang launch flags, notebook engine cell।

আপনার pinned SGLang 0.5.21 source review অনুযায়ী:
--min-free-slots-delay 1


দিলে সংশ্লিষ্ট min-free-slots delayer disabled হয়।

Engine flag builder-এ:
flags += [    "--min-free-slots-delay", "1",]



কেন: mrr13-এ automatic threshold প্রায় 3 হলে 11-running + queued pattern তৈরি হয়। Delayer disable করলে missing 12th slot admit হওয়ার সুযোগ পাওয়া যায়।

তবে আগের nodelay arm 13-running-এ crash করেছিল। তাই #37-এর matching graph/sampling capacity fix ছাড়া এই flag production-এ দেবেন না।

Tests: Launch argv-এ flag 1; resolved delayer log disabled; 12, 14 ও 16 concurrent requests-এ no tensor mismatch।
৩৭. 16 streams, fallback 14/12
প্রথমে একটি centralized flag builder তৈরি করুন। বিভিন্ন notebook cell-এ আলাদা hardcoded সংখ্যা রাখবেন না।
def engine_flags_for_streams(n: int) -> list[str]:    assert n in (12, 14, 16)    mrr = n + 1    graphs = list(range(1, mrr + 1))    mamba = 4 * mrr    return [        "--max-running-requests", str(mrr),        "--max-mamba-cache-size", str(mamba),        "--min-free-slots-delay", "1",        "--cuda-graph-bs-decode",        *map(str, graphs),    ]



Derived configuration:
Client streams	Max running	Graph batch sizes	Mamba slots
12	13	1–13	52
14	15	1–15	60
16	17	1–17	68

Correction: 12-stream fallback যদি known-good W1 হয়, W1-এর পুরোনো flags (delayer/graphs) বজায় রাখতে হতে পারে। উপরের 12-stream candidate হলো new delayer-disabled profile, পুরোনো W1-এর exact copy নয়। দুটিকে আলাদা named profiles হিসেবে রাখবেন।

Test:
assert engine_flags_for_streams(16)[0:2] == [    "--max-running-requests", "17"]assert engine_flags_for_streams(14)[0:2] == [    "--max-running-requests", "15"]



আর actual parser থেকে graph-list যাচাই করুন, শুধু raw argument string নয়।
৩৮. Client capacity gate যেন 16-কে সত্যিই অনুমতি দেয়
Notebook configuration-এ:
E12_STREAMS = ENGINE_STREAMSE12_WORKERS = ENGINE_STREAMS



Actual code call:
workers=effective_workers



effective_workers isolation preflight ও server capacity থেকে আসবে।
def effective_capacity(streams, isolation_ok, caps):    if not isolation_ok:        return 6    if caps["max_running_requests"] < streams:        raise RuntimeError("server concurrency cap too small")    if caps["mamba_slots"] < 4 * streams:        raise RuntimeError("Mamba capacity too small")    return streams



এই sample-এ 4 slots/request conservative planned policy; real SGLang allocation requirement startup-resolved config দিয়ে যাচাই করবেন।

KV gate আগের মতো 0.95 থাকবে:
kv_admit_limit=0.95



কিন্তু KV metrics stale হলে high-concurrency run-এ fail-open করা বিপজ্জনক হতে পারে। তাই telemetry-loss event record এবং bounded recovery/worker reduction যোগ করা উচিত—server KV overcommit করবেন না।

Test: Server caps 17 / Mamba 68 / isolation PASS →16 workers; capacity 12 → reject 16-profile; isolation INCONCLUSIVE after retry →6 workers।
৩৯. Real running 16 হয়েছে কি না
Current running count sample নিয়ে শুধু sample average করবেন না। Time-weighted histogram প্রয়োজন।
from collections import defaultdictdef time_weighted_running(samples):    # samples: [(monotonic_time, running, queued), ...]    buckets = defaultdict(float)    for (t0, n, q), (t1, _, _) in zip(        samples, samples[1:]    ):        buckets[n] += max(0.0, t1 - t0)    total = sum(buckets.values())    mean = (        sum(n * sec for n, sec in buckets.items()) / total        if total else 0.0    )    return mean, dict(buckets)



Pass rule: ১৬টি requests নিরবচ্ছিন্নভাবে offered/queued থাকা representative window-তে mean running ≥15.5 এবং no crash।

গুরুত্বপূর্ণ: Queue খালি হয়ে গেলে workload-এর অভাবে running 12 হওয়াকে 16-capacity failure বলা যাবে না। তাই backlog-positive intervals আলাদা করে মাপবেন।

Tests: Uniform 16-running → mean16; half 15/half16 →15.5; idle intervals excluded from saturation comparison।
৪০. 16-stream KV fit
আপনার revised projected pool ~865k tokens। 16×45k ≈720k, অর্থাৎ প্রায় 83%।
def kv_projection(streams, tokens_per_stream, pool_tokens):    demand = streams * tokens_per_stream    return {        "demand": demand,        "fraction": demand / pool_tokens,        "fits": demand < pool_tokens,    }



Example:
assert kv_projection(    16, 45_000, 865_000)["fits"]



কিন্তু এই হিসাব memory safety proof নয়।

একই সময়ে 16×55k হলে 880k >865k। Prompt lengths, graph allocation, speculative verify buffers, radix cache ও in-flight context growth প্রভাব ফেলবে।

Recommended capacity report:
kv_report = {    "kv_pool_tokens": actual_pool,    "kv_usage_peak": observed_peak,    "mamba_usage_peak": observed_mamba_peak,    "running_peak": observed_running_peak,    "retractions": observed_retractions,}



Test: High-context synthetic pressure-এ no OOM, no unbounded retraction। KV >95% হলে new admission held। Already-running requests-এর remaining context-growth margin আলাদাভাবে পর্যবেক্ষণ।
৪১. Graph capture startup time
graphs 1..17 initial capture বাড়াতে পারে। তাই actual time split মাপতে হবে।
startup_record = {    "launch_started": launch_started,    "model_loaded": model_loaded_time,    "graphs_ready": graphs_ready_time,    "server_ready": server_ready_time,}



Computed fields:
startup_record["ready_sec"] = (    server_ready_time - launch_started)startup_record["graph_sec"] = (    graphs_ready_time - model_loaded_time)



Stop rule: 16-profile launch timeout বা excessive headroom consumption হলে 14-profile; তারপর 12।

Tests: Fake delayed launch 1800s cap; timeout হলে orphan server process cleanup; selected profile actual memory ও capacity logged।
৪২. Engine report block
সব engine metrics এক জায়গায় রাখুন।
engine_report = {    "effective_streams": effective_workers,    "max_running": resolved_max_running,    "running_histogram_sec": running_histogram,    "mean_running_under_pressure": mean_under_pressure,    "queue_positive_seconds": queue_sec,    "decode_tok_s_by_running": tok_s_by_level,    "decode_step_ms": step_ms_by_level,    "kv_peak": kv_peak,    "mamba_peak": mamba_peak,    "server_ready_sec": ready_sec,    "graph_capture_sec": graph_sec,    "retractions": retract_count,}



Test: 16 configured কিন্তু 6 isolation fallback → report 6 effective workers; label 16 configured আলাদা।
৪৩. Unit tests + harness + Kaggle CPU kit
CPU harness-এ সব profile simulate করুন।
for n in (12, 14, 16):    args = engine_flags_for_streams(n)    assert flag_value(args, "--min-free-slots-delay") == "1"    assert int(flag_value(args, "--max-running-requests")) == n + 1    assert graph_sizes(args) == list(range(1, n + 2))    assert int(flag_value(args, "--max-mamba-cache-size")) >= 4 * (n + 1)



এখানে flag_value ও graph_sizes small CLI parser test helpers; actual argparse resolution সম্ভব হলে সেটিই ব্যবহার করুন।

Required: Commit, rerun এবং EVAL120 profile-এর exact notebook SHA-তে tests। GPU smoke-এর code আলাদা SHA-তে হলে সেই difference documented।
৪৪. Short GPU smoke — প্রায় 30-minute hard stop
এটি প্রস্তাবিত GPU test, এখন চালানো হয়নি।

পরীক্ষার ক্রম:
1. 16-stream launch।
2. READY ও memory record।
3. ≥10-minute saturated Duck workload।
4. Running mean ≥15.5, no OOM/retract।
5. 12-request target-only correctness/parity sanity gate।
6. Fail → 14; আবার fail →12। প্রতিটি extra GPU stage অনুমতি ও quota অনুযায়ী।
একটি সংশোধন: 12-request short parity দিয়ে 16-stream DFlash numerical correctness পুরোপুরি প্রতিষ্ঠিত হয় না। 16-way concurrent sample এবং অন্তত কিছু long-context outputsও পরের full experiment-এ যাচাই করতে হবে।

GO conditions: Actual running capacity, engine stability, bounded startup, usable KV headroom এবং non-regressing aggregate tok/s। শুধু “১৬টি request admit হয়েছে” যথেষ্ট নয়।
Phase F — Rev 2b-এর নতুন blocker ও অতিরিক্ত fixes (#45–55)
৪৫. Fair-share system দ্বিতীয় attempt-কে chunk করে ফেলবে
ফাইল: arc2_duck.py, solve_all_r6()।

এটাই Config B-এর সবচেয়ে জরুরি scheduler correction।

Current code stage-2-তে share_tokens() নিয়ে pause_at_tokens বসায়। B-তে সেটি চলবে না।
solve_all_r6(    ...,    first_exposure_tokens=None,    fair_share=False,    max_generated_tokens=35_000,)



Worker-এর ভেতর:
# E15B onlypause = Nones.run(    pause_at_tokens=pause,    tag=tag,)



Test: b0 এবং b1—প্রতিটি branch-এর scheduler run count 1। Stage-2 min chunk 12,288 বা dynamic fair share কার্যকর নয়।

LOCAL_ANALYZER_YIELD_TOKENS-এর internal analyzer yield আলাদা; সেটি #51-তে।
৪৬. Window formula 16 streams হলে ~54k-তে নেমে যায়
ফাইল: Notebook CELL 6।

বর্তমান:
E12_RUN["rev6_window"] = min(    131072,    int(kv_pool / streams))



16-stream-এ window প্রায় 54k হবে। এতে নতুন no-trim objective ভেঙে যাবে।

E15B profile-এ:
if CONFIG_VARIANT == "E15B":    E12_RUN["rev6_window"] = 100_000else:    E12_RUN["rev6_window"] = old_window_formula()



os.environ["E12_DUCK_CONTEXT_WINDOW"] = str(    E12_RUN["rev6_window"])



Test: Streams 12/14/16 যাই হোক B-এর configured window 100k। Other profiles আগের logic ব্যবহার করে।

Caution: 100k হলো context-management ceiling, memory reservation নয়। Actual KV capacity checks #40 থেকে বাদ দেওয়া যাবে না।
৪৭. Request timeout 1,800 → 3,000 seconds
ফাইল: Notebook config।
if CONFIG_VARIANT == "E15B":    E12_DUCK_REQUEST_TIMEOUT = 3000



একটি long 35k request-এ per-stream throughput ~20–30 tok/s হলে 1,200–1,750 seconds লাগতে পারে; slow tail আরও বেশি হতে পারে।

তবে absolute solver deadline clamp সবসময় জিতবে।
effective_timeout = min(    3000,    max(0.0, deadline - time.monotonic()))



Deadline ইতিমধ্যে পেরিয়ে গেলে নতুন request শুরু নয়। max(0.5, ...) দিয়ে expired deadline-এ request চালানোর অনুমতি দেওয়া ঠিক নয়।

Test: 2,200-second synthetic request allowed when enough deadline; deadline 300s away হলে timeout ≤300s; expired deadline → no request।
৪৮. শেষ 3k tokens commit turn-এর জন্য reserve
এই পরিবর্তনটি আপনার মূল plan-এর সবচেয়ে মূল্যবান behavioral experiment হতে পারে।

কারণ 35k hard cutoff-এ Qwen <escape></think></escape>-এর আগে output শেষ করলে tool call আর আসবে না।

Policy:
- Ordinary reasoning requests সর্বোচ্চ 32k-এর cumulative threshold পর্যন্ত যেতে পারে।
- অবশিষ্ট 3k দিয়ে একটি final candidate-commit opportunity।
- Total generated tokens সর্বোচ্চ 35k।
_arc2_apply_cap():
E15B_TOTAL = 35_000E15B_COMMIT_RESERVE = 3_000def e15b_request_cap(used, commit_started=False):    left = max(0, E15B_TOTAL - used)    if left == 0:        return 0    if commit_started or left <= E15B_COMMIT_RESERVE:        return left    return max(1, left - E15B_COMMIT_RESERVE)



Usage:
cap = e15b_request_cap(    self.generated_tokens,    self._e15b_commit_started,)self._max_output_tokens = capself._reply_reserve_tokens = cap



যখন remaining ≤3,000 এবং next request শুরু হচ্ছে, একবার এই message যুক্ত করুন:
COMMIT_MESSAGE = """Your reasoning budget is almost exhausted.Do not restart the analysis.Use the best rule you have so far.Write an executable one-grid transform(grid)and call check(transform) using the python tool.If the rule is imperfect, submit a provisionalcandidate only after testing it.Avoid another long verbal analysis."""



Tool-message/history ordering বজায় রেখে existing follow-up prompt path দিয়ে এটি inject করা সবচেয়ে নিরাপদ।

Tests:
assert e15b_request_cap(0) == 32_000assert e15b_request_cap(20_000) == 12_000assert e15b_request_cap(32_000) == 3_000assert e15b_request_cap(34_900) == 100assert e15b_request_cap(35_000) == 0



আর commit_started=True-এর পরে আবার commit message পাঠানো যাবে না।

সতর্কতা: 3k-তে model tool call দেবে, এমন guarantee নেই। তাই end harvest (#8) বজায় থাকবে। Accuracy লাভ পরীক্ষায় মাপতে হবে।
৪৯. Minimum admission guard correction
#6-এর revised implementation-ই final।
E15B_MIN_ADMIT_SEC = 300



worker()-এ fixed guard ছাড়াও stage-1 completion ও submission fallback নিশ্চিত করতে হবে।

Test: Deadline থেকে 20 মিনিট বাকি থাকলে worker exit নয়; শেষ কয়েক মিনিটে new work admission বন্ধ হলেও running work cleanup হবে।
৫০. Reasoning text-এর ভেতরে লেখা code উদ্ধার
কিছু branch-এ def transform reasoning-এর ভেতরে লেখা হয়েছে, কিন্তু Python tool call হয়নি।

এটি optional, কিন্তু useful salvage mechanism।

গুরুত্বপূর্ণ: Model reasoning untrusted code। এটিকে host notebook-এ exec() করবেন না।

Conservative extraction:
import astdef extract_transform_block(text):    try:        tree = ast.parse(text)    except SyntaxError:        return None    for node in tree.body:        if isinstance(node, ast.FunctionDef):            if node.name in {"transform", "trans", "candidate"}:                return ast.get_source_segment(text, node)    return None



বাস্তবে reasoning text-এর মধ্যে prose থাকায় ast.parse(text) প্রায়ই fail করবে। তাই line-scoped code blocks বা fenced Python code আগে বের করে, তারপর AST validate করতে হবে।

তারপর:
1. _r6_single_grid_arg(source)।
2. Existing sandbox-এ bounded execution।
3. Existing clean verifier।
4. Accepted হলে candidate।
5. Timeout/import error হলে rejected।
Test: Complete fenced def transform উদ্ধার; incomplete function reject; reasoning text-এ malicious import sandbox reject; no direct host execution।
৫১. yield_tokens=2048 কি বন্ধ করা যায়?
ফাইল: Duck tool_agent.py configuration।

বর্তমান 2048-token yield-ই rollback path trigger করতে পারে। তবে global deadline, tool loop এবং output accounting-এর সঙ্গে এর সম্পর্ক আছে।

তাই আমার decision:

প্রথমে #10 fix ও tests। তারপর CPU-controlled arm-এ yield disabled পরীক্ষা।
# Candidate only, after rollback regression passes:os.environ["LOCAL_ANALYZER_YIELD_TOKENS"] = "0"



এটি ভিন্ন performance/behavior arm। প্রথম build-এ একাধিক variable একসঙ্গে বদলানো না চাইলে 2048 রেখে rollback fix করুন।

Test: Long thinking with 2 tools; no lost history; no budget bypass; deadline stop; no tool duplication। Yield disabled হলে model turn bounded থাকে কি না।
৫২. Stage-2 class ranking আগেই আছে
এটি নতুন fix নয়। stage2_class() যথাযথভাবে reuse করুন।
priority = stage2_class(task_sessions)



Remove old resume0 paths for E15B; retain class ordering for fresh start1 jobs।

Test: Class 0,1,2,3 order; no resume0; accepted tasks only last when D2 enabled।
৫৩. Preemption/retraction নতুন 16-stream-এ মাপা
Older 16-slot precedent-এ long context-এর জন্য preemption বেড়েছিল। E15B context কিছুটা ছোট হলেও 16-stream safe নিশ্চিত নয়।
if report["retractions"] > 0:    report["capacity_warning"] = "retractions observed"



Smoke GO criterion হিসেবে zero retractions রাখা যায়।

Test: Simulated retract log → flagged; only config string retraction_policy → not counted।
৫৪. Deadline-এ in-flight candidate harvest
Deadline কাটলে নতুন model response নিষিদ্ধ। কিন্তু already retained program-এর host verification সুযোগ থাকলে সেটি bounded finalization-এ ব্যবহার করা যেতে পারে।
def finalize_attempt(session, verify_deadline):    left = verify_deadline - time.monotonic()    if left <= 0:        return {"skipped": "verification deadline"}    return bounded_existing_host_harvest(        session,        max_seconds=min(left, 30),    )



bounded_existing_host_harvest existing verifier ব্যবহার করে implement করতে হবে; নতুন untrusted function execution host-এ নয়।

Test: Request cancelled কিন্তু retained transform আছে → verified before freeze; no transform → immediate exit; late straggler callback frozen final record overwrite করতে পারবে না।
৫৫. Notebook size, traces ও facts cap
এই অংশ deployment checklist।

Notebook lightweight:
notebook_bytes = Path(    NOTEBOOK_PATH).stat().st_sizeassert notebook_bytes < 400_000



এটি আপনার নিজস্ব target, Kaggle-এর universally imposed 400KB limit নয়।

Facts view:
facts = arc2_facts_view(task)assert len(facts) <= configured_max_chars



Actual formatter যদি intentional truncation করে, assert করার বদলে:
report["facts_truncated"] = (    original_facts_chars > len(facts))



বড় ARC grids-এ 3,600-character / 800-token cap কার্যকর problem information হারাচ্ছে কি না পরীক্ষা করতে হবে। Full grid দেখার জন্য existing trusted show() helper থাকবে।

Full traces: Request ID, generated reasoning, tool calls, check results ও finish reason সংরক্ষণ করুন; API secrets/log headers বাদ দিয়ে archive করুন।

Tests: Large-grid facts truncation honest label; tokenizer estimate; complete 240-task report; output archive writable; no secret leakage।
৫৬. এখন সব patches কোন ক্রমে implement করবেন?
নম্বর ১–৫৫ তালিকার reference number। কিন্তু code লিখতে হবে dependency অনুযায়ী। সব একসঙ্গে লিখলে regression-এর উৎস ধরা কঠিন হবে।
Stage 1
নতুন E15B copy ও configuration

পুরোনো pushed notebooks freeze। আলাদা builder ও config। আগে #36–38, #41, #43, #46–47।



Stage 2
Attempt scheduler

#1–6, #19, #45, #49, #52। First-stage completion barrier, no resume, 35k cap।



Stage 3
Tools ও candidate integrity

#7–14, #21–23, #48, #50–51, #54। এটি regression-sensitive অংশ।



Stage 4
Reliability ও reporting

#15–18, #24–35, #39–42, #53, #55।



Stage 5
CPU validation ও GPU approval

#43, তারপর #44। CPU results এবং exact SHA যাচাই ছাড়া GPU নয়।



৫৭. select_attempts_b()-এর ব্যবহারযোগ্য implementation
#7-এ দেখানো selection policy-টি সম্পূর্ণ করার জন্য নিচের function নতুন E15B-specific selector-এর starting implementation হতে পারে। Existing select_attempts_r5() অপরিবর্তিত থাকবে।
def select_attempts_b(task, records):    by_branch = {        r.get("branch"): r        for r in records    }    def usable_outputs(record, qi, accepted_only=True):        if record is None:            return []        submissions = sorted(            record.get("submissions") or [],            key=lambda x: x.get("t", 0),            reverse=True,        )        checks = sorted(            record.get("checked") or [],            key=lambda x: x.get("t", 0),            reverse=True,        )        # Explicit accepted submission before check-only.        entries = submissions + checks        result = []        for entry in entries:            if accepted_only and not entry.get("accepted"):                continue            if not accepted_only and entry.get("accepted"):                continue            outputs = entry.get("test_outputs") or []            if qi >= len(outputs):                continue            grid = outputs[qi]



এই function ব্যবহারের আগে দুটি refinement প্রয়োজন: প্রতিটি branch-এর best accepted candidate আগের Rev 5 policy অনুযায়ী (accepted submission আগে, না হলে accepted check) নির্ধারণ করতে হবে; তারপর accepted alternates। উপরের code সব accepted submissions প্রথমে বসায়, ফলে একই b0-এর দুটি accepted alternate থাকলে b1-এর output দ্বিতীয় slot না-ও পেতে পারে। তাই final version-এ priority হবে best_b0 → best_b1 → other accepted → partial → fallback।

এটি final integration test ছাড়া production-ready ঘোষণা করছি না। বিশেষত multi-test-input task ও b0/b1 output diversity যাচাই প্রয়োজন।
৫৮. একটি কেন্দ্রীয় E15B configuration
বিভিন্ন ফাইলে conflicting constant এড়াতে একটি config ব্যবহার করুন।
E15B_CONFIG = {    "variant": "E15B",    "branches": 2,    "b1_view": "facts",    "effort": "xhigh",    "max_generated_tokens": 35_000,    "commit_reserve_tokens": 3_000,    "first_exposure_tokens": None,    "fair_share": False,    "min_admit_sec": 300,    "context_window": 100_000,    "request_timeout_sec": 3000,    "stream_candidates": [16, 14, 12],    "kv_admit_limit": 0.95,    "e16_nudge": False,    "end_harvest": True,    "attempt2_accepted_last": True,    "preserve_baseline_submission": True,}



Notebook builder-এ exact values embed করুন; runtime-এর environment overrides থাকলে resolved config ও checksum রিপোর্টে লিখুন।
Integration regression gate
Test group	Mandatory result
Two-attempt lifecycle	Independent histories, two real attempts
Scheduler	All b0 complete before b1; no scheduler resume
Budget	35k total, 3k commit reserve
History	No executed tool result lost
Tool contract	submit alias safe, python primary
Verification	Clean host verifier authoritative
Context	No unexpected trim; no overflow
Server	16→14→12 fallback works
Submission	Initial fallback + final atomic output
Reporting	True effective streams, effort, candidate provenance
Notebook	Exact SHA, compiled cells, embedded source parity
Kaggle CPU kit	PASS before any GPU experiment


আমার চূড়ান্ত নির্দেশনা
Implementation GO: E15B build-এর কাজ শুরু করা যুক্তিসংগত। তবে সবচেয়ে আগে #45–49-এর scheduling/budget corrections এবং #10–11-এর tool correctness ঠিক করুন। এগুলো বাদ দিয়ে শুধু 16 streams ও 2×35k চালালে পুরোনো সমস্যাই ফিরে আসতে পারে।

Engine GO এখনও conditional: 16 streams-এর +12% speed একটি projection, measurement নয়। Nodelay + graphs17 + Mamba68-এর short GPU smoke পাস করলে 16 গ্রহণ করবেন; অন্যথায় 14 বা 12।

Accuracy GO এখনও conditional: আগের historical data-তে বহু সঠিক solution 35k-এর পরে এসেছে। তাই new B-এর প্রথম GPU test-এ প্রথম attempt বনাম দ্বিতীয় attempt-এর incremental correct outputs, 35k cutoff losses, এবং candidate production বাধ্যতামূলকভাবে মাপতে হবে।

D2 YES, D3 OFF, D4 KEEP, D6 KEEP, D7 smoke→EVAL120, D9 try16→14→12—এই সিদ্ধান্তগুলোই recommend করছি।

এখানে ১–৫৫ পর্যন্ত সব items-এর implementation guidance, code patterns ও tests দেওয়া হলো। কোনো source file, builder, notebook, GitHub commit বা Kaggle submission পরিবর্তন করিনি।
