"

<pasted_content id="1891">
Q27-AGENT v1.8.1 — Deep Pre-GPU Audit
১২ অক্টোবর ২০২৬ | README-first | ARC-AGI-2 | Qwen3.8-27B-FP8
১. প্রধান পর্যবেক্ষণ
v1.6-এর পর আপনারা যথেষ্ট গুরুত্বপূর্ণ engineering পরিবর্তন করেছেন। বিশেষ করে আগের GPU crash দুটির কারণ শনাক্ত করে v1.8.1-এ সংশোধন করা হয়েছে।
সর্বশেষ DIRECT public LB
26.94%
v1.4.4 — reported result
Real AGENT answers
0
এখনো সফল GPU inference নেই


আগের দুই GPU failure-এর forensic verdict
Version	কী হয়েছিল	v1.8.1-এ অবস্থা
v1.7	KeyError: 'base_url' — rescue smoke	direct_client()-এ default দেওয়া হয়েছে
v1.8	Server চালু, কিন্তু readiness poll-এ UnboundLocalError	Local import সরানো হয়েছে এবং negative regression test যোগ হয়েছে
v1.8.1	Real GPU answer এখনো নেই	Local fake-server rehearsal PASS বলে bundled evidence-এ রয়েছে
গুরুত্বপূর্ণ পার্থক্য: v1.8.1-এর rehearsal startup, request parsing, tool execution ও evaluation reporting পরীক্ষা করে। কিন্তু real Qwen model-এর reasoning, token throughput, tool-call reliability কিংবা generalization এখনো প্রমাণ করে না।
অডিটে একটি নতুন দুর্বল test assertion-ও শনাক্ত হয়েছে: hidden_path_rescue_ran যাচাইয়ে nlines(...) >= 0 ব্যবহার করা হয়েছে। এটি খালি বা অনুপস্থিত log-এর ক্ষেত্রেও true হতে পারে। অর্থাৎ DRYRUN-এর সব PASS হলেও ওই নির্দিষ্ট test থেকে rescue execution নিশ্চিত হয় না।
২. Fresh test results
আমি ZIP-এর source থেকে notebook modules বের করে নিজস্ব Linux environment-এ পরীক্ষা করেছি।
Test	ফল	প্রমাণের সীমা
test_agent_local.py	PASS	Mock agent ও evaluation logic
test_submit_local.py	PASS	Hidden submission update path
test_v16_regress.py	PASS	আগের regression checks
test_gpu_path_local.py	PASS	Fake vLLM দিয়ে startup, smoke, e2e, reporting এবং পুরোনো crash-এর negative control
test_v18_local.py	FAIL	Windows-specific msvcrt dependency Linux-এ চলে না
Builder reproducibility	PASS	Rebuilt notebook byte-identical
Rebuilt notebook-এর SHA-256:
6d478a4501bbdca39ad21948bff75d3e307cab50e952049094008a4c81faf9d1
Kaggle CPU DRYRUN-এর সংরক্ষিত রিপোর্টে ২৭/২৭ PASS রয়েছে। সেই পরীক্ষাকে আমি আলাদা Kaggle machine-এ পুনরায় চালাইনি।
৩. নতুন HIGH সমস্যা — Agent e2e ব্যর্থ হলেও full evaluation চলে
Evidence: Reproduced locally · submit_main.py, main()
এটি আমার পাওয়া সবচেয়ে গুরুত্বপূর্ণ নতুন GPU-path সমস্যা।
বর্তমান code-এ:
elif not e2e["ok"]:    print(        "WARNING: agent end-to-end probe failed "        "on every effort; running with the configured effort"    )



অর্থাৎ real Qwen model xhigh, medium ও low—কোনো configuration-এই python → submit roundtrip করতে না পারলেও program AGENT evaluation চালিয়ে দেয়।
আমি agent_e2e()-কে controlled failure দিয়ে replace করে মূল submit_main.main() চালিয়েছি। ফলাফল:
WARNING: agent end-to-end probe failed on every effort;
running with the configured effort

E2E FAILURE FOLLOWED BY EVAL:
[('ran_eval', False, True)]


কেন এটি গুরুত্বপূর্ণ?
AGENT এখনো real GPU-তে একটিও solve করেনি। যদি e2e পরীক্ষাই ব্যর্থ হয়, তাহলে পরবর্তী ১.৫ GPU-hour AGENT evaluation চালানোর যথেষ্ট ভিত্তি নেই।
Fix: EVAL12-এর ক্ষেত্রে e2e সম্পূর্ণ ব্যর্থ হলে structured AGENT_E2E_NO_GO রিপোর্ট দিয়ে agent evaluation বন্ধ করুন। Hidden rerun-এর ক্ষেত্রে DIRECT rescue healthy থাকলে AGENT বাদ দিয়ে DIRECT-কে পুরো অবশিষ্ট সময় দিন।
উদাহরণস্বরূপ submit_main.main()-এর e2e section-এর পরে:
if report.get("agent_e2e", {}).get("ok") is False:    report["agent_gate"] = "AGENT_E2E_NO_GO"    REPORT_PATH.write_text(        json.dumps(report, indent=1, default=str)    )    if cfg.get("eval"):        print(            "EVAL NO-GO: no working agent tool roundtrip",            flush=True        )        return    # Competition rerun: preserve the DIRECT rescue.    # run_submit needs an explicit agent_disabled branch.    if cfg.get("rescue_smoke_ok") is True:        cfg["agent_disabled"] = True    else:        print(            "Both agent e2e and rescue smoke failed",            flush=True        )        return



এটি সম্পূর্ণ fix-এর প্রথম অংশ। run_submit.main()-এ agent_disabled=True হলে solve_repl() বাদ দিয়ে সরাসরি run_rescue() চালাতে হবে, প্রতিটি result আগের মতো atomic submission update-সহ। শুধু agent_frac=0 করলেই যথেষ্ট নয়।
Validation test: E2E failure inject করুন; EVAL12-এ run_eval.main() call count শূন্য হবে। Hidden rerun-এ solve_repl() call count শূন্য, কিন্তু DIRECT rescue চালিয়ে valid submission.json তৈরি হবে।
Verdict: গুরুত্বপূর্ণ reliability fix। এটি না থাকলে আবার একটি GPU run AGENT-এর কার্যকারিতা সম্পর্কে কোনো তথ্য না দিয়েই শেষ হতে পারে।
৪. HIGH — AGENT-এর thinking fallback DIRECT-কে নষ্ট করতে পারে
Evidence: Reproduced · submit_main.py + rescue.py
submit_main.py-তে AGENT thinking smoke ব্যর্থ হলে:
cfg["thinking"] = Falsecfg["thinking_token_budget"] = None



কিন্তু rescue.direct_client()-ও একই global cfg["thinking"] ব্যবহার করে।
আমি direct_client()-এ দুই configuration দিয়ে পরীক্ষা করেছি:
AGENT setting	DIRECT thinking	DIRECT thinking budget
thinking=True	Enabled	49,152
thinking=False	Disabled	None
ফলে AGENT-এর tool-thinking path কাজ না করলেও DIRECT-এর independent reasoning path কার্যকর থাকতে পারে—কিন্তু বর্তমান code তাকে বন্ধ করে দেয়।
এটি বিশেষভাবে গুরুত্বপূর্ণ, কারণ DIRECT-এর 26.94 public LB xhigh long-reasoning pipeline থেকে এসেছে। সেই method-এর সবচেয়ে প্রয়োজনীয় অংশটাই fallback-এ হারিয়ে যেতে পারে।
Fix: দুই solver-এর configuration আলাদা করুন।
# rescue.pydef direct_client(cfg):    think = int(cfg.get("rescue_think", 49152))    # Independent from the agent's thinking fallback.    use_think = bool(cfg.get("rescue_thinking", True))    return ChatClient(        cfg.get("base_url", "http://127.0.0.1:1234/v1"),        cfg.get("model", "qwen"),        max_tokens=think + int(            cfg.get("rescue_answer", 8192)        ),        temperature=cfg.get("rescue_temperature", 0.6),        top_p=0.95,        top_k=20,        thinking=use_think,        thinking_token_budget=(            think if use_think else None        ),        template_kwargs={            "reasoning_effort":                cfg.get("rescue_effort", "xhigh")        },    )



এর সঙ্গে DIRECT-এর independent smoke result দেখে fallback নির্ধারণ করতে হবে। rescue_thinking=True রাখলেই যদি server budget reject করে, তখনও DIRECT ঠিক হয়েছে বলা যাবে না।
Validation: AGENT thinking_ok=False, DIRECT probe সফল—এই case-এ DIRECT request-এ enable_thinking=True, thinking_token_budget=49152 থাকতে হবে।

৫. MEDIUM–HIGH — Recorded DIRECT comparison অকারণে বন্ধ হতে পারে
Evidence: Reproduced · submit_main.py
EVAL12-এ eval_direct="ref" থাকলে DIRECT-কে GPU-তে চালানো হয় না। আগের সংরক্ষিত DIRECT answers load করা হয়।
কিন্তু rescue-parameter smoke ব্যর্থ হলে:
cfg["rescue"] = False



এটি recorded DIRECT loading-ও বন্ধ করে দেয়।
আমি এই case reproduce করেছি:
WARNING: DIRECT rescue requests rejected -> rescue disabled

EVAL configuration:
rescue=False
eval_direct='ref'
rescue_smoke_ok=False


অর্থাৎ DIRECT-এর HTTP request shape ব্যর্থ হলেও, disk-এ থাকা recorded baseline ব্যবহার করা সম্ভব ছিল। সেটিও হারিয়ে যাচ্ছে।
Fix:
live_rescue_required = (    cfg.get("rescue")    and (        not cfg.get("eval")        or cfg.get("eval_direct") == "live"    ))if (    live_rescue_required    and chosen_sm.get("rescue_params_ok") is False):    cfg["rescue"] = False    report["rescue_disabled"] = (        "live rescue request rejected"    )



Recorded EVAL mode-এ চাইলে rescue smoke informational হিসেবে রাখা যায়, কিন্তু সেটির ফল দিয়ে recorded reference বন্ধ করা যাবে না।
Validation: eval_direct="ref" এবং failed rescue smoke inject করে দেখুন run_eval এখনো run7 reference load করছে এবং three-way comparison সম্পন্ন হচ্ছে।

৬. আরও চারটি সমস্যা ও নির্দিষ্ট সমাধান
Priority	সমস্যা	Verdict	GPU score-এর সম্ভাব্য প্রভাব
HIGH	Hidden rescue-এর আগে AGENT failure fallback অসম্পূর্ণ	Confirmed from code	বড় সময় অপচয় ঠেকাতে পারে
MEDIUM	Retained function-এর documentation ভুল নির্দেশ দেয়	Confirmed + tested	Python program verification
MEDIUM	DRYRUN-এর rescue assertion কার্যত অকেজো	Confirmed	False PASS ঠেকানো
MEDIUM	DIRECT-এর নতুন runtime policy পুরোপুরি পরীক্ষা হয়নি	Unmeasured	Accuracy ও coverage
6.1 Hidden run-এ AGENT-disabled branch যোগ করা
আগের e2e failure patch-এর বাকি অংশ run_submit.py-তে বসাতে হবে।
refresh() function-এর পরে এবং solve_repl() call-এর আগে:
# No functional agent tool roundtrip:# preserve the entire remaining budget for DIRECT.if cfg.get("agent_disabled", False):    print(        "AGENT disabled by e2e gate. "        "Running DIRECT rescue on all tasks.",        flush=True    )    def rescue_done(tid, rec):        with lock:            direct_recs[tid] = dict(rec)            refresh(tid)    run_rescue(        list(tasks),        tasks,        cfg,        deadline,        log=lambda m: print(m, flush=True),        on_result=rescue_done,    )    return



এর জন্য run_submit.py-তে lock, done, direct_recs, tagged() ও refresh() আগের মতো initialize করা থাকতে হবে।
Validation: Agent e2e fail → agent requests শূন্য → DIRECT answers generated → file checkpoint প্রতিটি completed task-এর পরে update → final submission valid।
এই fix-এর generation-token overhead নেই। বরং AGENT-এ অকারণে কয়েক ঘণ্টা নষ্ট হওয়া আটকাতে পারে।
6.2 Retained function-এর instructions পরস্পরবিরোধী
repl.py-তে RETAIN_DOC model-কে বলছে:
A kept function may use builtins, the preloaded names...

কিন্তু repl_agent.verify() tool-only global ব্যবহার করা function reject করে।
আমি পরীক্ষা করেছি:
tool_only_globals(    "def transform(grid):\n"    "    return train_inputs[0]")



ফল:
{'train_inputs'}


অর্থাৎ model-কে যে কাজ করার অনুমতি দেওয়া হচ্ছে বলে মনে হচ্ছে, submit-এ সেটিই নিষিদ্ধ।
Fix: RETAIN_DOC-এ স্পষ্ট লিখুন, exploratory python call-এ train_inputs, train_outputs, test_inputs, train ব্যবহার করা যাবে; কিন্তু submit-এ ব্যবহৃত retained transform() বা তার dependencies-এ এই tool-only globals চলবে না। Program must be self-contained and input-dependent.
Validation: নতুন tool description-এ এই নিষেধাজ্ঞা উপস্থিত আছে কি না পরীক্ষা, তারপর actual agent trace-এ nonportable global rejection count মাপুন।
6.3 DRYRUN-এর ভুল assertion
cell_18.py-তে:
ck(    "hidden_path_rescue_ran",    nlines(W / "dryrun" / "direct_attempts.jsonl") >= 0,    "see the log")



nlines() exception হলে শূন্য দেয়। ফলে log অনুপস্থিত হলেও 0 >= 0 true।
Fix:
rescue_log = W / "dryrun" / "direct_attempts.jsonl"ck(    "hidden_path_rescue_ran",    rescue_log.is_file()    and nlines(rescue_log) > 0,    {        "exists": rescue_log.is_file(),        "records": nlines(rescue_log),    })



Validation: Missing, empty এবং one-record JSONL—তিনটি case-এ যথাক্রমে FAIL, FAIL, PASS।
এই bug সত্যিকারের Kaggle DRYRUN-এর ২৭টি check সব ভুল প্রমাণ করে না। তবে ২৭/২৭ সংখ্যাটি প্রতিটি claim সমান শক্তভাবে যাচাই করেছে, এমন বলা যাবে না।
6.4 Recorded DIRECT reproducibility
আমি ZIP-এর reference JSON, EVAL solutions এবং দুই রানের records.jsonl মিলিয়েছি।
Reference	Correct test outputs	Match
DIRECT run 4	2/22 = 9.09%	মিলে গেছে
DIRECT run 7	3/22 = 13.64%	মিলে গেছে
এই অংশ নির্ভরযোগ্য।
তবে v1.8.1-এর live DIRECT rescue, DIRECT v1.4.4-এর একই accuracy দেবে, সেটি এখনো প্রমাণ হয়নি। Prompt ও parser একই হলেও forced-close mechanism এবং inference serving configuration পৃথক।
তাই eval_direct="ref" থেকে পাওয়া 13.64% হলো historical reference result, live v1.8.1 rescue-এর independently reproduced score নয়।
EVAL report-এ direct_source-এর LB-18.75 label-ও সংশোধন করা উচিত; run7 হলো 26.94 pipeline।

৭. Rerun split: সবচেয়ে গুরুত্বপূর্ণ design সিদ্ধান্ত
আমার মতে, বর্তমান 50% AGENT / 50% DIRECT split এখনই freeze করা ঠিক হবে না।
কারণ DIRECT-এর public LB 26.94 প্রমাণিত, কিন্তু AGENT-এর real model success rate এখনো অজানা।
হিসাব
DIRECT run7-এর measured scheduler result:
Throughput274.6
generated tok/s


Generation1.06M
tokens / EVAL12


Correct3/22
exact test outputs



DIRECT run7-এর measured generation-only সময় প্রায় 64.3 মিনিট। Setup, request scheduling ও অন্যান্য overhead আলাদা।
বর্তমান HYBRID যদি প্রথম প্রায় অর্ধেক inference window AGENT-কে দেয়, তাহলে DIRECT-এর জন্য অবশিষ্ট সময় যথেষ্ট হবে কি না সেটি নির্ভর করবে AGENT-এর actual throughput এবং কতগুলো task DIRECT থেকে বাদ যাচ্ছে তার ওপর।
সঠিক comparison metric
শুধু agent verified programs per GPU-hour যথেষ্ট নয়।
প্রধান metric হওয়া উচিত:
\[
\frac{\text{HYBRID correct}-\text{DIRECT baseline correct}}
{\text{additional or displaced GPU-hours}}
\]

অর্থাৎ net exact-output gain per GPU-hour।
কারণ train-verified program ভুলও হতে পারে। AGENT যদি এমন task-কে verified ঘোষণা করে, DIRECT rescue সেটি skip করে এবং correctness হারানোর সম্ভাবনা থাকে।
EVAL12 report-এ তাই পাশাপাশি রাখতে হবে:
Metric	উদ্দেশ্য
AGENT-only hits	Agent-এর নিজস্ব solve rate
DIRECT reference hits	প্রমাণিত baseline
AGENT-only unique hits	DIRECT-এর বাইরে নতুন correctness
DIRECT-only hits	Agent যে correctness দিতে পারেনি
Verified-but-wrong	False confidence / generalization failure
Deployed HYBRID hits	প্রকৃত merge-policy score
Estimated displaced DIRECT capacity	Agent চালানোর opportunity cost
Kaggle-এর scoring-এ প্রতি test output-এর দুটি candidate-এর যেকোনো একটি exact correct হলে point পাওয়া যায়। তাই task count নয়, test-output-level hit count ব্যবহার করতে হবে। 

Kaggle



DIRECT-first না AGENT-first?
বর্তমান প্রমাণে DIRECT-first বেশি conservative। কিন্তু পুরো ১২ ঘণ্টা DIRECT শেষ করে শুধু leftover-এ AGENT চালানোর policy-ও সেরা হবে—এমন নিশ্চিত প্রমাণ নেই।
আমি তিনটি scheduling policy তুলনা করব:
Policy	সুবিধা	ঝুঁকি
AGENT-first, then DIRECT	Agent দ্রুত verified task বাদ দিতে পারে	Agent সময় নষ্ট করলে DIRECT coverage কমে
DIRECT-first, then AGENT	26.94 method-এর coverage আগে সুরক্ষিত	Agent-এর জন্য সময় কম থাকতে পারে
Adaptive two-stage	Score gain ও coverage অনুযায়ী সময় বরাদ্দ	Scheduler design জটিল
আমার পছন্দ: Hybrid adaptive scheduler, যেখানে প্রথমে coverage guarantee এবং পরে high-value refinement। তবে বাস্তব agent score না পাওয়া পর্যন্ত এটি design recommendation, proven best policy নয়।

৮. EVAL12 GPU run-এ কোন ফল দেখে সিদ্ধান্ত নেব?
v1.8.1-এ ১.৫ ঘণ্টার EVAL12 diagnostic চালানো যুক্তিসংগত হবে preflight fixes-এর পরে।
Suggested decision gates
Startup gate	Server ready, smoke PASS, real e2e PASS, agent budget retained
Execution gate	Actual ARC tasks-এ tool calls, submits, cuts ও errors দেখা যাচ্ছে
Correctness gate	Agent-only exact hits এবং verified-but-wrong আলাদা গণনা
Efficiency gate	Measured tokens/s, GPU-minutes, attempts completed, second-pass coverage
Hybrid gate	3/22 DIRECT reference-এর ওপর net gain, এবং কোনো DIRECT-correct output হারিয়েছে কি না

২২টি test output খুব ছোট sample। তাই +১ hit মানেই architecture নিশ্চিতভাবে ভালো নয়। ১টি hit এখানে ৪.৫৫ percentage points। এটি engineering gate হিসেবে ব্যবহার করবেন; hidden LB prediction হিসেবে নয়।
বিশেষভাবে, AGENT যদি ২ বা ৩টি train-verified program দেয়, কিন্তু test-এ সব ভুল হয়, তাহলে verified count দেখেই AGENT-এ আরও compute বরাদ্দ দেওয়া উচিত হবে না।
৯. Residual risks
Security guard আগের তুলনায় শক্তিশালী, কিন্তু Python audit hook kernel-enforced sandbox নয়। Kaggle-এ unshare permission denied হওয়ায় এই সীমা আরও গুরুত্বপূর্ণ। Evaluation solutions inference environment-এ উপস্থিত থাকলে contamination prevention শুধু regex বা audit hook-এর ওপর নির্ভর করা নিরাপদ নয়। Public EVAL correctness scoring-কে generated programs-এর code-access audit থেকে আলাদা রাখতে হবে।
এছাড়া আরও তিনটি বিষয় live GPU data ছাড়া সিদ্ধান্তযোগ্য নয়: Qwen-এর real xhigh tool-call reliability, 6,144-token thinking limit-এর effect, এবং MTP/FP8 configuration-এর actual throughput। Fake vLLM-এর token accounting দিয়ে এসব extrapolate করবেন না।

১০. চূড়ান্ত GO / NO-GO
Current v1.8.1, unchanged
NO-GO

E2E-failure gate ও DIRECT thinking-coupling বিশেষ করে ঠিক করা উচিত। এগুলো CPU-তে reproducible; আবার GPU খরচ করে আবিষ্কার করার দরকার নেই।

EVAL12, after targeted patches
CONDITIONAL GO

Targeted fixes, full local regression এবং real configuration preflight পাস করলে ১.৫ ঘণ্টার AGENT diagnostic run সমর্থনযোগ্য।

12-hour hidden rerun
NOT READY

AGENT-এর real accuracy, throughput, verified-but-wrong rate ও DIRECT displacement না মেপে 50/50 split freeze করা যাবে না।

Priority অনুযায়ী কাজ: প্রথমে e2e failure gate ও hidden DIRECT fallback, দ্বিতীয়ত AGENT/DIRECT thinking আলাদা করা, তৃতীয়ত recorded-EVAL rescue handling, এরপর test assertions ও documentation সংশোধন।
সবকিছু builder source-এ বসিয়ে notebook rebuild করতে হবে এবং SHA-256 পুনরায় যাচাই করতে হবে। বর্তমান builder byte-identical output দিচ্ছে—এটি একটি শক্তিশালী দিক।
সবচেয়ে গুরুত্বপূর্ণ ফলাফল: v1.8.1 আগের GPU startup failures মোকাবিলায় বাস্তব অগ্রগতি করেছে। এখন প্রয়োজন first successful Qwen inference ও measurable AGENT correctness। সেই প্রমাণ ছাড়া ৬০%-এর পরিকল্পনায় AGENT-কে কত ঘণ্টা দেওয়া উচিত, সেটি নির্ভরযোগ্যভাবে নির্ধারণ করা যায় না।
</pasted_content id="1891">

"