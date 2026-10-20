"

<pasted_content id="1891">
27-AGENT v1.6 — Final Pre-GPU Audit
১১ অক্টোবর ২০২৬ | README-first | CPU verification + code audit
প্রাথমিক সিদ্ধান্ত: বর্তমান v1.6-কে অপরিবর্তিত অবস্থায় EVAL12 GPU-তে চালানোর অনুমোদন দিচ্ছি না। কিছু গুরুত্বপূর্ণ সমস্যা রয়েছে যেগুলো কম খরচে CPU-তেই ঠিক করা যায়। তবে architecture বাতিল করার প্রয়োজন নেই।
১. কী যাচাই করেছি
ZIP-এর README, v1.5→v1.6 diffs, notebook থেকে extracted modules, builder এবং tests পরীক্ষা করেছি। বর্তমান notebook-এর SHA-256:
956b6437ba5652f7fe6efa182d823b1b9d54fec1c1519b5d36ebd252f0c3d5be
Fresh CPU verification
v1.6 local tests
PASS

v1.5 regression suite on v1.6
PASS

Agent local test
PASS

Submission-path test
PASS


এগুলো বাস্তবে পুনরায় চালিয়ে পাওয়া ফলাফল। তবে CPU mock test পাস করা আর বাস্তবে Qwen3.8-27B দিয়ে কাজ করা এক নয়। v1.6-এর real GPU inference এখনো unverified।
২. নিশ্চিত হওয়া প্রথম গুরুত্বপূর্ণ সমস্যা
HIGH — verified_but_wrong_tasks রিপোর্ট ভুলভাবে গণনা হচ্ছে
Status: Confirmed from code · run_eval.py, প্রায় line 137
বর্তমান code:
"verified_but_wrong_tasks": [    t for t in verified_tasks    if per.get(t, 0) < 1]



সমস্যা: per হচ্ছে HYBRID submission-এর score। অর্থাৎ AGENT ভুল test grid দিলেও DIRECT rescue সঠিক উত্তর দিলে task-টি এখানে ভুল হিসেবে ধরা হবে না।
উদাহরণ:
Candidate	Test correctness
AGENT verified program	Wrong
DIRECT rescue	Correct
HYBRID	Correct
বর্তমান verified_but_wrong_tasks	ভুলটি শনাক্ত করবে না
ফলে AGENT generalization কতটা দুর্বল, সেটি মাপার প্রধান diagnostic-ই misleading হবে।
Fix: AGENT-only submission-এর score আলাদা গণনা করে এই field তৈরি করতে হবে।
_, _, agent_per = score(agent_sub, sol_ids)verified_but_wrong_tasks = [    t for t in verified_tasks    if agent_per.get(t, 0) < 1]



আরও ভালো হবে verified_but_wrong_queries রাখা, যাতে একাধিক test input-যুক্ত task-এ কোন output ভুল হয়েছে বোঝা যায়।
Validation: Mock AGENT-কে train-verified কিন্তু wrong test output দিন; DIRECT-কে correct output দিন। Expected: HYBRID correct, কিন্তু verified-but-wrong diagnostic-এ task থাকবে।
এই bug leaderboard submission বদলায় না, তবে EVAL12-এর সিদ্ধান্ত ভুল পথে নিতে পারে।
BLOCKER — EVAL12-এর HYBRID score বাস্তবে deploy করা policy-এর score নয়
Status: Reproduced · run_eval.py, run_submit.py, rescue.py
এটি প্রথম সমস্যার চেয়েও বেশি গুরুত্বপূর্ণ।
- EVAL12-এ rescue_all=True, তাই verified task-এও DIRECT candidate যোগ হয়।
- Hidden rerun-এ verified task-এ DIRECT rescue চলে না।
- কিন্তু EVAL12-এর official_score_pct-কে deployed HYBRID score হিসেবে রিপোর্ট করা হচ্ছে।
আমি একটি ছোট counterexample চালিয়েছি:
	EVAL12	Hidden policy
AGENT train-verified prediction	ভুল	ভুল
DIRECT prediction	সঠিক	চালানো হয় না
দুই attempt-এর ফল	সঠিক	ভুল
একই candidates ব্যবহার করে local reproduction-এ EVAL policy ১/১ এবং hidden policy ০/১ পেয়েছি।
অর্থাৎ বর্তমান EVAL12 score hidden policy-কে অতিরিক্ত ভালো দেখাতে পারে।
সুনির্দিষ্ট সমাধান
run_eval.py-তে all-task DIRECT rescue চালু থাকুক—এটি complementarity মাপার জন্য মূল্যবান। কিন্তু দুটি আলাদা HYBRID submission তৈরি করুন:
# Counterfactual: agent + DIRECT on every task.all_rescued_sub = hybrid_submission(    answers, rescue_recs, tasks)# Actual competition policy:# DIRECT only when the agent did not verify.deployed_rescue = {    tid: rec    for tid, rec in rescue_recs.items()    if tid not in verified0}deployed_sub = hybrid_submission(    answers, deployed_rescue, tasks)# This must be the primary EVAL score.sub = deployed_sub# Both scores remain available for diagnosis.deployed_hits = hits(deployed_sub, sol_ids, ids)all_rescued_hits = hits(all_rescued_sub, sol_ids, ids)deployed_pct = (    100.0 * sum(deployed_hits.values())    / max(len(deployed_hits), 1))all_rescued_pct = (    100.0 * sum(all_rescued_hits.values())    / max(len(all_rescued_hits), 1))



তারপর report-এ official_score_pct = deployed_pct এবং অতিরিক্ত hybrid_all_rescued_pct = all_rescued_pct রাখুন। Existing agent_only_official_pct ও direct_only_official_pct অপরিবর্তিত থাকতে পারে।
এই patch-এ আরেকটি বিষয় নিশ্চিত করতে হবে: eval_submission.json যেন deployed_sub রাখে, এবং optional eval_all_rescued_submission.json-এ counterfactual থাকে।
Validation: Verified task-এ AGENT ভুল, DIRECT সঠিক—এমন mock test চালিয়ে নিশ্চিত করুন:
- official_score_pct = ০%
- hybrid_all_rescued_pct = ১০০%
- verified_but_wrong_tasks-এ task উপস্থিত
বর্তমান test_v16_local.py-এর mock-এ verified task-এর AGENT-ই সঠিক, তাই পরীক্ষাটি এই bug ধরতে পারেনি।
৩. Isolation Audit — P0-B এখনো সম্পূর্ণ সমাধান হয়নি
Status: Reproduced · sandbox.py এবং repl.py
v1.6-এর audit hook সাধারণ file access আটকাচ্ছে—এটি ভালো উন্নতি। কিন্তু এটিকে README-তে fail-closed isolation বলা ঠিক নয়।
আমি একটি অস্থায়ী, নিরীহ sentinel file তৈরি করে পরীক্ষা করেছি। কোনো competition solution file পড়িনি।
Local experiment	ফলাফল
Python open() দিয়ে নিষিদ্ধ path পড়া	Blocked
pathlib ও os.open()	Existing tests PASS
Native library-এর file API দিয়ে একই sentinel পড়া	Bypassed
Python tool-এ একই native-access পরীক্ষা	Bypassed
Native-access পরীক্ষায় sandbox sentinel থেকে পাওয়া তথ্যের ভিত্তিতে [[8]] output তৈরি করেছে। অর্থাৎ model-written code ওই তথ্য ব্যবহার করতে সক্ষম।
এটি শুধু theoretical নয়, reproduced escape।
Python-এর official documentation-ও জানায়, sys.addaudithook() secure sandbox তৈরির উপযুক্ত ব্যবস্থা নয়; বিশেষ করে ctypes-এর মতো native-memory interfaces ঝুঁকিপূর্ণ। 

Python 3.12.15 documentation
+1



আরেকটি code problem:
try:    ...    sys.addaudithook(_q27_hook)except Exception:    pass



Hook installation ব্যর্থ হলেও execution চলতে থাকবে। এটি আক্ষরিক অর্থেই fail-open।
সমাধান
প্রথমত, hook initialization failure-এ child execution বন্ধ করতে হবে এবং installation-এর পরে harmless denied-path canary audit চালিয়ে hook সক্রিয় কি না পরীক্ষা করতে হবে।
দ্বিতীয়ত, arbitrary Python execution-এর জন্য সম্ভব হলে kernel-enforced restriction ব্যবহার করতে হবে—rootless filesystem namespace, seccomp-based restriction অথবা আগে থেকে উপলব্ধ isolated runner। Kaggle environment-এ এগুলো কার্যকর কি না CPU preflight-এ পরীক্ষা করতে হবে; available ধরে নেওয়া যাবে না।
তৃতীয়ত, এর বিকল্প না পাওয়া গেলে generated code-এর জন্য restrictive AST allowlist, native module restrictions এবং forbidden I/O checks যোগ করা যেতে পারে। তবে এটিকে secure sandbox দাবি করা যাবে না। বিশেষত ctypes, cffi, dynamic imports, reflection ও native-extension pathways শুধু কয়েকটি regex দিয়ে পুরোপুরি আটকানো যায় না।
এই পরিস্থিতিতে EVAL12-এর score integrity conditionally trusted, security-guaranteed নয়। সন্দেহজনক executed code থেকে পাওয়া prediction contamination review ছাড়া গ্রহণ করা যাবে না।
Hidden-run-এ test answer file উপস্থিত না থাকার কারণে একই evaluation-answer leakage risk নেই; তবে arbitrary code-এর broader security risk থেকেই যায়।

৪. HIGH — train global এখনো nonportable
Status: Reproduced · repl.py, repl_agent.py
README-এর সিদ্ধান্ত অনুযায়ী train blocked names-এ রাখা হয়নি, কারণ সাধারণ Python variable হিসেবে এই নাম ব্যবহার হতে পারে। কিন্তু repl.py-তে এটি বাস্তবেই preloaded:
ns["train"] = list(zip(    ns["train_inputs"],    ns["train_outputs"]))



তাই train একটি tool-only global-ও হতে পারে।
আমার reproduction:
def transform(grid):    return train[0][1].tolist()



Python tool-এ সফল হয়েছে, function retain-ও হয়েছে। কিন্তু submit sandbox-এ ফল:
NameError: name 'train' is not defined


Fix: _nonportable() এবং _absorb()-এ scope-aware global-name checking প্রয়োজন। সাধারণ local variable train অনুমোদিত থাকবে, কিন্তু tool-preloaded global train-এর ওপর নির্ভরশীল function reject হবে।
সবচেয়ে সহজ conservative fix হলো train-কেও blocked names-এ যোগ করা। তবে এতে legitimate local variable reject হওয়ার ঝুঁকি আছে। Better fix-এর জন্য Python-এর symtable ব্যবহার করে .is_global() এবং .is_referenced() পরীক্ষা করুন, আর source-এর module scope-এ train নিজস্বভাবে define করা আছে কি না দেখুন।
Validation: তিনটি case চালাতে হবে—preloaded global train rejected, function-local train accepted, এবং standalone program-defined global train accepted।
এটি GPU run-এর আগে ঠিক করা উচিত; খরচ negligible।

৫. অন্য গুরুত্বপূর্ণ সমস্যা ও সমাধান
Priority	Finding	Evidence	Required action
HIGH	AGENT throughput misleading	Confirmed from code	Stage-specific throughput
HIGH	Rescue smoke failure nonfatal	Confirmed from code	Required rescue preflight
HIGH	Hybrid merge answer হারাতে পারে	Reproduced	Loss diagnostics + policy A/B
MEDIUM	EVAL rescue outputs checkpoint হয় না	Confirmed from code	Incremental rescue records
MEDIUM	Circuit breaker শুধু প্রথম ৬টি attempt দেখে	Confirmed from code	Sliding error-rate check
MEDIUM	Global token ledger নেই	Confirmed	Run-level telemetry; later enforcement
5.1 AGENT throughput ভুলভাবে কম দেখানো হচ্ছে
run_eval.py-তে:
gen = sum(a["gen_tokens"] for a in atts)wall = time.time() - t0



এরপর gen/wall হচ্ছে reported gen_tok_per_s।
কিন্তু gen শুধু AGENT stage-এর generated tokens, আর wall-এ DIRECT rescue-এর সময়ও অন্তর্ভুক্ত।
Fix:
agent_gen = sum(a["gen_tokens"] for a in atts)agent_wall = report["wall_s"]rescue_gen = sum(    r.get("gen_tokens", 0)    for r in rescue_recs.values())total_wall = time.time() - t0# Save rescue start/end timestamps in run_eval.rescue_wall = max(    0.0, rescue_end - rescue_start)summary.update({    "agent_generated_tokens": agent_gen,    "agent_wall_s": round(agent_wall, 2),    "agent_gen_tok_per_s": round(        agent_gen / max(agent_wall, 1e-6), 2    ),    "rescue_generated_tokens": rescue_gen,    "rescue_wall_s": round(rescue_wall, 2),    "rescue_gen_tok_per_s": round(        rescue_gen / max(rescue_wall, 1e-6), 2    ),    "combined_gen_tok_per_s": round(        (agent_gen + rescue_gen)        / max(total_wall, 1e-6), 2    ),})



rescue_start ও rescue_end নির্ধারণ করতে run_rescue() call-এর আগে-পরে time.time() সংরক্ষণ করুন।
Validation: Mock AGENT ১,০০০ tokens/১০ seconds এবং rescue ২,০০০ tokens/২০ seconds দিলে expected agent throughput ১০০, rescue throughput ১০০, combined throughput ১০০ tok/s।
5.2 Rescue-parameter smoke failure উপেক্ষিত হচ্ছে
submit_main.py-তে rescue_params_ok=False হলেও overall smoke["ok"] true থাকতে পারে।
AGENT চালানো যুক্তিসংগত হতে পারে, কিন্তু তিন-পদ্ধতির তুলনা করার জন্য DIRECT rescue-র request shape কার্যকর হওয়া জরুরি।
Fix: EVAL12-এ rescue_all=True থাকলে rescue-parameter failure-কে diagnostic blocker হিসেবে ধরুন। সম্ভব হলে supported fallback parameters পরীক্ষা করুন। না হলে AGENT-only run চালানো যেতে পারে, কিন্তু three-way score report-কে complete দাবি করবেন না।
এছাড়া একটি full-sized xhigh tool-call roundtrip preflight অন্তর্ভুক্ত করা উচিত। কেবল max_tokens=32 request acceptance যথেষ্ট end-to-end প্রমাণ নয়।
5.3 HYBRID merge-এর score সবসময় AGENT বা DIRECT-এর চেয়ে ভালো হবে না
rescue.merge()-এর priority intentional হলেও তার trade-off আছে।
আমি দুটি case reproduce করেছি:
- AGENT-এর দ্বিতীয় unverified candidate সঠিক, কিন্তু DIRECT ভুল: DIRECT ঢোকানোর ফলে সঠিক AGENT candidate বাদ যায়।
- দুটি verified candidate ভুল, DIRECT সঠিক: দুটি verified candidate জায়গা দখল করায় DIRECT বাদ যায়।
এটি coding exception নয়, বরং selection-policy weakness।
দুটি attempt-এর সীমার কারণে তিন বা তার বেশি ভিন্ন prediction একসঙ্গে রাখা যায় না। তাই কোনো একক ranking দিয়ে union-of-correct score নিশ্চিত করা অসম্ভব।
Fix: EVAL report-এ আরও তিনটি field যোগ করুন:
hybrid_lost_vs_agent = sum(    h_agent[k] and not h_hybrid[k]    for k in h_agent)hybrid_lost_vs_direct = sum(    h_direct[k] and not h_hybrid[k]    for k in h_direct)oracle_union_hits = sum(    h_agent[k] or h_direct[k]    for k in h_agent)



এখানে oracle_union_hits শুধু retrospective upper bound; hidden submission policy হিসেবে ব্যবহার করা যাবে না।
Validation: Verified+DIRECT disagreement এবং unverified+DIRECT disagreement—উভয় ধরনের mock case যুক্ত করুন। তারপর ২২টি EVAL12 output-এ কোন policy কতবার correctness হারাচ্ছে মাপুন।
5.4 EVAL rescue-এর incremental checkpoint দরকার
AGENT attempts attempts.jsonl-এ সংরক্ষিত হচ্ছে। কিন্তু DIRECT rescue results report তৈরির আগে durable JSONL-এ জমা হচ্ছে না।
Run মাঝপথে কাটা পড়লে AGENT transcript থাকবে, DIRECT rescue diagnostics অসম্পূর্ণ থাকতে পারে।
Fix: run_rescue(..., on_result=...) ব্যবহার করে প্রতিটি DIRECT result rescue_attempts.jsonl-এ flush করুন। Record-এ task_id, status, grids, gen_tokens, wall, finish ও rescue_finish রাখুন। eval_report.json তৈরির আগে অন্তর্বর্তী merged submission snapshot-ও সংরক্ষণ করুন।
Validation: Mock run-এর rescue চলার সময় controlled interruption ঘটিয়ে দেখুন শেষ পর্যন্ত সম্পন্ন task-গুলোর results checkpoint-এ আছে কি না।

৬. EVAL12-এর time split যথেষ্ট কি?
README-এর ২ ঘণ্টার configured deadline এবং আগের GPU run-এর প্রায় ৯ মিনিট server startup/smoke ধরে একটি illustrative calculation করছি।
EVAL12 time allocation
Startup + smoke
~9 min




AGENT, 60% of remaining time
~66.6 min




DIRECT rescue
~44.4 min




Server initialization পরিবর্তিত হলে প্রকৃত সময়ও বদলাবে।
EVAL12-এ সবগুলো AGENT task দুই attempt-এর পূর্ণ cap ব্যবহার করলে মোট ৭২০k tokens, আর ১২টি DIRECT task ৩৬,৮৬৪ + সর্বোচ্চ ৪,০৯৬ commit tokens ব্যবহার করলে মোট প্রায় ৪৯২k tokens।
প্রাথমিকভাবে দেখা যাচ্ছে ৬০/৪০ split গ্রহণযোগ্য, যদি AGENT ও DIRECT stage-এর effective throughput যথেষ্ট থাকে। DIRECT stage-এর worst-case generation budget শেষ করতে শুধু decoding-এর জন্য প্রায় ১৮৫ tok/s প্রয়োজন, prefill ও অন্যান্য overhead বাদে।
তাই agent_frac=0.6 নিজে blocker নয়। কিন্তু DIRECT rescue-র actual throughput মাপা বাধ্যতামূলক।
Hidden 240-task run-এর সমস্যা বড়
আগের DIRECT aggregate throughput ২৫০–২৭৫ tok/s হলেও AGENT-এ একই throughput পাওয়া যাবে, তা নিশ্চিত নয়।
ধরি hidden run-এর প্রথম AGENT stage প্রায় ৫ ঘণ্টা ৪৩ মিনিট। ২৭৫ tok/s হলে generation capacity প্রায় ৫.৬৬M tokens।
কিন্তু সব ২৪০ task-এ ২৪k scout cap পর্যন্ত যেতে হলে ৫.৭৬M tokens প্রয়োজন—tool overhead ও prompt processing-এর আগেই।
এখানে কঠিন সিদ্ধান্ত:
EVAL12-এ 24k scout রাখা যায়, কিন্তু hidden rerun-এর জন্য 12–16k first-pass scout বেশি যুক্তিসংগত starting policy। তারপর unverified এবং high-potential tasks-এ অতিরিক্ত budget দেওয়া যায়। তবে এটি EVAL12-এ পরীক্ষা করা 24k policy থেকে আলাদা হবে; সেই পার্থক্য স্পষ্টভাবে রিপোর্ট ও validate করতে হবে।
আর hidden rescue-তে todo prioritization বদলাতে হবে। বর্তমানে cheapest prompt first। আমি আগে যেসব task-এ AGENT একটি attempt-ও শেষ করতে পারেনি, তারপর attempted-but-unverified task rescue করার A/B test চাই। কারণ সম্পূর্ণ অনুত্তরিত task-এ DIRECT সবচেয়ে বেশি coverage value দিতে পারে। এটি প্রমাণিত optimum নয়।

৭. README-এর পাঁচটি প্রশ্নের সরাসরি উত্তর
প্রশ্ন	অডিটের সিদ্ধান্ত
১. Full xhigh smoke mandatory?	হ্যাঁ, একটি bounded, actual-shape xhigh tool roundtrip preflight যোগ করা উচিত। 32-token parameter probe একা যথেষ্ট নয়।
২. EVAL12 agent_frac=0.6?	আপাতত গ্রহণযোগ্য। 12 task-এ throughput/verification মাপার জন্য সময় আছে বলে মনে হচ্ছে; runtime measurement প্রয়োজন।
৩. Hidden agent_frac=0.5?	শর্তসাপেক্ষ। 24k scout এবং 240 task-এর worst-case coverage অনুযায়ী এটি ঝুঁকিপূর্ণ। Adaptive first-pass budget দরকার।
৪. DIRECT rescue 32k thinking?	EVAL12-এর জন্য reasonable experimental point। Hidden run-এ সবার জন্য 32k বরাদ্দ করার আগে coverage পরীক্ষা করতে হবে।
৫. Stale merge/deadlock/submission degradation?	Existing lock path-এ নিশ্চিত deadlock পাইনি; local callback test PASS। কিন্তু candidate eviction-এর কারণে score degradation reproduce করেছি।
৮. কী এখনই করতে হবে, কী পরে?
GPU preflight checklist
[ ] EVAL all-rescued বনাম deployed-policy score আলাদা করা
[ ] verified-but-wrong diagnostic ঠিক করা
[ ] train tool-only global শনাক্ত করা
[ ] Isolation claim ও execution safeguards সংশোধন
[ ] Stage-specific throughput reporting
[ ] Rescue parameter failure স্পষ্টভাবে handle করা
[ ] Bounded real xhigh tool-roundtrip smoke
[ ] নতুন regression tests PASS ও builder থেকে notebook rebuild
0/8 marked completeReset checklist

এই checklist-এ checked মানে আপনি কাজটি সম্পন্ন বলে চিহ্নিত করছেন; আমি ZIP-এর original source পরিবর্তন করিনি।
GPU data পাওয়ার পরে করা যায়: global token ledger-এর sophisticated version, leave-one-out critic, independent second hypotheses, context-memory redesign, grid-format experiments এবং test-time training। এগুলো এই মুহূর্তে GPU run শুরু করার blocker নয়।
Builder-এও পরিবর্তন বসাতে হবে। কেবল generated notebook edit করলে পরবর্তী rebuild-এ fix হারিয়ে যাবে। আমি builder চালিয়ে SHA-identical notebook পুনর্গঠন করতে পেরেছি—এই reproducibility mechanism কাজ করছে।
৯. চূড়ান্ত GO / NO-GO
Current v1.6
NO-GO

কারণগুলো হলো deploy-policy score mismatch, verified-error masking এবং incomplete score-integrity protections। অন্য কয়েকটি ছোট reliability issue-ও GPU run-এর আগে সংশোধন করা উচিত।
After preflight fixes
CONDITIONAL GO

CPU regressions PASS, real tool-roundtrip PASS, এবং remaining isolation limitations স্পষ্টভাবে স্বীকার ও নিয়ন্ত্রণ করা হলে একমাত্র EVAL12 GPU run-টি diagnostic experiment হিসেবে যুক্তিসংগত।
সবশেষে, scoring methodology-ও যাচাই করেছি: Kaggle-এর নিয়ম অনুযায়ী প্রতি test output-এর দুটি attempt-এর যেকোনো একটি exact match করলেই ১, এবং মোট test outputs দিয়ে গড় হয়। v1.6-এর micro-score গণনার মূল পদ্ধতি ঠিক আছে; সমস্যা হচ্ছে কোন submission-কে deployed HYBRID নামে গণনা করা হচ্ছে। 

Kaggle



সবচেয়ে গুরুত্বপূর্ণ সিদ্ধান্ত: v1.6-এর HYBRID direction রাখুন। আগে scoring-policy mismatch ও train bug ঠিক করুন, isolation-এর বাস্তব সীমা মোকাবিলা করুন, তারপর GPU-তে AGENT বনাম DIRECT-এর complementarity মাপুন। সেই ফল ছাড়া ৬০%-এর পথে পরবর্তী বড় architecture পরিবর্তন করা সমর্থনযোগ্য নয়।
</pasted_content id="1891">

"