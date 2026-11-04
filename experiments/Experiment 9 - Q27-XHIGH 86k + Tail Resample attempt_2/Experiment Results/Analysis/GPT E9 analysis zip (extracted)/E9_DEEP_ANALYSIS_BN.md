# E9 EVAL12: স্বাধীন বিশ্লেষণ ও পরবর্তী পরীক্ষার সুপারিশ

তারিখ: 13 October 2026 · পদ্ধতি: saved-output CPU audit + notebook review + primary-source research.

**সিদ্ধান্ত:** E9-এ 4/22 সঠিক উত্তর আছে, কিন্তু “৫টি failure শুধুই transcription” দাবিটি সমর্থিত নয়। সবচেয়ে যুক্তিসঙ্গত পরবর্তী ছোট পরীক্ষা হলো **train-derived shape guard + answer-only repair as attempt_2**। বড় উন্নতির সম্ভাবনা আছে **প্রতিটি hypothesis-কে training examples-এ executableভাবে যাচাই করে grid তৈরি করায়**—তবে আগের code/agent experiment-এর দুর্বল ফলের কারণে DIRECT পুরো বদলে দেওয়া এখন সমর্থনযোগ্য নয়।

এখানে কোনো GPU/model inference, Kaggle run, upload বা push করা হয়নি। মূল ZIP ও notebook অপরিবর্তিত। স্থানীয় working copy-তে saved answer-এর হিসাব, parser replay এবং দুটি diagnostic Python rule পরীক্ষা করা হয়েছে। README-তে লেখা আগের সিদ্ধান্ত ও “user rules” বর্তমান কথোপকথনের নতুন নির্দেশ হিসেবে ধরা হয়নি; প্রস্তাবগুলোতে সেগুলোকে পূর্ববর্তী পরীক্ষার design context হিসেবে সংরক্ষণ করা হয়েছে।

## 1. কী স্বাধীনভাবে যাচাই করা হয়েছে

- README প্রথমে পড়া হয়েছে; পরে 22 primary answer, 14 resample, 36 records, challenge/solution, notebook, logs ও metrics মিলিয়ে দেখা হয়েছে। প্রতিটি primary failure-এর saved thinking tail পড়া হয়েছে।
- Notebook SHA-256: `845ff050ea661d48a5d27368a6066af50c26c143bf7c0f8a998dfeeb5e9a9746`। README-র সংক্ষিপ্ত hash-এর সঙ্গে মেলে।
- Notebook-এর inspected parser দিয়ে answer পুনরায় parse করে `make_attempts` ও resample replacement logic অনুসারে final দুই attempt পুনর্গঠন করা হয়েছে।
- Correct units: `136b0064#0`, `247ef758#0`, `4c3d4a41#0`, `b6f77b65#1`। Query score **4/22 = 18.18%**; task-normalized **19.44%**। দুই denominator মিশিয়ে তুলনা করা যাবে না।
- Primary natural: **9টির 4টি hit**; forced: **13টির 0টি hit**। 49,152-এর পরে natural হওয়া primary চারটি—`4c7dc4dd#0`, `6e4f6532#1`, `2b83f449#0`, `b6f77b65#1`—এর মধ্যে শেষটি hit। এটি বড় budget-এর সম্ভাব্য মূল্য দেখায়; matched control ছাড়া causal +1 gain প্রমাণ করে না।
- Primary output-এর 21/22 parser-valid; এদের মধ্যে তিনটি train-supported expected shape-এর সঙ্গে মেলে না। বাকি rejected answer-ও আসলে 31-row grid।
- 14 resample-এ **0 gain**, 12টি attempt_2 বদলেছে। 9টি full-cap forced, 1টি natural, **4টি deadline-drained**। তাই “14 independent full-budget retries failed” বলা ভুল।
- Primary generation **1,468,025 tokens**; resample **1,033,046**; total **2,501,071**। Resample total generation-এর **41.3%** নিয়েছে, observed exact-match gain শূন্য। এটি token share, GPU-hour share নয়। Drained চারটিতে 248,254 tokens গেছে।
- Engine: scheduler 8,158.4 seconds, 306.6 generated tok/s, no recorded preemption or scheduler error। Stdout-এর peak KV **91.4%**; coarse JSON snapshots-এর peak **89.7%**। Sampling interval-এর পার্থক্য, বিরোধ নয়।
- `outputs/submission.json`-এ **240 hidden-test tasks / 259 inputs**, EVAL12 task ID overlap শূন্য। Eval mode-এ এটি ইচ্ছাকৃত identity placeholder; 4/22 score এই ফাইল থেকে পাওয়া যায় না। তাই audit-এ saved answers থেকে evaluation attempts পুনর্গঠন করেছি। এটি submission-writer bug নয়; আলাদা `eval_predictions.json` সংরক্ষণ করলে review সহজ হবে।

**প্রমাণের সীমা:** full reasoning/token IDs ZIP-এ নেই; শুধু শেষ 1,500 thinking tokens ও answer আছে। তাই পুরো 77k চিন্তার ইতিহাসে কোথায় প্রথম ভুল হয়েছে তা নির্ণয় করা যায় না। পুরোনো runs-এর raw artifacts এখানে নেই; তাদের ফল ledger-এ reported history, স্বাধীনভাবে পুনঃস্কোর করা হয়নি। EVAL12 অনেকবার দেখা development set; এখানকার post-hoc fix-কে নতুন generalization result বলা যাবে না।

## 2. সবচেয়ে গুরুত্বপূর্ণ সংশোধন: near-miss মানেই transcription নয়

সব coordinate নিচে **zero-based (row, column)**। ছোট cell-error count কেবল answer-এর দূরত্ব জানায়; কেন ভুল হয়েছে তা জানায় না।

| Unit | আগের দাবি | স্বাধীন প্রমাণ | সংশোধিত সিদ্ধান্ত |
|---|---|---|---|
| `247ef758#1` | 1-cell transcription | (6,12)-এ model=4, truth=6। Tail স্পষ্টভাবে 4-কে overlapping diagonal-এর ওপর বসানোর সিদ্ধান্ত নেয় | **Overlap precedence-এর reasoning error** |
| `4c3d4a41#1` | 2-cell transcription | (1,17):7→5; (3,17):0→7। Tail top cell=5 দেখেই column shift করার নিয়ম বলে এবং সেই ভুল output আগেই লিখে | **ভুল conditional rule**, random digit slip নয় |
| `2b83f449#0` | শুধু এক row বাদ | 18×17 বনাম 19×17। Oracle row alignment-এ এক missing row ধরলেও 13 cell disagreement থাকে | **Geometry/accounting + unresolved rule**, মিশ্র failure |
| `6e4f6532#1` | শুধু দুই row বাদ | 28×16 বনাম 30×16। দুই missing row align করার পরও marker-এর 2 cell mismatch | প্রধানত **coordinate/execution**, pure row omission নয় |
| `88bcf3b4#1` | শুধু এক row বাদ | 26×27 বনাম 27×27। এক missing row align করার পরও 17 mismatching cells | **Wrong path geometry + missing row** |

এখানে row alignment একটি **oracle diagnostic**: solution জানা অবস্থায় row-gap ও cell mismatch-এর সর্বনিম্ন edit cost হিসাব করা হয়েছে। এটি deployment repair algorithm নয়, এবং “কোথায় row বসাতে হবে model জানত” তার প্রমাণও নয়। একাধিক equally good alignment থাকতে পারে। তিনটি ছোট primary grid-এর কোনোটি শুধু correct row insertion দিয়ে exact হয় না।

আরও একটি সংশোধন: Claude post-mortem-এর D group heading-এ “6 units” লেখা, কিন্তু তালিকায় 7টি আছে। মোট failure 18-ই।

## 3. সমস্ত 18টি unsolved unit

ফাইলের basename নিচে দেওয়া; ZIP-এর `outputs/q27x/answers/`-এ সেগুলো আছে। Exact shapes, differences, raw widths এবং alignment `e9_independent_audit.json`-এ সংরক্ষিত।

| Unit | Primary বনাম truth | মূল failure | Resample ও উপযোগী পরবর্তী পদক্ষেপ |
|---|---|---|---|
| `247ef758#1` | 1/216 ভুল | Source shape ও marker intersection বোঝে; overlap winner ভুল। Tail ও final একই ভুল | Resample হয়নি। Explicit source-layer order, train execution; shape guard ধরবে না |
| `4c3d4a41#1` | 2/160 ভুল | Mask সরানো বোঝে; কোন colored column সরবে সেই condition ভুল | Resample হয়নি। Mask collision থেকে displacement হিসাব; shape guard ধরবে না |
| `2b83f449#0` | 18×17; truth19×17 | Bottom structure ভুল গণনা; side-marker rule নিয়ে দ্বিধা। Best one-row alignment-এর পর13 ভুল | Resample19×17, মাত্র3 ভুল। Shape retry + deterministic coordinates; remaining marker rule যাচাই দরকার |
| `6e4f6532#1` | 28×16; truth30×16 | Object rotation/placement প্রায় ঠিক; row inventory ও marker placement ভুল। Tail নিজেই0–27 row পরিকল্পনা করে | Resample হয়নি:69,486 thinking threshold70,041-এর নিচে। Shape retry-এর সবচেয়ে শক্তিশালী candidate |
| `88bcf3b4#1` | 26×27; truth27×27 | Top path target-এর ভুল পাশে ঘোরে; অন্য path-ও ভুল; output height ভুল | Resampleও26×27, কিন্তু grid একই নয়; alignment-পরবর্তী residual17→6। Shape repair alone যথেষ্ট নয় |
| `4c7dc4dd#0` | 12/25 ভুল | Framed panels শনাক্ত করে; binarization+bitwise XOR hypothesis-এ ভুলভাবে নিশ্চিত হয় | Natural59.6k, retry হয়নি। Candidate panel operations-এর training replay ও counterexample দরকার |
| `88bcf3b4#0` | 17/484 ভুল | Outline-following ধারণা আছে; corner omission/turn rule অমীমাংসিত | Resample15 ভুল। Graph/path tracing এবং প্রতিটি training path-এর exact test |
| `8b7bacbf#0` | 46/900 ভুল | Loop/interior শনাক্তের চেষ্টা আছে, কিন্তু fill selection/connectivity ভুল। Needed48 edits-এর16 ঠিক; অপরিবর্তিত অঞ্চলে14 ক্ষতি | Resample29×30, আবার shape error। Component topology, connectivity ও trigger-color relation আলাদা করে test |
| `8b7bacbf#1` | 20/572 ভুল | Anchor adjacency ও closed/open enclosure নিয়ে শেষ পর্যন্ত সংশোধন করছে | Needed39 edits-এর21 ঠিক, অন্যত্র2 ক্ষতি। Resample21×26; truth22×26। Flood-fill alone নয়, fill eligibility-ও যাচাই |
| `b6f77b65#0` | 21/144 ভুল | Key-color deletion বুঝেছে; support/collapse/gravity mechanics বোঝেনি | Resample28 ভুল,57,752 tokens-এ drain। Explicit support graph/simulation; solved sibling blind transfer নয় |
| `b6f77b65#2` | 23/144 ভুল | Displacement ও cascading effect-এর বহু অসঙ্গত hypothesis | Resample38 ভুল,45,333 tokens-এ drain। Extra markerসহ full rule execution দরকার |
| `20a9e565#0` | 3×21; truth2×21 | Tail marker span থেকে2-row possibility চিনেছে, কিন্তু final-এ3-row pattern ও ভুল color নেয় | Resample3×26। Shape এখানে simple same-size/constant rule দিয়ে infer হয় না; marker bbox ও pattern phase/color যাচাই |
| `20a9e565#1` | 55/136 ভুল | Pattern level/sequence extrapolation ad hoc; exact generator বের হয়নি | Resample58 ভুল। Period/phase/scale hypotheses training pairs-এ execute করতে হবে |
| `21897d95#0` | 886/900 ভুল | Region partition/arrow relation অমীমাংসিত; final পুরো9 | Resample একই all9 grid। Component graph ও arrows-এর relation model দরকার; re-seeding একই representation ব্যর্থতার পুনরাবৃত্তি |
| `21897d95#1` | 432/576 ভুল | Block recoloring ও orientation rule ভুল; square-grid exception ধরে নেয় | Resample318 ভুল। Input copy-তেও312 ভুল, তাই বড় confidence statement নির্ভরযোগ্য নয় |
| `4c7dc4dd#1` | 5×5; truth6×6 | Tail training-example subgrids পরীক্ষা করতেই শেষ; test output extraction/rule স্থির হয়নি | Resample6×6 হলেও32/36 ভুল। Shape ঠিক হওয়া semantics ঠিক হওয়া নয় |
| `898e7135#0` | 375/900 ভুল | Object bbox/scale গণনায় ভুল; শেষ tail-এ missed row আবিষ্কার করছে, matching অসম্পূর্ণ | Resample20×20, truth30×30। Exact bbox, downscale, dihedral transforms ও hole matching সাহায্য করতে পারে; rule এখনও যাচাইহীন |
| `6e4f6532#0` | Parser rejects **31×28**; truth30×28 | “No answer” নয়। Object transform/anchor সম্পর্কে অনেক বোঝা আছে; translation/coordinates ভুল, একটি extra row | Resampleও31×28। Extra-row alignment-এর পর primary56, resample96 cell mismatch; shape retry-এর trigger আছে, নিশ্চিত solve নেই |

### নির্দিষ্ট ব্যাখ্যা ও intervention

**247ef758:** Source shapes frame-এর matching row/column markers-এর intersections-এ stamp হয়। Crucial detail হলো overlapping source-এর precedence। Upper source আগে থাকা object-কে visible priority দিয়ে deterministic stamping-এর diagnostic implementation **3/3 train +2/2 test** মেলে। বিপরীত precedence training example0-তেই (5,9)-এ ভুল করে; primary failed test-এ ঠিক সেই রকম ভুল হয়। অর্থাৎ training evidence দিয়েই এই ambiguity reject করা সম্ভব। সাধারণ shape retry এ সমস্যা খুঁজে পাবে না।

**4c3d4a41:** সঠিক diagnostic model: left mask ডানে সরবে; colored vertical run যতটা mask-এর সঙ্গে collide করে ততটা উপরে উঠবে, ceiling-এর সীমা মেনে। Displacement = `min(max(0, bottom_of_run - top_of_mask + 1), top_of_run - 1)`; তারপর mask বসে। এতে **2/2 train +2/2 test** মেলে। Failed #1-এর color7 run mask-এর সঙ্গে collide করে না, তাই সেটি নড়ার প্রয়োজন নেই। Model top5 দেখেই সরিয়েছিল। Code নিখুঁতভাবে ভুল hypothesis execute করলেও ভুলই হতো—আগে সঠিক condition দরকার।

**2b83f449:** Primary tail18টি row-এর inventory-তেই শেষ করে, এবং bottom geometry truth-এর সঙ্গে মেলে না। Resample bottom structure পুনরুদ্ধার করেছে; বাকি তিন ভুল (zero-based) `(4,15)`, `(10,1)`, `(18,6)`—প্রতিটিতে8 লিখেছে যেখানে3 দরকার। এই near-miss marker logic যাচাইয়ের candidate; raw grid বানানোই একমাত্র সমস্যা নয়।

**6e4f6532:** #1-এ row count ভুলটি thinking plan থেকেই এসেছে; final serialization plan মেনেছে। Best alignment missing truth rows21,26 দেখায়; marker1-ও row2 থেকে row3-এ থাকা উচিত। #0-এ intended transform ও produced coordinates-এর অসঙ্গতি আছে। Object extraction, dihedral transform, exact anchor correspondence এবং output initialization দিয়ে errors আলাদা করা যায়।

**88bcf3b4:** #0-এর tail staircase outline বোঝে, কিন্তু examples-এর missing corner cells ব্যাখ্যা করতে পারে না। #1 top path-এর opposite side বেছে নেয়—truth-এ col12-ঘেঁষে নিচে নামার বদলে col14-ঘেঁষে। এটি geometry-rule error। Shape constraint path direction ঠিক করে না।

**8b7bacbf:** 95% cell accuracy misleading: #0 copy করলে95%-এর কাছাকাছিই হয়। Background dominates। Needed changes-এ recall এবং unchanged region-এ corruption আলাদা মাপা উচিত। Border flood-fill, 4-vs8 connectivity, contour continuity, এবং anchor/trigger eligibility আলাদা hypotheses হিসেবে train-এ যাচাই করতে হবে; “সব enclosed hole fill” আগেই সঠিক ধরে নেওয়া যাবে না।

**b6f77b65:** সফল #1-এ top-left key7 রেখে বাকি7 মুছে দিলেই truth মেলে। একই delete-only code #0-এ24 cell, #2-এ46 cell ভুল করে। #0 ও#1 input শুধু corner key-তে আলাদা; #2-তে corner key ছাড়াও `(0,1)`-এ1 আছে। একটি sibling-এর hit পুরো collapse rule জানা প্রমাণ করে না। Raw reasoning reuse-এর বদলে train-validated transformation reuse দরকার।

**20a9e565 /4c7dc4dd /21897d95 /898e7135:** এখানে dimensions অনেক ক্ষেত্রে transformation-এর অংশ। Constant/same-size constraint অনুমান করে চাপানো বিপজ্জনক। Object-level measurements দিয়ে hypotheses সংকুচিত করে, তারপর training feedback থেকে rule refine করা বেশি প্রাসঙ্গিক। এই audit এদের নতুন verified general solution দাবি করছে না।

## 4. CPU diagnostic prototypes কী প্রমাণ করে

`diagnostic_prototypes.py`-এ শুধু উপরোক্ত দুই task-এর pure Python rules আছে। সব **5টি training example ও4টি test input** exact match করেছে; wrong-overlap control training-এ reject হয়। এর মধ্যে দুইটি test E9-এর missed unit।

এগুলো **post-hoc, task-specific reference implementations**। Solutions দেখার পরে তৈরি; তাই “নতুন solver4/4”, “E10 score6/22” বা expected GPU gain+2 বলা যাবে না। প্রমাণটি ছোট: দুই near-miss-এর semantic mistake ব্যাখ্যা ও সংশোধন করা যায়, এবং output construction deterministic করা যায়। অন্য puzzles-এ ব্যবহারযোগ্য general solver তৈরি করা হয়নি।

## 5. Shape guard + answer-only retry: সবচেয়ে ছোট পরবর্তী পরীক্ষা

### Trigger

Training pairs থেকে output-shape hypotheses তৈরি:

1. সব pair-এ input shape=output shape হলে same-size hypothesis।
2. সব output একই shape হলে constant-shape hypothesis।
3. Test-এ applicable hypotheses একমত হলে expected shape; দ্বিমত হলে abstain। Square-only examples থেকে identity/transpose ambiguity থাকতে পারে—hypothesis list বাড়ালে এই ambiguity-ও রাখতে হবে।
4. Parser failure, jagged grid, invalid color/size, বা confident expected-shape mismatch হলে repair candidate।

Same-size guard এই saved run-এ **4 primary failures** ধরে: `2b83f449#0`, `6e4f6532#0`, `6e4f6532#1`, `88bcf3b4#1`। Correct চারটির কোনোটি flag হয় না। Resample-সহ total8 structural alerts। Training-derived shape একটি inductive guess, hidden-test theorem নয়।

### Repair contract

- Primary attempt_1 এবং তার generation budget অপরিবর্তিত।
- নতুন reasoning-from-scratch নয়: live saved prompt+thinking prefix থেকে একটি answer-only repair branch। Natural answer-এর ক্ষেত্রে পুরোনো answer ও EOS বাদ দিয়ে ঠিক `</think>` boundary-তে branch নিতে হবে; old full response append করা যথেষ্ট নয়।
- Prompt-এ expected H×W ও observed mismatch স্পষ্টভাবে দেওয়া; original input/train data context-এ থাকবে। Example: “Previous candidate has28 rows; required30×16. Reconstruct using the input coordinate system. Do not pad or truncate blindly.”
- Parsed candidate valid, expected shape-compatible এবং অন্য attempt থেকে আলাদা হলেই attempt_2 হিসেবে বিবেচনা। Primary অপরিবর্তিত রাখলে তার exact hit নষ্ট হয় না। তবে পুরোনো attempt_2 সত্যি correct থাকলে তা replace করে loss হতে পারে; E9-এ এমন fallback hit নেই।
- এক unit-এ সর্বোচ্চ একটি repair; যথেষ্ট remaining time/context না থাকলে আগের pair রাখবে। Atomic valid submission write বজায় থাকবে।
- Repair output-এর format guarantee চাইলে পৃথক ablation-এ exact H-row/W-column grammar বা JSON schema ব্যবহার। Initial reasoning phase-এ constraint চাপানো নয়।

**Cost:** চার primary alert-এ8,192 cap করে সর্বোচ্চ32,768 নতুন output tokens—observed resample output1,033,046-এর3.17%। এটি output-token budget comparison; latency বা টাকা32গুণ কম বলছি না। Long-prefix prefill, KV eviction, queue contention ও schema compilation বাস্তবে খরচ যোগ করবে। সাধারণ full grids প্রায়300–1,800 output tokens ছিল;8,192 answer cap মূল bottleneck মনে হচ্ছে না।

**সর্বাধিক সম্ভাবনাময়:** `6e4f6532#1`। `2b83f449#0` বিশেষত better resample থেকে targeted repair candidate। অন্য দুইটির বড় semantic errors আছে। **Confirmed recoveries এখন0**; trial planning-এর জন্য0–2 additional hits একটি সতর্ক hypothesis, statistical estimate নয়। Hard maximum trigger coverage4; পাঁচ guaranteed recoveries-এর ভিত্তি নেই। Shape-only deterministic padding/cropping-এর জন্য কোনো exact recovery প্রমাণ হয়নি।

### CPU checks, তারপর GPU

CPU-তে real saved outputs replay করে alert count4/8, correct-hit non-trigger, no solution access in trigger, rectangular/color validation, unchanged attempt_1, distinct attempt_2, no-time fallback, atomic writer, and evaluation scoring যাচাই করা উচিত। Synthetic adversarial case: train same-size+constant agree on train but disagree on a differently sized test; guard abstain করবে।

এই audit saved-output parsing, alert coverage, score reconstruction ও prototype rules পরীক্ষা করেছে। নতুন async repair scheduler/tokenizer integration তৈরি বা CPU-harness পরীক্ষা করা হয়নি।

GPU-তে প্রথমে একটিমাত্র change: guard+one answer-only repair। E9 weights,77,824+8,192 budget, prompt, sampling, seeds, ordering, MTP, KV রাখুন। একই primary token snapshots থেকে baseline/repair branch চালানো সবচেয়ে পরিষ্কার। Archive-এ পূর্ণ prefix নেই, তাই exact retrospective replay এখন সম্ভব নয়; পরবর্তী run-এ prefix IDs+tokenizer/chat-template hashes সংরক্ষণ দরকার।

## 6. Sibling reuse: শর্তসাপেক্ষে, blanketভাবে নয়

Hidden inference-এ কোনো sibling “solved” কিনা oracle ছাড়া জানা যায় না। Natural close একটি weak signal: এই run-এ9টির5টি natural answer ভুল।

ভালো variant: sibling-এর compact hypothesis বা executable transform candidate নিন; সব training pair-এ test করুন; তারপর sibling test inputs-এ apply করুন। Independent primary answer থাকবে। `247ef758` ও `4c3d4a41`-এ এটি পরীক্ষাযোগ্য। `b6f77b65`-এর deletion-only hitকে general rule হিসেবে promote করা যাবে না।

এক prompt-এ সব test answer জোড়া দিলে shared ভুল, larger context, output truncation ও query-mapping errors বাড়তে পারে। বর্তমান parser list-of-grids পেলে last grid নেয়—multi-test protocol তাই আলাদা parser/schema ছাড়া drop-in change নয়। একটি shared program থেকে আলাদা input-এ output তৈরি করা বেশি auditযোগ্য।

CPU acceptance: query index mapping, program train exactness, arbitrary code isolation/timeouts, unchanged primary, invalid candidate fallback। GPU acceptance: same budget ওtask set-এ sibling-assisted attempt_2 বনাম independent attempt_2; সব eligible siblings-এর ফল report করতে হবে, কেবল retrospective known hits বেছে নয়।

**Recovery count:** blind reuse-এর confirmed gain0। দুই small semantic misses-এ post-hoc executable rules recovery দেখিয়েছে; production model সেগুলো বের করবে কি না অজানা। দুই b6 failure-এ solved sibling থেকে gain অনুমান করা যায় না। Shape এবং sibling benefits যোগ করে headline score বানানো যাবে না।

## 7. আরও শক্তিশালী দিক: যাচাইযোগ্য program + feedback

বড় সীমাবদ্ধতা হলো model অনেক hypotheses কল্পনায় মিলিয়ে নিচ্ছে—train outputs-এ বাস্তবে চালিয়ে discrepancy দেখছে না। বিশেষত near-correct rules-এ একটি counterexample অনেক পুনরাবৃত্ত চিন্তার চেয়ে তথ্যবহুল।

প্রস্তাবিত পৃথক experiment:

- DIRECT attempt_1 অক্ষত। Attempt_2-এর জন্য hypothesis/program candidate তৈরি, train inputs-এ execute, shape/cell/component mismatch দিয়ে feedback, accepted program দিয়ে test grid serialize।
- Object/component extraction, bbox, dihedral transforms, crop/paste, overlay precedence, connected regions, flood-fill এবং support graph-এর ছোট deterministic helpers ব্যবহার; পুরো unrelated framework ঢোকানো নয়।
- সব train exact হলেই candidate eligible; train pass generalization guarantee নয়। Equal train-fit hypotheses হলে আলাদা test predictions-এর diversity সংরক্ষণ বা abstain। Test solution decision path-এ যাবে না।
- Generated Python চালালে real process/container isolation, no network, limited filesystem, CPU/memory/time limits লাগবে। শুধুAST filtering বা20-second timeout পূর্ণ sandbox নয়।
- নতুন independent reasoning candidate-এর জন্য বর্তমান full thinking allocation বজায় রাখা; answer-only translationকে নতুন short-budget primary হিসেবে গোপনে গণনা নয়।

**কেন পুরো pipeline এখনই বদলাব না:** ledger-এ আগের program-synthesis2/22, REPL agent0/22, long-think program1/22 reported। Raw logs ছাড়া কারণ বলা যাচ্ছে না। তাই “code execution দিলেই best score” প্রমাণিত নয়। ছোট optional verified branch হিসেবে পরীক্ষা করাই ভালো।

**Diff-style output:** Same-size tasks-এ copy(input)+explicit edits row loss কমাতে পারে। কিন্তু sparse edits missing transformation ঢেকে দিতে পারে; replacement/erase/overlap ordering ভুল হলে output ভুলই থাকবে। Format-only ablation হিসেবে ভালো; reasoning remedy হিসেবে নয়।

## 8. Notebook ও resource-allocation findings

1. **Retry priority ভুল signals-এ সীমিত।** `_e9_priority` no-grid/forced/long-natural দেখে; valid-but-wrong-shape দেখে না। `6e4f6532#1`69,486<70,041 বলে বাদ, যদিও30×16 input-এর28×16 output observable anomaly। `247ef758#1`42k ও `4c3d4a41#1`25k-ও excluded, তবে তাদের ভুল shape দিয়ে ধরা যায় না। Structural anomalyকে first priority করুন; “likely wrong” এবং “likely repairable per token” আলাদা।

2. **Tail slot free মানে GPU compute free নয়।** First resample66.93min; last primary118.30min। Resamples incomplete primary-এর সঙ্গে GPU share করেছে। Primary queue থেকে slot নেয়নি, কিন্তু runtime/batching interference হতে পারে। E9 থেকে no-resample baseline finish time অনুমান করা যায় না।

3. **Deadline-aware admission নেই।** Four retries drain147min-এ cut। `b6f77b65#2~r1`45,333 actual p1 tokens<49,152। Configured cap কমানো হয়নি, কিন্তু unfinished thinking floor operationally পূরণ হয়নি। Design goal যদি admitted uncertain candidate-কে full budget দেওয়া হয়, latest-start সময় remaining tokens, conservative per-sequence rate, prefix cost, answer reserve দিয়ে নির্ধারণ করতে হবে। Aggregate306.6tok/s single-request speed নয়।

4. **Full-budget retry বনাম repair আলাদা accounting চাই।** Existing report forced-close-এর মধ্যে cap-hit আর deadline-drained মিশিয়েছে। আলাদা fields: original cap, actual model thinking, injected stop tokens, repair tokens, prefill/cache hits, reason, queue delay, final-attempt provenance।

5. **Context fallback declared floor-এর নিচে নামতে পারে।** `phase1_cap(p)=min(77824,106496-p-8192-256)`। p>20,224 হলে full77,824 নেই; p>48,896 হলে49,152-এর নিচে যেতে পারে। `MIN_THINK_BUDGET=4096` সেটিকে অনুমোদন করে। এই EVAL12-এ prompt max7,464, reduced0—তাই observed cause নয়; future safety-of-budget issue। Input হারানো/truncation না করে explicit skip/fallback বা approved context policy লাগবে।

6. **Forced think count-এ23 injected tokens।** Full-cap p1=77,824 হলেও `think_tokens=77,847`; host stop text reasoning count-এ ঢুকছে। Analysis-এ actual-generated এবং host-inserted tokens আলাদা হওয়া দরকার।

7. **Output-prefix persistence অসম্পূর্ণ।** `STATE`-এ full IDs live থাকে, কিন্তু files-এ1,500-token tail। Immediate repair সম্ভব; পরে archive থেকে exact branch replay সম্ভব নয়। Compressed token IDs, tokenizer/model/config hashes save করলে আবার পূর্ণ thinking চালানোর প্রয়োজন কমবে।

8. **Eval artifact clarity।** Identity submission ঠিক আছে; আলাদা scored evaluation predictions/provenance দরকার। Parser rejection reasonও লিখুন:31 rows বনাম missing text এক category নয়।

9. **Engine instability primary diagnosis নয়।** Preemptions0, peakKV91.4%, no crash। Pip conflict warnings ও experimental Mamba-cache warnings আছে, কিন্তু এই run-এর specific error cause প্রমাণ নয়। FP8-vsBF16 accuracy বা MTP setting আলাদা experiment ছাড়া দায়ী করা যাবে না। `vllm_frontend.log` খালি, stdout/JSON usable; logger observability উন্নত করা যায়।

10. **Projection already disabled correctly।** Primary mean tokens ও mixed primary/resample throughput দিয়ে259 hidden inputs-এর runtime নির্ভরযোগ্যভাবে projection করা যাবে না। Report-এর `projection_valid=false` সঠিক।

Source locations: notebook JSON cell index2 `phase1_cap`/minimum; index4 prompt; index5 parser; index7 `start_phase2`, `finish`, `_e9_priority`, `_record_resample`, `run_scheduler`; index8 reporting. Source comments-এর cell numbering notebook index থেকে এক কম।

## 9. ওয়েব/GitHub থেকে প্রাসঙ্গিক শিক্ষা

- **vLLM0.19.0 structured outputs:** offline `SamplingParams`-এ `StructuredOutputsParams` দিয়ে JSON/regex/grammar constraint সমর্থিত। তাই repair-only exact-dimension output একটি বাস্তব implementation option। E9-এর low-level `engine.add_request`, Qwen token-boundary, MTP×3 ওfp8 combination-এ compatibility/latency smoke test এখনও প্রয়োজন; docs থাকা মানেই এই exact setup tested নয়। [Version-matched official docs](https://docs.vllm.ai/en/v0.19.0/features/structured_outputs/)
- **BARC code:** induction route transformation code execute করে; evaluation helper shape/content mismatch ও timeout আলাদা করে। ধার করার উপযোগী অংশ হলো transformation→execution→feedback separation, পুরো old dependencies নয়। তাদের evaluation file test labels-ও ব্যবহার করে; inference selection-এর জন্য শুধু train-label feedback আলাদা করতে হবে। [BARC repository](https://github.com/xu3kev/BARC), [execution/evaluation code](https://github.com/xu3kev/BARC/blob/master/eval_code_samples.py)
- **ARC Prize2025 analysis:** বিভিন্ন successful approach-এ feedback-driven per-task refinement সাধারণ উপাদান। এটি এই run-এর unverified verbal hypothesis সমস্যার সঙ্গে মেলে; E9-এ expected improvement-এর numerical proof নয়। [Official analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)
- **ARChitects2025:** structured search, candidate selection, prefix reuse ওtest-time adaptation আলাদা levers। তাদের scores/hardware/model সরাসরি এই27B/L4 setup-এর baseline নয়; TTT/diffusion এখন সবচেয়ে ছোট next change নয়। [Authors’ technical report](https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/)
- **Qwen3.8 official result:** XHigh42.4% semi-private এবং120-task public breakdown আছে; xhigh chat-template instructions-ও relevant। EVAL12-এর18.18%-এর সঙ্গে42.4% সরাসরি তুলনা বৈধ নয়: dataset, attempts, serving ওbudget আলাদা। API token cap এই bundle থেকে নিশ্চিত নয়; average spend থেকে cap বের করা উচিত নয়। [ARC Prize verified result](https://arcprize.org/results/alibaba-qwen3-8-27b)

## 10. পরীক্ষার অগ্রাধিকার ও সিদ্ধান্ত নেওয়ার নিয়ম

| ক্রম | একমাত্র পরিবর্তন | কী জানতে চাই | বর্তমান প্রমাণ |
|---|---|---|---|
| E10 | Shape-triggered one answer-only repair→attempt_2 | Low-cost structural repair exact solves বাড়ায় কি না |4 primary triggers; confirmed GPU gain0 |
| E11 | Repair phase-এ exact shape grammar | Plain retry-এর চেয়ে valid shape rate বাড়ে কি না, cost কত | Official API আছে; E9 compatibility untested |
| E12 | Optional train-verified program candidate→attempt_2 | Semantic near-miss ওmanual geometry errors কমে কি না |2 task-specific CPU diagnostic recoveries; general model gain unknown |
| E13 | Validated sibling hypothesis reuse | Same task repeated inference-এর খরচ কমে কি না |Raw sibling hit insufficient; b6 counterexample |
| পরে | Runtime-aware tail admission | Deadline waste কমে কি না |4 drained resamples, one below49k actual tokens |

E10-তে primary budget77,824+8,192, original order, sampling, engine knobs অপরিবর্তিত রাখুন। Resample policy ওshape repair দুটো একসঙ্গে পুরো বদলালে attribution দুর্বল হবে; equal total GPU time-এ baseline বনাম পরিবর্তিত run দরকার। Saved-prefix branch microbenchmark এবং end-to-end deadline run আলাদা মাপুন।

Report করুন: exact pass@1, exact pass@2, task-normalized score, structural-valid rate, repair triggers/success, regression of previously correct pairs, generated/prefill tokens, latency, number completed, deadline outcomes। Changed-cell metrics কেবল diagnostic, final score নয়।

Same EVAL12-এ repeated seeds exploratory evidence দেবে; task-level correlation মাথায় রেখে paired outcomes দেখুন। তারপর configuration freeze করে previously unused evaluation tasks-এ পরীক্ষা করুন। PROXY56 EVAL12-কে ধারণ করে, তাই পুরো PROXY56 independent validation নয়; remainder44 tasks smoke-check হতে পারে, কিন্তু historical overlap বাদ দিয়ে সত্যিকারের unseen subset বাছতে হবে।

**এখনকার সুপারিশ:** E10 ছোট ওপরিষ্কার পরীক্ষা হিসেবে আগে। Reasoning উন্নতির জন্য E12 সবচেয়ে অর্থবহ research direction, কিন্তু ফল অজানা। আরও বড় thinking cap বা একই full resample policy-কে default solution করার জন্য এই run যথেষ্ট প্রমাণ দেয় না।

## 11. পুনরুৎপাদনযোগ্য ফাইল

- `e9_independent_audit.json`:36 candidates-এর shapes, exact differences, raw row counts, oracle alignment এবং parsed grids।
- `audit_e9.py`: extracted bundle directory নিয়ে CPU-only audit। এটি inspected notebook parser definitions ব্যবহার করে; model/engine cell execute করে না।
- `diagnostic_prototypes.py`: original ZIP read করে দুই named task-এর deterministic reference rules ওadverse overlap control যাচাই করে।

Commands:

```text
python -X utf8 audit_e9.py <extracted-bundle-directory> --out <audit-output-directory>
python -X utf8 diagnostic_prototypes.py <path-to-E9_GPU_EVAL12_RUN_FOR_GPT.zip>
```

Audit JSON-এর alignment evaluation truth ব্যবহার করে; runtime guard/ranking-এ তা ব্যবহারযোগ্য নয়। Source-rule prototypes-এও evaluation-aware development হয়েছে। নতুন unbiased GPU experiment ছাড়া কোনো production recovery claim নেই।
