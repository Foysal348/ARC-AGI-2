ওকে চলো পরের কাজ শুরু করি।
"

<pasted_content id="1891">
KV Capacity Research — Qwen3.8-Flash-Next + DFlash2 + Merlin 0.32
মূল ফলাফল: তোমাদের 540-block DFlash2 pool-কে বড় করাই একমাত্র সমাধান নয়। সবচেয়ে গুরুত্বপূর্ণ হলো প্রতি request-এর fixed admission cost (F≈12) কমানো, DFlash2 যোগ হওয়ার কারণে KV group-layout overhead কমানো এবং Mamba checkpoint storage-এর waste উদ্ধার করা।

GitHub-এর upstream PR/issue এবং তোমাদের ZIP-এ bundled Merlin cache-manager source মিলিয়ে কয়েকটি concrete port path পাওয়া যাচ্ছে। নিচে estimated লাভ, বাস্তব upstream evidence এবং port complexity আলাদা করেছি।
১. Executive comparison
Capacity estimates হলো engineering scenarios, GPU measurements নয়। একই memory saving বিভিন্ন lever-এ overlap করলে লাভ যোগ করা যাবে না।
Lever	Expected capacity/resident effect	Confidence	Port effort	Upstream status
A. Hybrid KV group-layout optimizer #58207	Padding/group overhead কমলে blocks বাড়বে; K100-specific gain UNVERIFIED	Medium	Medium	Open PR
B. Packed BLHNC layout #58380	Group count ও padded pages কমতে পারে; exact gain UNVERIFIED	Medium	High	Open PR
C. Mamba page padding removal #60967	২১% padded component উদ্ধার করার সম্ভাবনা; মোট pool +২১% নয়	Medium	Medium–High	Open PR
D. Admission cap F=12 → lower safe bound	24 streams-এ F 12→10 হলে 48-block admission headroom	Medium	Medium	Custom port
E. Hybrid boundary-aware admission #62752	একই pool-এ avoidable preemption কমতে পারে; +0 physical blocks	High for CPU finding	Medium	Open issue
F. Shared KV storage / bucket redesign #59117	DFlash bucket fragmentation কমাতে পারে; gain UNVERIFIED	Medium–Low	High	Experimental POC
G. Per-group page sizes / internal paging #55196, #64508	Uniform-page waste কমানোর structural opportunity	Medium for diagnosis	Very high	RFC
H. Generated-tail Mamba checkpoint #53803, #50551	Physical blocks বাড়ায় না; usable cached prefix বাড়ায়	High for relevance	High	Draft/open
I. FP8/int8 GDN recurrent state #55196	State bytes কমার সম্ভাবনা, কিন্তু quality risk বড়	Low–Medium	Very high	RFC/third-party
J. Draft SW block 1024 বনাম 2048 #58345	F এবং pool—দুটিই বদলাবে; net resident gain UNVERIFIED	Medium	Low–Medium	PR

সবচেয়ে গুরুত্বপূর্ণ capacity calculation
তোমাদের reported configuration অনুযায়ী:
MTP480
blocks · F=6

DFlash2540
blocks · F≈12

Difference+60
blocks, কিন্তু F দ্বিগুণ


২৪টি simultaneous requests-এ fixed admission accounting অনুযায়ী:
- MTP: \(24\times6=144\) block units
- DFlash2: \(24\times12=288\) block units
- অতিরিক্ত admission cost: 144 block units
তাই DFlash2-তে pool 60 blocks বাড়লেও admission overhead আরও 144 units হতে পারে। এই সংখ্যাগুলো একটি upper-bound reservation comparison, actual simultaneously allocated physical blocks নয়। F=12 যদি scheduling-time bound হয়, সেটি resident memory-তে সবসময় 12 block দখল করে আছে—এমন ধরে নেওয়া ভুল হবে।

এই distinction-ই প্রথম port-এর মূল বিষয়: reservation, actual holding এবং peak in-flight demand আলাদাভাবে মাপতে হবে।
২. Fixed admission F — সবচেয়ে দ্রুত পরীক্ষা করার মতো lever
২.১ F≈12 কেন হচ্ছে
Merlin-এর bundled source থেকে তিনটি নির্দিষ্ট জায়গা পাওয়া গেছে।

File 1: vllm/config/vllm.py

Properties:
- max_concurrent_batches
- max_in_flight_tokens
V2 + PP2 + async scheduling-এর upstream formula:
max_concurrent_batches = pp_size + 1max_in_flight_tokens = (    max_concurrent_batches    * max_num_batched_tokens)



PP2 হলে তিনটি concurrent/in-flight batch-এর allowance। এটি তিনটি batch সবসময় simultaneously GPU-তে execute হচ্ছে—তার প্রমাণ নয়; allocation safety-এর upper bound। 
GitHub


Source: https://github.com/vllm-project/vllm/blob/main/vllm/config/vllm.py

File 2: vllm/v1/kv_cache_interface.py

SlidingWindowSpec.max_admission_blocks_per_request()-এর কার্যকর formula:
num_tokens = min(    self.sliding_window - 1    + self.extra_retained_tokens    + max_in_flight_tokens,    max_model_len,)return cdiv(num_tokens, self.block_size) + 1



শেষের +1 একটি misaligned sliding-window start-এর জন্য। extra_retained_tokens speculative replay-তে প্রয়োজন হতে পারে।

অর্থাৎ SW group-এর ৬ block-কে শুধু ceil(2048/1024)=2 ধরে কমানো যাবে না। PP scheduling lag, queued work, speculative tokens এবং window-start alignment হিসাব করতে হবে।

File 3: vllm/v1/core/kv_cache_manager.py

KVCacheManager.allocate_slots()-এ full_sequence_must_fit হলে:
self.coordinator.get_num_blocks_to_allocate(    ...,    apply_admission_cap=True,)



এর মাধ্যমে full-sequence admission হিসাব হয়। HybridKVCacheCoordinator.get_num_blocks_to_allocate() প্রতিটি cache group-এর estimated blocks যোগ করে।

File 4: vllm/v1/core/single_type_kv_cache_manager.py

SingleTypeKVCacheManager.get_num_blocks_to_allocate()-এ recycling-aware cap প্রয়োগ হয়।

Merlin copy-তে এই চারটি path মিলেছে। তাই F-এর source কেবল memory-allocation code নয়; PP async scheduling-এর safety model-ও।
২.২ প্রথমে যে upstream fix যাচাই করব
PR #57050 — Mamba block admission overcount

https://github.com/vllm-project/vllm/pull/57050
- Status: Merged, 16 September 2026।
- GPU reproduction: Kimi K3, 8×B300, Mooncake-backed prefix loading।
- Problem: get_num_blocks_to_allocate() ভুলভাবে প্রয়োজনীয় Mamba blocks বেশি ধরে request-কে waiting queue-তে আটকে দিতে পারত।
- এটি F=12→F=8 optimization নয়; একটি incorrect allocation estimate-এর bug fix। 
  GitHub
Merlin port inspection:
single_type_kv_cache_manager.py
  MambaManager.get_num_blocks_to_allocate()
  SingleTypeKVCacheManager.get_num_blocks_to_allocate()

kv_cache_coordinator.py
  HybridKVCacheCoordinator.get_num_blocks_to_allocate()

kv_cache_manager.py
  KVCacheManager.allocate_slots()


আগে Merlin ad42e37-এ #57050-এর equivalent logic আছে কি না diff করতে হবে। Present/absent verification ছাড়াই patch বসানো উচিত নয়।
২.৩ Live-block admission design
আমি full-sequence admission সম্পূর্ণ বাদ দিতে বলছি না। বরং running request ও waiting request-এর admission আলাদা করতে বলছি।

বর্তমান worst-case gate-এর পাশাপাশি next-transition demand গণনা করতে হবে:
def next_step_demand(request, coordinator, planned_tokens):    demand = 0    for manager in coordinator.single_type_managers:        demand += manager.get_num_blocks_to_allocate(            request_id=request.request_id,            num_tokens=planned_tokens,            # Remaining arguments from the actual scheduler.            ...        )    return demand



এটি interface-level sketch, সরাসরি paste-ready patch নয়।

নতুন policy:
- Waiting request: bounded peak check বহাল।
- Already-running request: actual live blocks + known next-step demand।
- Boundary-crossing request: imminent Mamba-state allocation-এর headroom আগে সংরক্ষণ।
- Admission accounting-এ evictable/reclaimable blocks-কে free blocks থেকে আলাদা দেখানো।
- Full in-flight reservation কমালে producer/consumer completion delay-এর upper bound অবশ্যই যাচাই।
২.৪ #62752 — সবচেয়ে প্রাসঙ্গিক reproducer
https://github.com/vllm-project/vllm/issues/62752

Status: Open issue; repair PR প্রকাশিত নয়।

একটি CPU reproducer-এ seven running requests, Attention + circular + তিনটি Mamba group এবং 148/163/164-block pools ব্যবহৃত হয়েছে। Baseline-এ avoidable preemption হয়; experimental headroom-preserving scheduling-এ সব requests preemption ছাড়া শেষ হয়।

এটি GPU benchmark নয়। কিন্তু একই pool size-এ scheduling order বদলে useful residency বজায় রাখার বাস্তব reproducer। 
GitHub


Merlin patch surface:
- sched/scheduler.py: schedule(), running requests selection, allocation failure handling।
- kv_cache_manager.py: allocate_slots() ও actual shortfall।
- kv_cache_coordinator.py: group-wise next-step block demand।
- block_pool.py: get_num_free_blocks(), refcount/reclaimable ledger।
Test: #62752-এর CPU fixture Merlin-এর APIs অনুযায়ী port করো, তারপর 24 concurrent DFlash requests-এ boundary crossing, all-accepted/all-rejected speculation এবং simultaneous tool-return prefill পরীক্ষা করো।

Expected gain: zero additional physical blocks; preemption/recompute কমলে effective resident streams বাড়তে পারে। তোমাদের reported zero-preemption MTP baseline-এ এই optimization থেকে বড় লাভ আশা করার ভিত্তি নেই; DFlash2-তে pressure দেখা গেলে তবেই এটি decisive।

৩. Mamba/GDN padding — ২১% waste-এর নির্দিষ্ট solution
৩.১ দুটি padding সংজ্ঞা আলাদা করতে হবে
তোমাদের Mamba page-এর দেওয়া পরিমাণ:
- Actual state payload: প্রায় 3,385 units
- Padded allocation: 4,096 units
সুতরাং:
\[
\frac{4096-3385}{4096}=17.36\%
\]

এটি allocated page-এর wasted fraction।

অন্যভাবে:
\[
\frac{4096}{3385}-1=21.00\%
\]

অর্থাৎ padded page actual payload-এর তুলনায় ২১% বড়।

540-block pool ২১% বেড়ে 653 হবে—এটি সরাসরি বলা যায় না। কারণ total group bytes-এ QSA KV, drafter SW, group padding ও অন্যান্য state আছে। Mamba page padding পুরো dominant component না হলে gain অনেক কম হবে।
৩.২ PR #60967 — সবচেয়ে নির্দিষ্ট padding fix
https://github.com/vllm-project/vllm/pull/60967

Status: Open, 1 commit।

Changed design:
- vllm/platforms/interface.py — _align_hybrid_block_size()
- vllm/v1/core/kv_cache_utils.py — _get_packed_kv_cache_groups()
- vllm/v1/kv_cache_interface.py — page-size and unpadded-state representation integration
PR-এর মূল পদ্ধতি হলো packed layout-এ Mamba state-এর natural page size ব্যবহার করা, যেখানে backend memory layout সমর্থন করে।

প্রকাশিত পরীক্ষা:
- GLM-5.3-Flash real configuration
- SM90/SM100/SM120, TP1/2/4/8
- 960 CPU allocation configurations
- 140 improved, বাকিগুলো unchanged
- সর্বোচ্চ reported concurrency improvement প্রায় +26%, নির্দিষ্ট SM90/SM100 cases-এ
- Actual kernel serving throughput নয়। 
  GitHub
K100 applicability: Merlin-এর actual QSA/GDN backend packed block layout এবং attention-kernel block constraints সমর্থন করে কি না যাচাই করতে হবে। SM89 এই PR-এর reported test matrix-এ নেই।
৩.৩ Exact port sketch
kv_cache_utils.py-তে দুইটি planner output তুলনা করো:
uniform = build_uniform_groups(specs)packed = build_packed_groups(specs)for candidate in (uniform, packed):    validate_backend_layout(candidate)    calculate_total_group_bytes(candidate)    calculate_max_request_capacity(candidate)chosen = min(valid_candidates, key=objective)



এখানে objective কেবল page padding নয়; group count, metadata overhead এবং resident request capacity-ও বিবেচনা করা প্রয়োজন।

kv_cache_interface.py-এর MambaSpec-এ ইতিমধ্যে natural/padded page-size differentiation রয়েছে:
spec.unpadded_page_size_bytesspec.page_size_bytes



তবে unpadded_page_size_bytes দিয়ে arithmetic করলেই kernel stride বদলে যাবে না। compute_layer_kv_cache_shape_bytes(), tensor layout, group offsets ও runner metadata-তে consistent physical layout প্রয়োজন।

Validation:
1. CPU allocator-এ 4096 → natural page-accounting difference।
2. All ranks-এ unchanged model-state shapes and dtype।
3. FULL CUDA graph capture।
4. 2k/8k/40k/90k/116k teacher-forced oracle।
5. Memory pool bytes, blocks এবং resident count।
6. No overlapping page addresses or misaligned state pointers।
৩.৪ Alternative: Separate per-group pools
https://github.com/vllm-project/vllm/issues/55196

এই RFC-এ A10G/Falcon-H1 hybrid FP8 capacity সীমার analysis রয়েছে। Attention-only model-এ 2.00×, hybrid model-এ 32k-এ 1.84× এবং short context-এ 1.00× পাওয়া গেছে—কারণ Mamba state uniform page size নির্ধারণ করে দেয়। 
GitHub


KVCacheCoordinator-এ আলাদা sub-pool, BlockPool-এ group-local addresses ও admission accounting, এবং model runner-এ group-specific block IDs দরকার হবে।

এটি উচ্চ-effort redesign, কিন্তু FP32 recurrent state অপরিবর্তিত রেখে attention/drafter pages ছোট রাখার স্থায়ী পথ।
৩.৫ #64508 — checkpoint-internal paging
https://github.com/vllm-project/vllm/issues/64508

Status: RFC, open।

এতে একটি logical Mamba checkpoint বহু physical page-এ ভাগ করার proposal আছে। SSM member pages এবং convolution page আলাদাভাবে tracked হবে। প্রাথমিক implementation target Falcon-H1/Mamba2, single CUDA GPU, TP1, eager execution।

তোমাদের PP2 + DFlash2 + RecoverSSM + FULL graph-এ সরাসরি port-ready নয়। State mapping, kernel, PP ownership এবং graph metadata সব পরিবর্তন প্রয়োজন। 
GitHub


এটিকে competition-এর immediate patch নয়, future architecture branch হিসেবে রাখব।
৪. Draft/target KV bucket sharing — বড় capacity upside, কিন্তু correctness trap
৪.১ #58207 — group-size optimization
https://github.com/vllm-project/vllm/pull/58207

Status: Open।

Changed function:
vllm/v1/core/kv_cache_utils.py
    _get_kv_cache_groups_uniform_page_size()


বর্তমান heuristic-এর পরিবর্তে group size-এর candidate range পরীক্ষা করে সবচেয়ে কম worst-case padding bytes নির্বাচন করে। Full-attention padding long context-এর সঙ্গে বাড়ে, কিন্তু Mamba/SW padding bounded; তাই এই পার্থক্য গুরুত্বপূর্ণ।

প্রকাশিত example-এ Qwen3.6 + DFlash-এর grouping উন্নত হয়। কিন্তু অন্য real deployment-এর #59117 issue-এ #58207 backport করে মাত্র +4% থেকে −12% outcome পাওয়া গেছে। তাই এটি guaranteed capacity improvement নয়। 
GitHub+1

৪.২ #58380 — BLHNC packing
https://github.com/vllm-project/vllm/pull/58380

Status: Open, stacked on #58207।

Target/drafter layer-এর page sizes ভিন্ন হলেও block-outermost layout-এ একই packed block-এর বিভিন্ন page slots ব্যবহার করা যায়।

Qwen3.6 + DFlash-এর প্রকাশিত layout example-এ:
- 17 groups → 5 groups
- Target full attention ও drafter full attention একই packed group-এ
- Full-attention padding eliminated
এটি metadata-overhead ও KV layout improvement। এই exact five-group result তোমাদের Flash-Next-এর জন্য নয়। 
GitHub


Merlin port: kv_cache_utils.py-এ packed group construction, kv_cache_interface.py-এ layout compatibility ও physical block mapping, kv_cache_coordinator.py-এ group-index translation এবং runner block-table construction পরীক্ষা করতে হবে।
৪.৩ #58345 — sliding-window packed-block bug
https://github.com/vllm-project/vllm/pull/58345

Status: Closed, not shown as merged।

এটি অবহেলা করা যাবে না। Packed group-এ drafter SW 16-token kernel block ব্যবহার করলে প্রতিটি ছোট SW block পুরো packed allocation unit consume করতে পারে।

Reported Qwen3.6-35B + DFlash benchmark, 1×GB10, 40 GiB KV:
Layout	KV token capacity
Default layer-outermost	1,043,626
Packed, broken SW sizing	442,705
Packed, fixed SW sizing	1,538,467

Fix-এ SlidingWindowSpec-এর block size packed target block-এর সঙ্গে consistent করা হয়েছে। Benchmark-এ c=16 step time 147.94 → 146.43 ms; এই gain ছোট, কিন্তু capacity improvement বিশাল। 
GitHub


K100 implication: DFLASH_DRAFT_SW_BLOCK=1024 রাখার পরও packed planner drafter block size silently 16/32/64 করে দিচ্ছে কি না read-back করো। এটি হলে DFlash2 pool collapse হতে পারে।
৪.৪ #59117 — dedicated draft pool POC
https://github.com/vllm-project/vllm/issues/59117

Status: Open RFC/POC।

অন্য hybrid-model deployment-এ DFlash যোগ করে:
- RTX 5090 TP2: 369,121 → 33,530 KV tokens
- A800 TP1: 455,717 → 142,137 KV tokens
- RTX 5090 c=8 throughput: −21%
- c=16-এ OOM
এটি draft model-এর actual KV footprint-এর তুলনায় group-layout/admission accounting-এর অনেক বড় penalty দেখিয়েছে। 
GitHub


Port direction: target এবং drafter-এর জন্য logical allocators পৃথক, কিন্তু global byte budget shared।
Global KV budget
  ├─ Target attention / GDN groups
  └─ DFlash sliding-window groups


Coordinator-এ group-local free-space tracking, simultaneous allocation reservation এবং transactional rollback প্রয়োজন। শুধু দুইটি BlockPool বানালে unified memory constraint রক্ষা হবে না।
৪.৫ NEW HIGH warning — #60560 / #60767
এটি তোমাদের DFlash2 SW-1024 port-এর জন্য অত্যন্ত প্রাসঙ্গিক।

https://github.com/vllm-project/vllm/issues/60560

https://github.com/vllm-project/vllm/pull/60767

PR #60767 19 October 2026-এ merged।

Root cause: Sliding-window manager পুরোনো block free করে, কিন্তু worker-এর persistent block table-এ stale block ID থেকে যায়। Reused block-এর NaN bytes masked attention computation-এ ঢুকে draft acceptance নষ্ট করতে পারে।

Reported B300 DFlash2 deployment-এ 32 sessions-এ draft acceptance প্রায় 40% থেকে 27–29%-এ পড়ে। Fix forward-pass gather_block_tables-এ window-এর নিচের entries null block দিয়ে replace করে। 
GitHub


Merlin port location:
- vllm/v1/worker/gpu/model_runner.py — V2 gather_block_tables-equivalent path।
- single_type_kv_cache_manager.py — SlidingWindowManager.remove_skipped_blocks()।
- kv_cache_interface.py — SlidingWindowSpec.num_retained_tokens।
- PP DFlash metadata construction ও draft block-table read-back।
Validation: D3s/D4s 1→8→16→24 streams, long-context sliding-window wrap, forced block reuse, NaN/Inf checks, acceptance-by-position এবং baseline oracle। এই test না থাকলে capacity improvement-এর পরে hidden acceptance regression মিস হতে পারে।
৪.৬ FP8 DFlash KV এবং SW 2048 বনাম 1024
FP8 draft KV-এর feasibility Merlin SM89/TRITON_ATTN configuration-এ UNVERIFIED। dtype গ্রহণ করা এবং actual FP8 kernel চালানো এক নয়।

SW block 2048 ব্যবহার করলে SW F কমতে পারে, কিন্তু pool size ও physical page utilization বদলাবে। তাই 1024 এবং 2048-এর জন্য একই CPU-kit matrix দরকার:
Arm	SW block	Draft KV	Required evidence
D3s	1024	BF16	Existing baseline
D3-2048	2048	BF16	F, pool, resident
D3-FP8	1024	FP8	Actual kernel, oracle
D3-2048-FP8	2048	FP8	Combined impact

Draft/target KV pages সরাসরি share করা: DFlash2 target-এর attention KV ও drafter-এর attention KV একই representation নয়। Identical token positions হলেও key/value tensors বিনিময়যোগ্য নয়। তাই zero-copy sharing বলতে physical allocator sharing বোঝানো যেতে পারে; actual KV tensors alias করা যাবে না।
৫. Generated-tail checkpoints — তোমাদের agent workload-এ সবচেয়ে বড় prefill-saving lever
তোমাদের নতুন measurement অনুযায়ী fresh prefill-এর 72.7% আগের turn-এর generated tokens। এটি physical KV pool বাড়ানোর lever নয়, কিন্তু multi-turn request-এ repeated compute এবং cache-residency pressure কমানোর বড় সুযোগ।
৫.১ Upstream implementation comparison
Source	Status	Evidence	Merlin applicability
#52789 internal prefill checkpoints	Merged, 22 Aug 2026	9–25% TTFT improvement, hybrid/KDA setup	Partial; generated tail নয়
#53803 align-mode decode checkpoints	Draft PR	Decode-end retention logic, unresolved dependencies	Research reference
#50551 finalized decode checkpoints	Open PR	Finish-only publish design	Best direct starting point
#55697 application-directed checkpoints	Open RFC	Producer/consumer checkpoint coordination	Useful for agent workloads
SGLang MambaRadixCache	Implemented	Hybrid recurrent-state prefix caching	Alternative architecture
Marconi	Published paper	Up to 34.4× token-hit improvement in studied workloads	Admission/eviction research

Sources: 
GitHub+4


Raw URLs:

https://github.com/vllm-project/vllm/pull/52789

https://github.com/vllm-project/vllm/pull/53803

https://github.com/vllm-project/vllm/pull/50551

https://github.com/vllm-project/vllm/issues/55697

https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/mamba_checkpoint_pool.py

https://proceedings.mlsys.org/paper_files/paper/2025/hash/7c180af017258d239bac6248d1eb26ac-Abstract-Conference.html
৫.২ Merlin-এ সবচেয়ে গুরুত্বপূর্ণ existing safety guard
তোমাদের bundled MERLIN__scheduler.py:88–106:
configure_recoverssm_pp_checkpoints(managers)



RecoverSSM + pipeline parallelism-এ এটি:
manager.publish_decode_checkpoints = False



সেট করে।

এটি সরিয়ে দিলেই generated-tail caching চালু হবে না।

কারণ PP0 একটি delayed step process করতে পারে। PP1-এর state committed হলেও PP0-এর state পুরোনো হলে published checkpoint-এর attention KV ও GDN state এক token boundary-তে থাকবে না।

সুতরাং এই guard remove করা unsafe।
৫.৩ #50551 থেকে port-এর সঠিক শুরু
এই PR finalized Mamba decode checkpoint শুধু FINISHED_STOPPED অবস্থায় publish করার design দেয়। Aborted ও length-capped requests-এর ক্ষেত্রে checkpoint discard করার উদ্দেশ্য রয়েছে। 
GitHub


Merlin changes:

single_type_kv_cache_manager.py

MambaManager-এ private checkpoint record:
@dataclassclass PendingCheckpoint:    request_id: str    generation: int    token_position: int    state_block_id: int    producer_step: int    ready_ranks: set[int]



kv_cache_manager.py
- cache_blocks()
- pop_blocks_for_free()
- free()
- get_computed_blocks()
এখানে pending state-এর refcount, hash publication এবং cache lookup consistency রাখতে হবে।

kv_cache_coordinator.py

HybridKVCacheCoordinator-এ checkpoint-এর attention/GDN shared boundary নিশ্চিত করতে হবে।

sched/scheduler.py
- schedule()
- update_from_output()
- request finish path
- configure_recoverssm_pp_checkpoints()
এসবের মধ্যে pending completion fence যোগ করতে হবে।
৫.৪ PP-safe publication sketch
def try_publish_tail_checkpoint(pending, commit_state):    if pending.generation != commit_state.request_generation:        return False    if not commit_state.all_pp_stages_finished(pending.producer_step):        return False    if not commit_state.state_materialized_at(        pending.token_position    ):        return False    if not commit_state.attention_kv_valid_at(        pending.token_position    ):        return False    publish_atomic_checkpoint(pending)    return True



এগুলো proposed interfaces, Merlin-এর বর্তমান function names নয়।

আরও গুরুত্বপূর্ণ: Finished response-এর text থেকে পরের prompt বানানোর সময় exact token-ID prefix identity নিশ্চিত করতে হবে। Tool-call formatter, reasoning tokens বা stop-token truncation বদলে গেলে checkpoint reuse unsafe।
৫.৫ Validation matrix
- 4096 boundary-এর আগে/পরে finish।
- PP0 delayed commit।
- Both micro-batches active।
- MTP all-accepted/all-rejected।
- Abort during boundary crossing।
- Request index reuse।
- Cached block eviction before publication।
- Next-turn cold versus warm recurrent-state equivalence।
- 90k/116k long-context greedy oracle।
Expected gain: physical blocks +0; saved fresh-prefill tokens potentially substantial, কিন্তু exact reduction UNVERIFIED। 72.7% generated-tail fraction-এর সবটুকু recoverable নয়—alignment, formatting, cache eviction এবং actual checkpoint eligibility সীমা আরোপ করবে।

৬. আরও একটি বড় lever — cached recurrent state INT8, active state FP32/BF16 অপরিবর্তিত
SGLang #28185
https://github.com/sgl-project/sglang/pull/28185

Merged: 18 June 2026।

File:

python/sglang/srt/mem_cache/mamba_checkpoint_pool.py

Classes:
- Int8CheckpointStore
- MambaCheckpointPool
এটি active recurrent state quantize করে প্রতিটি decode step চালায় না। বরং inactive, radix-cached checkpoint INT8-এ রাখে; cache hit হলে dequantize করে active full-precision pool-এ ফিরিয়ে দেয়।

SGLang-এর design-এ cached checkpoint capacity প্রায় 2× হওয়ার দাবি আছে, কিন্তু এটি active running streams 2× করবে না। 
GitHub


এটাই তোমাদের জন্য সবচেয়ে নিরাপদ quantization architecture direction:

Merlin port sketch
kv_cache_manager.py ও block_pool.py-তে আলাদা checkpoint store object রাখো:
class CachedStateStore:    def store(self, state, key):        q, scale = quantize_per_channel_int8(state)        self.insert(key, q, scale)    def restore(self, key, native_dtype):        q, scale = self.lookup(key)        return dequantize(q, scale).to(native_dtype)



এই sketch-এ eviction/refcount/PP fence অনুপস্থিত; production-ready নয়।

Merlin implementation gate:
- Active FP32 SSM অপরিবর্তিত।
- INT8 conversion শুধুমাত্র published cached checkpoint-এ।
- Same-key immutable hash।
- All-rank completion fence।
- Quantize/dequantize error বনাম full-precision reference।
- Multi-turn ARC quality comparison।
Related third-party research:

https://github.com/tomalmog/carrykernel

vLLM #55196-এ আলোচিত INT8+error-feedback implementation-এ Qwen3.5-4B-তে uniform INT8 GSM8K 81%→41% নামিয়েছে, error feedback দিয়ে baseline পর্যন্ত ফিরে এসেছে বলে author report করেছেন। এটি গুরুত্বপূর্ণ warning: active-state quantization quality-safe ধরে নেওয়া যাবে না। 
GitHub

আরেকটি concrete memory optimization
SGLang PR #28302:

https://github.com/sgl-project/sglang/pull/28302

Merged: 18 June 2026।

এটি speculative GDN convolution-window intermediate buffers sliding-window layout দিয়ে deduplicate করেছে। নির্দিষ্ট D=4, convolution width=4-এ ওই buffer-এর footprint অর্ধেক হয়; Qwen3.5-0.8B-তে bit-exact kernel tests ও matched output tests প্রকাশিত। 
GitHub


এটি main KV group capacity-এর পরিবর্তে GPU runtime buffer memory কমাতে পারে। সেই memory যদি profiling-এর আগে allocation হয়, তবে liberated VRAM KV pool-এ যোগ করার সুযোগ থাকতে পারে। তোমাদের --kv-cache-memory-bytes fixed হওয়ায় সেই savings স্বয়ংক্রিয়ভাবে pool বাড়াবে না।

Port locations: Merlin RecoverSSM/GDN verification buffer allocation, ops_recoverssm.py এবং runner-side intermediate convolution state scatter। Exact kernel compatibility এখনও UNVERIFIED।

৭. Priority order: Merlin 0.32-এ বাস্তবায়ন
আমি কাজগুলো তিন স্তরে ভাগ করব।
Order	Experiment	GPU প্রয়োজন?	Decisive measurement
1	Existing F/group/block ledger	প্রথমে CPU	Per-group real pages, reserve, free
2	#58207 group-size selector	CPU first	Group count, pool, worst-case bytes
3	SW 1024 vs 2048, #58345 check	CPU + smoke	F and actual residents
4	#60767 stale SW block-table fix audit	CPU + GPU	NaN and acceptance
5	#60967 packed Mamba padding	CPU first	Unpadded bytes, pool gain
6	#62752 live boundary headroom	CPU reproducer + GPU	Avoided preemptions
7	#58380 packed layout	CPU + GPU	KV groups, pool, latency
8	Generated-tail PP-safe checkpoint	GPU required	Re-prefill tokens avoided
9	INT8 cached checkpoint pool	GPU required	Cached-prefix capacity, quality
10	Separate draft/target allocators	CPU + GPU	Physical pool and residency

Required per-rank capacity ledger
প্রতি rank থেকে এই fields collect করো:
{    "rank": rank,    "num_blocks": num_blocks,    "uniform_page_bytes": page_bytes,    "num_kv_groups": len(groups),    "group_specs": [        {            "type": spec_type,            "block_size": block_size,            "page_bytes": page_bytes,            "unpadded_page_bytes": unpadded_bytes,            "max_admission_blocks": max_admission_blocks,            "live_blocks": live_blocks,            "reclaimable_blocks": reclaimable_blocks,            "retained_checkpoint_blocks": checkpoint_blocks,        }    ],    "free_blocks": free_blocks,    "running_requests": running_requests,    "preemptions": preemptions,    "recompute_prefill_tokens": recompute_tokens,}



এতে F=12 থেকে কতটুকু genuinely live memory, কতটুকু future reserve এবং কতটুকু cache-retention overhead—সেটি বোঝা যাবে।
৮. Port-location matrix
File	Functions / surface	কী পরিবর্তন
v1/core/kv_cache_utils.py	_get_kv_cache_groups_uniform_page_size, _get_packed_kv_cache_groups, unification/planner	#58207, #58380, #60967 group layout
v1/kv_cache_interface.py	MambaSpec, SlidingWindowSpec, max_admission_blocks_per_request, page_size_bytes	F, state padding, group-specific block sizing
v1/core/kv_cache_manager.py	KVCacheManager.allocate_slots, get_computed_blocks, cache_blocks, free	Admission ledger, pending checkpoint ownership
v1/core/kv_cache_coordinator.py	HybridKVCacheCoordinator.get_num_blocks_to_allocate, allocate_new_blocks, free, find_longest_cache_hit	Group-wise allocation and common prefix boundary
v1/core/single_type_kv_cache_manager.py	SingleTypeKVCacheManager.get_num_blocks_to_allocate, SlidingWindowManager.remove_skipped_blocks, MambaManager._retire_state_blocks	Reclamation, F accounting, checkpoint retention
v1/core/block_pool.py	BlockPool, free/evict/refcount handling	Separate pools, pending references, eviction safety
v1/core/sched/scheduler.py	schedule, update_from_output, _preempt_request, configure_recoverssm_pp_checkpoints	Transition-aware scheduling and PP checkpoint fences
v1/worker/gpu/model_runner.py	gather_block_tables equivalent	Stale SW entry masking (#60767)

এগুলো exact upstream or locally inspected functions যেখানে পাওয়া গেছে; block_pool.py ও kv_cache_utils.py-এর সম্পূর্ণ Merlin source এই research bundle-এ ছিল না। তাই সেসব জায়গার exact Merlin line numbers এবং সব implementation differences UNVERIFIED।
৯. Final research verdict
প্রথমে সবচেয়ে মূল্যবান তিনটি কাজ:
1. #58207 + #58380 + #60967-এর CPU planner comparison, তবে একটি combined patch নয়—প্রতিটির incremental result।
2. F=12-এর per-group accounting এবং SW 1024/2048 comparison, সঙ্গে #60767 correctness check।
3. Generated-tail checkpoint Option B, কারণ তোমাদের 72.7% generated-history re-prefill data অনুযায়ী এটি whole-run efficiency-এর সম্ভাবনাময় lever।
#64508 internal paging, FP32 active-state compression এবং fully separate target/draft allocators বড় architectural projects; এগুলো competition deadline-এর আগে production-এ merge করার পক্ষে এখন যথেষ্ট evidence নেই।

সবচেয়ে গুরুত্বপূর্ণ সতর্কতা হলো: KV blocks বাড়লেই resident streams বাড়বে না, এবং resident streams বাড়লেই 1500 tok/s আসবে না। প্রতিটি experiment-এর final metric হওয়া উচিত identical 8k/40k/90k/116k workload-এ actual simultaneously resident streams, preemptions, useful output tok/s এবং ARC correctness।

GitHub upstream PRs, SGLang implementation, Hugging Face checkpoint এবং Marconi research source যাচাই করেছি। নতুন Merlin GPU execution করিনি; reported external improvements-কে তোমাদের SM89 TP2×PP2 results হিসেবে উপস্থাপন করিনি।
</pasted_content id="1891">

"