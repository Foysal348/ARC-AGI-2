Research task: find concrete, source-level ways to raise KV capacity (resident streams) for Qwen3.8-Flash-Next + DFlash2 on our Merlin setup (a vLLM 0.32 fork). Our notebooks are in the zips you already have (NBDF-K100, NB-S1, NB-S2, the E25 progress report).

SETUP (exact)
- Engine: Merlin v3.0.0 = a vLLM 0.32.0 fork (github.com/halt95/merlin-engine, commit ad42e37).
- Hardware: Kaggle 4x L4 (24 GB, sm_89, PCIe, no P2P), TP2 x PP2 + EP, V2 model runner, async scheduling.
- Model: Qwen3.8-Flash-Next klee100 3bpw-MTP. 48 layers: 36 GDN linear-attention + 12 QSA sparse attention. 512 experts, top-10. Hyper-connections. PLE on host.
- KV config:
  - --kv-cache-memory-bytes 7,933,526,016;
  - block 4096; prefix_match_unit 64; mamba_cache_mode align;
  - RecoverSSM + use_replayssm;
  - FP8 E4M3 QSA KV;
  - QSA indexer KV bf16 (fp8 gives 480 -> 508 blocks).
- Drafter: DFlash2 (incoai/Qwen3.8-Flash-Next-DFlash2), a 5-layer BF16 drafter with a sliding-window KV group. DFLASH_DRAFT_SW_BLOCK=1024, so the engine block becomes 1024 for the drafter group.
- Measured pools:
  - MTP: 480 blocks (page PP0 14,680,064 B / PP1 16,515,072 B), F = 6 fixed blocks per request.
  - DFlash2 SW-1024: 540 blocks, but F ~ 12 (6 Mamba + 6 sliding-window). So it holds fewer long-context residents than MTP: 64k: 18-20 vs 21; 116k: ~13.
  - The Mamba page is padded ~21 % (min block 3,385 -> padded).
- Workload: 24 concurrent agent streams, contexts 8k-116k (mean ~33k, p90 ~79k), max-model-len 131072.
- 72.7 % of fresh prefill is the previous turn's generated tokens. Cause: under RecoverSSM + PP, Mamba checkpoints are published only at prompt positions.

FIND (GitHub vLLM / SGLang / Merlin / forks first, then HF, then papers)
For EACH item give:
- exact PR / issue / commit URLs;
- the exact files and functions changed;
- merged or not;
- measured numbers and the hardware they were measured on;
- how to port it to vLLM 0.32 / Merlin: which functions in vllm/v1/core/kv_cache_utils.py, kv_cache_manager.py, kv_cache_coordinator.py, single_type_kv_cache_manager.py, block_pool.py, sched/scheduler.py, kv_cache_interface.py.

1. F (fixed admission blocks per request) for hybrid Mamba/GDN + a sliding-window drafter.
   - Why do max_admission_blocks_per_request and in-flight tokens ((PP+1) x chunk) reserve 6+6 blocks?
   - Any upstream change that reserves less, allocates SW or Mamba state lazily, or shares the SW ring across requests.
   - Live-block (not worst-case) admission.
   - vLLM #62752 (hybrid block-transition preemption) and RFC #64508 (hybrid state internal paging).
2. Mamba / GDN page padding.
   - Upstream work on unifying hybrid page sizes without padding waste: unify_hybrid_kv_cache_specs, separate pools per group, variable page size, "hybrid allocator" / "mamba page size" PRs.
   - Block-size choice for GDN state vs attention.
   - Anything that cuts the 21 % waste while keeping the FP32 SSM state.
3. Draft / target KV bucket sharing.
   - vLLM #58207 (KV-group layout optimization; Qwen3.5 + DFlash groups 14 -> 5, usable KV +27 %).
   - #58638 (KV group explosion with DFlash on hybrid GDN).
   - #58345 (sliding-window block size).
   - Any DFlash / EAGLE work that shares drafter KV or puts drafter KV on target pages.
   - FP8 drafter KV for DFlash (TRITON_ATTN on sm_89), and the effect of drafter SW block 2048 vs 1024.
4. Generated-tail prefix checkpoint for hybrid models: publishing the Mamba state at the end of generation, so the next multi-turn request hits the cache past the previous prompt.
   - vLLM PR #52789 (shared-prefix checkpoint), draft PR #53803, RFC #55697 (application-directed checkpoints).
   - SGLang hybrid radix / Mamba radix cache ("MambaRadixCache"), the Marconi paper.
   - Especially anything that works under pipeline parallelism + speculative decoding (state-commit timing / fences).
5. Anything else that raises resident long-context streams on 24 GB GPUs for hybrid SSM + attention MoE models:
   - KV quantization of the SSM state;
   - state offload of idle (tool-waiting) agent requests;
   - pause / park of requests between agent turns;
   - prefix-cache eviction policies for agent workloads.

OUTPUT
- First a table: idea, expected blocks / residents gain, confidence, port effort, URLs.
- Then one section per item, with the exact code locations and a port sketch for vLLM 0.32.
- Mark anything unverified as UNVERIFIED. No generic advice.
