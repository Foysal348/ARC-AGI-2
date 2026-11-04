"""Flash-Next patches to SGLang 0.5.20: the one patch set that Kaggle, phi1 and local servers apply.

Every patched file is pinned by the SHA-256 of its stock 0.5.20 source, and every anchor must occur
exactly as often as expected, so any other SGLang fails closed. The stock file is kept beside it as
``.py.before_serving``; patching again re-derives the file from it. Files are replaced, never
truncated, so a stock file hard-linked with uv's cache keeps its bytes there.

Always on (fixes):
- Marlin MoE under expert parallelism (gptq_kernels.py, awq_kernels.py): the runner receives the
  dispatcher's local expert map, so each EP rank runs only its own experts.
- BF16 expert scales (gptq_moe.py): upstream allocates FP16 scales even for BF16 activations.
- PLE loading (qwen4_exp.py): the PLE shard loader stored its warning flag on itself; the
  self-reference kept its captured parameter snapshot alive while weights were repacked.
- PLE offload (qwen4_exp.py): 0.5.20 allocates each offloaded PLE table on the GPU (23.84 GiB per
  rank at TP4) before moving it to pinned host memory, which exceeds an L4. The table is built on
  the meta device instead, as in upstream 89f21671; the host wrapper reads only its metadata.

Decode throughput (v14), each inert unless enabled:
- HC sharding (SGLANG_FLASH_NEXT_HC_TP=1 at TP > 1): each rank keeps a lowrank slice of the mix down
  projection and a hidden slice of the up projection. It all-gathers the [T, lowrank] activation,
  computes the sigmoid branch weights and branch mix for its hidden slice, then all-gathers
  [T, hidden], instead of all-reducing a [T, hc * hidden] partial at the same weight memory.
- TP hot-token draft head (--speculative-token-map): upstream indexes the local vocab-parallel
  lm_head shard with global hot ids, which is valid only at TP1. Every rank fills the hot rows it
  owns, one all-reduce assembles them and each rank keeps an equal slice; hot_token_id becomes the
  rank-major padded order, so the draft logits all-gather and hot_token_id[topk_index] mapping hold.
  Only draft proposals change; the target still verifies over the full vocabulary.
- Expert placement (--init-expert-location): FusedMoE loads every MoE layer whose layer_id lies
  inside the map (the MTP draft's included) into the mapped physical slots, but qwen2_moe.py remaps
  router ids only on its DeepEP path, so the normal path ran tokens through the wrong experts.
  Routing now remaps on exactly the layers the loader placed. The map balances activated experts
  per GPU from phi1 decode routing (flash_next_expert_location.json).

- Local HC up projection (SGLANG_FLASH_NEXT_HC_LOCAL_UP=1 with HC sharding): every rank keeps the whole up projection
  in INT8 (one fp16 scale per row and 32 lowrank columns, +0.18 GB per L4 at TP4) and computes the branch weights and
  mix over the full hidden width in one Triton kernel, so each mix drops its [T, hidden] all-gather (97 a decode step,
  7.3 of 85 ms at 10 running on L4). The [T, lowrank] all-gather stays. With SGLANG_FLASH_NEXT_HC_MIX_CUDA=1, batches
  of 1-64 tokens take flash_next_hc_local_mix.cuh instead (installed beside hyperconnection.py; ForgeKernel's
  hc_local_mix, cold-L2 L4 20.7 vs 46.0 µs a call at 40 tokens; bit-identical to the Triton kernel).

Memory reclaim, each inert unless enabled:
- Draft vocabulary sharing (SGLANG_FLASH_NEXT_DRAFT_SHARED_VOCAB=1, qwen4_exp.py, qwen4_exp_mtp.py,
  qwen3_5_mtp.py): the MTP draft builds its own full-vocab embedding and lm_head (0.64 GB per L4 at TP4),
  then init_lm_head replaces both with the target's only after the KV pool was sized, so they cost pool
  slots. The draft builds them on the meta device and skips loading its embedding; the target's tensors
  replace them before any draft forward, as before. 0.5.20's post-load step (loader.py) refuses meta
  tensors, so it skips modules whose own parameters are all on meta: they hold nothing to process.
- Host embedding (SGLANG_FLASH_NEXT_HOST_EMBED=1, qwen4_exp.py, eagle_worker_v2.py): the target's
  embed_tokens shard lives in pinned host memory through the same Qwen4ExpPinnedHostEmbedding (gather
  kernel plus TP all-reduce) that serves the PLE tables in CUDA graphs. Only T rows are read per forward.
  The draft shares that module instead of its CPU weight.

Collectives, inert unless enabled:
- Host-memory all-reduce (SGLANG_FLASH_NEXT_HOST_AR=1, parallel_state.py): the four-rank TP group's bf16
  all-reduces of at most 320 KiB per rank (decode) run through flash_next_host_ar.py, which patch_sglang
  installs beside SGLang's device communicators, instead of NCCL's SHM transport; larger ones (prefill)
  stay on NCCL. Every rank holds the same bits: the fp32 sum in rank order, rounded once.
- Fused MoE down projection + all-reduce (SGLANG_FLASH_NEXT_MOE_DOWN_AR=1 with the host all-reduce; parallel_state.py,
  qwen2_moe.py, fused_marlin_moe.py): for batches the host all-reduce carries (m <= 64), the MoE block runs the shared
  expert and its gate value beside the routed experts (two streams in decode-graph capture, as stock); fused_marlin_moe
  hands the down projection, after an event wait for the shared output, to
  flash_next_moe_down_ar.py (installed with its .cuh beside the host all-reduce), whose one kernel also does the top-k
  sum, the gated shared-expert add and the four-rank all-reduce while the GEMM runs. Its tiles are Marlin's without
  split K (1.9e-5 of elements differ from stock Marlin in the last bits); everything after them is bit-identical to the
  stock tail. Kaggle 4xL4, 48 random-weight layers: 271 -> 221 µs a layer at 40 tokens, 349 -> 283 at 64.

Logits exchange, inert unless enabled:
- Top-k logits (SGLANG_FLASH_NEXT_TOPK_LOGITS=K, logits_processor.py): the target's vocab-parallel logits all-gather
  (0.5 MB per token in bf16; 3.4 ms of a 78 ms step at 10 running on L4) becomes each rank's top K values and ids,
  scattered into a -inf row of the full vocabulary, so the sampler and the speculative verify run unchanged. Exact
  when every request samples with top_k <= K - n, where n is the number of tokens it biases (only negative biases),
  and asks for no logprobs: a token outside its rank's top K has K tokens above it there, at most n of them lowered.
  Shards narrower than the full vocabulary (the draft's hot-token head) and input-logprob batches keep the gather.

Prefill CUDA graph, inert unless enabled:
- tc_piecewise prefill (SGLANG_FLASH_NEXT_PREFILL_GRAPH=1 with --cuda-graph-backend-prefill tc_piecewise
  --cuda-graph-tc-compiler eager; cuda_graph_setup.py, qwen4_exp.py, hyperconnection.py, grouped_gemma_rmsnorm.py,
  hc_combine.py, tc_piecewise_cuda_graph_backend.py). Stock 0.5.20 refuses an EAGLE target's tc_piecewise prefill for
  FP4/TRTLLM-MoE replay state (#28386), which Marlin W4A16 MoE does not keep, so the env drops that skip.
  - The QSA indexer (host-side, length-dependent metadata) and PLE (conv state, host gathers) run as eager split ops
    beside GDN and QSA attention.
  - The tvm-ffi HC norm and HC combine become mutating custom ops that Dynamo can trace.
  - The compile pass leaves ops with no torch forward (the indexer) on their dispatch, and resolves each fused op's
    lazy forward once.
  - Measured on L4 (one 512 bucket): 585 -> 516 ms per prefill batch, 0.15 GB per GPU.
- Breakable prefill (--cuda-graph-backend-prefill breakable instead; qwen4_exp.py, eagle_worker_v2.py,
  breakable_cuda_graph_backend.py): the indexer and PLE ops become the backend's eager break points, the HC state becomes
  an explicit captured output (replay otherwise skips the attribute that carries it), and the MTP draft's prefill stays
  eager. vast.ai 4xL4 at 1024-token chunks: -3.4 to -3.8% time against tc_piecewise, 0.36 against 0.29 GB per GPU.

Decode context parallelism, inert unless enabled:
- QSA DCP (SGLANG_FLASH_NEXT_QSA_DCP=1 with --decode-context-parallel-size > 1; qwen_sparse_attn_backend.py,
  qsa_kv_pool.py, kv_cache_configurator.py, pool_configurator.py): stock QSA has no DCP path, so every rank kept
  every token at its logical slot and crashed once slots passed its physical rows. The target's QSA backend now
  writes each token only on the rank that owns it (slot % dcp at row slot // dcp) and merges the ranks' sparse
  attention through their LSE (flash_next_qsa_dcp.py, installed beside QSA); the indexer cache stays replicated
  over the logical slots, and the pool's per-token cost counts it once per DCP rank.

QSA indexer decode scoring, inert unless enabled:
- SGLANG_FLASH_NEXT_QSA_DECODE=1 (qsa_indexer.py): speculative verify and draft-extend CUDA graphs score the indexer
  through flash_next_qsa_index_decode.cuh (ForgeKernel's qsa_index_decode, installed beside the indexer), one CTA per
  request's 4 verify rows and 64-key tile, instead of TileLang's CTAs per row over the whole context limit. It reads
  page-table row 4i for rows 4i .. 4i + 3, which graph-built rows guarantee only at exactly 4 rows per request, so it is
  chosen at capture from that shape; draft decode and eager batches keep TileLang. Same math and order as TileLang
  (MMA k order, ReLU, head sum, fast-math division by sqrt(128)). Cold-L2 L4, 40 rows: 82.8 against 326 us at 32k mean
  context.
"""
import hashlib
import json
import os
import subprocess
from pathlib import Path

SGLANG_VERSION = "0.5.20"
HOT_TOKENS = "flash_next_hot_tokens.json"
EXPERT_LOCATION = "flash_next_expert_location.json"
HOST_ALL_REDUCE_MODULE = "flash_next_host_ar.py"
QSA_DCP_MODULE = "flash_next_qsa_dcp.py"
MOE_DOWN_AR_MODULE = "flash_next_moe_down_ar.py"
MOE_DOWN_AR_KERNEL = "flash_next_moe_down_ar.cuh"
HC_MIX_KERNEL = "flash_next_hc_local_mix.cuh"
QSA_DECODE_KERNEL = "flash_next_qsa_index_decode.cuh"
MODULES = {HOST_ALL_REDUCE_MODULE: "srt/distributed/device_communicators", QSA_DCP_MODULE: "srt/layers/attention/qsa",
           MOE_DOWN_AR_MODULE: "srt/distributed/device_communicators",
           MOE_DOWN_AR_KERNEL: "srt/distributed/device_communicators", HC_MIX_KERNEL: "srt/layers",
           QSA_DECODE_KERNEL: "srt/layers/attention/qsa"}
# The patched QSA backend imports the port as sglang.srt.layers.attention.qsa.flash_next_dcp.
INSTALLED_NAMES = {QSA_DCP_MODULE: "flash_next_dcp.py"}
PINS = {
    "srt/layers/hyperconnection.py": "338bf29366565b357a15e9650a0fd7f83cdf0456f0d995b0006ad3c1edaa9a8f",
    "srt/hardware_backend/gpu/quantization/gptq_kernels.py": "4134148fcc97957eee13f0e3f4b0bd4ddaca52a5fd36ccc81e8b43c8242493a7",
    "srt/hardware_backend/gpu/quantization/awq_kernels.py": "281b7d7282dcbebe22f208454c6bd2dea20b346b8bc01133fccc7bd30ef4609f",
    "srt/layers/quantization/gptq/schemes/gptq_moe.py": "135e004f7613a59aa0765cbe3d779c9bb0c459f3a199b868358759e020aaeb5d",
    "srt/models/qwen4_exp.py": "1f85793336bad76ce8cb4841068a0b5e8302ec962799972c8d12291f9c64068e",
    "srt/speculative/eagle_worker_v2.py": "8fc9285c458d746874c96da7b16bef2733509dea3d1170cbad9e787bcde67754",
    "srt/models/qwen2_moe.py": "f37a31ea813d04241879614a6c0d011d22e18551814c4809d9321714dbf37795",
    "srt/models/qwen4_exp_mtp.py": "243531fbb3d256fff68db299a2f902ba3cc38227735f2ab0a90fb5be311f9f60",
    "srt/models/qwen3_5_mtp.py": "1ebc9643333046d5621ad8b1bd6bd005686988c0104576bd1ea6c3d5e51acf24",
    "srt/model_loader/loader.py": "69043753c6158f200c17ffda174282cef62430320d3d7caab361c39426a1902e",
    "srt/distributed/parallel_state.py": "bcfbc3386b58d6d7ce0d426a50d6ebcd8e8a9bfe22d3e0d1172b7ab041d25f92",
    "srt/layers/attention/qwen_sparse_attn_backend.py": "e4f0c2f3d5295d7865d293d5e64c41ab30c3e48fc17373900edb2250e2ff77b9",
    "srt/mem_cache/qsa_kv_pool.py": "eaab189e2f70d18495a2e6e2d158efcbcbebbb951475fc4cf0b426c4c5598fa8",
    "srt/mem_cache/kv_cache_configurator.py": "691bef2acb4a9dfbc815bac5d130723b1b0ec8b9cb45fd751d586b17a1b633a6",
    "srt/model_executor/pool_configurator.py": "0d5c9ac48c15c95e63247306b0d9d220e9f2b9dd60b4f76472a690eb701eed21",
    "srt/layers/logits_processor.py": "f6fa7a79bf9be1ac4bcb8591939341b8418dfd7612ba2e042becf8c8b411fb9c",
    "srt/model_executor/model_runner_components/cuda_graph_setup.py": "ff04e7af8871c650932bdcaffb6982ce0e46fcb65f6fc9d7a0d80bead5341597",
    "srt/model_executor/runner_backend/tc_piecewise_cuda_graph_backend.py": "5c6ab2dbebe91699f2e3fd84f7081fe4a825575923607c39849ef86e0126628b",
    "kernels/ops/layernorm/grouped_gemma_rmsnorm.py": "e4aa93c0662df1a5bb4413a08a7768ca062eb5b3eceb7d0519078a68cd3ed835",
    "kernels/ops/elementwise/hc_combine.py": "12f8df2c0ceda53425f3a9c44c39815f1c91c1eb3d8043b04cefdfc685ec1275",
    "srt/layers/moe/fused_moe_triton/fused_marlin_moe.py": "884a088ee56ad1ea8786c7a9ba1349f01a99442a5adf1f53a445acc6f3f9581c",
    "srt/layers/attention/qsa/qsa_indexer.py": "1ca45f9ed46c088bd8ba734f34bc085d852c0141f5e7f5f0f933603b33e14b9c",
    "srt/model_executor/runner_backend/breakable_cuda_graph_backend.py": "2172447619e3af9fc4c024b016d5a18b7806139762d2252233163547d593860f",
}

HC_SHARDING = [
    ('''        norm_dim = (
''', '''        # FLASH_NEXT_SERVING: sharded HC mixes
        import os

        self._hc_tp_group = None
        mix_lowrank, mix_out = config.hc_lowrank, self.hc_count * self.hidden_size
        if use_mix and os.environ.get("SGLANG_FLASH_NEXT_HC_TP") == "1":
            from sglang.srt.distributed import get_tp_group

            group = get_tp_group()
            if group.world_size > 1:
                if mix_lowrank % group.world_size or self.hidden_size % group.world_size:
                    raise ValueError("HC lowrank and hidden size must divide the TP world size")
                self._hc_tp_group = group
                mix_lowrank //= group.world_size
                self._hc_width = self.hidden_size // group.world_size
                self._hc_start = group.rank_in_group * self._hc_width
                mix_out = self.hc_count * self._hc_width
        norm_dim = (
''', 1),
    ('''            self.input_mix_weight_down = nn.Linear(
                self.hidden_size * self.hc_count,
                self.config.hc_lowrank,
''', '''            self.input_mix_weight_down = nn.Linear(
                self.hidden_size * self.hc_count,
                mix_lowrank,
''', 1),
    ('''            self.input_mix_weight_up = nn.Linear(
                self.config.hc_lowrank,
                self.hc_count * self.hidden_size,
''', '''            self.input_mix_weight_up = nn.Linear(
                self.config.hc_lowrank,
                mix_out,
''', 1),
    ('''            lowrank = self.config.hc_lowrank
''', '''            if self._hc_tp_group is not None:
                self.input_mix_weight_down.weight.weight_loader = self._hc_load_down
                self.input_mix_weight_up.weight.weight_loader = self._hc_load_up
            lowrank = self.config.hc_lowrank
''', 1),
    ('''        self._mix_compute = torch.compile(_mix_compute)
''', '''        def _hc_down(x, down, hc):
            return F.silu(F.linear(x, down) / hc)

        def _hc_finish(lowrank, up, x, hc, hs, start, width):
            weights = torch.sigmoid(F.linear(lowrank, up)).unflatten(-1, (hc, width))
            branches = x.unflatten(-1, (hc, hs))[..., start:start + width]
            return (weights * branches).mean(dim=-2)

        self._hc_down = torch.compile(_hc_down)
        self._hc_finish = torch.compile(_hc_finish)
        self._mix_compute = torch.compile(_mix_compute)
''', 1),
    ('''    def combine(self, block_output: torch.Tensor, residuals) -> torch.Tensor:
''', '''    def _hc_load_down(self, param, loaded_weight):
        group = self._hc_tp_group
        expected = [param.shape[0] * group.world_size, param.shape[1]]
        if list(loaded_weight.shape) != expected:
            raise ValueError(f"HC checkpoint shape {list(loaded_weight.shape)} != {expected}")
        width = param.shape[0]
        with torch.no_grad():
            param.copy_(loaded_weight.narrow(0, group.rank_in_group * width, width))

    def _hc_load_up(self, param, loaded_weight):
        expected = [self.hc_count * self.hidden_size, self.config.hc_lowrank]
        if list(loaded_weight.shape) != expected:
            raise ValueError(f"HC checkpoint shape {list(loaded_weight.shape)} != {expected}")
        shard = loaded_weight.unflatten(0, (self.hc_count, self.hidden_size))
        shard = shard[:, self._hc_start:self._hc_start + self._hc_width]
        with torch.no_grad():
            param.copy_(shard.reshape(param.shape))

    def combine(self, block_output: torch.Tensor, residuals) -> torch.Tensor:
''', 1),
    ('''        if (
            self._jit_mix_ok
''', '''        if self._hc_tp_group is not None:
            group = self._hc_tp_group
            lowrank = group.all_gather(
                self._hc_down(hyper_input_normed, self.input_mix_weight_down.weight, self.hc_count),
                dim=-1,
            )
            part = self._hc_finish(
                lowrank, self.input_mix_weight_up.weight, hyper_input_normed,
                self.hc_count, self.hidden_size, self._hc_start, self._hc_width,
            )
            mixed_input = group.all_gather(part.to(self.params_dtype).contiguous(), dim=-1)
            return mixed_input, (hyper_input, hyper_input_normed)
        if (
            self._jit_mix_ok
''', 1),
]

GPTQ_EXPERT_MAP = [
    ('''            is_k_full=self.is_k_full,
        )

        return self.runner.run(dispatch_output, quant_info)''', '''            is_k_full=self.is_k_full,
            expert_map=getattr(layer.dispatcher, 'local_expert_mapping', None),
        )

        return self.runner.run(dispatch_output, quant_info)''', 1),
]

AWQ_EXPERT_MAP = [
    ('''            weight_bits=self.quant_config.weight_bits,
        )
        return self.runner.run(dispatch_output, quant_info)''', '''            weight_bits=self.quant_config.weight_bits,
            expert_map=getattr(layer.dispatcher, 'local_expert_mapping', None),
        )
        return self.runner.run(dispatch_output, quant_info)''', 1),
]

BF16_SCALES = [("dtype=torch.half", "dtype=params_dtype", 2)]

PLE_ON_META = [
    ('''        self.ngram_embedding = VocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim_per_ngram,
            params_dtype=(
                torch.float8_e4m3fn
                if _ple_table_is_fp8(config, quant_config, ngram_prefix)
                else torch.bfloat16
            ),
            output_dtype=torch.bfloat16,
            use_attn_tp_group=self.use_attn_tp_ngram,
        )
''', '''        # FLASH_NEXT_SERVING: an offloaded table is never allocated on the device
        with torch.device("meta") if getattr(config, "ple_offload_embedding", False) else nullcontext():
            self.ngram_embedding = VocabParallelEmbedding(
                padded_vocab_size,
                self.head_dim_per_ngram,
                params_dtype=(
                    torch.float8_e4m3fn
                    if _ple_table_is_fp8(config, quant_config, ngram_prefix)
                    else torch.bfloat16
                ),
                output_dtype=torch.bfloat16,
                use_attn_tp_group=self.use_attn_tp_ngram,
            )
''', 1),
]

PLE_LOADER = [
    ('''        def load_qwen4_exp_ple_shard(name: str, loaded_weight: torch.Tensor) -> bool:
''', '''        warned_ple_downcast = False

        def load_qwen4_exp_ple_shard(name: str, loaded_weight: torch.Tensor) -> bool:
            nonlocal warned_ple_downcast
''', 1),
    ('if not getattr(load_qwen4_exp_ple_shard, "_warned_downcast", False):',
     "if not warned_ple_downcast:", 1),
    ("load_qwen4_exp_ple_shard._warned_downcast = True", "warned_ple_downcast = True", 1),
]

TP_HOT_HEAD = [
    ('''            if self.hot_token_id is not None and head is not None:
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]
''', '''            if self.hot_token_id is not None and head is not None:
                # FLASH_NEXT_SERVING: TP-balanced hot-token draft head
                head, self.hot_token_id = _tp_hot_head(head, self.hot_token_id, target_lm_head)
''', 1),
]

TP_HOT_HEAD_FUNCTION = '''

def _tp_hot_head(head, hot_token_id, target_lm_head):
    """Balanced per-rank slice of the hot lm_head rows and the rank-major hot id order."""
    from sglang.srt.distributed import get_tp_group

    group = get_tp_group()
    hot = hot_token_id.to(head.device)
    if group.world_size == 1:
        return head[hot].clone(), hot
    indices = getattr(target_lm_head, "base_layer", target_lm_head).shard_indices
    start, end = indices.org_vocab_start_index, indices.org_vocab_end_index
    per_rank = -(-hot.numel() // group.world_size)
    hot = torch.cat([hot, hot[:1].expand(per_rank * group.world_size - hot.numel())])
    rows = torch.zeros(hot.numel(), head.shape[1], dtype=head.dtype, device=head.device)
    local = (hot >= start) & (hot < end)
    rows[local] = head[hot[local] - start]
    rows = group.all_reduce(rows)  # every hot row is nonzero on exactly its owning rank
    rank = group.rank_in_group
    return rows[rank * per_rank:(rank + 1) * per_rank].contiguous(), hot
'''

EXPERT_ROUTING = [
    ('''        router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        if defer_finalize:
''', '''        router_logits, _ = self.gate(hidden_states)
        # FLASH_NEXT_SERVING: remap router ids wherever the loader placed experts by the map
        topk_output = self.topk(
            hidden_states,
            router_logits,
            expert_location_dispatch_info=_flash_next_expert_dispatch(self.layer_id),
        )
        if defer_finalize:
''', 1),
]

EXPERT_ROUTING_FUNCTION = '''

def _flash_next_expert_dispatch(layer_id):
    """Dispatch info for exactly the MoE layers FusedMoE loaded through the expert map.

    The loader places every layer with layer_id below the map's layer count, the MTP draft's included,
    and keeps identity beyond it. Without --init-expert-location the dispatch algorithm is unset and
    init_new returns None.
    """
    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata

    metadata = get_global_expert_location_metadata()
    if metadata is None or layer_id >= metadata.physical_to_logical_map.shape[0]:
        return None
    return ExpertLocationDispatchInfo.init_new(layer_id=layer_id)
'''

EMBED_PLACEMENT = [
    ('''    def _build_embed_tokens(self, config: Qwen4ExpTextConfig) -> nn.Module:
        return VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            use_attn_tp_group=is_dp_attention_enabled(),
        )
''', '''    def _build_embed_tokens(self, config: Qwen4ExpTextConfig) -> nn.Module:
        # FLASH_NEXT_SERVING: a draft table the target's replaces is never allocated, and the
        # target's may live in pinned host memory like the PLE tables.
        placement = _flash_next_embed_placement(config)
        with torch.device("meta") if placement != "gpu" else nullcontext():
            embedding = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                use_attn_tp_group=is_dp_attention_enabled(),
            )
        if placement == "host":
            # The host wrapper keeps the PLE tables' per-tensor scale; a bf16 table's is 1 and unused.
            embedding.register_buffer("weight_scale", torch.ones(1, dtype=torch.bfloat16), persistent=False)
            return Qwen4ExpPinnedHostEmbedding(embedding, backend="pinned")
        return embedding
''', 1),
]

EMBED_PLACEMENT_FUNCTION = '''

def _flash_next_embed_placement(config):
    """Where an embedding table (and the MTP draft's lm_head) lives.

    "meta" for an MTP draft whose tables init_lm_head replaces with the target's, "host" for a target
    table served from pinned host memory, "gpu" otherwise.
    """
    import os

    if getattr(config, "tie_word_embeddings", False):
        return "gpu"
    if getattr(config, "_flash_next_draft", False):
        shared = not getattr(config, "mtp_use_dedicated_embeddings", False)
        return "meta" if shared and os.environ.get("SGLANG_FLASH_NEXT_DRAFT_SHARED_VOCAB") == "1" else "gpu"
    return "host" if os.environ.get("SGLANG_FLASH_NEXT_HOST_EMBED") == "1" else "gpu"
'''

DRAFT_META = [
    ('''        config.ple_layer_ids = []
''', '''        config.ple_layer_ids = []
        # FLASH_NEXT_SERVING: the draft's embedding and lm_head may be built on the meta device.
        config._flash_next_draft = True
''', 1),
    ('''        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("model.shared_head.head", prefix),
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
        )
''', '''        from sglang.srt.models.qwen4_exp import _flash_next_embed_placement

        with torch.device("meta") if _flash_next_embed_placement(config) == "meta" else ExitStack():
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("model.shared_head.head", prefix),
                use_attn_tp_group=get_parallel().enable_dp_lm_head,
            )
''', 1),
]

DRAFT_EMBED_LOAD = [
    ('''                param_name = "model.embed_tokens.weight"
                if param_name in params_dict:
''', '''                param_name = "model.embed_tokens.weight"
                # FLASH_NEXT_SERVING: a meta draft table is replaced by the target's; nothing to load.
                if param_name in params_dict and not params_dict[param_name].is_meta:
''', 1),
]

SHARE_HOST_EMBED = [
    ('''            # Share the embedding and lm_head
            self.draft_runner.model.set_embed_and_head(embed, head)
''', '''            # Share the embedding and lm_head
            # FLASH_NEXT_SERVING: a host-resident target table is shared as its module.
            embed = _flash_next_share_host_embed(
                self.target_worker.model_runner.model, self.draft_runner.model, embed
            )
            self.draft_runner.model.set_embed_and_head(embed, head)
''', 1),
]

SHARE_HOST_EMBED_FUNCTION = '''

def _flash_next_share_host_embed(target_model, draft_model, embed):
    """Give the draft the target's host embedding module; a CPU table cannot back a CUDA lookup."""
    if embed is None or embed.device.type != "cpu":
        return embed
    owner = next(module for module in target_model.modules() if getattr(module, "weight", None) is embed)
    draft_model.model.embed_tokens = owner
    return None
'''

META_PLACEHOLDERS = [
    ('''    def postprocess_weights(model, target_device):
        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
''', '''    def postprocess_weights(model, target_device):
        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            # FLASH_NEXT_SERVING: a draft table left on meta (the target's replaces it) has nothing to process.
            if quant_method is not None and not _flash_next_meta_placeholder(module):
''', 1),
]

META_PLACEHOLDERS_FUNCTION = '''

def _flash_next_meta_placeholder(module):
    """Whether every parameter the module owns is on the meta device (the shared-vocab draft tables)."""
    parameters = list(module.parameters(recurse=False))
    return bool(parameters) and all(parameter.is_meta for parameter in parameters)
'''

HOST_ALL_REDUCE = [
    ('''        if use_custom_allreduce and self.world_size > 1:
            # Initialize a custom fast all-reduce implementation.
''', '''        # FLASH_NEXT_SERVING: host-memory decode all-reduce for the four-rank TP group
        self.flash_next_host_ar = None
        if (
            os.environ.get("SGLANG_FLASH_NEXT_HOST_AR") == "1"
            and group_name == "tp"
            and self.world_size == 4
        ):
            from sglang.srt.distributed.device_communicators.flash_next_host_ar import (
                create_host_all_reduce,
            )

            self.flash_next_host_ar = create_host_all_reduce(
                self.cpu_group, self.device, self.rank_in_group
            )
        # FLASH_NEXT_SERVING: fused MoE down projection + host all-reduce (qwen2_moe.py, fused_marlin_moe.py)
        self.flash_next_moe_down_ar = None
        if self.flash_next_host_ar is not None and os.environ.get("SGLANG_FLASH_NEXT_MOE_DOWN_AR") == "1":
            from sglang.srt.distributed.device_communicators.flash_next_moe_down_ar import (
                create_moe_down_all_reduce,
            )

            self.flash_next_moe_down_ar = create_moe_down_all_reduce(
                self.cpu_group, self.device, self.rank_in_group, self.flash_next_host_ar
            )
        if use_custom_allreduce and self.world_size > 1:
            # Initialize a custom fast all-reduce implementation.
''', 1),
    ('''        should_use_pymscclpp_allreduce: Optional[bool] = None,
    ) -> Optional[str]:
''', '''        should_use_pymscclpp_allreduce: Optional[bool] = None,
    ) -> Optional[str]:
        host_ar = getattr(self, "flash_next_host_ar", None)
        if host_ar is not None and host_ar.should_all_reduce(input_):
            return "flash_next_host"
''', 1),
    ('''        ca_comm = self.ca_comm
        qr_comm = self.qr_comm
''', '''        if outplace_all_reduce_method == "flash_next_host":
            return self.flash_next_host_ar.all_reduce(input_)
        ca_comm = self.ca_comm
        qr_comm = self.qr_comm
''', 1),
]

FUSED_MOE_DOWN_AR = [
    ('''        use_fused_gate = (
            self.shared_expert_gate is not None
            and not use_intel_amx_backend(self.shared_expert_gate)
            and not is_npu()
        )

        if hidden_states.shape[0] == 0:
''', '''        use_fused_gate = (
            self.shared_expert_gate is not None
            and not use_intel_amx_backend(self.shared_expert_gate)
            and not is_npu()
        )

        # FLASH_NEXT_SERVING: fused down projection + all-reduce for batches the host all-reduce carries
        fused = _flash_next_fused_moe(self, hidden_states, use_fused_gate, defer_finalize)
        if fused is not None:
            return fused.view(num_tokens, hidden_dim)

        if hidden_states.shape[0] == 0:
''', 1),
]

FUSED_MOE_DOWN_AR_FUNCTION = '''

def _flash_next_fused_moe(block, hidden_states, use_fused_gate, defer_finalize):
    """The MoE layer through flash_next_moe_down_ar (SGLANG_FLASH_NEXT_MOE_DOWN_AR=1), or None for the stock path.

    Taken for batches the host all-reduce would carry, outside torch.compile, when the block's tail is the stock one:
    a separate shared expert with the fused gate, then the TP all-reduce. fused_marlin_moe's down projection returns
    the all-reduced layer output. In decode-graph capture the streams follow stock forward_normal_dual_stream: the
    shared expert and its gate value on the current stream, the router and routed experts on the alt stream, whose
    fused down projection waits for them on an event (elsewhere all of it runs on the current stream, as stock does).
    If the experts did not take the fused kernel (another weight format, as the draft's may be), the stock tail
    finishes the layer here.
    """
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers.moe.utils import should_skip_post_experts_all_reduce

    comm = getattr(get_tp_group(), "flash_next_moe_down_ar", None)
    m = hidden_states.shape[0]
    if (
        comm is None
        or defer_finalize
        or not use_fused_gate
        or block.shared_expert is None
        or block.enable_shared_expert_fusion
        or m == 0
        or m > comm.max_m
        or block.tp_size <= 1
        or hidden_states.dtype != torch.bfloat16
        or not hidden_states.is_contiguous()
        or torch.compiler.is_compiling()
        or not get_moe_a2a_backend().is_none()
        or should_skip_post_experts_all_reduce(is_tp_path=True)
    ):
        return None
    from sglang.srt.distributed.device_communicators import flash_next_moe_down_ar as fused

    gate_weight = block.shared_expert_gate.weight.squeeze()
    if block.alt_stream is not None and get_is_capture_mode():
        current = torch.cuda.current_stream()
        block.alt_stream.wait_stream(current)
        shared_output = block._forward_shared_experts(hidden_states.clone(), apply_gate=False)
        gate = fused.shared_gate(hidden_states, gate_weight)
        shared_output.record_stream(block.alt_stream)
        gate.record_stream(block.alt_stream)
        ready = current.record_event()
        with torch.cuda.stream(block.alt_stream):
            with fused.down_context(comm, shared_output, gate, ready) as context:
                routed = block._forward_router_experts(hidden_states)
        current.wait_stream(block.alt_stream)
    else:
        shared_output = block._forward_shared_experts(hidden_states, apply_gate=False)
        gate = fused.shared_gate(hidden_states, gate_weight)
        with fused.down_context(comm, shared_output, gate) as context:
            routed = block._forward_router_experts(hidden_states)
    if context.out is not None:
        return routed
    fused_gate_sigmoid_mul_add(hidden_states, gate_weight, shared_output, routed)
    return tensor_model_parallel_all_reduce(routed)
'''

MARLIN_DOWN_HOOK = [
    ('''    if expert_map is not None:
        intermediate_cache3.zero_()

    intermediate_cache3 = moe_wna16_marlin_gemm(
''', '''    if expert_map is not None:
        intermediate_cache3.zero_()

    # FLASH_NEXT_SERVING: inside qwen2_moe's fused path, the down projection, top-k sum, gated shared-expert add and
    # TP all-reduce in one kernel; None when the call is not the one the kernel was built for.
    from sglang.srt.distributed.device_communicators.flash_next_moe_down_ar import try_fused_down

    fused_output = try_fused_down(
        intermediate_cache2, intermediate_cache3, w2, w2_scale, sorted_token_ids, expert_ids,
        num_tokens_post_padded, topk_weights, top_k=topk, block_size_m=block_size_m, expert_map=expert_map,
        num_bits=num_bits, w2_zeros=w2_zeros, w2_bias=w2_bias, g_idx2=g_idx2, w2_global_scale=w2_global_scale,
        routed_scaling_factor=routed_scaling_factor, activation=activation, is_gated=is_gated,
        clamp_limit=clamp_limit, gemm1_alpha=gemm1_alpha, is_k_full=is_k_full,
    )
    if fused_output is not None:
        return fused_output

    intermediate_cache3 = moe_wna16_marlin_gemm(
''', 1),
]

QSA_DCP_BACKEND = [
    ('''        self._graph_extend_lens = None
        self._graph_extend_lens_pin = None
''', '''        self._graph_extend_lens = None
        self._graph_extend_lens_pin = None
        # FLASH_NEXT_SERVING: the target's decode context parallelism (None for the draft or without DCP)
        from sglang.srt.layers.attention.qsa.flash_next_dcp import create_qsa_dcp

        self.flash_next_dcp = create_qsa_dcp(self, runner)
''', 1),
    ('''            self.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )
''', '''            self.token_to_kv_pool.set_kv_buffer(
                layer,
                (
                    forward_batch.out_cache_loc
                    if self.flash_next_dcp is None
                    else self.flash_next_dcp.write_loc(forward_batch.out_cache_loc)
                ),
                k,
                v,
            )
''', 2),
    ('''        # The validated chunk-prefill kernel consumes tightly packed full-context
        # K/V. Current-chunk K/V has already been committed to the cache above.
''', '''        if self.flash_next_dcp is not None:
            output = self.flash_next_dcp.chunk_prefill(
                q, layer, forward_batch, topk_indices, sequence_lens, cu_seqlens_q
            )
            return self._pad_extend_output(output, num_output_rows)
        # The validated chunk-prefill kernel consumes tightly packed full-context
        # K/V. Current-chunk K/V has already been committed to the cache above.
''', 1),
    ('''        metadata = self._resolve_metadata(forward_batch)
        topk_indices = topk_indices.to(torch.int32).contiguous()
        trtllm_decode = _resolve_trtllm_sparse_decode()
''', '''        metadata = self._resolve_metadata(forward_batch)
        topk_indices = topk_indices.to(torch.int32).contiguous()
        if self.flash_next_dcp is not None:
            return self.flash_next_dcp.paged(
                q, layer, forward_batch, metadata, topk_indices
            )
        trtllm_decode = _resolve_trtllm_sparse_decode()
''', 1),
]

QSA_DCP_POOL = [
    ('''        quant_method=None,
        post_capture_active: bool = False,
    ):
        if page_size <= 1 or page_size % qsa_compress_ratio != 0:
''', '''        quant_method=None,
        post_capture_active: bool = False,
        qsa_index_scale: int = 1,
    ):
        if page_size <= 1 or page_size % qsa_compress_ratio != 0:
''', 1),
    ('''        state_size = size + page_size
''', '''        # FLASH_NEXT_SERVING: under DCP the replicated indexer cache spans the logical slots
        state_size = (size + page_size) * qsa_index_scale
''', 1),
]

QSA_DCP_CONFIGURATOR = [
    ('''        from sglang.srt.mem_cache.qsa_kv_pool import (
            QSATokenToKVPool,
        )

        qsa_profile = parse_qsa_profile(self.model_config.hf_config)
''', '''        from sglang.srt.mem_cache.qsa_kv_pool import (
            QSATokenToKVPool,
        )
        from sglang.srt.layers.attention.qsa.flash_next_dcp import qsa_index_copies

        qsa_profile = parse_qsa_profile(self.model_config.hf_config)
''', 1),
    ('''                qsa_token_topk=qsa_profile.budget,
                num_request_slots=req_to_token_pool.req_to_token.shape[0],
            )
''', '''                qsa_token_topk=qsa_profile.budget,
                num_request_slots=req_to_token_pool.req_to_token.shape[0],
                # FLASH_NEXT_SERVING: replicated indexer cache under decode context parallelism
                qsa_index_scale=qsa_index_copies(self.is_draft_worker),
            )
''', 1),
]

QSA_DCP_CELL = [
    ('''        cell_size += self._compute_qsa_cell_size(
            hf_config=model_config.hf_config, num_layers=num_layers
        )
        return cell_size
''', '''        # FLASH_NEXT_SERVING: one replicated indexer-cache copy per logical slot a DCP row stands for
        from sglang.srt.layers.attention.qsa.flash_next_dcp import qsa_index_copies

        cell_size += qsa_index_copies(kvc.is_draft_worker) * self._compute_qsa_cell_size(
            hf_config=model_config.hf_config, num_layers=num_layers
        )
        return cell_size
''', 1),
]

TOPK_LOGITS = [
    ('''            else:
                logits = self._logits_gatherer(logits)
''', '''            elif _flash_next_topk_applies(self, logits, logits_metadata):
                logits = _flash_next_topk_gather(self, logits)
            else:
                logits = self._logits_gatherer(logits)
''', 1),
]

TOPK_LOGITS_FUNCTION = '''

# FLASH_NEXT_SERVING: top-k logits exchange (SGLANG_FLASH_NEXT_TOPK_LOGITS=K)
_FLASH_NEXT_TOPK = int(__import__("os").environ.get("SGLANG_FLASH_NEXT_TOPK_LOGITS") or 0)


def _flash_next_topk_applies(processor, logits, logits_metadata):
    """Full-vocabulary shards only (the draft's hot-token head keeps its gather), and no input logprobs."""
    return (_FLASH_NEXT_TOPK > 0 and logits.dim() == 2
            and logits.shape[-1] * get_tp_group().world_size >= processor.vocab_size
            and not logits_metadata.extend_return_logprob)


def _flash_next_topk_gather(processor, logits):
    """Each rank's top K logits and global ids, scattered into a -inf row of the full vocabulary on every rank."""
    group = get_tp_group()
    rows, shard, k = logits.shape[0], logits.shape[-1], _FLASH_NEXT_TOPK
    start = group.rank_in_group * shard
    if start + shard > processor.vocab_size:  # Vocabulary padding never competes.
        logits = logits.clone()
        logits[:, max(processor.vocab_size - start, 0):] = float("-inf")
    values, index = torch.topk(logits, k, dim=-1)
    packed = torch.cat((values.float().view(torch.int32), (index + start).to(torch.int32)), dim=-1)
    from sglang.srt.distributed import tensor_model_parallel_all_gather

    gathered = tensor_model_parallel_all_gather(packed.contiguous(), dim=-1).view(rows, group.world_size, 2, k)
    full = torch.full((rows, shard * group.world_size), float("-inf"), dtype=logits.dtype, device=logits.device)
    full.scatter_(1, gathered[:, :, 1].reshape(rows, -1).long(),
                  gathered[:, :, 0].reshape(rows, -1).view(torch.float32).to(logits.dtype))
    return full
'''

# Prefill CUDA graph (tc_piecewise), SGLANG_FLASH_NEXT_PREFILL_GRAPH=1; edits made on phi1 2026-10-29
# (inducer/results/prefill_graph_phi1_20261029/). Each applies after this file's earlier edits to the same source.

PREFILL_GRAPH_SETUP = [
    ('''    if (
        model_runner.spec_algorithm.is_eagle()
        and not model_runner.is_draft_worker
        and get_server_return_hidden_states_mode() < CaptureHiddenMode.FULL
        and check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
    ):''', '''    if (
        __import__("os").environ.get("SGLANG_FLASH_NEXT_PREFILL_GRAPH") != "1"  # FLASH_NEXT_SERVING prefill graph: Marlin W4A16 MoE
        and model_runner.spec_algorithm.is_eagle()
        and not model_runner.is_draft_worker
        and get_server_return_hidden_states_mode() < CaptureHiddenMode.FULL
        and check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
    ):''', 1),
]

PREFILL_GRAPH_QWEN4 = [
    ('''        overlap_indexer = (
            self.is_qsa
            and self.alt_stream is not None
''', '''        flash_next_pg = self.is_qsa and _flash_next_pg_tc_piecewise(forward_batch)
        overlap_indexer = (
            self.is_qsa
            and not flash_next_pg
            and self.alt_stream is not None
''', 1),
    ('''        elif self.is_qsa:
            attention_kwargs["topk_indices"] = self._compute_qsa_topk_indices(
                hidden_states, positions, forward_batch
            )
''', '''        elif flash_next_pg:
            # FLASH_NEXT_SERVING prefill graph: the indexer runs eagerly between captured pieces.
            topk_indices = hidden_states.new_empty(
                (
                    hidden_states.shape[0],
                    self.indexer.token_topk + self.indexer.compress_ratio - 1,
                ),
                dtype=torch.int32,
            )
            flash_next_qsa_topk_with_output(hidden_states, positions, topk_indices, self.layer_id)
            attention_kwargs["topk_indices"] = topk_indices
        elif self.is_qsa:
            attention_kwargs["topk_indices"] = self._compute_qsa_topk_indices(
                hidden_states, positions, forward_batch
            )
''', 1),
    ('''        self.is_qsa = is_qwen_qsa(config)
        if self.is_qsa:
            self.indexer = build_qsa_indexer(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.indexer" if prefix else "indexer",
                rotary_emb=self.rotary_emb,
            )
''', '''        self.is_qsa = is_qwen_qsa(config)
        if self.is_qsa:
            self.indexer = build_qsa_indexer(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.indexer" if prefix else "indexer",
                rotary_emb=self.rotary_emb,
            )
            if not is_nextn:
                # FLASH_NEXT_SERVING prefill graph: the target's layers, for the eager indexer op (first model wins).
                _FLASH_NEXT_PG_QSA_LAYERS.setdefault(layer_id, self)
''', 1),
    ('''        self.hyper_connection_mixer = GatedResidual(hc_config, use_combine=False)
''', '''        self.hyper_connection_mixer = GatedResidual(hc_config, use_combine=False)
        # FLASH_NEXT_SERVING prefill graph: the target model, for the eager PLE op (first model wins; one PLE layer).
        ple_layers = [
            i for i, layer in enumerate(self.layers) if getattr(layer, "ple", None) is not None
        ]
        self._flash_next_pg_last_ple_layer = ple_layers[-1] if ple_layers else None
        if self.has_ple and not is_nextn:
            if len(ple_layers) != 1 and _FLASH_NEXT_PREFILL_GRAPH:
                raise NotImplementedError("Flash-Next tc_piecewise PLE supports exactly one PLE layer")
            _FLASH_NEXT_PG_PLE_MODELS.setdefault("target", self)
''', 1),
    ('''        ple_batch = (
            _prepare_ple_batch(
                input_ids,
                forward_batch,
                ngram_size=self.ple_ngram_size,
                ngram_eos_token_id=self.ple_ngram_eos_token_id,
            )
            if self.has_ple
            else None
        )
''', '''        flash_next_pg = self.has_ple and _flash_next_pg_tc_piecewise(forward_batch)
        ple_batch = (
            _FLASH_NEXT_PG_PLE_DEFERRED
            if flash_next_pg
            else _prepare_ple_batch(
                input_ids,
                forward_batch,
                ngram_size=self.ple_ngram_size,
                ngram_eos_token_id=self.ple_ngram_eos_token_id,
            )
            if self.has_ple
            else None
        )
''', 1),
    ('''                next_ple = getattr(self.layers[i + 1], "ple", None)
                if next_ple is not None:
''', '''                next_ple = getattr(self.layers[i + 1], "ple", None)
                if next_ple is not None and not flash_next_pg:
''', 1),
    ('''        _commit_ple_batch(ple_batch, forward_batch)

        hc_hidden_states = hidden_states
''', '''        if not flash_next_pg:
            _commit_ple_batch(ple_batch, forward_batch)

        hc_hidden_states = hidden_states
''', 1),
    ('''                self.ple.forward_idle(forward_batch)
            else:
''', '''                self.ple.forward_idle(forward_batch)
            elif ple_batch is _FLASH_NEXT_PG_PLE_DEFERRED:
                # FLASH_NEXT_SERVING prefill graph: PLE (length-dependent conv state, host gathers) runs eagerly.
                ple_query = (
                    hidden_states if residual is None else hidden_states + residual
                )
                ple_out = torch.empty_like(ple_query)
                flash_next_ple_with_output(ple_query, ple_out, self.layer_id)
                hidden_states = hidden_states + ple_out
            else:
''', 1),
]

PREFILL_GRAPH_QWEN4_FUNCTION = '''

# FLASH_NEXT_SERVING prefill graph: tc_piecewise prefill support (QSA indexer and PLE as eager split ops).
from sglang.srt.compilation.compilation_config import register_split_op as _flash_next_pg_register_split_op
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context as _flash_next_pg_get_tc_context,
)
from sglang.srt.utils.custom_op import register_custom_op as _flash_next_pg_register_custom_op

_FLASH_NEXT_PREFILL_GRAPH = __import__("os").environ.get("SGLANG_FLASH_NEXT_PREFILL_GRAPH") == "1"
_FLASH_NEXT_PG_QSA_LAYERS = {}
_FLASH_NEXT_PG_PLE_MODELS = {}
_FLASH_NEXT_PG_PLE_DEFERRED = object()


def _flash_next_pg_tc_piecewise(forward_batch) -> bool:
    return forward_batch.forward_mode.is_extend() and _flash_next_pg_get_tc_context() is not None


@_flash_next_pg_register_custom_op(mutates_args=["topk_out"])
@_flash_next_pg_register_split_op()
def flash_next_qsa_topk_with_output(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    topk_out: torch.Tensor,
    layer_id: int,
) -> None:
    forward_batch = _flash_next_pg_get_tc_context().forward_batch
    real = forward_batch.global_num_token_non_padded_cpu
    if real is None:
        real = hidden_states.shape[0]
    real = int(real)
    if real == 0:
        topk_out.fill_(-1)
        return
    layer = _FLASH_NEXT_PG_QSA_LAYERS[layer_id]
    topk = layer._compute_qsa_topk_indices(
        hidden_states[:real], positions[..., :real], forward_batch
    )
    rows = topk.shape[0]
    topk_out[:rows].copy_(topk)
    if rows < topk_out.shape[0]:
        topk_out[rows:].fill_(-1)


@_flash_next_pg_register_custom_op(mutates_args=["ple_out"])
@_flash_next_pg_register_split_op()
def flash_next_ple_with_output(ple_query: torch.Tensor, ple_out: torch.Tensor, layer_id: int) -> None:
    forward_batch = _flash_next_pg_get_tc_context().forward_batch
    model = _FLASH_NEXT_PG_PLE_MODELS["target"]
    batch = _prepare_ple_batch(
        forward_batch.input_ids,
        forward_batch,
        ngram_size=model.ple_ngram_size,
        ngram_eos_token_id=model.ple_ngram_eos_token_id,
    )
    if batch is None:
        ple_out.zero_()
        return
    output = model.layers[layer_id].ple(ple_query, forward_batch, batch)
    ple_out.copy_(output)
    if layer_id == model._flash_next_pg_last_ple_layer:
        _commit_ple_batch(batch, forward_batch)
'''

PREFILL_GRAPH_RMSNORM = [
    ('''    hidden_size = input.size(-1)
    x = input.reshape(-1, hidden_size)
    if out is None:
        out = torch.empty_like(x)
    else:
        out = out.reshape(-1, hidden_size)

    module = _jit_grouped_gemma_rmsnorm_module(group_size, input.dtype)
    module.grouped_gemma_rmsnorm(x, weight, out, eps)
    return out.reshape(input.shape)
''', '''    if _FLASH_NEXT_PREFILL_GRAPH:
        # FLASH_NEXT_SERVING prefill graph: a mutating custom op, so torch.compile (tc_piecewise) can trace through it.
        if out is None:
            out = torch.empty(input.shape, dtype=input.dtype, device=input.device)
        _flash_next_pg_grouped_gemma_rmsnorm_op(input, weight, out, group_size, eps)
        return out.reshape(input.shape)
    hidden_size = input.size(-1)
    x = input.reshape(-1, hidden_size)
    if out is None:
        out = torch.empty_like(x)
    else:
        out = out.reshape(-1, hidden_size)

    module = _jit_grouped_gemma_rmsnorm_module(group_size, input.dtype)
    module.grouped_gemma_rmsnorm(x, weight, out, eps)
    return out.reshape(input.shape)


from sglang.srt.utils.custom_op import register_custom_op as _flash_next_pg_register_custom_op

_FLASH_NEXT_PREFILL_GRAPH = __import__("os").environ.get("SGLANG_FLASH_NEXT_PREFILL_GRAPH") == "1"


@_flash_next_pg_register_custom_op(mutates_args=["out"])
def _flash_next_pg_grouped_gemma_rmsnorm_op(
    input: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, group_size: int, eps: float
) -> None:
    hidden_size = input.size(-1)
    module = _jit_grouped_gemma_rmsnorm_module(group_size, input.dtype)
    module.grouped_gemma_rmsnorm(
        input.reshape(-1, hidden_size), weight, out.view(-1, hidden_size), eps
    )
''', 1),
]

PREFILL_GRAPH_HC_COMBINE = [
    ('''    y = block_output.reshape(-1, hidden_size)
    r = residual.reshape(-1, hc_count * hidden_size)
    n = normed_residual.reshape(-1, hc_count * hidden_size)
    if out is None:
        out = torch.empty_like(r)
    else:
        out = out.reshape(-1, hc_count * hidden_size)

    module = _jit_hc_combine_module(hc_count, hidden_size, residual.dtype)
    module.hc_combine(y, r, n, inject_weight, out)
    return out.reshape(residual.shape)
''', '''    if _FLASH_NEXT_PREFILL_GRAPH:
        # FLASH_NEXT_SERVING prefill graph: a mutating custom op, so torch.compile (tc_piecewise) can trace through it.
        if out is None:
            out = torch.empty(residual.shape, dtype=residual.dtype, device=residual.device)
        _flash_next_pg_hc_combine_op(block_output, residual, normed_residual, inject_weight, out, hc_count, hidden_size)
        return out.reshape(residual.shape)
    y = block_output.reshape(-1, hidden_size)
    r = residual.reshape(-1, hc_count * hidden_size)
    n = normed_residual.reshape(-1, hc_count * hidden_size)
    if out is None:
        out = torch.empty_like(r)
    else:
        out = out.reshape(-1, hc_count * hidden_size)

    module = _jit_hc_combine_module(hc_count, hidden_size, residual.dtype)
    module.hc_combine(y, r, n, inject_weight, out)
    return out.reshape(residual.shape)


from sglang.srt.utils.custom_op import register_custom_op as _flash_next_pg_register_custom_op

_FLASH_NEXT_PREFILL_GRAPH = __import__("os").environ.get("SGLANG_FLASH_NEXT_PREFILL_GRAPH") == "1"


@_flash_next_pg_register_custom_op(mutates_args=["out"])
def _flash_next_pg_hc_combine_op(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    normed_residual: torch.Tensor,
    inject_weight: torch.Tensor,
    out: torch.Tensor,
    hc_count: int,
    hidden_size: int,
) -> None:
    y = block_output.reshape(-1, hidden_size)
    r = residual.reshape(-1, hc_count * hidden_size)
    n = normed_residual.reshape(-1, hc_count * hidden_size)
    module = _jit_hc_combine_module(hc_count, hidden_size, residual.dtype)
    module.hc_combine(y, r, n, inject_weight, out.view(-1, hc_count * hidden_size))
''', 1),
]

PREFILL_GRAPH_HC_SPLIT = [
    ('''            if self._split_combine_ok and block_output.shape[0] <= 32:''', '''            if (
                self._split_combine_ok
                and block_output.shape[0] <= 32
                and not torch.compiler.is_compiling()  # FLASH_NEXT_SERVING prefill graph
            ):''', 1),
]

PREFILL_GRAPH_TC_BACKEND = [
    ('''    for sub in model._modules.values():
        if isinstance(sub, BaseFusedOp):
''', '''    from sglang.srt.layers.utils.multi_platform import MultiPlatformOp as _FlashNextMPO

    for sub in model._modules.values():
        # FLASH_NEXT_SERVING prefill graph: ops without a torch forward (QSA indexer) keep their dispatch.
        if isinstance(sub, BaseFusedOp) and not (
            isinstance(sub, _FlashNextMPO) and type(sub).forward_native is _FlashNextMPO.forward_native
        ):
''', 1),
    ('''            if reverse:
                sub.leave_torch_compile()
''', '''            if reverse:
                sub.leave_torch_compile()
                # FLASH_NEXT_SERVING prefill graph: resolve the lazy dispatch now; a None -> method change at the first
                # eager call fails Dynamo's `_forward_method is None` guard after capture, and the runtime
                # recompile then runs without CUDA graphs ("PCG capture stream is not set").
                if sub._forward_method is None:
                    sub._forward_method = sub._resolve_forward_method()
''', 1),
]

# Local HC up projection, SGLANG_FLASH_NEXT_HC_LOCAL_UP=1 with SGLANG_FLASH_NEXT_HC_TP=1 (2026-10-29,
# inducer/results/deepinfer_review_20261029/). Applies after the HC sharding edits.
HC_LOCAL_UP = [
    ('''        self._hc_tp_group = None
''', '''        self._hc_tp_group = None
        self._hc_local_up = False
''', 1),
    ('''                mix_out = self.hc_count * self._hc_width
''', '''                mix_out = self.hc_count * self._hc_width
                # FLASH_NEXT_SERVING local HC up: a full INT8 up projection on every rank instead of a hidden slice.
                self._hc_local_up = os.environ.get("SGLANG_FLASH_NEXT_HC_LOCAL_UP") == "1"
                if self._hc_local_up and (config.hc_lowrank % _FLASH_NEXT_HC_GROUP or self.hidden_size % 64):
                    raise ValueError("local HC up needs lowrank % 32 == 0 and hidden % 64 == 0")
''', 1),
    ('''                self.input_mix_weight_up.weight.weight_loader = self._hc_load_up
''', '''                self.input_mix_weight_up.weight.weight_loader = self._hc_load_up
            if self._hc_local_up:
                rows = self.hc_count * self.hidden_size
                up = self.input_mix_weight_up
                up.weight = nn.Parameter(
                    torch.empty(rows, self.config.hc_lowrank, dtype=torch.int8, device=torch.cuda.current_device()),
                    requires_grad=False,
                )
                up.weight.weight_loader = self._hc_load_up_int8
                up.register_buffer("weight_scale", torch.empty(
                    rows, self.config.hc_lowrank // _FLASH_NEXT_HC_GROUP, dtype=torch.float16,
                    device=torch.cuda.current_device(),
                ))
                if _FLASH_NEXT_HC_MIX_CUDA:
                    _flash_next_hc_mix_module()  # compile at load: a build failure stops every rank, never one
''', 1),
    ('''            part = self._hc_finish(
''', '''            if self._hc_local_up:
                mixed_input = _flash_next_hc_local_mix(
                    lowrank, self.input_mix_weight_up.weight, self.input_mix_weight_up.weight_scale,
                    hyper_input_normed, self.hc_count, self.hidden_size,
                )
                return mixed_input, (hyper_input, hyper_input_normed)
            part = self._hc_finish(
''', 1),
    ('''    def _hc_load_up(self, param, loaded_weight):
''', '''    def _hc_load_up_int8(self, param, loaded_weight):
        # FLASH_NEXT_SERVING local HC up: symmetric INT8, one scale per row and _FLASH_NEXT_HC_GROUP lowrank columns,
        # stored x1024 in fp16 and quantized with the rounded scale.
        expected = [self.hc_count * self.hidden_size, self.config.hc_lowrank]
        if list(loaded_weight.shape) != expected:
            raise ValueError(f"HC checkpoint shape {list(loaded_weight.shape)} != {expected}")
        # On the CPU: GPU temporaries would stay in the caching allocator and count against the KV pool.
        weight = loaded_weight.detach().to("cpu", torch.float32).unflatten(1, (-1, _FLASH_NEXT_HC_GROUP))
        scale = (weight.abs().amax(dim=2).clamp(min=1e-8) / 127 * _FLASH_NEXT_HC_SCALE_SHIFT).to(torch.float16)
        quantized = torch.round(weight / (scale.float()[..., None] / _FLASH_NEXT_HC_SCALE_SHIFT)).clamp_(-127, 127)
        with torch.no_grad():
            param.copy_(quantized.flatten(1).to(torch.int8))
            self.input_mix_weight_up.weight_scale.copy_(scale)

    def _hc_load_up(self, param, loaded_weight):
''', 1),
]
HC_LOCAL_UP_FUNCTION = '''

# FLASH_NEXT_SERVING local HC up: the branch mix over the full hidden width from the rank's INT8 up projection, fused with
# its sigmoid and branch mean. A mutating custom op, so tc_piecewise prefill traces through it.
import functools as _flash_next_functools
import os as _flash_next_os
from pathlib import Path as _FlashNextPath

import triton as _flash_next_triton
import triton.language as tl

from sglang.srt.utils.custom_op import register_custom_op as _flash_next_hc_register_custom_op

_FLASH_NEXT_HC_GROUP = 32  # lowrank columns per scale
_FLASH_NEXT_HC_SCALE_SHIFT = 1024.0  # fp16 scales are stored x1024: the smallest (2.6e-5) would be subnormal
# SGLANG_FLASH_NEXT_HC_MIX_CUDA=1: batches of 1-64 tokens at hc 4, hidden 2560, lowrank 320 take
# flash_next_hc_local_mix.cuh (installed beside this file), the rest this Triton kernel. Every rank decides from shapes
# and alignment alone, so the replicated mix output stays identical across ranks.
_FLASH_NEXT_HC_MIX_CUDA = _flash_next_os.environ.get("SGLANG_FLASH_NEXT_HC_MIX_CUDA") == "1"


@_flash_next_functools.cache
def _flash_next_hc_mix_module():
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "flash_next_hc_local_mix",
        cuda_files=[str(_FlashNextPath(__file__).with_name("flash_next_hc_local_mix.cuh"))],
        cuda_wrappers=[("hc_local_mix", "hc_local_mix")],
        # The sigmoid matches the Triton kernel bit for bit only with fast math.
        extra_cuda_cflags=["--use_fast_math"],
    )


def _flash_next_hc_mix_cuda_fits(lowrank, x, hc, hidden):
    tokens, lowrank_dim = lowrank.shape
    return (_FLASH_NEXT_HC_MIX_CUDA and 1 <= tokens <= 64 and (hc, hidden, lowrank_dim) == (4, 2560, 320)
            and lowrank.is_contiguous() and lowrank.data_ptr() % 16 == 0 and x.data_ptr() % 16 == 0)


@_flash_next_triton.jit
def _flash_next_hc_local_mix_kernel(lr_ptr, w_ptr, s_ptr, x_ptr, out_ptr, T, stride_lr, stride_x, stride_out,
                                    H: tl.constexpr, L: tl.constexpr, HC: tl.constexpr, G: tl.constexpr,
                                    BM: tl.constexpr, BN: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, G)
    mask_m = rm < T
    out = tl.zeros((BM, BN), dtype=tl.float32)
    for c in tl.static_range(HC):
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in tl.static_range(0, L, G):
            a = tl.load(lr_ptr + rm[:, None] * stride_lr + (k0 + rk)[None, :], mask=mask_m[:, None], other=0.0)
            w = tl.load(w_ptr + (c * H + rn)[None, :] * L + (k0 + rk)[:, None])
            s = tl.load(s_ptr + (c * H + rn) * (L // G) + k0 // G).to(tl.float32)
            acc += tl.dot(a, w.to(tl.bfloat16)) * s[None, :]
        g = tl.sigmoid(acc * (1.0 / 1024.0))
        xv = tl.load(x_ptr + rm[:, None] * stride_x + (c * H + rn)[None, :], mask=mask_m[:, None], other=0.0)
        out += g * xv.to(tl.float32)
    out = out * (1.0 / HC)
    tl.store(out_ptr + rm[:, None] * stride_out + rn[None, :], out.to(out_ptr.dtype.element_ty), mask=mask_m[:, None])


@_flash_next_hc_register_custom_op(mutates_args=["out"])
def _flash_next_hc_local_mix_op(lowrank: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, x: torch.Tensor,
                                out: torch.Tensor, hc: int, hidden: int) -> None:
    if _flash_next_hc_mix_cuda_fits(lowrank, x, hc, hidden):
        _flash_next_hc_mix_module().hc_local_mix(lowrank, weight, scale, x, out)
        return
    tokens, lowrank_dim = lowrank.shape
    # Up to 64 tokens (decode) take one row block, so each weight byte is read once; narrow column blocks fill the SMs.
    bm, bn = (16, 32) if tokens <= 16 else (32, 32) if tokens <= 32 else (64, 32) if tokens <= 64 else (64, 64)
    grid = (hidden // bn, _flash_next_triton.cdiv(tokens, bm))
    _flash_next_hc_local_mix_kernel[grid](
        lowrank, weight, scale, x, out, tokens, lowrank.stride(0), x.stride(0), out.stride(0),
        H=hidden, L=lowrank_dim, HC=hc, G=_FLASH_NEXT_HC_GROUP, BM=bm, BN=bn, num_warps=4,
    )


def _flash_next_hc_local_mix(lowrank, weight, scale, x, hc, hidden):
    out = torch.empty((lowrank.shape[0], hidden), dtype=x.dtype, device=x.device)
    _flash_next_hc_local_mix_op(lowrank, weight, scale, x.contiguous(), out, hc, hidden)
    return out
'''

# Breakable prefill CUDA graph (--cuda-graph-backend-prefill breakable with SGLANG_FLASH_NEXT_PREFILL_GRAPH=1;
# 2026-10-30, Hau's adapter, codex/v30-inference-speed scripts/vastai/experiments/v30_breakable_patch.py).
# Upstream breakable also captures the MTP draft's prefill, which the target-only QSA/PLE registry cannot serve
# (KeyError at capture); the draft prefill stays eager.
BREAKABLE_DRAFT = [
    ('''            if check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE):
                self.draft_runner.init_prefill_cuda_graph(force_for_draft_worker=True)
''', '''            # FLASH_NEXT_SERVING breakable prefill: the draft's prefill stays eager; the target-only QSA/PLE
            # registry of the prefill graph cannot serve the MTP draft.
''', 1),
]
BREAKABLE_QWEN4_FUNCTION = '''

# FLASH_NEXT_SERVING breakable prefill: the QSA indexer and PLE ops read the live prefill context and write
# fixed-address outputs; under the breakable backend they run eagerly between captured segments, as its attention and
# GDN layers do. Other backends call them unchanged.
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph as _flash_next_eager_on_graph,
)

flash_next_qsa_topk_with_output = _flash_next_eager_on_graph(True)(flash_next_qsa_topk_with_output)
flash_next_ple_with_output = _flash_next_eager_on_graph(True)(flash_next_ple_with_output)
'''
BREAKABLE_HC_FUNCTION = '''

# FLASH_NEXT_SERVING breakable prefill: the HC state is an explicit captured output. Qwen4ExpVLModel.forward otherwise
# exposes it only through a Python attribute, which body-only replay bypasses and an intervening eager request can
# overwrite.
_flash_next_bcg_capture_one = BreakableCudaGraphBackend.capture_one
_flash_next_bcg_replay = BreakableCudaGraphBackend.replay


def _flash_next_bcg_target(backend):
    model = backend._model_runner.model
    return model if type(model).__name__ == "Qwen4ExpForConditionalGeneration" else None


def _flash_next_bcg_capture_hc(self, shape_key, forward_fn, capture_inputs=None, post_warmup_hook=None):
    target = _flash_next_bcg_target(self)
    if target is not None:
        original_forward = forward_fn

        def forward_fn():
            body_output = original_forward()
            hc = target.model.last_hc_hidden_states
            if hc is None:
                raise RuntimeError("Flash-Next graph body did not expose HC state")
            return body_output, hc

    return _flash_next_bcg_capture_one(self, shape_key, forward_fn, capture_inputs, post_warmup_hook)


def _flash_next_bcg_replay_hc(self, shape_key, static_forward_batch, **kwargs):
    output = _flash_next_bcg_replay(self, shape_key, static_forward_batch, **kwargs)
    target = _flash_next_bcg_target(self)
    if target is not None:
        body_output, target.model.last_hc_hidden_states = output
        return body_output
    return output


BreakableCudaGraphBackend.capture_one = _flash_next_bcg_capture_hc
BreakableCudaGraphBackend.replay = _flash_next_bcg_replay_hc
'''

# QSA indexer decode scoring, SGLANG_FLASH_NEXT_QSA_DECODE=1 (2026-10-31, ForgeKernel qsa_index_decode;
# inducer/results/l4_kernel_handoffs_20261030/qsa_index_decode.md).
QSA_DECODE = [
    ('''        self.block_topk = self.token_topk // self.compress_ratio
''', '''        self.block_topk = self.token_topk // self.compress_ratio
        # FLASH_NEXT_SERVING QSA decode: compile at load, so a build failure stops every rank, never one.
        self._flash_next_qsa_decode = _FLASH_NEXT_QSA_DECODE and (
            self.index_n_heads, self.index_kv_heads, self.index_head_dim) == (4, 1, 128)
        if self._flash_next_qsa_decode:
            _flash_next_qsa_decode_module()
''', 1),
    ('''        query_positions: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        logits = qsa_mqa_decode(
            q,
            compressed_cache,
            compressed_page_table,
            compressed_lengths,
            max_model_len,
        )
''', '''        query_positions: torch.Tensor,
        sequence_lengths: torch.Tensor,
        flash_next_grouped: bool = False,
    ) -> torch.Tensor:
        if flash_next_grouped:
            # FLASH_NEXT_SERVING QSA decode: one CTA scores a request's 4 verify rows (flash_next_qsa_index_decode.cuh).
            logits = torch.empty((q.shape[0], max_model_len), dtype=torch.float32, device=q.device)
            _flash_next_qsa_decode_module().qsa_index_decode(
                q, compressed_cache, compressed_page_table, compressed_lengths, logits
            )
        else:
            logits = qsa_mqa_decode(
                q,
                compressed_cache,
                compressed_page_table,
                compressed_lengths,
                max_model_len,
            )
''', 1),
    ('''                logical_positions,
                indexer_metadata.get_seqlens_int32(),
            )
''', '''                logical_positions,
                indexer_metadata.get_seqlens_int32(),
                flash_next_grouped=self._flash_next_qsa_decode and _flash_next_qsa_decode_grouped(
                    forward_batch, indexer_metadata, is_target_verify or is_draft_extend,
                    q, compressed_cache, page_table, compressed_lengths, max_model_len,
                ),
            )
''', 1),
]
QSA_DECODE_FUNCTION = '''

# FLASH_NEXT_SERVING QSA decode: speculative verify and draft-extend graphs score through flash_next_qsa_index_decode.cuh
# (installed beside this file) instead of the TileLang kernel, which gives every row its own CTAs over the whole context
# limit and reads each request's keys once per verify row. The kernel reads page-table row 4i for rows 4i .. 4i + 3, so it
# serves only graph-built rows (qsa/graph_metadata.py): verify and draft-extend lay out each request's rows together,
# and a graph whose rows are exactly 4 per request has 4 rows of one request in each group. Draft decode (one row per
# request) and eager batches keep TileLang. The choice is made at capture, from shapes, identically on every rank.
import functools as _flash_next_functools
import logging as _flash_next_logging
import os as _flash_next_os
from pathlib import Path as _FlashNextPath

_FLASH_NEXT_QSA_DECODE = _flash_next_os.environ.get("SGLANG_FLASH_NEXT_QSA_DECODE") == "1"
_FLASH_NEXT_QSA_DECODE_SEEN = set()


@_flash_next_functools.cache
def _flash_next_qsa_decode_module():
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "flash_next_qsa_index_decode",
        cuda_files=[str(_FlashNextPath(__file__).with_name("flash_next_qsa_index_decode.cuh"))],
        cuda_wrappers=[("qsa_index_decode", "qsa_index_decode")],
        # TileLang builds its kernel with fast math; the logits match it bit for bit only with fast math.
        extra_cuda_cflags=["--use_fast_math"],
    )


def _flash_next_qsa_decode_grouped(forward_batch, indexer_metadata, speculative, q, k_cache, page_table, lengths,
                                   max_model_len):
    rows = q.shape[0]
    grouped = (
        speculative
        and indexer_metadata.is_cuda_graph
        and rows > 0
        and rows == 4 * forward_batch.batch_size
        and q.dim() == 3 and q.shape[1] >= 4 and q.shape[2] == 128 and q.dtype == torch.bfloat16 and q.is_contiguous()
        and tuple(k_cache.shape[1:]) == (16, 1, 128) and k_cache.dtype == torch.bfloat16 and k_cache.is_contiguous()
        and page_table.dtype == torch.int32 and page_table.is_contiguous() and page_table.shape[0] == rows
        and lengths.dtype == torch.int32 and lengths.is_contiguous() and lengths.shape[0] == rows
        and max_model_len == 16 * page_table.shape[1]
        and all(t.data_ptr() % 16 == 0 for t in (q, k_cache, page_table, lengths))
    )
    key = (forward_batch.forward_mode.name, rows, grouped)
    if key not in _FLASH_NEXT_QSA_DECODE_SEEN:
        _FLASH_NEXT_QSA_DECODE_SEEN.add(key)
        _flash_next_logging.getLogger(__name__).info(
            "flash_next_qsa_decode: %s rows=%d bs=%d -> %s", key[0], rows, forward_batch.batch_size,
            "cuda" if grouped else "tilelang",
        )
    return grouped
'''

PATCHES = {
    "srt/layers/hyperconnection.py": (HC_SHARDING + PREFILL_GRAPH_HC_SPLIT + HC_LOCAL_UP, HC_LOCAL_UP_FUNCTION),
    "srt/hardware_backend/gpu/quantization/gptq_kernels.py": (GPTQ_EXPERT_MAP, ""),
    "srt/hardware_backend/gpu/quantization/awq_kernels.py": (AWQ_EXPERT_MAP, ""),
    "srt/layers/quantization/gptq/schemes/gptq_moe.py": (BF16_SCALES, ""),
    "srt/models/qwen4_exp.py": (PLE_ON_META + PLE_LOADER + EMBED_PLACEMENT + PREFILL_GRAPH_QWEN4,
                                EMBED_PLACEMENT_FUNCTION + PREFILL_GRAPH_QWEN4_FUNCTION + BREAKABLE_QWEN4_FUNCTION),
    "srt/speculative/eagle_worker_v2.py": (TP_HOT_HEAD + SHARE_HOST_EMBED + BREAKABLE_DRAFT,
                                           TP_HOT_HEAD_FUNCTION + SHARE_HOST_EMBED_FUNCTION),
    "srt/models/qwen2_moe.py": (EXPERT_ROUTING + FUSED_MOE_DOWN_AR,
                                EXPERT_ROUTING_FUNCTION + FUSED_MOE_DOWN_AR_FUNCTION),
    "srt/models/qwen4_exp_mtp.py": (DRAFT_META, ""),
    "srt/models/qwen3_5_mtp.py": (DRAFT_EMBED_LOAD, ""),
    "srt/model_loader/loader.py": (META_PLACEHOLDERS, META_PLACEHOLDERS_FUNCTION),
    "srt/distributed/parallel_state.py": (HOST_ALL_REDUCE, ""),
    "srt/layers/attention/qwen_sparse_attn_backend.py": (QSA_DCP_BACKEND, ""),
    "srt/mem_cache/qsa_kv_pool.py": (QSA_DCP_POOL, ""),
    "srt/mem_cache/kv_cache_configurator.py": (QSA_DCP_CONFIGURATOR, ""),
    "srt/model_executor/pool_configurator.py": (QSA_DCP_CELL, ""),
    "srt/layers/logits_processor.py": (TOPK_LOGITS, TOPK_LOGITS_FUNCTION),
    "srt/model_executor/model_runner_components/cuda_graph_setup.py": (PREFILL_GRAPH_SETUP, ""),
    "srt/model_executor/runner_backend/tc_piecewise_cuda_graph_backend.py": (PREFILL_GRAPH_TC_BACKEND, ""),
    "kernels/ops/layernorm/grouped_gemma_rmsnorm.py": (PREFILL_GRAPH_RMSNORM, ""),
    "kernels/ops/elementwise/hc_combine.py": (PREFILL_GRAPH_HC_COMBINE, ""),
    "srt/layers/moe/fused_moe_triton/fused_marlin_moe.py": (MARLIN_DOWN_HOOK, ""),
    "srt/layers/attention/qsa/qsa_indexer.py": (QSA_DECODE, QSA_DECODE_FUNCTION),
    "srt/model_executor/runner_backend/breakable_cuda_graph_backend.py": ([], BREAKABLE_HC_FUNCTION),
}


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def replace_text(path, text):
    """Write ``path`` through a sibling temporary file: an inode it shares with another path is never changed."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def transform(name, source):
    """The patched source of one pinned stock file."""
    if sha256_text(source) != PINS[name]:
        raise RuntimeError(f"{name} is not the SGLang {SGLANG_VERSION} source this patch set pins")
    replacements, suffix = PATCHES[name]
    for old, new, count in replacements:
        if source.count(old) != count:
            raise RuntimeError(f"{name}: anchor found {source.count(old)} times, not {count}: {old.strip()[:60]!r}")
        source = source.replace(old, new)
    source += suffix
    compile(source, name, "exec")
    return source


def patch_sglang(package):
    """Patch the sglang package directory in place; a patched package is re-derived from its originals.

    The host all-reduce, fused MoE down + all-reduce and QSA DCP modules are installed beside the code that imports
    them; each is inert unless enabled.
    """
    package = Path(package)
    staged = []
    for name in PINS:
        path = package / name
        backup = path.with_suffix(".py.before_serving")
        current = path.read_text()
        original = backup.read_text() if backup.exists() else current
        changed = transform(name, original)
        if current not in (original, changed):
            raise RuntimeError(f"{name} was changed outside this patch set")
        staged.append((name, path, backup, original, changed))
    for _, path, backup, original, changed in staged:
        replace_text(backup, original)
        replace_text(path, changed)
    modules = []
    for source, directory in MODULES.items():
        module = Path(__file__).with_name(source).read_text()
        installed = f"{directory}/{INSTALLED_NAMES.get(source, source)}"
        replace_text(package / installed, module)
        modules.append({"file": installed, "before_sha256": None, "after_sha256": sha256_text(module)})
    return [{"file": name, "before_sha256": PINS[name], "after_sha256": sha256_text(changed)}
            for name, _, _, _, changed in staged] + modules


def check_expert_location(document, model_path):
    """The map must match the target's MoE shape and, where its REAP export is present, its experts."""
    config = json.loads((Path(model_path) / "config.json").read_text())
    config = config.get("text_config", config)
    layers, experts = len(document["physical_to_logical_map"]), len(document["physical_to_logical_map"][0])
    if (layers, experts) != (config["num_hidden_layers"], config["num_experts"]):
        raise ValueError(f"expert map {layers}x{experts} does not fit the target "
                         f"({config['num_hidden_layers']}x{config['num_experts']})")
    if any(sorted(row) != list(range(experts)) for row in document["physical_to_logical_map"]):
        raise ValueError("expert map rows must be permutations")
    for name in ("phi1_reap_export.json", "reap_export.json"):
        export = Path(model_path) / name
        if export.is_file():
            digest = hashlib.sha256(export.read_bytes()).hexdigest()
            if digest != document["reap_export_sha256"]:
                raise ValueError(f"expert map was recorded on another REAP export ({name})")
            return name
    return None


def prepare_serving(python, environment, work, runtime_root, model_path):
    """Write the draft hot-token map and the expert map for the patched server.

    Returns the paths for --speculative-token-map and --init-expert-location.
    """
    token_map = Path(work) / "flash_next_hot_tokens.pt"
    expert_location = Path(work) / "flash_next_expert_location.pt"
    placement = json.loads((Path(runtime_root) / EXPERT_LOCATION).read_text())
    verified_export = check_expert_location(placement, model_path)
    subprocess.run(
        [python, "-c", "import json, sys, torch; "
         "torch.save(json.load(open(sys.argv[1]))['token_ids'], sys.argv[2]); "
         "torch.save({'physical_to_logical_map': torch.tensor("
         "json.load(open(sys.argv[3]))['physical_to_logical_map'])}, sys.argv[4])",
         str(Path(runtime_root) / HOT_TOKENS), str(token_map),
         str(Path(runtime_root) / EXPERT_LOCATION), str(expert_location)],
        env={**environment, "CUDA_VISIBLE_DEVICES": ""}, check=True, timeout=600,
    )
    print(json.dumps({"stage": "serving_maps", "token_map": str(token_map),
                      "expert_location": str(expert_location), "verified_reap_export": verified_export}),
          flush=True)
    return token_map, expert_location


if __name__ == "__main__":
    import sys

    print(json.dumps(patch_sglang(sys.argv[1]), indent=2))
