import os
import sys
from pathlib import Path
legacy_root = Path(os.environ["PROGRAM063_LEGACY_ROOT"]).resolve()
overlay_root = Path(os.environ["PROGRAM063_OVERLAY_ROOT"]).resolve()
clean = []
for item in sys.path:
    if not item:
        clean.append(item)
        continue
    try:
        resolved = Path(item).resolve()
    except OSError:
        continue
    if resolved == legacy_root or legacy_root in resolved.parents or resolved == overlay_root:
        continue
    clean.append(item)
sys.path[:] = [str(overlay_root)] + clean

import hashlib
import json
import math
import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F

BASE_SEED = int(os.environ["PROGRAM063_BASE_SEED"])
if os.environ.get("PYTHONHASHSEED") != "0":
    raise RuntimeError(f"PROGRAM063_SMOKE_PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')!r}")
random.seed(BASE_SEED)
np.random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(BASE_SEED)
print("PROGRAM063_SMOKE_SEED_READY=" + json.dumps({
    "base_seed": BASE_SEED,
    "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    "hash_probe": hash("arc-program-seed-probe-v1"),
    "python_rng_sha256": hashlib.sha256(pickle.dumps(random.getstate(), protocol=4)).hexdigest(),
    "numpy_rng_sha256": hashlib.sha256(pickle.dumps(np.random.get_state(), protocol=4)).hexdigest(),
    "torch_cpu_rng_sha256": hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
    "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
    "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
}, sort_keys=True))

from unsloth import FastLanguageModel, UnslothTrainingArguments
from peft import get_peft_model_state_dict
from datasets import Dataset
from arc_loader import QwenFormatter
from arc_solver import ARC_TOKENS, QwenDataCollatorForCompletionOnlyLM, UnslothFixedTrainer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
    full_finetuning=False,
    load_in_4bit=False,
    local_files_only=True,
    use_gradient_checkpointing=False,
    max_seq_length=8192,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=256,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing=False,
    random_state=42,
    use_rslora=True,
    loftq_config=None,
)
for _, parameter in model.named_parameters():
    if parameter.dtype == torch.float32:
        parameter.data = parameter.data.to(torch.bfloat16)
state = get_peft_model_state_dict(model, adapter_name="default")
if not state:
    raise RuntimeError("PROGRAM063_LORA_STATE_EMPTY")

formatter = QwenFormatter(tokenizer)
smoke_text = formatter.fmt_train([
    {"input": [[1, 0], [0, 1]], "output": [[1, 0], [0, 1]]},
    {"input": [[2, 0], [0, 2]], "output": [[2, 0], [0, 2]]},
])
collator = QwenDataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False)
args = UnslothTrainingArguments(
    output_dir="/kaggle/working/program063_smoke_output",
    per_device_eval_batch_size=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    max_steps=1,
    warmup_steps=0,
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    learning_rate=5e-5,
    optim="adamw_torch",
    weight_decay=0.0,
    lr_scheduler_type="cosine",
    seed=42,
    report_to="none",
    save_strategy="no",
    eval_strategy="no",
    logging_strategy="no",
    fp16=False,
    bf16=True,
    fsdp="",
    ddp_find_unused_parameters=False,
    dataloader_num_workers=0,
    gradient_checkpointing=False,
)
model = FastLanguageModel.for_training(model)
trainer = UnslothFixedTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=collator,
    train_dataset=Dataset.from_list([{"text": smoke_text}]),
    dataset_text_field="text",
    max_seq_length=8192,
    args=args,
)
trainer.create_optimizer_and_scheduler(num_training_steps=1)
batch = next(iter(trainer.get_train_dataloader()))
batch = {
    key: (value.to(model.device) if torch.is_tensor(value) else value)
    for key, value in batch.items()
}
if "labels" not in batch or not torch.any(batch["labels"] != -100):
    raise RuntimeError("PROGRAM063_SMOKE_NO_SUPERVISED_LABELS")
loss = trainer.compute_loss(model, batch)
if loss is None or not math.isfinite(float(loss.detach().cpu())):
    raise RuntimeError(f"PROGRAM063_SMOKE_INVALID_LOSS={loss}")
loss.backward()
torch.cuda.synchronize()
grad_tensors = sum(
    1 for parameter in model.parameters()
    if parameter.requires_grad and parameter.grad is not None
)
if grad_tensors == 0:
    raise RuntimeError("PROGRAM063_SMOKE_NO_GRADIENTS")
trainer.optimizer.step()
trainer.optimizer.zero_grad(set_to_none=True)
model = FastLanguageModel.for_inference(model)
def cache_full_parity(ids, label):
    prefix_mask = torch.ones_like(ids)
    prefix_output_local = model(
        input_ids=ids,
        attention_mask=prefix_mask,
        return_dict=True,
        use_cache=True,
    )
    if prefix_output_local.past_key_values is None or not torch.isfinite(prefix_output_local.logits).all():
        raise RuntimeError(f"PROGRAM063_SMOKE_PREFIX_INFERENCE_FAILED label={label}")
    next_input_local = ids[:, -1:]
    next_position_local = torch.full(
        (ids.shape[0], 1),
        ids.shape[1],
        dtype=torch.long,
        device=ids.device,
    )
    cached_output = model(
        input_ids=next_input_local,
        position_ids=next_position_local,
        past_key_values=prefix_output_local.past_key_values,
        return_dict=True,
        use_cache=True,
    )
    full_ids = torch.cat([ids, next_input_local], dim=1)
    full_output = model(
        input_ids=full_ids,
        attention_mask=torch.ones_like(full_ids),
        return_dict=True,
        use_cache=False,
    )
    cached_logits = cached_output.logits[:, -1, :].float()
    full_logits = full_output.logits[:, -1, :].float()
    if cached_logits.shape != full_logits.shape:
        raise RuntimeError(
            f"PROGRAM063_SMOKE_CACHE_FULL_SHAPE label={label} cached={cached_logits.shape} full={full_logits.shape}"
        )
    if not torch.isfinite(cached_logits).all() or not torch.isfinite(full_logits).all():
        raise RuntimeError(f"PROGRAM063_SMOKE_CACHE_FULL_NONFINITE label={label}")
    delta = cached_logits - full_logits
    absolute_delta = delta.abs()
    max_abs = float(absolute_delta.max().cpu())
    mean_abs = float(absolute_delta.mean().cpu())
    rmse = float(torch.sqrt(torch.mean(delta.square())).cpu())
    reference_rms = float(torch.sqrt(torch.mean(full_logits.square())).cpu())
    relative_rmse = rmse / max(reference_rms, 1e-6)
    cosine = F.cosine_similarity(cached_logits, full_logits, dim=-1)
    cosine_min = float(cosine.min().cpu())
    top1_agreement = float(
        (cached_logits.argmax(dim=-1) == full_logits.argmax(dim=-1)).float().mean().cpu()
    )
    arc_token_ids = torch.as_tensor(ARC_TOKENS, dtype=torch.long, device=cached_logits.device)
    if int(arc_token_ids.max().item()) >= cached_logits.shape[-1]:
        raise RuntimeError(
            f"PROGRAM063_SMOKE_ARC_TOKEN_RANGE max_token={int(arc_token_ids.max().item())} "
            f"vocab={cached_logits.shape[-1]}"
        )
    cached_arc_nll = (
        torch.logsumexp(cached_logits, dim=-1, keepdim=True)
        - cached_logits.index_select(-1, arc_token_ids)
    )
    full_arc_nll = (
        torch.logsumexp(full_logits, dim=-1, keepdim=True)
        - full_logits.index_select(-1, arc_token_ids)
    )
    arc_nll_delta = (cached_arc_nll - full_arc_nll).abs()
    dfs_max_score = -math.log(0.2)
    cached_alive = cached_arc_nll < dfs_max_score
    full_alive = full_arc_nll < dfs_max_score
    threshold_mismatch_count = int((cached_alive != full_alive).sum().item())
    threshold_decision_count = int(cached_alive.numel())
    threshold_decision_agreement = 1.0 - (
        threshold_mismatch_count / max(threshold_decision_count, 1)
    )
    arc_nll_max_abs = float(arc_nll_delta.max().cpu())
    arc_nll_mean_abs = float(arc_nll_delta.mean().cpu())
    full_threshold_margin_min = float((full_arc_nll - dfs_max_score).abs().min().cpu())
    top_k = min(4, cached_arc_nll.shape[-1])
    cached_topk = torch.topk(-cached_arc_nll, k=top_k, dim=-1).indices
    full_topk = torch.topk(-full_arc_nll, k=top_k, dim=-1).indices
    topk_position_agreement = float((cached_topk == full_topk).float().mean().cpu())
    # Bfloat16 cached and full kernels need not be bit-identical. Scale-aware
    # error, cosine direction, and the production batch's argmaxes must agree.
    if max_abs > 4.0 or relative_rmse > 0.08 or cosine_min < 0.99 or top1_agreement < 0.75:
        raise RuntimeError(
            "PROGRAM063_SMOKE_CACHE_FULL_PARITY_FAILED="
            + json.dumps({
                "label": label,
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "rmse": rmse,
                "reference_rms": reference_rms,
                "relative_rmse": relative_rmse,
                "cosine_min": cosine_min,
                "top1_agreement": top1_agreement,
            }, sort_keys=True)
        )
    return prefix_output_local, cached_output, {
        "label": label,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "reference_rms": reference_rms,
        "relative_rmse": relative_rmse,
        "cosine_min": cosine_min,
        "top1_agreement": top1_agreement,
        "arc_nll_max_abs": arc_nll_max_abs,
        "arc_nll_mean_abs": arc_nll_mean_abs,
        "dfs_threshold": dfs_max_score,
        "threshold_mismatch_count": threshold_mismatch_count,
        "threshold_decision_count": threshold_decision_count,
        "threshold_decision_agreement": threshold_decision_agreement,
        "full_threshold_margin_min": full_threshold_margin_min,
        "arc_top4_position_agreement": topk_position_agreement,
    }

with torch.no_grad():
    base_prefix = batch["input_ids"][:, : min(8, batch["input_ids"].shape[1])]
    prefix_ids = base_prefix.repeat(4, 1)
    prefix_output, next_output, batch4_parity = cache_full_parity(prefix_ids, "batch4")
    _, batch1_output, batch1_parity = cache_full_parity(base_prefix[:1], "batch1")
    if next_output.logits.shape[0] != 4 or next_output.logits.shape[1] != 1:
        raise RuntimeError(f"PROGRAM063_SMOKE_CACHED_SHAPE={next_output.logits.shape}")
    if batch1_output.logits.shape[0] != 1 or batch1_output.logits.shape[1] != 1:
        raise RuntimeError(f"PROGRAM063_SMOKE_BATCH1_CACHED_SHAPE={batch1_output.logits.shape}")
print("PROGRAM063_MODEL_SMOKE_OK=" + json.dumps({
    "loss": float(loss.detach().cpu()),
    "lora_tensors": len(state),
    "gradient_tensors": grad_tensors,
    "batch_tokens": int(batch["input_ids"].numel()),
    "supervised_tokens": int((batch["labels"] != -100).sum().item()),
    "prefix_logits_shape": list(prefix_output.logits.shape),
    "cached_logits_shape": list(next_output.logits.shape),
    "batch1_cached_logits_shape": list(batch1_output.logits.shape),
    "batch4_cache_full_parity": batch4_parity,
    "batch1_cache_full_parity": batch1_parity,
    "allocated_mb": round(torch.cuda.memory_allocated(0) / 1024 / 1024, 1),
    "reserved_mb": round(torch.cuda.memory_reserved(0) / 1024 / 1024, 1),
    "device": torch.cuda.get_device_name(0),
}, sort_keys=True))
