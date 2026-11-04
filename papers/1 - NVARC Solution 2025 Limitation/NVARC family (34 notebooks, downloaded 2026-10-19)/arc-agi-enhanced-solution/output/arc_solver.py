
"""
arc_solver.py
=============
Per-puzzle fine-tuning (LoRA / test-time training) + turbo DFS beam search
+ augmented verification scoring.
"""

from unsloth import FastLanguageModel, UnslothTrainingArguments, UnslothTrainer
from arc_loader import ArcDataset, QwenFormatter

import gc
import os
import io
import time
import torch
import numpy as np
from datasets import Dataset
from collections import defaultdict
from typing import Any, Union
from transformers import DataCollatorForLanguageModeling
from contextlib import redirect_stdout, redirect_stderr
from peft import get_peft_model_state_dict, set_peft_model_state_dict
import logging
import bz2
import pickle

logging.disable(logging.WARNING)


# ── Token constants ───────────────────────────────────────────────────────────
# These match the Qwen tokenizer's special token IDs for the ARC grid format.

ARC_VOCAB = {str(d): d for d in range(10)}
ARC_VOCAB.update({"Ċ": 10, "<|im_end|>": 15})

ARC_TOKENS        = list(ARC_VOCAB.values())
USER_TOKEN_ID     = 11
ASSISTANT_TOKEN_ID = 12
PAD_ID            = 13
EOS_ID            = 15


# ── Trainer fix ───────────────────────────────────────────────────────────────

class FixedUnslothTrainer(UnslothTrainer):
    """
    Patches a view-tensor issue in Unsloth's DDP loss scaling
    (https://github.com/unslothai/unsloth/issues/2435).
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels") if (self.label_smoother and "labels" in inputs) else None
        outputs = model(**inputs)

        if labels is not None:
            raw = self.accelerator.unwrap_model(model)
            shift = hasattr(raw, "_get_name") and "unsloth" in raw._get_name().lower()
            loss  = self.label_smoother(outputs, labels, shift_labels=shift) if shift \
                    else self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Clone to detach view before DDP in-place multiply
        if hasattr(loss, "clone"):
            loss = loss.clone()

        if self.accelerator.num_processes > 1:
            loss = loss * self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss


# ── Data collator ─────────────────────────────────────────────────────────────

class ArcDataCollator(DataCollatorForLanguageModeling):
    """
    Masks labels so that loss is only computed on *assistant* turns
    (the grid output), not on the user prompts.
    """

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)

        for i, _ in enumerate(examples):
            ids   = batch["input_ids"][i]
            labels = ids.clone()

            u_pos  = np.where(ids.numpy() == USER_TOKEN_ID)[0].tolist()
            a_pos  = np.where(ids.numpy() == ASSISTANT_TOKEN_ID)[0].tolist()
            starts = sorted(u_pos + a_pos)
            ends   = np.where(ids.numpy() == EOS_ID)[0]

            batch["labels"][i, :] = -100

            for turn_idx, (s, e) in enumerate(zip(starts, ends)):
                assert s < e
                if turn_idx % 2 == 1:          # assistant turn
                    s += 2                      # skip <|im_start|>assistant
                    e += 1                      # include <|im_end|>
                    batch["labels"][i, s:e] = labels[s:e]

        return batch


# ── DFS beam search ───────────────────────────────────────────────────────────

def _dfs_step(model, logits, max_new_tokens, max_score, scores, pos,
              cache, start_time, end_time) -> dict:
    """
    One recursive step of the turbo DFS: expand each beam by one token,
    collecting completed sequences (EOS reached) into *suffixes*.
    """
    n   = logits.size(0)
    nll = torch.tensor(scores, dtype=torch.float32).view(n, 1) - logits.float().cpu().log_softmax(-1)

    suffixes   = defaultdict(list)
    candidates = {i: [] for i in range(n)}

    for i in range(n):
        for t in ARC_TOKENS:
            sc = nll[i, t].item()
            if sc >= max_score:
                continue
            if t == EOS_ID:
                suffixes[i].append((sc, [t]))
            elif max_new_tokens > 1:
                candidates[i].append((sc, t))
        candidates[i].sort(key=lambda x: x[0])

    while time.time() - start_time < 540 and time.time() < end_time:
        batch_toks   = []
        batch_scores = []
        n_alive      = 0

        for i in range(n):
            if candidates[i]:
                sc, t = candidates[i].pop(0)
                batch_toks.append(t)
                batch_scores.append(sc)
                n_alive += 1
            else:
                batch_toks.append(PAD_ID)
                batch_scores.append(1e6)

        if n_alive == 0:
            break

        out = model(
            input_ids      = torch.tensor(batch_toks, device=model.device, dtype=torch.long).view(-1, 1),
            position_ids   = torch.full((n, 1), pos, device=model.device),
            past_key_values= cache,
            return_dict    = True,
            use_cache      = True,
        )

        child_suffixes = _dfs_step(
            model,
            logits        = out.logits[:, -1],
            max_new_tokens= max_new_tokens - 1,
            max_score     = max_score,
            scores        = batch_scores,
            pos           = pos + 1,
            cache         = out.past_key_values,
            start_time    = start_time,
            end_time      = end_time,
        )

        for bid, beams in child_suffixes.items():
            for sc, toks in beams:
                toks.insert(0, batch_toks[bid])
                suffixes[bid].append((sc, toks))

    return suffixes


@torch.no_grad()
def beam_search(model, prefix_token_lists, max_new_tokens, max_score, end_time):
    """
    Run turbo DFS from multiple prefix sequences in a single forward pass.
    Returns a list of (batch_index, sorted_beams) pairs.
    """
    input_ids = torch.tensor(prefix_token_lists, device=model.device, dtype=torch.long)
    out       = model(input_ids=input_ids, return_dict=True, use_cache=True)

    raw = _dfs_step(
        model,
        logits        = out.logits[:, -1],
        max_new_tokens= max_new_tokens,
        max_score     = max_score,
        scores        = [0.0] * input_ids.size(0),
        pos           = input_ids.size(1),
        cache         = out.past_key_values,
        start_time    = time.time(),
        end_time      = end_time,
    )

    return [
        (bid, sorted(beams, key=lambda x: x[0]))
        for bid, beams in raw.items()
    ]

# Legacy alias
inference_turbo_dfs = beam_search


@torch.no_grad()
def score_candidates(queries, answers, tokenizer, model) -> list:
    """
    Compute per-sample NLL for (query, answer) pairs using a single batched
    forward pass.  Returns a list of positive scalars (lower = better).
    """
    q_toks, a_toks, all_toks, lengths = [], [], [], []
    for q, a in zip(queries, answers):
        qt = tokenizer.encode(q)
        at = tokenizer.encode(a)
        q_toks.append(qt)
        a_toks.append(at)
        all_toks.append(qt + at)
        lengths.append(len(qt) + len(at))

    max_len     = max(lengths)
    padded      = [t + [PAD_ID] * (max_len - len(t)) for t in all_toks]
    input_ids   = torch.tensor(padded, device=model.device, dtype=torch.long)
    out         = model(input_ids=input_ids, return_dict=True, use_cache=True)
    log_probs   = out.logits.float().cpu().log_softmax(-1)

    results = []
    for lp, qt, at in zip(log_probs, q_toks, a_toks):
        ql  = len(qt)
        ans_lp = lp[ql - 1: ql - 1 + len(at)]
        nll    = -ans_lp[torch.arange(len(at)), at].sum().item()
        results.append(nll)
    return results

# Legacy alias
calc_scores = score_candidates


# ── Per-puzzle worker ─────────────────────────────────────────────────────────

def worker(rank: int, queue, end_time: float):
    """
    Main per-GPU worker.  Loads the base model once, then for each puzzle:
      1. Resets LoRA weights.
      2. Fine-tunes on augmented training examples.
      3. Runs turbo DFS beam search over augmented test views.
      4. Scores each unique hypothesis with augmented NLL.
      5. Saves results to disk.
    """

    is_rerun = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))

    # ── LoRA config ───────────────────────────────────────────────────────────
    peft_cfg = dict(
        r                      = 256,
        target_modules         = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "embed_tokens", "lm_head",
        ],
        lora_alpha             = 32,
        lora_dropout           = 0.0,
        bias                   = "none",
        use_gradient_checkpointing = False,
        random_state           = 42,
        use_rslora             = True,
        loftq_config           = None,
    )

    # ── Training hyperparameters ──────────────────────────────────────────────
    train_cfg = dict(
        per_device_train_batch_size  = 1,
        per_device_eval_batch_size   = 1,
        gradient_accumulation_steps  = 1,
        num_train_epochs             = 1,
        warmup_steps                 = 0,
        warmup_ratio                 = 0.1,
        max_grad_norm                = 1.0,
        learning_rate                = 5e-5,
        optim                        = "adamw_torch",
        weight_decay                 = 0.0,
        lr_scheduler_type            = "cosine",
        seed                         = 42,
        report_to                    = "none",
        save_strategy                = "no",
        eval_strategy                = "no",
        logging_strategy             = "no",
        fp16                         = False,
        bf16                         = True,
        fsdp                         = "",
        ddp_find_unused_parameters   = False,
        dataloader_num_workers       = 0,
        gradient_checkpointing       = False,
    )

    MAX_SEQ = 8192

    # ── Load base model ───────────────────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name             = "/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
        full_finetuning        = False,
        load_in_4bit           = False,
        local_files_only       = True,
        use_gradient_checkpointing = False,
        max_seq_length         = MAX_SEQ,
    )
    model = FastLanguageModel.get_peft_model(model, **peft_cfg)

    # Ensure uniform bfloat16 precision
    for _, p in model.named_parameters():
        if p.dtype == torch.float32:
            p.data = p.data.to(torch.bfloat16)

    # Cache pristine LoRA weights for per-puzzle reset
    init_weights = {
        k: v.clone().detach()
        for k, v in get_peft_model_state_dict(model, adapter_name="default").items()
    }

    collator  = ArcDataCollator(tokenizer=tokenizer, mlm=False)
    formatter = QwenFormatter(tokenizer=tokenizer)
    MAX_NEW   = formatter.max_new_tokens()
    MAX_SCORE = -np.log(0.2)   # threshold ≈ NLL for p=0.2

    test_path = (
        "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
        if is_rerun else
        "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"
    )
    arc_test = ArcDataset.from_file(test_path)
    out_dir  = "/kaggle/inference_outputs"
    os.makedirs(out_dir, exist_ok=True)

    # ── Puzzle loop ───────────────────────────────────────────────────────────
    while not queue.empty():
        if time.time() > end_time:
            print(f"[GPU {rank}] time limit reached — stopping.")
            break

        key = queue.get()
        if key is None:
            break

        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        print(f"[GPU {rank}] starting puzzle {key}")

        # Reset LoRA to initial weights
        set_peft_model_state_dict(model, init_weights.copy(), adapter_name="default")
        model = FastLanguageModel.for_training(model)

        # ── Fine-tune on augmented training examples ───────────────────────
        puzzle_ds = arc_test.change_keys([key])
        train_ds  = puzzle_ds.augment(n=16, shfl_keys=True, seed=1)
        train_ds  = train_ds.cut_to_len(formatter=formatter, name="text", max_len=MAX_SEQ)

        with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
            trainer = FixedUnslothTrainer(
                model              = model,
                tokenizer          = tokenizer,
                data_collator      = collator,
                train_dataset      = Dataset.from_list(train_ds.as_list(formatter)),
                dataset_text_field = "text",
                max_seq_length     = MAX_SEQ,
                args               = UnslothTrainingArguments(**train_cfg),
            )
            stats = trainer.train()
            model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
            del trainer

        model = FastLanguageModel.for_inference(model)
        gc.collect()
        torch.cuda.empty_cache()

        mem_train = torch.cuda.max_memory_allocated() // 1024 ** 2
        print(f"[GPU {rank}] training peak memory: {mem_train} MB  stats: {stats}")
        torch.cuda.reset_peak_memory_stats()

        # ── Build inference batches ────────────────────────────────────────
        puzzle_multi = puzzle_ds.split_multi_replies()
        eval_ds      = puzzle_multi.augment(n=2, seed=2)
        eval_ds      = eval_ds.cut_to_len(formatter=formatter, name="input",
                                           max_len=MAX_SEQ - MAX_NEW)

        # Group sub-keys by test-input index, then arrange into batches that
        # pair geometrically complementary augmentations (for memory efficiency).
        by_test = defaultdict(list)
        for sk in sorted(eval_ds.keys):
            tid = sk.split(".")[0].split("_")[1]
            by_test[tid].append(sk)

        batches = []
        for sks in by_test.values():
            # Group: plain rotations (0,4) and (2,6); then transposed (8,12) and (10,14)
            for offsets in ([0, 4], [2, 6]):
                batch = []
                for off in offsets:
                    batch.extend(sks[off: off + 2])
                batches.append(batch)
            for offsets in ([8, 12], [10, 14]):
                batch = []
                for off in offsets:
                    batch.extend(sks[off: off + 2])
                batches.append(batch)

        # ── Beam search + augmented scoring ───────────────────────────────
        with torch.inference_mode():
            scored_cache = {}   # (base_key, grid_hash) → aug scores

            for batch_sks in batches:
                elapsed = time.time() - t0
                if elapsed > 1200 or time.time() > end_time:
                    print(f"[GPU {rank}] puzzle {key} timed out after {elapsed:.0f}s")
                    break

                print(f"[GPU {rank}] decoding {batch_sks}")
                prefix_toks = [
                    tokenizer.encode(eval_ds.get(sk, formatter)["input"])
                    for sk in batch_sks
                ]

                dfs_out = beam_search(model, prefix_toks, MAX_NEW, MAX_SCORE, end_time)

                for bid, beams in dfs_out:
                    sk   = batch_sks[bid]
                    bk   = sk.split(".")[0]
                    hits = []

                    for beam_score, toks in beams:
                        arr = formatter.tokens_to_grid(toks)
                        if arr is None:
                            continue

                        sol    = puzzle_multi.invert_mod(arr, sk, inv_perm=True)
                        gid    = (bk, tuple(map(tuple, sol)))

                        if gid in scored_cache:
                            aug_scores = scored_cache[gid]
                        else:
                            print(f"[GPU {rank}] verifying {sk} candidate #{len(hits)}")
                            aug_ds = ArcDataset(
                                keys    = [bk],
                                queries = {bk: puzzle_multi.queries[bk]},
                                replies = {bk: [sol.tolist()]},
                            )
                            aug_ds = aug_ds.augment(seed=hash(bk) % (1024 ** 2))
                            aug_ds = aug_ds.cut_to_len(
                                formatter=formatter, name="input",
                                max_len=MAX_SEQ - MAX_NEW
                            )
                            qs, ans = zip(*[
                                (s["input"], s["reply"])
                                for s in aug_ds.as_list(formatter)
                            ])
                            aug_scores = (
                                score_candidates(list(qs[:4]),  list(ans[:4]),  tokenizer, model)
                                + score_candidates(list(qs[4:]), list(ans[4:]), tokenizer, model)
                            )
                            scored_cache[gid] = aug_scores

                        hits.append({
                            "beam_score": beam_score,
                            "score_aug":  aug_scores,
                            "solution":   sol,
                        })

                    if hits:
                        out_path = os.path.join(out_dir, sk)
                        with bz2.BZ2File(out_path, "w") as fh:
                            pickle.dump(hits, fh)

        mem_infer = torch.cuda.max_memory_allocated() // 1024 ** 2
        elapsed   = time.time() - t0
        print(f"[GPU {rank}] puzzle {key} done — {elapsed:.0f}s  infer peak {mem_infer} MB")
