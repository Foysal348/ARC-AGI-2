"""Shared request plumbing and the puzzle driver for the codex_lite policy.

``solve_puzzle`` runs the selected ``codex_lite`` attempts of one puzzle, votes their
outputs into Kaggle's two submission candidates and reports each physical attempt as an
intact trace. Every request is rendered locally with the Qwen3.8 chat template and fails
closed unless SGLang returns the identical prompt token IDs. ``codex_lite`` imports the
request helpers below lazily.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import os
import struct
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from codex_lite import (CODEX_LITE_ALTERNATE_POLICIES, CODEX_LITE_CARRY_POLICIES, CODEX_LITE_EFFORTS,
                        CODEX_LITE_LOGIT_BIASES, FILLER_TOKENS, TEMPERATURE, TOP_P,
                        CODEX_LITE_FIRST_REQUEST_CAPS, CODEX_LITE_IMAGE_POLICIES,
                        CODEX_LITE_POLICIES,
                        CODEX_LITE_REQUEST_CAPS, CODEX_LITE_TOKEN_CAP,
                        CODEX_LITE_TOOL_POLICIES, carry_path, run_codex_lite_episode)
Grid = list[list[int]]

RUNNABLE_POLICY_IDS = CODEX_LITE_POLICIES
RUNNABLE_POLICY_TOKEN_CEILINGS = dict.fromkeys(CODEX_LITE_POLICIES, CODEX_LITE_TOKEN_CAP)

CTX = 262144
REQ_TIMEOUT = 900.0

# Sampled at solve_puzzle's temperature and top_p (codex_lite's TEMPERATURE and TOP_P by default).
QWEN38_TOP_K = 20
QWEN38_MIN_P = 0.0

def _fmt_grid(grid: Grid) -> str:
    return "\n".join(" ".join(str(value) for value in row) for row in grid)


def render_user_prompt(
    train: list[dict],
    test_inputs: list[Grid],
    allowed_tools: tuple[str, ...],
) -> str:
    """Render the task grids in the NVIDIA-style layout that ends codex_lite's task message."""
    parts = [
        "Please solve this ARC-AGI problem:",
        f"**Allowed Tools:** {list(allowed_tools)}",
        "",
    ]
    for index, pair in enumerate(train, 1):
        parts += [
            f"Train Example {index}:",
            "",
            "Input:",
            _fmt_grid(pair["input"]),
            "",
            "Output:",
            _fmt_grid(pair["output"]),
            "",
        ]
    parts.append("")
    if len(test_inputs) == 1:
        parts += ["Test Input:", _fmt_grid(test_inputs[0])]
    else:
        for index, test_input in enumerate(test_inputs, 1):
            parts += [f"Test Input {index}:", _fmt_grid(test_input), ""]
    return "\n".join(parts)

class PromptContractError(RuntimeError):
    """A locally rendered continuation differed from SGLang's prompt contract."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _ids_summary(token_ids: Sequence[int]) -> dict[str, Any]:
    raw = b"".join(
        int(token).to_bytes(4, "little", signed=False) for token in token_ids
    )
    return {
        "count": len(token_ids),
        "sha256_u32le": hashlib.sha256(raw).hexdigest(),
    }


def _get_extra(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    value = getattr(obj, name, None)
    if value is not None:
        return value
    extra = getattr(obj, "model_extra", None)
    return extra.get(name) if isinstance(extra, dict) else None


def _response_token_ids(response: Any, names: Sequence[str]) -> Optional[list[int]]:
    choices = _get_extra(response, "choices") or []
    candidates = [response]
    if choices:
        candidates.extend((choices[0], _get_extra(choices[0], "message")))
    for candidate in candidates:
        for name in names:
            value = _get_extra(candidate, name)
            if hasattr(value, "tolist"):
                value = value.tolist()
            if isinstance(value, list) and all(
                isinstance(token, int) and not isinstance(token, bool)
                for token in value
            ):
                return [int(token) for token in value]
    return None


def _verify_prompt_ids(
    local_ids: Sequence[int], server_ids: Optional[Sequence[int]], image_token: Optional[int] = None
) -> dict[str, Any]:
    if server_ids is None:
        raise PromptContractError(
            "SGLang response omitted requested prompt token IDs"
        )
    if list(server_ids) != list(local_ids):
        # SGLang may put per-image cache ids where the prompt has image pads; every other token must match.
        if (image_token is not None and image_token in local_ids and len(server_ids) == len(local_ids)
                and all(lhs == rhs for lhs, rhs in zip(local_ids, server_ids) if lhs != image_token)):
            return {
                "status": "match_outside_images",
                "local": _ids_summary(local_ids),
                "server": _ids_summary(server_ids),
            }
        common = 0
        for lhs, rhs in zip(local_ids, server_ids):
            if lhs != rhs:
                break
            common += 1
        raise PromptContractError(
            "server/local rendered prompt token IDs differ "
            f"(local={len(local_ids)}, server={len(server_ids)}, lcp={common})"
        )
    return {
        "status": "exact_match",
        "local": _ids_summary(local_ids),
        "server": _ids_summary(server_ids),
    }


def png_size(url: str) -> tuple[int, int]:
    """Width and height of a base64 PNG data URL (codex_lite.task_images)."""
    prefix = "data:image/png;base64,"
    if not url.startswith(prefix):
        raise PromptContractError("only base64 PNG data URLs render locally")
    header = base64.b64decode(url[len(prefix):len(prefix) + 44])  # The signature and IHDR.
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise PromptContractError("image data URL is not a PNG")
    return struct.unpack(">II", header[16:24])


def message_image_sizes(messages: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """(width, height) of every image part, in the order the chat template renders them."""
    return [png_size(item["image_url"]["url"]) for message in messages if isinstance(message.get("content"), list)
            for item in message["content"] if isinstance(item, dict) and item.get("type") == "image_url"]


def image_token_count(width: int, height: int, processor: dict[str, Any]) -> int:
    """Vision tokens for one image: Qwen2-VL's resize to multiples of patch x merge, one token per merged patch."""
    factor = processor["patch_size"] * processor["merge_size"]
    min_pixels, max_pixels = processor["size"]["shortest_edge"], processor["size"]["longest_edge"]
    h_bar, w_bar = max(factor, round(height / factor) * factor), max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar, w_bar = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return (h_bar // factor) * (w_bar // factor)


def expand_image_pads(token_ids: list[int], sizes: list[tuple[int, int]], image_token: int,
                      processor: dict[str, Any]) -> list[int]:
    """The template renders one pad per image; the server expands each to that image's vision tokens."""
    counts = [image_token_count(width, height, processor) for width, height in sizes]
    if token_ids.count(image_token) != len(counts):
        raise PromptContractError(f"{token_ids.count(image_token)} image pads for {len(counts)} images")
    expanded, remaining = [], iter(counts)
    for token in token_ids:
        expanded += [token] * next(remaining) if token == image_token else [token]
    return expanded


class Qwen38PromptRenderer:
    """Mirror SGLang's Qwen3.8 text chat rendering exactly (images: pads expanded as the server does)."""

    def __init__(self, tokenizer_path: Path):
        import transformers
        from transformers import AutoTokenizer

        self.path = tokenizer_path.resolve()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.path), trust_remote_code=True, local_files_only=True
        )
        self._lock = threading.Lock()
        template = str(self.tokenizer.chat_template)
        self.identity = {
            "path": str(self.path),
            "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "tokenizer_class": type(self.tokenizer).__name__,
            "transformers_version": transformers.__version__,
            "server_tokenization_contract": "sglang_render_then_encode_v1",
        }
        self.identity["sha256"] = hashlib.sha256(
            _canonical_json(self.identity)
        ).hexdigest()
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        processor = self.path / "preprocessor_config.json"
        self.image_processor = json.loads(processor.read_text()) if processor.is_file() else None

    @staticmethod
    def _normalize_tool_arguments(messages: list[dict[str, Any]]) -> None:
        for message in messages:
            if message.get("role") != "assistant" or not isinstance(
                message.get("tool_calls"), list
            ):
                continue
            for tool_call in message["tool_calls"]:
                function = (
                    tool_call.get("function")
                    if isinstance(tool_call, dict)
                    else None
                )
                if not isinstance(function, dict) or not isinstance(
                    function.get("arguments"), str
                ):
                    continue
                try:
                    arguments = json.loads(function["arguments"])
                except json.JSONDecodeError as exc:
                    raise PromptContractError(
                        "assistant tool-call arguments are not valid JSON"
                    ) from exc
                if not isinstance(arguments, dict):
                    raise PromptContractError(
                        "assistant tool-call arguments must decode to an object"
                    )
                function["arguments"] = arguments

    def render(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        reasoning_effort: str,
        enable_thinking: bool = True,
    ) -> list[int]:
        render_messages = copy.deepcopy(messages)
        self._normalize_tool_arguments(render_messages)
        with self._lock:
            rendered_prompt = self.tokenizer.apply_chat_template(
                render_messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=copy.deepcopy(tools),
                reasoning_effort=reasoning_effort,
                enable_thinking=enable_thinking,
                preserve_thinking=True,
            )
            if not isinstance(rendered_prompt, str):
                raise PromptContractError(
                    "tokenizer did not return one rendered prompt string"
            )
            token_ids = self.tokenizer.encode(
                rendered_prompt,
                add_special_tokens=False,
            )
            if not isinstance(token_ids, list) or not all(
                isinstance(token, int) and not isinstance(token, bool)
                for token in token_ids
            ):
                raise PromptContractError(
                    "tokenizer did not return one integer token sequence"
                )
            token_ids = [int(token) for token in token_ids]
        sizes = message_image_sizes(render_messages)
        if sizes:
            if self.image_processor is None:
                raise PromptContractError("image prompts need the model's preprocessor_config.json")
            token_ids = expand_image_pads(token_ids, sizes, self.image_token_id, self.image_processor)
        return token_ids


_RENDERERS: dict[str, Qwen38PromptRenderer] = {}
_RENDERERS_LOCK = threading.Lock()


def _prompt_renderer(tokenizer_path: str | os.PathLike[str]) -> Qwen38PromptRenderer:
    key = str(Path(tokenizer_path).resolve())
    with _RENDERERS_LOCK:
        renderer = _RENDERERS.get(key)
        if renderer is None:
            renderer = Qwen38PromptRenderer(Path(key))
            _RENDERERS[key] = renderer
        return renderer


def _usage_tokens(response: Any, requested: int) -> tuple[Any, Any, int, str]:
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = (
        getattr(usage, "completion_tokens", None) if usage else None
    )
    if (
        isinstance(completion_tokens, int)
        and not isinstance(completion_tokens, bool)
        and completion_tokens >= 0
    ):
        if completion_tokens > requested:
            raise PromptContractError(
                f"reported completion tokens {completion_tokens} exceed cap {requested}"
            )
        return prompt_tokens, completion_tokens, completion_tokens, "reported"
    return prompt_tokens, completion_tokens, requested, "requested_cap_fallback"



def _tool_call_dicts(message: Any) -> list[dict[str, Any]]:
    """The message's native tool calls as plain dicts; nothing is recovered from text."""
    result: list[dict[str, Any]] = []
    for call in getattr(message, "tool_calls", None) or []:
        function = getattr(call, "function", None)
        arguments = getattr(function, "arguments", "")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        result.append(
            {
                "id": str(getattr(call, "id", None) or f"call_{len(result)}"),
                "type": "function",
                "function": {
                    "name": str(getattr(function, "name", "")),
                    "arguments": str(arguments or ""),
                },
            }
        )
    return result


def _qwen_extra_body(*, enable_thinking: bool, reasoning_effort: str) -> dict[str, Any]:
    return {
        "top_k": QWEN38_TOP_K,
        "min_p": QWEN38_MIN_P,
        "reasoning_effort": reasoning_effort,
        "chat_template_kwargs": {
            "enable_thinking": enable_thinking,
            "preserve_thinking": True,
        },
    }


def _hashable(grid: Grid) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in grid)


def _vote_top2(grids: list[Grid]) -> list[Grid]:
    if not grids:
        return []
    counts = Counter(_hashable(grid) for grid in grids)
    return [list(map(list, value)) for value, _ in counts.most_common(2)]


def submission_candidates(outputs_by_sample: Sequence[Optional[list[Optional[Grid]]]],
                          alternates_by_sample: Sequence[Optional[list[Optional[Grid]]]],
                          ntest: int) -> list[list[Grid]]:
    """Each test's candidates from its solved samples: the top two votes, or a lone answer and an alternate.

    ``solve_puzzle`` returns these as its submission; ``compare_runs`` rebuilds them from stored attempts.
    """
    per_test_grids: list[list[Grid]] = [[] for _ in range(ntest)]
    for outputs in outputs_by_sample:
        for test_index, grid in enumerate(outputs or []):
            if grid is not None:
                per_test_grids[test_index].append(grid)
    submission = [_vote_top2(grids) for grids in per_test_grids]
    for test_index, candidates in enumerate(submission):
        # Only an empty second place takes an attempt's own alternate, so voting is unchanged.
        if len(candidates) != 1:
            continue
        for alternates in alternates_by_sample:
            grid = (alternates or [None] * ntest)[test_index]
            if grid is not None and grid != candidates[0]:
                candidates.append(grid)
                break
    return submission


def policy_logit_bias(policy: str, renderer: Any) -> Optional[dict[str, float]]:
    """The policy's fixed logit bias, after checking each token id still decodes to its intended text."""
    bias = CODEX_LITE_LOGIT_BIASES.get(policy)
    tokenizer = getattr(renderer, "tokenizer", None)
    if bias and tokenizer is not None:
        for token in bias:
            if tokenizer.decode([int(token)]) != FILLER_TOKENS[token]:
                raise PromptContractError(f"token {token} is not {FILLER_TOKENS[token]!r} in this tokenizer")
    return bias


def solve_puzzle(
    client: Any,
    model: str,
    puzzle: dict[str, Any],
    *,
    policies: list[str],
    time_budget: float,
    tokenizer_path: str | os.PathLike[str],
    seed_base: int = 1234,
    label: str = "",
    context_window: Optional[int] = None,
    request_timeout: Optional[float] = None,
    trace_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    attempt_limiter: Any = None,
    sample_indices: Optional[list[int]] = None,
    completion_token_cap: Optional[int] = None,
    provisional_callback: Optional[Callable[[int, list], None]] = None,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    logit_bias: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Run the selected ``codex_lite`` attempts and return Kaggle-formatted candidates.

    ``provisional_callback(sample_index, outputs)`` receives an attempt's saved outputs while
    it is still running, so a caller can persist them before a hard stop.

    ``policies`` defines both sample count and per-sample policy.  ``sample_indices``
    may select a coverage-first phase from that full schedule without changing
    sample/seed identity. ``tokenizer_path`` renders every request locally so its
    exact server-side prompt token IDs can be verified. ``completion_token_cap`` may
    only lower the policy's per-attempt budget.
    """
    if not policies:
        raise ValueError("at least one policy is required")
    unknown = sorted(set(policies) - RUNNABLE_POLICY_IDS)
    if unknown:
        raise ValueError(
            f"unknown policies {unknown}; choose from {sorted(RUNNABLE_POLICY_IDS)}"
        )
    samples = len(policies)
    selected_samples = (
        list(range(samples)) if sample_indices is None else list(sample_indices)
    )
    if (
        not selected_samples
        or len(selected_samples) != len(set(selected_samples))
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= samples
            for index in selected_samples
        )
    ):
        raise ValueError("sample_indices must be unique valid sample indices")
    train = puzzle.get("train")
    tests = puzzle.get("test")
    if not isinstance(train, list) or not train:
        raise ValueError("puzzle must contain at least one train pair")
    if not isinstance(tests, list) or not tests:
        raise ValueError("puzzle must contain at least one test input")
    try:
        test_inputs = [test["input"] for test in tests]
    except (KeyError, TypeError) as exc:
        raise ValueError("every puzzle test item must contain an input grid") from exc
    renderer = _prompt_renderer(tokenizer_path)

    ntest = len(test_inputs)
    ctx = int(context_window or CTX)
    timeout = float(request_timeout or REQ_TIMEOUT)
    outputs_by_sample: list[Optional[list[Optional[Grid]]]] = [None] * samples
    # Second candidates an attempt saved beside its answer (codex_lite.Alternates).
    alternates_by_sample: list[Optional[list[Optional[Grid]]]] = [None] * samples
    selected_policy_ids = sorted({policies[index] for index in selected_samples})
    stats: dict[str, Any] = {
        "calls": 0,
        "tool_calls": 0,
        "gen_tokens": 0,
        "samples_solved": 0,
        "samples_failed": 0,
        "sample_errors": [],
        "attempts": [],
        "best_n_passed": 0,
        "n_train": len(train),
        "policies": {
            policy: {
                "samples": sum(
                    policies[index] == policy for index in selected_samples
                ),
                "samples_solved": 0,
                "samples_failed": 0,
            }
            for policy in selected_policy_ids
        },
        "prompt_renderer": copy.deepcopy(renderer.identity),
    }

    def one_sample(sample_index: int) -> tuple[Any, ...]:
        policy = policies[sample_index]
        attempt_stats: dict[str, Any] = {
            "calls": 0,
            "tool_calls": 0,
            "gen_tokens": 0,
            "best_n_passed": 0,
            "n_train": len(train),
            "prompt_renderer": copy.deepcopy(renderer.identity),
        }
        trace: list[dict[str, Any]] = []
        acquired = False
        started = time.time()
        try:
            if attempt_limiter is not None:
                attempt_limiter.acquire()
                acquired = True
            started = time.time()
            deadline = started + time_budget
            # A carry policy's first attempt leaves its state if unverified; the task's second attempt resumes it.
            carry = carry_path(label) if policy in CODEX_LITE_CARRY_POLICIES and label else None
            outputs = run_codex_lite_episode(
                client, model, train, test_inputs, deadline, seed_base + sample_index,
                renderer=renderer, stats=attempt_stats, trace=trace, context_window=ctx,
                request_timeout=timeout, completion_token_cap=completion_token_cap, temperature=temperature, top_p=top_p,
                logit_bias=logit_bias if logit_bias is not None else policy_logit_bias(policy, renderer),
                reasoning_effort=CODEX_LITE_EFFORTS[policy],
                alternates=policy in CODEX_LITE_ALTERNATE_POLICIES,
                tools=policy in CODEX_LITE_TOOL_POLICIES, request_cap=CODEX_LITE_REQUEST_CAPS.get(policy),
                first_request_caps=CODEX_LITE_FIRST_REQUEST_CAPS.get(policy),
                images=policy in CODEX_LITE_IMAGE_POLICIES,
                carry_out=carry if sample_index == 0 else None, carry_in=carry if sample_index == 1 else None,
                on_outputs=(None if provisional_callback is None
                            else lambda outputs: provisional_callback(sample_index, outputs)),
            )
            return (
                outputs,
                attempt_stats,
                policy,
                None,
                sample_index,
                trace,
                round(time.time() - started, 3),
            )
        except Exception as exc:
            return (
                None,
                attempt_stats,
                policy,
                exc,
                sample_index,
                trace,
                round(time.time() - started, 3),
            )
        finally:
            if acquired:
                attempt_limiter.release(int(attempt_stats.get("gen_tokens") or 0))

    with ThreadPoolExecutor(max_workers=len(selected_samples)) as executor:
        futures = [executor.submit(one_sample, index) for index in selected_samples]
        for future in as_completed(futures):
            (
                outputs,
                attempt_stats,
                policy,
                error,
                sample_index,
                trace,
                attempt_wall,
            ) = future.result()
            for key in ("calls", "tool_calls", "gen_tokens"):
                stats[key] += int(attempt_stats.get(key) or 0)
            attempt = {
                "sample": sample_index,
                "seed": seed_base + sample_index,
                "policy": policy,
                "outputs": outputs,
                "completion_tokens_used": attempt_stats.get(
                    "completion_tokens_used", attempt_stats.get("gen_tokens", 0)
                ),
                "end_reason": attempt_stats.get("end_reason"),
                # Visible training pairs the attempt's final program passes (kaggle_runner.primary_verified).
                "best_n_passed": attempt_stats.get("best_n_passed", 0),
                "n_train": len(train),
            }
            if trace_callback is not None:
                trace_callback(
                    {
                        "schema_version": 3,
                        "task_id": label,
                        "model": model,
                        "sample": sample_index,
                        "seed": seed_base + sample_index,
                        "policy": policy,
                        "puzzle": puzzle,
                        "turns": trace,
                        "outputs": outputs,
                        "stats": attempt_stats,
                        "wall": attempt_wall,
                        "error": None if error is None else repr(error),
                    }
                )
            stats["best_n_passed"] = max(stats["best_n_passed"], attempt["best_n_passed"])
            if error is not None:
                error_text = " ".join(
                    f"{type(error).__name__}: {error}".split()
                )[:500]
                attempt["error"] = error_text
                stats["attempts"].append(attempt)
                stats["samples_failed"] += 1
                stats["policies"][policy]["samples_failed"] += 1
                stats["sample_errors"].append(
                    {
                        "sample": sample_index,
                        "policy": policy,
                        "seed": seed_base + sample_index,
                        "error": error_text,
                    }
                )
                if label:
                    print(
                        f"  [live] {label} attempt failed  policy={policy}  "
                        f"seed={seed_base + sample_index}  error={error_text}",
                        flush=True,
                    )
                continue
            stats["attempts"].append(attempt)
            solved = bool(outputs) and any(grid is not None for grid in outputs)
            if label:
                print(
                    f"  [live] {label} attempt done  policy={policy}  "
                    f"best_train={attempt_stats.get('best_n_passed', 0)}/{len(train)}  "
                    f"selected={solved}  gen_tok={attempt_stats['gen_tokens']}",
                    flush=True,
                )
            if not solved:
                continue
            assert outputs is not None
            outputs_by_sample[sample_index] = outputs
            alternates_by_sample[sample_index] = attempt_stats.get("alt_outputs")
            stats["samples_solved"] += 1
            stats["policies"][policy]["samples_solved"] += 1

    stats["sample_errors"].sort(key=lambda item: item["sample"])
    stats["attempts"].sort(key=lambda item: item["sample"])
    if stats["samples_failed"] == len(selected_samples):
        details = "; ".join(
            f"sample {item['sample']} ({item['policy']}, seed={item['seed']}): "
            f"{item['error']}"
            for item in stats["sample_errors"]
        )
        raise RuntimeError(
            f"all {len(selected_samples)} attempts failed: {details}"
        )
    return {
        "submission": submission_candidates(outputs_by_sample, alternates_by_sample, ntest),
        "stats": stats,
    }
