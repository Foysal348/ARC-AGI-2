"""ARC-AGI-2 adapter for Tufa Labs' Duck harness (Daniel Franzen's milestone-2 fork, vendored verbatim in vendor/).

Experiment 12 (user-approved plan, Rev 1). Only what differs between ARC-AGI-2 and ARC-AGI-3 lives here:
  1. system prompt = ARC-2 objective + the task itself, set after ToolAgent.__init__ (Rev 1 #1)
  2. a trusted pure-Python prelude (task grids + check/submit) prepended to the model's code (Rev 1 #2)
  3. submit() signals through Duck's existing action channel; the host re-verifies the retained function in a
     clean sandbox and never trusts the self-report (Rev 1 #3)
  4. no numpy: the sandbox is unchanged; transforms use lists (Rev 1 #4)
  5. a minimal runtime-state file via Duck's write_runtime_state + an ARC-2 tool description (Rev 1 #5)
  6. selection: host-verified train pass > agreement across branches > distinct attempt_2 (Rev 1 #6)
  8. Duck's inline no-tool-call nudge keeps its generic prefix; only the game tail is replaced (Rev 1 #8)
The driver loop follows Duck's HarnessSolver.play()/_play_inner() (priority gate, failure back-off, yield handling).

Import order matters: Duck reads its LOCAL_ANALYZER_* / ARC3_* settings at import time, so the caller sets the
environment (DUCK_ENV below, the milestone-2 values) before importing this module.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

# Milestone-2 settings copied from dfranzen/arc-agi-3-milestone-2-solution cell 4 (game-only switches left off).
DUCK_ENV = {
    "LOCAL_ANALYZER_PROVIDER": "vllm", "OPENAI_PROVIDER": "vllm",
    "LOCAL_ANALYZER_APP_NAME": "ARC2 Duck Harness",
    "ARC3_REASONING_HISTORY_KEY": "reasoning_content",
    "LOCAL_ANALYZER_ENABLE_THINKING": "true",
    "LOCAL_ANALYZER_TOOL_STEPS": "0",
    "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
    "LOCAL_ANALYZER_YIELD_SECONDS": "0",
    "LOCAL_ANALYZER_YIELD_TOKENS": "2048",
    "LOCAL_ANALYZER_TEMPERATURE": "0.7", "LOCAL_ANALYZER_TOP_P": "0.95", "LOCAL_ANALYZER_TOP_K": "20",
    "LOCAL_ANALYZER_CONTEXT_WINDOW": str((116 + 12) * 1024),
    "LOCAL_ANALYZER_MAX_OUTPUT": str(12 * 1024),
    "ARC3_CONTEXT_DRAIN_TOKENS": str(58 * 1024),
    "ARC3_HISTORY_ASSISTANT_TURNS": "150", "ARC3_HISTORY_TURN_DRAIN": "30", "ARC3_HISTORY_DRAIN_COALESCE": "1",
    "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "3072", "ARC3_MIDDLE_TRUNCATION": "1",
    "ARC3_MEMORY_SECTIONS": "off",
    "ARC3_PERSISTENT_FUNCTIONS": "1", "ARC3_PERSISTENT_FUNCTIONS_SCOPE": "game",
    "ARC3_PERSISTENT_FUNCTIONS_IMPORTS": "1",
    "ARC3_PRIORITY_REFRESH_QUEUE": "1", "ARC3_PRIORITY_PACE": "0",
    "ARC3_PRIORITY_TAIL_FADE": "1", "ARC3_PRIORITY_TAIL_FADE_FRACTION": "0.2",
    "ARC3_PRIORITY_TAIL_LOOKUP": "remaining", "ARC3_PRIORITY_SCORE_NORMALIZATION": "1",
    "ARC3_HTTP_RETRY_INITIAL_SECONDS": "900",
}


def duck_env(base_url: str, model_id: str, max_active_streams: int) -> dict[str, str]:
    env = dict(DUCK_ENV)
    env.update({"LOCAL_ANALYZER_BASE_URL": base_url, "OPENAI_BASE_URL": base_url,
                "LOCAL_ANALYZER_MODEL_ID": model_id, "INFERENCE_ANALYZER_MODEL": model_id,
                "ARC3_MAX_ACTIVE_STREAMS": str(int(max_active_streams))})
    return env


def _import_duck():
    """Deferred import: Duck reads the environment at import time."""
    from inference.agent import tool_agent as TA
    from inference.agent import prompts as DP
    from inference.agent.python_tool_sandbox import run_sandboxed_python
    from inference.agent.runtime_state import write_runtime_state
    return TA, DP, run_sandboxed_python, write_runtime_state


log = logging.getLogger("arc2_duck")

# ------------------------------------------------------------------ grids and the trusted prelude
PRELUDE_NAMES = frozenset({"train", "test_inputs", "check", "submit", "_arc2_grid_ok"})
SUBMIT_PREFIX = "SUBMIT:"

_HELPERS_SRC = '''def _arc2_grid_ok(g):
    if not isinstance(g, (list, tuple)) or not 1 <= len(g) <= 30 or not isinstance(g[0], (list, tuple)):
        return False
    w = len(g[0])
    if not 1 <= w <= 30:
        return False
    for r in g:
        if not isinstance(r, (list, tuple)) or len(r) != w:
            return False
        for v in r:
            if type(v) is not int or not 0 <= v <= 9:
                return False
    return True
def check(transform, verbose=True):
    results = []
    for i, p in enumerate(train):
        try:
            out = transform([list(r) for r in p["input"]])
            out = [list(r) for r in out] if isinstance(out, (list, tuple)) else out
        except Exception as e:
            results.append({"pair": i, "ok": False, "error": type(e).__name__ + ": " + str(e)[:300]})
            continue
        if not _arc2_grid_ok(out):
            results.append({"pair": i, "ok": False, "error": "not a valid grid (rectangular list of lists of ints 0-9, 1..30 per side)"})
            continue
        exp = p["output"]
        if len(out) != len(exp) or len(out[0]) != len(exp[0]):
            results.append({"pair": i, "ok": False, "shape": [len(out), len(out[0])], "expected_shape": [len(exp), len(exp[0])]})
            continue
        wrong = [(r, c, out[r][c], exp[r][c]) for r in range(len(exp)) for c in range(len(exp[0])) if out[r][c] != exp[r][c]]
        results.append({"pair": i, "ok": not wrong, "wrong_cells": len(wrong), "wrong_sample_row_col_got_expected": wrong[:12]})
    if not verbose:
        return results
    print("check:", sum(1 for r in results if r["ok"]), "/", len(results), "train pairs correct")
    for r in results:
        if not r["ok"]:
            print(" ", r)
def submit(transform):
    name = getattr(transform, "__name__", "")
    if not callable(transform) or not name or name == "<lambda>":
        raise ValueError("submit(transform) needs a function defined at top level with def")
    action({"action": "''' + SUBMIT_PREFIX + '''" + name})
    print("submitted", name, "- the harness now verifies it on every train pair in a clean sandbox")
'''


def task_prelude(task: dict) -> str:
    train = [{"input": p["input"], "output": p["output"]} for p in task["train"]]
    tests = [q["input"] for q in task["test"]]
    return ("train = " + json.dumps(train, separators=(",", ":")) + "\n"
            + "test_inputs = " + json.dumps(tests, separators=(",", ":")) + "\n" + _HELPERS_SRC)


def _grid_call_code(name: str, grid: list) -> str:
    """One clean verification call: only the retained functions and ONE input grid exist in the sandbox (no train
    outputs, no other grids, no host marker). The value comes back through the sandbox's structured `result` field,
    not stdout, so print/builtins tampering cannot forge it (GPT audit 2026-10-19, reproduced and fixed)."""
    return ("_arc2_in = " + json.dumps(grid, separators=(",", ":")) + "\n"
            f"result = {name}([list(r) for r in _arc2_in])\n")


# Rev 2 (GPT audit): optional per-process address-space limit for every Duck sandbox child (tool calls and
# verification). Applied from the host with prlimit right after the child starts: the child waits for its code on
# stdin, so the limit is in place before any model code runs. No preexec_fn (unsafe with many host threads); the
# vendored sandbox file is unchanged (its module-level `subprocess` reference is swapped for a proxy).
SANDBOX_LIMIT = {"mb": 0, "applied": 0, "failed": 0, "readback_ok": None, "last_error": ""}
_SANDBOX_LIMIT_LOCK = threading.Lock()


def install_sandbox_memory_limit(mb: int) -> dict:
    import subprocess
    import types
    from inference.agent import python_tool_sandbox as PTS
    SANDBOX_LIMIT["mb"] = int(mb or 0)
    if SANDBOX_LIMIT["mb"] <= 0:
        return SANDBOX_LIMIT
    try:
        import resource
        prlimit, rlim_as = resource.prlimit, resource.RLIMIT_AS
    except (ImportError, AttributeError) as exc:
        SANDBOX_LIMIT["last_error"] = f"not available on this platform: {exc}"
        return SANDBOX_LIMIT
    lim = SANDBOX_LIMIT["mb"] * 1024 * 1024
    base = subprocess.Popen

    class _LimitedPopen(base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            try:
                prlimit(self.pid, rlim_as, (lim, lim))
                actual = tuple(prlimit(self.pid, rlim_as))      # read back every child
                if actual != (lim, lim):
                    raise RuntimeError(f"RLIMIT_AS read-back mismatch: {actual}")
                with _SANDBOX_LIMIT_LOCK:
                    SANDBOX_LIMIT["applied"] += 1
                    SANDBOX_LIMIT["readback_ok"] = True
            except Exception as exc:
                # rev 3 (GPT re-audit, reproduced): fail closed. The child has not received any model code yet
                # (that is sent on stdin after Popen returns); kill it and raise OSError, which Duck's runner turns
                # into "Sandbox process could not start." for this one call.
                with _SANDBOX_LIMIT_LOCK:
                    SANDBOX_LIMIT["failed"] += 1
                    SANDBOX_LIMIT["readback_ok"] = False
                    SANDBOX_LIMIT["last_error"] = f"{type(exc).__name__}: {exc}"[:200]
                try:
                    self.kill()
                    self.wait(timeout=5)
                except Exception:
                    pass
                for _s in (self.stdin, self.stdout, self.stderr):
                    try:
                        if _s is not None:
                            _s.close()
                    except Exception:
                        pass
                raise OSError("sandbox RLIMIT_AS setup failed") from exc

    proxy = types.ModuleType("subprocess")
    proxy.__dict__.update(subprocess.__dict__)
    proxy.Popen = _LimitedPopen
    PTS.subprocess = proxy
    return SANDBOX_LIMIT


def valid_grid(g: Any) -> bool:
    """Host-side mirror of _arc2_grid_ok (never trust the sandbox's own flag)."""
    if not isinstance(g, list) or not 1 <= len(g) <= 30 or not isinstance(g[0], list):
        return False
    w = len(g[0])
    if not 1 <= w <= 30:
        return False
    return all(isinstance(r, list) and len(r) == w and all(type(v) is int and 0 <= v <= 9 for v in r) for r in g)


def grid_rows(g: list[list[int]]) -> str:
    return "\n".join(" ".join(str(v) for v in row) for row in g)


# ------------------------------------------------------------------ prompts (Duck wording kept where not game-specific)
def arc2_system_prompt(task: dict, DP, tool_output_tokens: int) -> str:
    parts = ["You are a coding agent solving an ARC-AGI-2 grid puzzle."]
    parts.append(
        "\n\nTask:\n"
        "- Each grid is a rectangular list of rows of integers 0-9 (colors). The training pairs below define one "
        "transformation from input grid to output grid.\n"
        "- Find the single rule that maps every training input to its output, implement it as a Python function "
        "`transform(grid)` that takes and returns a list of lists of ints, and submit it. It is then applied to each "
        "test input; the test outputs are unknown to everyone.\n"
        "- Prefer the simplest rule that explains all pairs, and make it general: derive positions, sizes, colors and "
        "counts from the input. Use a fixed coordinate, color or size only when the examples show it is part of the "
        "rule. Check that the rule also fits the structure of the test inputs.\n")
    for i, p in enumerate(task["train"], 1):
        parts.append(f"\nTraining pair {i} input:\n{grid_rows(p['input'])}\nTraining pair {i} output:\n{grid_rows(p['output'])}\n")
    for i, q in enumerate(task["test"], 1):
        parts.append(f"\nTest input {i}:\n{grid_rows(q['input'])}\n")
    parts.append(
        "\n\nPython tool guidance:\n"
        "- Provided globals in every `python` call: `train` (a list of {'input': grid, 'output': grid}), "
        "`test_inputs` (a list of grids), `check(transform)` and `submit(transform)`.\n"
        "- `check(transform)` runs your function on every training input and prints how many pairs are exact and, for "
        "each failing pair, the shape mismatch or the error, or a sample of wrong cells as (row, col, got, expected). "
        "`check(transform, verbose=False)` prints nothing and returns the per-pair results as a list.\n"
        "- `submit(transform)` submits your final function. The harness re-runs it on every training pair in a clean "
        "sandbox; if all pairs are exact, its test outputs become your answer and the task ends. Otherwise you get "
        "the failures and continue. Submit only a function that `check` shows exact on every pair.\n"
        "- Work in pure Python lists: numpy is not available.\n"
        "- The only importable standard-library modules are: bisect, collections, copy, difflib, fractions, functools, "
        "heapq, itertools, json, math, operator, random, re, statistics, string.\n"
        "- Never print full grids unless they are small. Return compact derived summaries such as object lists, "
        "diffs, coordinates or counts.\n"
        "- Use `print(...)` for compact summaries, or assign a final compact object to `result`.\n"
        "- A strong default loop is: inspect the pairs in Python, form a hypothesis, write `transform`, run "
        "`check(transform)`, fix it from the reported cells, then `submit(transform)`.\n")
    persistent = DP.PERSISTENT_FUNCTIONS_LINE.format(
        lifetime="for later calls on this task",
        import_guidance="Supported explicit top-level imports used by retained functions are restored with them; "
                        "imports inside a function also work.",
    ).replace("Python builtins, provided globals such as `current_frame`, and other retained functions",
              "Python builtins and other retained functions (not the provided globals)")
    parts.append(
        "\n\nTool session rules:\n"
        "- You have exactly one tool: `python`.\n"
        f"- {DP.TOOL_CALL_FORMAT_GUIDANCE}\n"
        f"{persistent}"
        "- Functions that read `train` or `test_inputs` directly are not retained; pass the data as arguments.\n"
        "- You can call the `python` tool as many times as you want. Investigate until your code has a clear plan.\n"
        "- Each `python` tool call has a hard time limit of 30 seconds.\n"
        f"- Tool responses are capped to about {tool_output_tokens} tokens. If a response is cut off, the tool "
        "result will tell you that.\n"
        "- Keep code snippets short and purpose-built rather than dumping large frameworks into one call.\n")
    return "".join(parts)


ARC2_TOOL_DESCRIPTION = (
    "Run one Python snippet. Provided globals: `train` (list of {'input': grid, 'output': grid}), `test_inputs` "
    "(list of grids), `check(transform)` (prints and returns per-pair exactness on the training pairs) and "
    "`submit(transform)` (submits a top-level function; the harness verifies it). Grids are lists of lists of ints "
    "0-9; numpy is not available. Use `print(...)` for compact output or assign final data to `result`."
)

ARC2_NUDGE_TAIL = (
    "Call the `python` tool: write or fix `transform(grid)`, run `check(transform)` on the training pairs, and "
    "`submit(transform)` once every pair is exact. "
)
_NUDGE_GAME_TAIL_START = "Then investigate and revise your working world model"


# ------------------------------------------------------------------ the agent
def make_agent_class():
    TA, DP, run_sandboxed_python, _ = _import_duck()

    class Arc2ToolAgent(TA.ToolAgent):
        def __init__(self, *, task: dict, task_id: str, branch: int, **kwargs):
            super().__init__(**kwargs)
            self.arc2_task = task
            self.arc2_task_id = task_id
            self.arc2_branch = branch
            # Rev 1 #1: the module-level prompt builder already ran in ToolAgent.__init__; replace its result
            self._system_prompt = arc2_system_prompt(task, DP, self._tool_output_tokens)
            self._arc2_prelude = task_prelude(task)
            self._arc2_prelude_lines = self._arc2_prelude.count("\n")
            self.arc2_submissions: list[dict] = []
            self.arc2_usage: list[dict] = []   # per served request (prefix-cache measurement, step 3.5)
            self.arc2_accepted: dict | None = None
            self._arc2_pending_submit: str | None = None
            # Rev 2 (GPT audit): Duck's 900 s first-request grace is for an ARC-3 server that is still loading. Here
            # every agent is built after the SGLang /health + smoke checks, and that grace ignores the run deadline
            # (reproduced: a dead server kept a 5 s-deadline branch alive past 90 s). Duck's normal retries remain.
            self._http_initial_grace_used = True

        def _accumulate_usage_tokens(self, usage, *, count_toward_turn: bool = True) -> None:
            super()._accumulate_usage_tokens(usage, count_toward_turn=count_toward_turn)   # Duck's counting, unchanged
            if isinstance(usage, dict):
                det = usage.get("prompt_tokens_details")
                self.arc2_usage.append({"t": round(time.time(), 1), "prompt": usage.get("prompt_tokens"),
                                        "completion": usage.get("completion_tokens"),
                                        "cached": det.get("cached_tokens") if isinstance(det, dict) else None})

        # Rev 1 #5: ARC-2 tool description (same schema as Duck's)
        def _tools(self, state_path: Path) -> list[dict[str, Any]]:
            tools = super()._tools(state_path)
            tools[0]["function"]["description"] = ARC2_TOOL_DESCRIPTION
            return tools

        # the task lives in the system prompt, which is never trimmed; openers stay short
        def _build_user_prompt(self, action_num, *, valid_actions, current_frame=None, history_entries=None,
                               previous_step_summary=None) -> str:
            # Duck's milestone runs the 'full' resume mode: the opener is rebuilt after every yield (2,048 tokens).
            # Duck's opener then restates unchanged state; the ARC-2 equivalent is a continuation, never a restart.
            if getattr(self, "_resume_after_yield", False):
                return ("Nothing has been submitted since the message above, so it still holds. Your tool results "
                        "above are still valid: continue from them rather than starting over.")
            if not self.arc2_submissions:
                return ("Solve the task in the system prompt. Start with the `python` tool: inspect `train` and "
                        "`test_inputs`, then build and `check` a `transform`.")
            last = self.arc2_submissions[-1]
            return ("Your last submission (`" + last["name"] + "`) was not accepted: " + last["summary"]
                    + " Continue from your earlier work: fix the rule or the code, run `check`, and submit again.")

        def _retained_function_context(self) -> str:  # Duck's opener note, scope wording -> ARC-2
            return super()._retained_function_context().replace(" in this game: ", " on this task: ")

        # Rev 1 #8: keep Duck's generic nudge prefix, replace only its game tail
        def _append_context_message(self, messages, message) -> None:
            if (isinstance(message, dict) and message.get(TA._CONTROL_MESSAGE_KEY) == "nudge"
                    and isinstance(message.get("content"), str) and _NUDGE_GAME_TAIL_START in message["content"]):
                prefix = message["content"].split(_NUDGE_GAME_TAIL_START, 1)[0]
                message = dict(message)
                message["content"] = prefix + ARC2_NUDGE_TAIL + DP.TOOL_CALL_FORMAT_GUIDANCE
            super()._append_context_message(messages, message)

        def _fix_line_numbers(self, text: str) -> str:
            off = self._arc2_prelude_lines

            def _sub(m):
                n = int(m.group(1))
                return f'File "<python_tool>", line {n - off}' if n > off else m.group(0)
            return re.sub(r'File "<python_tool>", line (\d+)', _sub, text)

        # Rev 1 #2/#3: prelude + model code through Duck's verbatim sandbox; submit -> host re-verification
        def _run_python_tool(self, state_path: Path, arguments: dict[str, Any]):
            self._ensure_session(state_path)
            code = str(arguments.get("code", "")).rstrip()
            if not code:
                return TA._ToolDispatchResult(json.dumps({"error": "python requires a non-empty `code` string."}, indent=2))
            try:
                compile(code, "<python_tool>", "exec")
            except SyntaxError as exc:
                return TA._ToolDispatchResult(json.dumps({"error": f"Python syntax error: {exc}"}, indent=2))
            self._arc2_pending_submit = None
            empty_state = {"current_frame": None, "history": [], "valid_actions": [], "last_action_call_result": {}}

            def _handle_action(actions, *, stale_after=None):
                names = [str((a or {}).get("action", "")) for a in actions]
                subs = [n[len(SUBMIT_PREFIX):] for n in names if n.startswith(SUBMIT_PREFIX)]
                if not subs:
                    raise RuntimeError("Only submit(transform) is available in this task.")
                self._arc2_pending_submit = subs[-1]
                result = {"executed": True, "submitted": subs[-1]}
                return {"action_result": result, "state": empty_state}

            sandbox_result = run_sandboxed_python(
                code=self._arc2_prelude + code,
                timeout_seconds=self._python_timeout,
                initial_state=empty_state,
                action_handler=_handle_action,
                kept_functions=self._retained_sources(),
                retain_imports=TA._get_env_bool("ARC3_PERSISTENT_FUNCTIONS_IMPORTS", False),
                repair_hints=TA._get_env_bool("ARC3_PERSISTENT_FUNCTIONS_REPAIR_HINTS", False),
            )
            # the prelude's helpers are harness-provided, never the model's retained functions
            sandbox_result = dict(sandbox_result)
            sandbox_result["keepable_functions"] = [
                f for f in (sandbox_result.get("keepable_functions") or []) if f.get("name") not in PRELUDE_NAMES]
            sandbox_result["retention_rejected"] = [
                r for r in (sandbox_result.get("retention_rejected") or []) if r.get("name") not in PRELUDE_NAMES]
            payload: dict[str, Any] = {"tool": "python"}
            self._record_retained_functions(sandbox_result, payload)
            if "function_retention" in payload:  # Duck's scope wording ("in this game") -> ARC-2
                payload["function_retention"] = payload["function_retention"].replace(" in this game.", " on this task.")
            rendered_stdout = self._fix_line_numbers(str(sandbox_result.get("stdout", "") or ""))
            rendered_error = self._fix_line_numbers(str(sandbox_result.get("error", "") or ""))
            if rendered_error:
                payload["error"] = rendered_error
                if rendered_stdout:
                    payload["stdout"] = rendered_stdout
            else:
                payload["returncode"] = 0
                if rendered_stdout:
                    payload["stdout"] = rendered_stdout
                elif sandbox_result.get("result") is not None:
                    payload["result"] = sandbox_result.get("result")
            step_executed = False
            accepted = False
            if self._arc2_pending_submit is not None:
                verdict = self._verify_submission(self._arc2_pending_submit)
                self.arc2_submissions.append(verdict)
                payload["submission"] = verdict["summary"]
                step_executed = True
                accepted = verdict["accepted"]
                if accepted:
                    self.arc2_accepted = verdict
            return TA._ToolDispatchResult(
                self._render_tool_payload(payload, truncate_fields=("stdout", "error", "result")),
                step_executed=step_executed, made_progress=accepted)

        def _verify_submission(self, name: str) -> dict:
            """Rev 2: one clean sandbox per grid (Duck's runner, retained sources + that one input only); the value is
            read from the structured `result`; the host validates the grid and compares it with the truth. Test grids
            are always run (they feed the best-partial fallback, Rev 1 #6)."""
            v = {"name": name, "accepted": False, "train_exact": [], "test_outputs": [], "summary": "",
                 "branch": self.arc2_branch, "t": time.time(), "tokens": self.generated_tokens}
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or "") or name in PRELUDE_NAMES:
                v["summary"] = f"`{name}` is not a valid top-level function name."
                return v
            if name not in (self._kept_functions or {}):
                v["summary"] = (f"`{name}` was not retained, so it cannot be verified. Define it at top level with def, "
                                "pass any data as arguments (not `train` directly), and submit again.")
                return v

            def _refuse(actions, *, stale_after=None):
                raise RuntimeError("actions are not available during verification")
            train, test = self.arc2_task["train"], self.arc2_task["test"]
            grids = [p["input"] for p in train] + [q["input"] for q in test]
            budget_end = time.monotonic() + 2 * self._python_timeout   # all grids (one 30 s call before Rev 2)
            outs, first_err, slow = [], "", False
            for g in grids:
                left = budget_end - time.monotonic()
                if slow or left < 1:
                    outs.append(None)
                    continue
                res = run_sandboxed_python(
                    code=_grid_call_code(name, g), timeout_seconds=max(1, min(self._python_timeout, int(left))),
                    initial_state={"current_frame": None, "history": [], "valid_actions": [],
                                   "last_action_call_result": {}},
                    action_handler=_refuse, kept_functions=self._retained_sources(),
                    retain_imports=TA._get_env_bool("ARC3_PERSISTENT_FUNCTIONS_IMPORTS", False), repair_hints=False)
                err = str(res.get("error", "") or "")
                got = None if err else res.get("result")
                outs.append(got if valid_grid(got) else None)
                if err and not first_err:
                    first_err = err.strip().splitlines()[-1][:300] if err.strip() else err[:300]
                if "timed out" in err:          # a slow function: do not spend the budget on every grid
                    slow = True
            exact = [bool(valid_grid(o) and o == p["output"]) for o, p in zip(outs[:len(train)], train)]
            tests = outs[len(train):]
            v.update(train_exact=exact, test_outputs=tests)
            if first_err and not any(o is not None for o in outs):
                v["summary"] = f"verification of `{name}` failed to run: {first_err}"
                return v
            if all(exact) and all(g is not None for g in tests):
                v["accepted"] = True
                v["summary"] = f"accepted: `{name}` is exact on all {len(exact)} training pairs."
            elif all(exact):
                bad = [i for i, g in enumerate(tests) if g is None]
                v["summary"] = (f"`{name}` is exact on all training pairs but produced no valid grid for test input(s) "
                                f"{bad} (a rectangular list of lists of ints 0-9 is required).")
            else:
                bad = [i for i, e in enumerate(exact) if not e]
                v["summary"] = (f"`{name}` is wrong on training pair(s) {bad} of {len(exact)} (clean re-run). "
                                "Use check() to see the wrong cells." + (f" First error: {first_err}" if first_err else ""))
            return v

    return Arc2ToolAgent


# ------------------------------------------------------------------ driver (follows HarnessSolver.play/_play_inner)
def run_branch(agent_cls, *, task_id: str, task: dict, branch: int, queue_position: int, out_dir: Path,
               deadline: float, base_url: str, model_id: str, save_request_logs: bool,
               analyzer_timeout: float = 900.0, max_failures: int = 10, backoff: float = 10.0,
               started_at: float | None = None, max_generated_tokens: int | None = None) -> dict:
    """max_generated_tokens mirrors Duck's max_generated_tokens_per_game (None in the milestone hidden run)."""
    TA, _, _, write_runtime_state = _import_duck()
    run_dir = out_dir / f"{task_id}_b{branch}"
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / TA.RUNTIME_STATE_FILENAME
    write_runtime_state(state_path, current_frame=None, history=[])
    agent = agent_cls(task=task, task_id=task_id, branch=branch, model=model_id, timeout=analyzer_timeout,
                      base_url=base_url, provider="vllm", save_request_logs=save_request_logs,
                      dispatch_index=queue_position)
    record = {"task_id": task_id, "branch": branch, "turns": 0, "failures": 0, "note": "", "t0": time.time()}

    def should_stop() -> bool:
        if agent.arc2_accepted is not None or time.monotonic() >= deadline:
            return True
        return bool(max_generated_tokens and max_generated_tokens > 0
                    and agent.generated_tokens >= max_generated_tokens)
    gate = TA._priority_gate()
    if gate is not None:
        gate.configure_clock(started_at if started_at is not None else time.monotonic(), deadline)
        gate.acquire(TA._PRIORITY_UNTRIMMED_BASE - queue_position)
    try:
        step, retry_step, consecutive = 0, None, 0
        while not should_stop():
            if retry_step is None:
                step += 1
            analysis_step = step if retry_step is None else retry_step
            remaining = deadline - time.monotonic()
            result = agent.analyze(state_path, 0, valid_actions=[], step_env=None,
                                   transcript_path=run_dir / "transcript.txt", analysis_step=analysis_step,
                                   request_timeout_seconds=max(0.1, min(analyzer_timeout, remaining)),
                                   should_stop=should_stop)
            record["turns"] += 1
            if result is None:
                record["note"] = "analyzer returned no result"
                break
            if result.retryable_failure:
                consecutive += 1
                record["failures"] += 1
                if 0 <= max_failures < consecutive:
                    record["note"] = f"analyzer failed {consecutive} times consecutively"
                    break
                retry_step = analysis_step
                if should_stop():
                    break
                time.sleep(backoff)
                continue
            consecutive = 0
            retry_step = analysis_step if (getattr(result, "yielded_control", False) and not result.step_executed) else None
    except Exception as exc:  # one branch must never take the run down
        record["note"] = f"{type(exc).__name__}: {exc}"[:500]
        log.warning("%s b%d: %s", task_id, branch, record["note"])
    finally:
        if gate is not None:
            gate.release()
    record.update(t1=time.time(), generated_tokens=agent.generated_tokens, total_tokens=agent.total_tokens,
                  submissions=agent.arc2_submissions, accepted=agent.arc2_accepted is not None, usage=agent.arc2_usage)
    (run_dir / "branch_result.json").write_text(json.dumps(record, default=str), encoding="utf-8")
    return record


# ------------------------------------------------------------------ selection (Rev 1 #6)
def select_attempts(task: dict, branch_records: list[dict]) -> list[dict]:
    """Per test input: host-verified train-exact programs first (grouped by identical test outputs, larger group =
    agreement first), then the best partial program (most exact train pairs), then the test input as fallback."""
    subs = [s for r in branch_records for s in (r.get("submissions") or [])]
    accepted = [s for s in subs if s.get("accepted")]
    partial = sorted([s for s in subs if not s.get("accepted")],
                     key=lambda s: (sum(bool(e) for e in s.get("train_exact") or []), -s.get("t", 0)), reverse=True)
    out = []
    for qi, q in enumerate(task["test"]):
        groups: dict[str, list[dict]] = {}
        for s in accepted:
            g = (s.get("test_outputs") or [None] * (qi + 1))[qi] if qi < len(s.get("test_outputs") or []) else None
            if g is not None:
                groups.setdefault(json.dumps(g), []).append(s)
        ordered = sorted(groups.items(), key=lambda kv: (len(kv[1]), -min(s.get("t", 0) for s in kv[1])), reverse=True)
        grids = [json.loads(k) for k, _ in ordered]
        for s in partial:
            outs = s.get("test_outputs") or []
            g = outs[qi] if qi < len(outs) else None
            if g is not None and g not in grids:
                grids.append(g)
        if not grids:
            grids = [q["input"]]
        a1 = grids[0]
        # Rev 2 (GPT audit): no invented guess (the mirror is gone): next distinct candidate, else the test input
        a2 = next((g for g in grids[1:] + [q["input"]] if g != a1), a1)
        out.append({"attempt_1": a1, "attempt_2": a2})
    return out


def build_submission(tasks: dict, records: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Kaggle submission.json: every task, one {attempt_1, attempt_2} per test input (fallback = the test input)."""
    sub = {}
    for tid, task in tasks.items():
        try:
            sub[tid] = select_attempts(task, records.get(tid) or [])
        except Exception as exc:
            log.warning("selection failed for %s: %s", tid, exc)
            sub[tid] = [{"attempt_1": q["input"], "attempt_2": q["input"]} for q in task["test"]]
    return sub


def solve_all(tasks: dict, order: list[str], *, out_dir: Path, deadline: float, base_url: str, model_id: str,
              branches: int = 2, save_request_logs: bool = False, max_generated_tokens: int | None = None,
              on_branch_done: Callable[[dict], None] | None = None) -> dict[str, list[dict]]:
    """Every branch is admitted at once (Duck); the priority gate keeps ARC3_MAX_ACTIVE_STREAMS active.
    Queue positions: all first branches in task order, then all second branches (untrimmed band = queue order)."""
    agent_cls = make_agent_class()
    jobs = [(b, i, tid) for b in range(branches) for i, tid in enumerate(order)]
    records: dict[str, list[dict]] = {tid: [] for tid in order}
    lock = threading.Lock()
    started = time.monotonic()
    setup_failed = [0]

    def _one(job):
        b, i, tid = job
        try:
            rec = run_branch(agent_cls, task_id=tid, task=tasks[tid], branch=b, queue_position=b * 100000 + i,
                             out_dir=out_dir, deadline=deadline, base_url=base_url, model_id=model_id,
                             save_request_logs=save_request_logs, started_at=started,
                             max_generated_tokens=max_generated_tokens)
        except Exception as exc:  # setup failure (before the gate): never take the run down
            rec = {"task_id": tid, "branch": b, "accepted": False, "submissions": [],
                   "note": f"setup {type(exc).__name__}: {exc}"[:500]}
            log.warning("%s b%d: %s", tid, b, rec["note"])
            with lock:
                setup_failed[0] += 1          # never reached the gate: do not wait for it below
        with lock:
            records[tid].append(rec)
        if on_branch_done is not None:
            try:
                on_branch_done(rec)
            except Exception as exc:
                log.warning("on_branch_done failed: %s", exc)
        return rec

    # Duck's gate admits whoever arrives first while slots are free, so the first ARC3_MAX_ACTIVE_STREAMS
    # admissions would be a thread race. Hold every slot until all branches are queued, then release: the
    # first admissions follow the queue priority too (all first branches before any second branch).
    TA = _import_duck()[0]
    gate = TA._priority_gate()
    held = 0
    if gate is not None:
        gate.configure_clock(started, deadline)
        for _ in range(TA._max_active_streams()):
            gate.acquire(TA._PRIORITY_UNTRIMMED_BASE + 1)
            held += 1
    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
        try:
            futures = [pool.submit(_one, job) for job in jobs]
            if gate is not None:
                t_wait = time.monotonic() + 120
                while len(gate._waiting) + setup_failed[0] < len(jobs) and time.monotonic() < t_wait:
                    time.sleep(0.05)
        finally:   # Rev 2 (GPT audit): the held slots are always given back, or the pool would wait forever
            while held > 0:
                gate.release()
                held -= 1
        for f in futures:
            f.result()
    return records
