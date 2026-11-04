"""Codex-equivalent ARC attempts in one process: the ``codex_lite`` policy.

Flash-Next sees exactly what the Codex CLI harness showed it (``codex_lite_interface``): the
arc_compact_v2 system prompt, environment context, task prompt, exec_command/write_stdin schemas,
tool-output format and hook prompts. The harness levers live in this loop instead of a proxy:
a per-request token cap with truncation nudges, a reminder when no solution.py exists, a
thinking-off finalize reserve, and visible-example completion checks. Commands run as
subprocesses in a per-attempt directory shown to the model as /workspace/task, with a /tmp of their
own, under per-command resource limits and, when running as root, a private unprivileged uid; without
root, an attempt's commands share a private PID namespace instead. There is no CLI process, proxy or
chroot per attempt.
"""
from __future__ import annotations

import base64
import copy
from datetime import datetime, timezone
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import re
import resource
import secrets
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, BinaryIO, Callable, Iterator
import zlib

from codex_lite_interface import ENVIRONMENT_TEMPLATE, SYSTEM_PROMPT, TASK_PREFIX, TOOLS

# Since 2026-10-19 production attempts save a second candidate and get arc_tools.py (codex_lite_compact_v2_medium
# before); see inducer/README.md.
CODEX_LITE_POLICY = "codex_lite_compact_v2_medium_alt_tools"
# Kernel v21 (2026-10-24): the production policy whose second attempt after an unverified first continues from that
# attempt's files and tool history without its reasoning (``carry_in``/``carry_out``).
CODEX_LITE_CARRY_POLICY = CODEX_LITE_POLICY + "_carry"
# Kernel v22 (2026-10-24): carry with a -1 logit bias (acting as -1/T) on Flash-Next's most frequent deliberation
# openers, "Hmm" and "Alternatively" in both token forms. On the rehearsal's seed 1234 it cut generated tokens 4.5%
# (inducer/results/sampling_sweep_phi1_20261024_v1/, filler_words_20261024_v1/).
CODEX_LITE_NOFILLER_POLICY = CODEX_LITE_CARRY_POLICY + "_nofiller"
# Kernel v27 (2026-10-27): v22 with the per-request output cap at 12,288 instead of 24,576, so the truncation nudge
# ("write solution.py now") fires after 12k tokens of one turn's thinking. Two public-120 runs against four of v22:
# 6.9% fewer tokens, +1.1 to +2.2 under Kaggle-sized budgets (inducer/results/ud_harness_20261026_v1/plan.md).
CODEX_LITE_CAP12K_POLICY = CODEX_LITE_NOFILLER_POLICY + "_cap12k"
# Candidate (2026-10-27): v27 with a carried second attempt's first request capped at 4,096. A paired replay of carried
# attempts saved ~4% of their tokens at equal credit (inducer/results/carry_note_ab_20261027_v1/).
CODEX_LITE_CARRYCAP_POLICY = CODEX_LITE_CAP12K_POLICY + "_carrycap"
# Kaggle since v28: v27 whose task message also shows every grid as an image (``task_images``), as Franzen's
# ARC-AGI-3 milestone-2 solution shows the board at 10x. The server must load the target's vision tower (no
# language_model_only override).
CODEX_LITE_IMAGES_POLICY = CODEX_LITE_CAP12K_POLICY + "_images"
CODEX_LITE_IMAGE_POLICIES = frozenset({CODEX_LITE_IMAGES_POLICY})
# The 12,288-cap family (v27, its carrycap candidate, v28's images) keeps v22's filler bias and the request cap.
# Retired experiments (filler2, phase caps, the 8k cap, the "act" carry note, low/xhigh effort, dropped reasoning)
# are in Git history and plan.md.
CAP12K_POLICIES = (CODEX_LITE_CAP12K_POLICY, CODEX_LITE_CARRYCAP_POLICY, CODEX_LITE_IMAGES_POLICY)
CODEX_LITE_CARRY_POLICIES = frozenset({CODEX_LITE_CARRY_POLICY, CODEX_LITE_NOFILLER_POLICY, *CAP12K_POLICIES})
# Qwen3.8 tokenizer ids, checked against their text before an attempt uses them (loop_driver.policy_logit_bias).
FILLER_TOKENS = {"77264": "Hmm", "85152": " Hmm", "88842": "Alternatively", "37201": " Alternatively"}
V22_FILLER_BIAS = {"77264": -1.0, "85152": -1.0, "88842": -1.0, "37201": -1.0}
CODEX_LITE_LOGIT_BIASES = {policy: dict(V22_FILLER_BIAS) for policy in (CODEX_LITE_NOFILLER_POLICY, *CAP12K_POLICIES)}
# Per-request output caps that replace STEERING["request_cap"].
CODEX_LITE_REQUEST_CAPS = dict.fromkeys(CAP12K_POLICIES, 12_288)
# Lower caps on an attempt's first request: (fresh attempt, carried second attempt).
CODEX_LITE_FIRST_REQUEST_CAPS = {CODEX_LITE_CARRYCAP_POLICY: (12_288, 4_096)}
CODEX_LITE_EFFORTS = dict.fromkeys((CODEX_LITE_POLICY, CODEX_LITE_CARRY_POLICY, CODEX_LITE_NOFILLER_POLICY,
                                    *CAP12K_POLICIES), "medium")
CODEX_LITE_POLICIES = frozenset(CODEX_LITE_EFFORTS)
# Policies whose attempts may leave a second candidate beside their answer (``Alternates``): all of them.
CODEX_LITE_ALTERNATE_POLICIES = CODEX_LITE_POLICIES
# Policies whose sandbox holds arc_tools.py, tested grid helpers the model can import: all of them.
CODEX_LITE_TOOL_POLICIES = CODEX_LITE_POLICIES
ARC_TOOLS_SOURCE = Path(__file__).with_name("arc_tools.py")
TOOLS_TASK_NOTE = ("arc_tools.py in this directory has tested grid helpers (load, show, diff, check, compare, "
                   "write_predictions, objects, conn_diff, fragments, bbox, crop, lines, split_by_lines, ray, "
                   "background_report, dihedral, shape_rules); python3 -c 'import arc_tools; "
                   "print(arc_tools.__doc__)' lists their signatures. Import them in your scripts and solution.py "
                   "instead of re-implementing grid utilities.")
ALTERNATE_TASK_NOTE = ("If several programs or rules reproduce every training pair but give different test "
                       "outputs, keep the one that treats every object uniformly with the fewest special cases "
                       "in predictions.json, and save the strongest different candidate as "
                       "predictions_alt.json in the same format.")
# The runner's --teacher-token-cap lowers this per run; Kaggle uses 65,536.
CODEX_LITE_TOKEN_CAP = 131_072
REASONING_EFFORT = "medium"
TEMPERATURE = 0.7
TOP_P = 0.95
STEERING = {"request_cap": 24_576, "reminder_after": 32_768, "finalize_reserve": 8_192,
            "truncation_continuations": 6}
# Requests stop this far short of the context window: SGLang counted 4 tokens beyond the rendered prompt and rejected
# a carried attempt's request at exactly the window (139,268 > 139,264), losing the attempt's outputs.
CONTEXT_MARGIN = 256
COMPLETION_CONTINUATIONS = 3
VIRTUAL_ROOT = "/workspace/task"
# Literal /tmp as a path of its own (not ~/tmp, ./tmp, /var/tmp or /tmpfoo); each attempt sees its own (``Workspace``).
TMP_PATH = r"(?<![\w.~/$`})\]-])/tmp(?![\w.-])"
HOOK = '<hook_prompt hook_run_id="stop:0:/home/runner/.codex/hooks.json">{}</hook_prompt>'
OUTPUT_TOKENS = 10_000
# Carry: a first attempt that ends unverified asks, thinking off, for this recap and leaves its state for the
# task's second attempt, which shows near misses (this share of pairs passed) a per-pair diff.
RECAP_REQUEST = ("Harness note: pause here. Do not call tools and do not analyze further. In at most 150 words, "
                 "recap your work so far: each transformation rule you considered, which training pairs its "
                 "program passed or failed and how, and the idea you would try next.")
RECAP_TOKENS = 512
CARRY_NEAR_MISS = 0.5
CARRY_DIFF_CELLS = 12
COMMAND_WALL_SECONDS = 900
SANDBOX_MARKER = "CODEX_LITE_SANDBOX"
# Saved work is published to the caller while the attempt runs, so a hard stop keeps it.
PROVISIONAL_REPLAY_SECONDS = 60
PROVISIONAL_REPLAY_TIMEOUT = 20
LIMITS = ((resource.RLIMIT_AS, 8 * 2**30), (resource.RLIMIT_CPU, 600),
          (resource.RLIMIT_FSIZE, 64 * 2**20), (resource.RLIMIT_NOFILE, 512))
# Private per-attempt uids under root: 1,024 per process (a 240-task run uses about 480), above `nobody`
# (65534) and within the 61k-125k range earlier Kaggle runs used.
_UIDS = itertools.count(70_000 + os.getpid() % 50 * 1024)
# Replays and carried work directories skip larger files; commands may write up to 64 MiB (LIMITS).
COPY_FILE_LIMIT = 16 * 2**20
# The task prompt promises numpy and scipy. Kaggle's runtime reaches them only through PYTHONPATH,
# so commands get the directories this interpreter imports them from.
PACKAGE_PATH = os.pathsep.join(sorted({str(Path(spec.origin).parents[1]) for spec in map(
    importlib.util.find_spec, ("numpy", "scipy")) if spec is not None and spec.origin}))
# Each input runs on its own and streams one marked line (argv[2] is the marker), so an exception or a timeout
# keeps the outputs already computed. The program's own prints go to stderr, which shares the pipe.
REPLAY = r'''
import contextlib, importlib.util, json, sys, traceback
puzzle, mark, out = json.load(sys.stdin), sys.argv[2], sys.stdout


def emit(record):
    sys.stderr.flush()
    out.write(f"\n{mark} {record}\n")
    out.flush()


with contextlib.redirect_stdout(sys.stderr):
    try:
        spec = importlib.util.spec_from_file_location("arc_solution", sys.argv[1])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except BaseException:
        module = None
        emit(json.dumps({"kind": "load", "error": traceback.format_exc()[-4000:]}))
    for kind in ("train", "test") if module is not None else ():
        for index, pair in enumerate(puzzle[kind]):
            try:
                grid = module.transform(pair["input"])
                record = json.dumps({"kind": kind, "index": index,
                                     "output": grid.tolist() if hasattr(grid, "tolist") else grid})
            except BaseException:
                record = json.dumps({"kind": kind, "index": index, "error": traceback.format_exc()[-4000:]})
            emit(record)
emit(json.dumps({"kind": "done"}))
'''


def replay_result(text: str, mark: str, puzzle: dict, stopped: str | None = None) -> dict:
    """REPLAY's streamed records as ordered outputs, None where an input raised or never ran, and what failed."""
    outputs = {"train": [None] * len(puzzle["train"]), "test": [None] * len(puzzle["test"])}
    failures, load_error, ran, done = [], None, 0, False
    for line in text.splitlines():
        if not line.startswith(mark + " "):
            continue
        try:
            record = json.loads(line[len(mark) + 1:])
        except ValueError:
            continue  # Cut off by a kill.
        kind, index = record.get("kind"), record.get("index")
        if kind == "done":
            done = True
        elif kind == "load":
            load_error = str(record.get("error"))
        elif kind in outputs and isinstance(index, int) and 0 <= index < len(outputs[kind]):
            ran += 1
            if "output" in record:
                outputs[kind][index] = record["output"]
            else:
                failures.append((kind, index, str(record.get("error"))))
    notes = [load_error.rstrip()] if load_error is not None else []
    if failures:
        notes.append(failures[0][2].rstrip() + "\nRaised on "
                     + ", ".join(f"{kind} input {index}" for kind, index, _ in failures) + ".")
    if not done and load_error is None:
        total = len(outputs["train"]) + len(outputs["test"])
        notes.append(f"{stopped or 'replay ended early'}; ran {ran} of {total} inputs.")
    result = {"train_outputs": outputs["train"], "test_outputs": outputs["test"]}
    if notes:
        result["error"] = "\n".join(notes)
    return result


# A fresh single-threaded interpreter applies the sandbox before exec; preexec_fn is unsafe in
# the threaded runner. Root keeps its user namespace so it can still drop to the attempt uid.
# Without root, a command joins its attempt's namespaces (ATTEMPT_INIT) and runs as a child in
# their PID namespace, while this interpreter waits outside and exits the way the command did.
LAUNCHER = f'''
import os, resource, sys
uid, isolate, init = int(sys.argv[1]), sys.argv[2] == "1", int(sys.argv[3])
if init > 0:
    cwd = os.getcwd()  # Joining a mount namespace moves to its root.
    os.setns(os.pidfd_open(init), os.CLONE_NEWUSER | os.CLONE_NEWNS | os.CLONE_NEWNET | os.CLONE_NEWPID)
    os.chdir(cwd)
elif isolate:
    os.unshare(os.CLONE_NEWNET if os.geteuid() == 0 else os.CLONE_NEWUSER | os.CLONE_NEWNET)
for kind, value in {[(int(kind), value) for kind, value in LIMITS]!r}:
    resource.setrlimit(kind, (value, value))
if init > 0 and (child := os.fork()):
    code = os.waitstatus_to_exitcode(os.waitpid(child, 0)[1])
    if code < 0:  # Die of the command's signal, without dumping this interpreter's core.
        import signal
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if -code != signal.SIGKILL:
            signal.signal(-code, signal.SIG_DFL)
        os.kill(os.getpid(), -code)
    os._exit(code % 256)
if uid >= 0:
    os.setgroups([])
    os.setgid(uid)
    os.setuid(uid)
os.execvp(sys.argv[4], sys.argv[4:])
'''

# Without root, all of an attempt's commands share one private user, network, mount and PID namespace
# with its own /proc, so they see and signal only the attempt's processes. On 2026-10-23 a model's
# `pkill -f solution` matched the local launcher's `--solutions` argument and killed it. The
# namespace's first process stays as its init, reaping orphans, until the runner closes the
# lifeline or kills it; the kernel then kills everything left inside.
ATTEMPT_INIT = r'''
import ctypes, os, select, signal, sys
lifeline = int(sys.argv[1])
os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWNET | os.CLONE_NEWNS | os.CLONE_NEWPID)
ready, report = os.pipe()
if child := os.fork():
    os.close(report)
    started = os.read(ready, 1) == b"1"
    if started:
        print(child, flush=True)  # Its pid outside the namespace.
    os._exit(0 if started else 1)
os.close(ready)
os.setsid()
signal.signal(signal.SIGINT, signal.SIG_DFL)  # An init ignores the signals it has no handler for.
nosuid_nodev_noexec = ctypes.c_ulong(2 | 4 | 8)
if ctypes.CDLL(None, use_errno=True).mount(b"proc", b"/proc", b"proc", nosuid_nodev_noexec, None):
    raise OSError(ctypes.get_errno(), "cannot mount /proc")
os.write(report, b"1")
null = os.open(os.devnull, os.O_RDWR)
for fd in (0, 1, 2):
    os.dup2(null, fd)
poller = select.poll()  # The lifeline keeps the runner's descriptor number, which may exceed select's limit.
poller.register(lifeline, select.POLLIN)
while not poller.poll(1000):
    try:
        while os.waitpid(-1, os.WNOHANG)[0]:
            pass
    except ChildProcessError:
        pass
'''


def sandboxed(argv: list[str], uid: int | None, isolate: bool, namespace: int | None = None) -> list[str]:
    return [sys.executable, "-I", "-S", "-c", LAUNCHER, str(-1 if uid is None else uid), str(int(isolate)),
            str(namespace or 0), *argv]


def start_attempt_namespace(environment: dict[str, str] | None = None) -> tuple[int, BinaryIO]:
    """Start an ATTEMPT_INIT; return its pid and the lifeline whose closing ends it."""
    alive, lifeline = os.pipe()
    lifeline = os.fdopen(lifeline, "wb")
    try:
        started = subprocess.run([sys.executable, "-I", "-S", "-c", ATTEMPT_INIT, str(alive)], cwd="/",
                                 env=environment, stdin=subprocess.DEVNULL, capture_output=True, timeout=30,
                                 pass_fds=(alive,))
        if started.returncode or not started.stdout.strip().isdigit():
            raise RuntimeError("codex_lite attempt namespace failed: "
                               + started.stderr.decode("utf-8", "replace")[-2000:])
    except BaseException:
        lifeline.close()
        raise
    finally:
        os.close(alive)
    return int(started.stdout), lifeline


def _probe_pid_isolation() -> bool:
    if os.geteuid() == 0:
        return False  # Root attempts get a private uid instead.
    try:
        init, lifeline = start_attempt_namespace()
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return False
    with lifeline:
        try:
            return subprocess.run(sandboxed(["true"], None, True, init), timeout=30,
                                  stderr=subprocess.DEVNULL).returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False


def _probe_network_isolation() -> bool:
    try:
        return subprocess.run(sandboxed(["true"], None, True), timeout=30,
                              stderr=subprocess.DEVNULL).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


PID_ISOLATED = _probe_pid_isolation()
# Like the Codex sandbox, commands get no network (not even the local inference server) where
# the host allows a private network namespace.
NETWORK_ISOLATED = PID_ISOLATED or _probe_network_isolation()


def live_processes() -> Iterator[tuple[int, int, int]]:
    """Yield (pid, real uid, session id) for every process that is not a zombie."""
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as stream:
                stat = stream.read()
            state, _, _, session = stat[stat.rindex(b")") + 2:].split()[:4]
            if state in (b"Z", b"X"):
                continue
            with open(f"/proc/{name}/status", "rb") as stream:
                uid = int(next(line for line in stream if line.startswith(b"Uid:")).split()[1])
        except (OSError, ValueError, StopIteration):  # It exited while being read.
            continue
        yield int(name), uid, int(session)


class BadArgument(ValueError):
    """A tool argument the model sent malformed; reported back to it like Codex does."""


def _number(args: dict, key: str, default: float, cast: type, low: float, high: float):
    """A numeric tool argument clamped to [low, high]; empty means the default."""
    try:
        value = cast(args.get(key) or default)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BadArgument(f"invalid `{key}`: {exc}") from None
    if not math.isfinite(value):
        raise BadArgument(f"invalid `{key}`: not finite")
    return min(max(value, low), high)


def valid_grid(grid: object) -> bool:
    """An ARC grid as the runner submits it: 1-30 rows and columns of integer colors 0-9."""
    return (isinstance(grid, list) and 1 <= len(grid) <= 30 and isinstance(grid[0], list) and 1 <= len(grid[0]) <= 30
            and all(isinstance(row, list) and len(row) == len(grid[0])
                    and all(type(color) is int and 0 <= color <= 9 for color in row) for row in grid))


class Session:
    """One command's process, its accumulated output, exit status and peak memory."""

    def __init__(self, process: subprocess.Popen):
        self.process, self.started = process, time.monotonic()
        self.buffer, self.read, self.lock = bytearray(), 0, threading.Lock()
        self.exit_code: int | None = None
        self.peak_rss_kb = 0
        self.done = threading.Event()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        threading.Thread(target=self._wait, daemon=True).start()

    def _read(self):
        for chunk in iter(lambda: self.process.stdout.read1(65536), b""):
            with self.lock:
                self.buffer += chunk
        self.process.stdout.close()

    def _wait(self):
        _, status, usage = os.wait4(self.process.pid, 0)
        self.exit_code = os.waitstatus_to_exitcode(status)
        self.process.returncode = self.exit_code  # Popen must not reap it again.
        self.peak_rss_kb = usage.ru_maxrss
        self.reader.join(1)  # Drain the pipe; a background child may still hold it open.
        try:
            self.process.stdin.close()
        except OSError:
            pass
        self.done.set()

    def take(self) -> bytes:
        with self.lock:
            data, self.read = bytes(self.buffer[self.read:]), len(self.buffer)
        return data


def uncopied_files(folder: str, names: list[str]) -> list[str]:
    """A ``shutil.copytree`` ignore for carried work: files above COPY_FILE_LIMIT, and sockets and FIFOs, which
    copytree cannot copy (one would fail the whole carry)."""
    skipped = []
    for name in names:
        path = Path(folder) / name
        if not (path.is_symlink() or path.is_dir()) and (not path.is_file() or path.stat().st_size > COPY_FILE_LIMIT):
            skipped.append(name)
    return skipped


class Workspace:
    """Per-attempt directory, command sessions and the /workspace/task and /tmp path views.

    Both paths are rewritten inward in commands and outward in outputs, so each attempt has its own /tmp. A private
    /tmp mount needs namespaces that Kaggle, phi1 and gpu-h200 refuse, and on the shared /tmp attempts imported
    each other's and a previous day's /tmp/solution.py, and a stray /tmp/enum.py broke standard-library imports
    (gpu-h200 and phi1 public runs, 2026-10-27/28). Under Kaggle's private uids a later attempt cannot even
    replace an earlier attempt's /tmp file, only run it.
    """

    def __init__(self, root: Path, puzzle: dict, tools: bool = False, files: Path | None = None,
                 tmp_files: Path | None = None, previous_root: str | None = None):
        self.root, self.work, self.tmp = root, root / "work", root / "tmp"
        self._inward = re.compile(re.escape(VIRTUAL_ROOT) + "|" + TMP_PATH)
        self._outward = re.compile(re.escape(str(self.work)) + "|" + re.escape(str(self.tmp)))
        for name in ("work", "home", "tmp", "bin"):
            (root / name).mkdir(parents=True)
        (self.work / "puzzle.json").write_text(json.dumps(puzzle))
        if tools:  # Copied into the work directory, so replays of solution.py can import it too.
            shutil.copyfile(ARC_TOOLS_SOURCE, self.work / ARC_TOOLS_SOURCE.name)
        if files is not None:  # A carried first attempt's work directory and /tmp.
            shutil.copytree(files, self.work, symlinks=True, dirs_exist_ok=True)
        if tmp_files is not None:
            shutil.copytree(tmp_files, self.tmp, symlinks=True, dirs_exist_ok=True)
        if previous_root is not None:  # Commands write real paths, so carried files may name the first attempt's.
            old = previous_root.encode()
            for path in (*self.work.rglob("*"), *self.tmp.rglob("*")):
                if path.is_file() and not path.is_symlink() and old in (data := path.read_bytes()):
                    try:
                        path.write_bytes(data.decode().replace(previous_root, str(root)).encode())
                    except UnicodeDecodeError:  # Binary files keep their bytes.
                        pass
        for name in ("python", "python3"):  # A symlink would lose the interpreter's venv.
            launcher = root / "bin" / name
            launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
            launcher.chmod(0o755)
        self.uid = next(_UIDS) if os.geteuid() == 0 else None
        if self.uid is not None:
            assert self.uid != os.getuid()
            for path in (root, *root.rglob("*")):
                os.lchown(path, self.uid, self.uid)
        # Without root every process shares the runner's uid; this inherited variable marks the
        # attempt's processes even after they leave the command's process group or session.
        self.marker = f"{SANDBOX_MARKER}={secrets.token_hex(16)}".encode()
        # Without root, where the host allows it, the attempt's commands share one PID namespace.
        self.namespace, self.lifeline = (start_attempt_namespace(self._environment()) if PID_ISOLATED
                                         else (None, None))
        self.sessions: dict[int, Session] = {}
        self.launched: list[Session] = []
        self.commands, self.peak_rss_kb = 0, 0

    def inward(self, text: str) -> str:
        return self._inward.sub(lambda match: str(self.work if match[0] == VIRTUAL_ROOT else self.tmp), text)

    def outward(self, text: str) -> str:
        return self._outward.sub(lambda match: VIRTUAL_ROOT if match[0] == str(self.work) else "/tmp", text)

    def _environment(self) -> dict[str, str]:
        return {"PATH": os.pathsep.join((str(self.root / "bin"), os.path.dirname(sys.executable),
                                         "/usr/local/bin", "/usr/bin", "/bin")),
                "HOME": str(self.root / "home"), "TMPDIR": str(self.tmp), "LANG": "C.UTF-8",
                "PYTHONPATH": PACKAGE_PATH, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
                "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", SANDBOX_MARKER: self.marker.split(b"=", 1)[1].decode()}

    def _spawn(self, argv: list[str], cwd: Path) -> subprocess.Popen:
        return subprocess.Popen(sandboxed(argv, self.uid, NETWORK_ISOLATED, self.namespace), cwd=cwd,
                                env=self._environment(),
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True)

    def owns(self, pid: int, uid: int) -> bool:
        """Whether a process belongs to this attempt: its private uid, or else its marker."""
        if self.uid is not None:
            return uid == self.uid
        if uid != os.getuid() or pid == os.getpid():
            return False
        try:
            with open(f"/proc/{pid}/environ", "rb") as stream:
                return self.marker in stream.read().split(b"\0")
        except OSError:
            return False

    def sweep(self, session: int | None = None) -> int:
        """SIGKILL the attempt's processes (or one command's session) until none remain.

        Process groups are not enough: ``timeout`` moves into its own group, and ``setsid`` or a
        daemonizing fork leaves the session, while the uid or marker always stays with them.
        """
        killed = 0
        for _ in range(50):
            targets = [pid for pid, uid, sid in live_processes()
                       if (session is None or sid == session) and self.owns(pid, uid)]
            if not targets:
                break
            for pid in targets:
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed += 1
                except (ProcessLookupError, PermissionError):
                    pass
            time.sleep(0.02)
        return killed

    def _kill(self, process: subprocess.Popen):
        """Stop one live command or replay, including descendants in its own process groups."""
        try:
            os.killpg(process.pid, signal.SIGKILL)  # Its leader is unreaped, so the id is still its own.
        except ProcessLookupError:
            pass
        self.sweep(session=process.pid)

    def _format(self, output: bytes, started: float, status: str, max_tokens: int) -> str:
        text = self.outward(output.decode("utf-8", "replace"))
        tokens = (len(text) + 3) // 4
        if tokens > max_tokens:
            keep = max_tokens * 2
            text = f"{text[:keep]}\n…{tokens - max_tokens} tokens truncated…\n{text[-keep:]}"
        return (f"Chunk ID: {secrets.token_hex(3)}\nWall time: {time.monotonic() - started:.4f} seconds\n"
                f"{status}\nOriginal token count: {tokens}\nOutput:\n{text}")

    def _collect(self, session: Session, number: int | None, started: float, yield_ms: float,
                 max_tokens: int) -> str:
        if session.done.wait(yield_ms / 1000):
            self.peak_rss_kb = max(self.peak_rss_kb, session.peak_rss_kb)
            if number is not None:
                self.sessions.pop(number, None)
            status = f"Process exited with code {session.exit_code}"
        else:
            if time.monotonic() - session.started > COMMAND_WALL_SECONDS:
                self._kill(session.process)
                session.done.wait(5)
            if number is None:
                number = secrets.randbelow(90_000) + 10_000
                self.sessions[number] = session
            status = f"Process running with session ID {number}"
        return self._format(session.take(), started, status, max_tokens)

    def exec_command(self, args: dict) -> str:
        if not isinstance(args.get("cmd"), str):
            return "failed to parse function arguments: missing field `cmd`"
        yield_ms = _number(args, "yield_time_ms", 10_000, float, 250, 30_000)
        max_tokens = _number(args, "max_output_tokens", OUTPUT_TOKENS, int, 0, OUTPUT_TOKENS)
        started = time.monotonic()
        workdir = Path(self.inward(str(args.get("workdir") or VIRTUAL_ROOT)))
        if not workdir.is_dir():
            return f"exec_command failed: workdir {self.outward(str(workdir))} is not a directory"
        self.commands += 1
        session = Session(self._spawn(["bash", "-c", self.inward(args["cmd"])], workdir))
        self.launched.append(session)
        return self._collect(session, None, started, yield_ms, max_tokens)

    def write_stdin(self, args: dict) -> str:
        number = args.get("session_id")
        if not isinstance(number, (int, float)) or isinstance(number, bool) or not math.isfinite(number):
            return "failed to parse function arguments: invalid type for `session_id`, expected i32"
        chars = args.get("chars") or ""
        yield_ms = _number(args, "yield_time_ms", 250 if chars else 5_000, float, 250, 30_000 if chars else 60_000)
        max_tokens = _number(args, "max_output_tokens", OUTPUT_TOKENS, int, 0, OUTPUT_TOKENS)
        session = self.sessions.get(int(number))
        if session is None:
            return f"write_stdin failed: Unknown session ID {int(number)}"
        started = time.monotonic()
        if chars and not session.done.is_set():
            try:
                session.process.stdin.write(self.inward(str(chars)).encode())
                session.process.stdin.flush()
            except (OSError, ValueError):  # The command exited and its stdin closed.
                pass
        return self._collect(session, int(number), started, yield_ms, max_tokens)

    def run_tool(self, name: str, arguments: str | dict) -> tuple[str, bool]:
        """Return the observation and whether the call itself was malformed."""
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
            if not isinstance(args, dict):
                raise ValueError("arguments must be an object")
        except (ValueError, TypeError) as exc:
            return f"failed to parse function arguments: {exc}", True
        try:
            if name == "exec_command":
                result = self.exec_command(args)
            elif name == "write_stdin":
                result = self.write_stdin(args)
            else:
                return f"unsupported call: {name}", True
        except BadArgument as exc:
            return f"failed to parse function arguments: {exc}", True
        return result, result.startswith("failed to parse function arguments")

    def replay(self, puzzle: dict, program: str = "solution.py", timeout: float = 60) -> dict:
        """Run transform(grid) on each visible input from a copy of the work directory (``replay_result``)."""
        source = self.work / program
        if source.is_symlink() or not source.is_file():
            return {"error": f"No readable {program}"}
        copy_root = Path(tempfile.mkdtemp(prefix="replay-", dir=self.tmp))
        for path in self.work.rglob("*"):
            if path.is_file() and not path.is_symlink() and path.stat().st_size <= COPY_FILE_LIMIT:
                target = copy_root / path.relative_to(self.work)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        (copy_root / "puzzle.json").write_text(json.dumps(puzzle))
        if self.uid is not None:
            for path in (copy_root, *copy_root.rglob("*")):
                os.lchown(path, self.uid, self.uid)
        mark = secrets.token_hex(8)
        process = self._spawn([sys.executable, "-c", REPLAY, program, mark], copy_root)
        try:
            out, _ = process.communicate(json.dumps(puzzle).encode(), timeout=timeout)
            stopped = None
        except subprocess.TimeoutExpired as exc:
            out = exc.output or b""  # The records that arrived in time.
            self._kill(process)
            process.wait()  # The leader is dead; an escaped child may still hold the pipes, so don't read them.
            for stream in (process.stdin, process.stdout):
                stream.close()
            stopped = f"replay exceeded {timeout:g} seconds"
        finally:
            shutil.rmtree(copy_root, ignore_errors=True)
        return replay_result(out.decode("utf-8", "replace"), mark, puzzle, stopped)

    def close(self):
        # Like the end of a Codex sandbox, this also stops background jobs of finished commands.
        for session in self.launched:
            if not session.done.is_set():
                self._kill(session.process)
        self.sweep()  # Without root this includes the namespace init, which takes the rest with it.
        if self.lifeline is not None:
            self.lifeline.close()
        self.sessions.clear()


class Provisional:
    """The answer saved so far, published before the attempt ends.

    Each changed solution.py is replayed (rate-limited), also once a predictions.json exists. As in the final
    selection, the latest valid predictions.json wins unless the current solution.py passes every visible training
    pair and disagrees with it; without one, the replay passing the most training pairs is kept.
    """

    def __init__(self, workspace: Workspace, puzzle: dict, on_outputs: Callable[[list], None] | None,
                 stats: dict):
        self.workspace, self.puzzle, self.on_outputs, self.stats = workspace, puzzle, on_outputs, stats
        self.outputs = self.source = None
        self.saved: list | None = None  # The latest valid predictions.json.
        self.best: tuple[int, list] | None = None  # Training pairs passed and outputs of the best replay so far.
        self.verified: tuple[tuple, list] | None = None  # The last replay's solution.py, if it passed every pair.
        self.signatures: dict[str, tuple | None] = {}
        self.last_replay = float("-inf")
        stats["provisional_updates"] = 0

    def _signature(self, name: str) -> tuple | None:
        path = self.workspace.work / name
        if path.is_symlink() or not path.is_file():
            return None
        info = path.stat()
        return info.st_mtime_ns, info.st_size

    def _valid(self, outputs: object) -> bool:
        return (isinstance(outputs, list) and len(outputs) == len(self.puzzle["test"])
                and all(valid_grid(grid) for grid in outputs))

    def _publish(self, outputs: list, source: str):
        self.source = source
        if outputs == self.outputs:
            return
        self.outputs = outputs
        self.stats["provisional_updates"] += 1
        if self.on_outputs is not None:
            try:
                self.on_outputs(copy.deepcopy(outputs))
            except Exception as exc:  # Publishing must never end the attempt.
                self.stats["provisional_callback_error"] = repr(exc)

    def refresh(self):
        signature = self._signature("predictions.json")
        if signature is not None and signature != self.signatures.get("predictions.json"):
            self.signatures["predictions.json"] = signature
            try:
                outputs = json.loads((self.workspace.work / "predictions.json").read_text())
            except (OSError, ValueError):
                outputs = None
            if self._valid(outputs):
                self.saved = outputs
        program = self._signature("solution.py")
        if (program is not None and program != self.signatures.get("solution.py")
                and time.monotonic() - self.last_replay >= PROVISIONAL_REPLAY_SECONDS):
            self.signatures["solution.py"], self.last_replay = program, time.monotonic()
            replay = self.workspace.replay(self.puzzle, timeout=PROVISIONAL_REPLAY_TIMEOUT)
            outputs, train = replay.get("test_outputs"), replay.get("train_outputs") or []
            passed = sum(i < len(train) and train[i] == pair["output"] for i, pair in enumerate(self.puzzle["train"]))
            self.verified = None
            if self._valid(outputs):
                if self.best is None or passed >= self.best[0]:
                    self.best = passed, outputs
                if passed == len(self.puzzle["train"]):
                    self.verified = program, outputs
        verified = self.verified[1] if self.verified is not None and self.verified[0] == program else None
        if verified is not None and verified != self.saved:
            self._publish(verified, "replay")
        elif self.saved is not None:
            self._publish(self.saved, "predictions.json")
        elif self.best is not None:
            self._publish(self.best[1], "replay")


class Alternates:
    """A second candidate per test beside the attempt's answer; the runner fills attempt 2 with it.

    The task prompt asks for ``predictions_alt.json`` when training-fitting rules disagree on a test.
    In the thinking-off finalize window every changed solution.py is replayed, so a passing answer
    the attempt later replaced is kept too and takes precedence. On the public tasks (2026-10-19)
    9 of 91 alternates were right where the answer was wrong, 8 of them from predictions_alt.json.
    """

    def __init__(self, workspace: Workspace, puzzle: dict):
        self.workspace, self.puzzle = workspace, puzzle
        self.passing: list[list] = []
        self.signature: tuple | None = None

    def finalize_turn(self):
        """Replay a changed solution.py and keep its test outputs if it passes every training pair."""
        path = self.workspace.work / "solution.py"
        if path.is_symlink() or not path.is_file():
            return
        info = path.stat()
        if (info.st_mtime_ns, info.st_size) == self.signature:
            return
        self.signature = info.st_mtime_ns, info.st_size
        replay = self.workspace.replay(self.puzzle, timeout=PROVISIONAL_REPLAY_TIMEOUT)
        train, tests = replay.get("train_outputs") or [], replay.get("test_outputs")
        if (len(train) == len(self.puzzle["train"])
                and all(output == pair["output"] for output, pair in zip(train, self.puzzle["train"]))
                and isinstance(tests, list) and len(tests) == len(self.puzzle["test"])
                and all(valid_grid(grid) for grid in tests)):
            self.passing.append(tests)

    def collect(self, outputs: list | None) -> tuple[list, list]:
        """Per test: a candidate that differs from the final output, and where it came from."""
        count = len(self.puzzle["test"])
        alternates, sources = [None] * count, [None] * count
        if outputs is None:
            return alternates, sources
        declared = None
        path = self.workspace.work / "predictions_alt.json"
        if path.is_file() and not path.is_symlink():
            try:
                declared = json.loads(path.read_text())
            except (OSError, ValueError):
                declared = None
        if not (isinstance(declared, list) and len(declared) == count):
            declared = [None] * count
        for index in range(count):
            for snapshot in reversed(self.passing):
                if snapshot[index] != outputs[index]:
                    alternates[index], sources[index] = snapshot[index], "passing_history"
                    break
            else:
                if valid_grid(declared[index]) and declared[index] != outputs[index]:
                    alternates[index], sources[index] = declared[index], "predictions_alt.json"
        return alternates, sources


def inspect_completion(workspace: Workspace, puzzle: dict) -> dict:
    """The Codex Stop-hook check: saved files and visible training examples only."""
    issues, outputs = [], None
    for name in ("solution.py", "predictions.json"):
        path = workspace.work / name
        if path.is_symlink() or not path.is_file():
            issues.append(f"{name} is missing or unreadable (FileNotFoundError).")
        elif name == "predictions.json":
            try:
                outputs = json.loads(path.read_text())
            except (OSError, ValueError) as exc:
                issues.append(f"{name} is missing or unreadable ({type(exc).__name__}).")
    structure_valid = (isinstance(outputs, list) and len(outputs) == len(puzzle["test"])
                       and all(valid_grid(grid) for grid in outputs))
    if not structure_valid:
        issues.append(f"predictions.json must contain exactly {len(puzzle['test'])} ordered test grids; "
                      "each must be a nonempty rectangular list of integer colors 0 through 9.")
    replay = workspace.replay(puzzle)
    train_outputs = replay.get("train_outputs", [])
    passed = [i for i, pair in enumerate(puzzle["train"])
              if i < len(train_outputs) and valid_grid(train_outputs[i]) and train_outputs[i] == pair["output"]]
    if replay.get("error"):
        issues.append("solution.py replay failed: " + workspace.outward(replay["error"])[-1600:])
    if len(passed) != len(puzzle["train"]):
        failed = [i for i in range(len(puzzle["train"])) if i not in passed]
        issues.append(f"transform(grid) failed visible training examples {failed} (zero-based).")
    if structure_valid and replay.get("test_outputs") != outputs:
        issues.append("Saved predictions differ from transform(grid) on the ordered test inputs.")
    return {"passed": not issues, "issues": issues, "outputs": outputs if structure_valid else None,
            "replay_test_outputs": replay.get("test_outputs"), "train_passed": passed,
            "n_train": len(puzzle["train"])}


def request_limits(used: int, cap: int, request_cap: int = STEERING["request_cap"]) -> tuple[int, bool]:
    """This request's output limit, and whether only the thinking-off reserve remains."""
    remaining, reserve = cap - used, STEERING["finalize_reserve"]
    if remaining <= reserve + 4096:
        return remaining, True
    return min(remaining - reserve, request_cap), False


def carry_path(label: str) -> Path:
    """Where a carry policy's first attempt leaves its state for the task's second attempt (this process)."""
    return (Path(os.environ.get("ARC_CODEX_LITE_ROOT") or tempfile.gettempdir())
            / "codex-lite-carry" / str(os.getpid()) / label)


def request_recap(client: Any, model: str, renderer, messages: list[dict], seed: int, *, stats: dict, trace: list,
                  timeout: float, context_window: int, reasoning_effort: str = REASONING_EFFORT,
                  temperature: float = TEMPERATURE, top_p: float = TOP_P,
                  logit_bias: dict[str, float] | None = None) -> str:
    """Ask, thinking off, for a short recap of the work in ``messages``; empty when no room is left.

    Tools stay offered: ``tool_choice="none"`` makes SGLang drop them from the prompt, which breaks the
    prompt-ID contract and the cached prefix. A tool call in the reply is not executed; the trace keeps it
    under ``unexecuted_tool_calls``. The trace turn keeps its own request, which later prompts do not extend.
    """
    from loop_driver import (PromptContractError, _qwen_extra_body, _response_token_ids, _tool_call_dicts,
                             _usage_tokens, _verify_prompt_ids)

    request = [*messages, {"role": "user", "content": HOOK.format(RECAP_REQUEST)}]
    local_ids = renderer.render(request, tools=TOOLS,
                                reasoning_effort=reasoning_effort, enable_thinking=False)
    limit = min(RECAP_TOKENS, context_window - len(local_ids) - CONTEXT_MARGIN)
    if limit <= 0:
        return ""
    started = time.time()
    response = client.chat.completions.create(
        model=model, messages=request, tools=TOOLS, tool_choice="auto", temperature=temperature, top_p=top_p,
        presence_penalty=0.0, seed=seed, max_tokens=limit, timeout=timeout, stream=False,
        **({"logit_bias": logit_bias} if logit_bias else {}),
        extra_body={**_qwen_extra_body(enable_thinking=False, reasoning_effort=reasoning_effort),
                    "return_prompt_token_ids": True})
    prompt_tokens, reported, accounted, accounting = _usage_tokens(response, limit)
    identity = _verify_prompt_ids(local_ids, _response_token_ids(response, ("prompt_token_ids", "input_token_ids")),
                                  getattr(renderer, "image_token_id", None))
    if isinstance(prompt_tokens, int) and prompt_tokens != len(local_ids):
        raise PromptContractError(f"usage prompt_tokens={prompt_tokens} but rendered {len(local_ids)}")
    choice = response.choices[0]
    text = (getattr(choice.message, "content", None) or "").strip()
    returned = _tool_call_dicts(choice.message) if getattr(choice.message, "tool_calls", None) else []
    stats["calls"] = stats.get("calls", 0) + 1
    stats["gen_tokens"] = stats.get("gen_tokens", 0) + accounted
    stats["recap_tokens"] = stats.get("recap_tokens", 0) + accounted
    trace.append({"step": len(trace) + 1, "phase": "carry_recap", "request_message_count": len(request),
                  "request_max_tokens": limit, "thinking": False, "prompt_token_identity": identity,
                  "reasoning": "", "content": text, "finish_reason": choice.finish_reason,
                  "prompt_tokens": prompt_tokens, "completion_tokens": reported, "accounted_completion_tokens": accounted,
                  "accounting_source": accounting, "tool_calls": [], "tool_results": [],
                  **({"unexecuted_tool_calls": returned, "unexecuted_reason": "recap"} if returned else {}),
                  "request_wall": round(time.time() - started, 3), "request_messages": copy.deepcopy(request),
                  "request_tools": copy.deepcopy(TOOLS)})
    return text


def carried_history(messages: list[dict]) -> list[dict]:
    """An earlier attempt's task prompt plus every tool call and output, without reasoning, replies or notes."""
    kept = [m for m in messages[3:] if m["role"] == "tool" or (m["role"] == "assistant" and m.get("tool_calls"))]
    return copy.deepcopy(messages[:3]) + [{k: v for k, v in m.items() if k != "reasoning_content"} for m in kept]


# The usual ARC palette for colors 0-9; cells are IMAGE_CELL pixels with a one-pixel dark line on their far edges.
ARC_COLORS = ((0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64), (255, 220, 0), (170, 170, 170),
              (240, 18, 190), (255, 133, 27), (127, 219, 255), (135, 12, 37))
GRID_LINE = (85, 85, 85)
IMAGE_CELL = 10
IMAGE_NOTE = ("\n\nThe same grids as images ({cell} pixels per cell; colors 0-9 are black, blue, red, green, yellow, "
              "grey, magenta, orange, azure and maroon):")


def grid_png(grid: list, cell: int = IMAGE_CELL) -> bytes:
    """An ARC grid as an RGB PNG (no imaging library: the harness also runs outside the SGLang environment)."""
    line = bytes(GRID_LINE)
    rows = []
    for row in grid:
        pixels = b"".join(bytes(ARC_COLORS[value]) * (cell - 1) + line for value in row)
        rows += [b"\0" + pixels] * (cell - 1) + [b"\0" + line * (len(row) * cell)]

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", len(grid[0]) * cell, len(grid) * cell, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"".join(rows), 9))
            + chunk(b"IEND", b""))


def task_images(train: list[dict], test_inputs: list) -> list[dict]:
    """Chat content parts showing a task's grids as images, labelled like ``render_user_prompt``."""
    def image(grid):
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64,"
                                                          + base64.b64encode(grid_png(grid)).decode()}}

    parts = [{"type": "text", "text": IMAGE_NOTE.format(cell=IMAGE_CELL)}]
    for index, pair in enumerate(train, 1):
        parts += [{"type": "text", "text": f"\nTrain Example {index} input:"}, image(pair["input"]),
                  {"type": "text", "text": f"\nTrain Example {index} output:"}, image(pair["output"])]
    for index, grid in enumerate(test_inputs, 1):
        label = "Test Input" if len(test_inputs) == 1 else f"Test Input {index}"
        parts += [{"type": "text", "text": f"\n{label}:"}, image(grid)]
    return parts


def grid_diff(got: list, expected: list) -> str:
    """``arc_tools.diff``'s summary of a program output against the expected grid."""
    if (len(got), len(got[0])) != (len(expected), len(expected[0])):
        return f"shape {len(got)}x{len(got[0])} vs {len(expected)}x{len(expected[0])}"
    cells = [(r, c, got[r][c], expected[r][c]) for r in range(len(got)) for c in range(len(got[0]))
             if got[r][c] != expected[r][c]]
    if not cells:
        return "ok"
    shown = ", ".join(f"({r},{c}) {a}->{b}" for r, c, a, b in cells[:CARRY_DIFF_CELLS])
    return f"{len(cells)} cells differ: {shown}" + (" ..." if len(cells) > CARRY_DIFF_CELLS else "")


def carry_note(state: dict, workspace: Workspace, puzzle: dict, cap: int) -> dict:
    """The harness note opening a carried second attempt; near misses get the saved program's per-pair diff."""
    passed = state["train_passed"]
    failed = [i for i in range(len(puzzle["train"])) if i not in passed]
    ended = "ran out of its token budget" if state["end_reason"] == "completion_token_budget" else "ended"
    if failed:
        result = f"with solution.py failing training pairs {failed} (zero-based)"
    else:  # Unverified only because a test input got no valid output.
        result = ("with solution.py passing every training pair but giving no valid output for test inputs "
                  f"{state.get('tests_without_output', [])} (zero-based)")
    text = ("This is a second attempt at the same puzzle. The workspace keeps every file of the first attempt, and "
            "the tool calls and outputs above are that attempt's work; its reasoning was removed. The first attempt "
            f"{ended} {result}. Its own recap:\n{state['recap']}\n\n")
    if len(passed) >= CARRY_NEAR_MISS * len(puzzle["train"]):
        replay = workspace.replay(puzzle)
        outputs = replay.get("train_outputs") or []
        lines = [f"train {i}: " + (grid_diff(outputs[i], pair["output"]) if i < len(outputs) and valid_grid(outputs[i])
                                   else "no valid output grid") for i, pair in enumerate(puzzle["train"])]
        if replay.get("error"):
            lines.append("replay error: " + workspace.outward(replay["error"])[-600:])
        text += "Training check of the saved solution.py:\n" + "\n".join(lines) + "\n\n"
    text += ("These failures may come from a bug or from a misread rule. Decide whether to repair solution.py or "
             "replace its rule, then write and validate solution.py and predictions.json. "
             f"You have {cap} generated tokens.")
    return {"role": "user", "content": HOOK.format(text)}


def run_codex_lite_episode(
    client: Any, model: str, train: list[dict], test_inputs: list, deadline: float, seed: int, *,
    renderer, stats: dict, trace: list, context_window: int, request_timeout: float,
    completion_token_cap: int | None = None, on_outputs: Callable[[list], None] | None = None,
    reasoning_effort: str = REASONING_EFFORT, alternates: bool = False, tools: bool = False,
    carry_in: Path | None = None, carry_out: Path | None = None,
    temperature: float = TEMPERATURE, top_p: float = TOP_P, logit_bias: dict[str, float] | None = None,
    request_cap: int | None = None, first_request_caps: tuple[int, int] | None = None,
    images: bool = False,
) -> list | None:
    """One codex_lite attempt.

    Carry policies only: an attempt that ends unverified saves its messages, work directory, /tmp and a thinking-off
    recap to ``carry_out``; an attempt given a ``carry_in`` holding that state starts from the saved files and
    tool history (without reasoning) under ``carry_note``, and deletes the state.

    ``images`` (Kaggle since v28): the task message also shows every grid as an image (``task_images``).
    """
    from loop_driver import (PromptContractError, _get_extra, _qwen_extra_body, _response_token_ids,
                             _tool_call_dicts, _usage_tokens, _verify_prompt_ids, render_user_prompt)

    cap = completion_token_cap or CODEX_LITE_TOKEN_CAP
    if not 0 < cap <= CODEX_LITE_TOKEN_CAP:
        raise ValueError(f"codex_lite token cap may only lower {CODEX_LITE_TOKEN_CAP}")
    if reasoning_effort not in CODEX_LITE_EFFORTS.values():
        raise ValueError(f"codex_lite reasoning effort must be one of {sorted(CODEX_LITE_EFFORTS.values())}")
    steering = {**STEERING, "request_cap": request_cap or STEERING["request_cap"]}
    if first_request_caps:
        steering["first_request_caps"] = list(first_request_caps)  # [fresh attempt, carried attempt]
    if images:
        steering["task_images"] = True
    puzzle = {"train": train, "test": [{"input": grid} for grid in test_inputs]}
    date = datetime.now(timezone.utc).date().isoformat()
    task = (TASK_PREFIX + (" " + ALTERNATE_TASK_NOTE if alternates else "") + (" " + TOOLS_TASK_NOTE if tools else "")
            + "\n\n" + render_user_prompt(train, test_inputs, ("exec_command", "write_stdin")))
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ENVIRONMENT_TEMPLATE.format(date=date)},
                {"role": "user", "content": ([{"type": "text", "text": task}, *task_images(train, test_inputs)]
                                             if images else task)}]
    stats.update(completion_token_budget=cap, completion_tokens_used=0, reasoning_effort=reasoning_effort,
                 temperature=temperature, top_p=top_p, steering=steering, notes=[], completion_checks=[],
                 truncation_nudges=0, completion_continuations=0, failed_tool_calls=0)
    if logit_bias:
        stats["logit_bias"] = dict(logit_bias)
    root = Path(tempfile.mkdtemp(prefix="codex-lite-", dir=os.environ.get("ARC_CODEX_LITE_ROOT")))
    workspace = None
    if carry_in is not None and (carry_in / "state.json").is_file():
        try:
            state = json.loads((carry_in / "state.json").read_text())
            workspace = Workspace(root, puzzle, tools=tools, files=carry_in / "work",
                                  tmp_files=carry_in / "tmp" if (carry_in / "tmp").is_dir() else None,
                                  previous_root=state.get("root"))
            note = carry_note(state, workspace, puzzle, cap)
            stats["carried"] = {key: state[key] for key in ("train_passed", "end_reason", "recap_tokens")}
            messages = carried_history(state["messages"]) + [note]
            stats["carried"]["context_messages"] = len(messages)
        except Exception as exc:  # noqa: BLE001 - a broken carry starts the attempt fresh instead of losing it.
            stats.pop("carried", None)
            stats["carry_in_error"] = repr(exc)
            if workspace is not None:
                workspace.close()
            shutil.rmtree(root, ignore_errors=True)
            root = Path(tempfile.mkdtemp(prefix="codex-lite-", dir=os.environ.get("ARC_CODEX_LITE_ROOT")))
            workspace = None
        finally:
            shutil.rmtree(carry_in, ignore_errors=True)
    if workspace is None:
        workspace = Workspace(root, puzzle, tools=tools)
    provisional = Provisional(workspace, puzzle, on_outputs, stats)
    second = Alternates(workspace, puzzle) if alternates else None
    snapshots: list[int] = []
    try:
        while True:
            used = stats["completion_tokens_used"]
            if used >= cap:
                stats["end_reason"] = "completion_token_budget"
                break
            if time.time() >= deadline:
                stats["end_reason"] = "deadline"
                break
            limit, finalizing = request_limits(used, cap, steering["request_cap"])
            if first_request_caps and not snapshots and not finalizing:
                limit = min(limit, first_request_caps[1 if "carried" in stats else 0])
            kinds = set(stats["notes"])
            note = None
            if finalizing and "finalize" not in kinds:
                note = "finalize", (
                    f"Harness note: only {cap - used} generated tokens remain. Do not analyze further. "
                    "Immediately write solution.py with your best current transform(grid) and "
                    "predictions.json for every test input, then check both files.")
            elif (not finalizing and used >= STEERING["reminder_after"] and "reminder" not in kinds
                  and not (workspace.work / "solution.py").exists()):
                note = "reminder", (
                    f"Harness note: {used} of {cap} generated tokens are spent and solution.py does not exist "
                    "yet. Stop analyzing in prose. Write solution.py with transform(grid) for your current best "
                    "hypothesis now, run it on every training pair, and refine it from the mismatches.")
            if note:
                stats["notes"].append(note[0])
                messages.append({"role": "user", "content": note[1]})
            if finalizing and second is not None:
                second.finalize_turn()  # The answer as finalize starts, or after a finalize edit.
            local_ids = renderer.render(messages, tools=TOOLS,
                                        reasoning_effort=reasoning_effort, enable_thinking=not finalizing)
            limit = min(limit, context_window - len(local_ids) - CONTEXT_MARGIN)
            if limit <= 0:
                stats["end_reason"] = "context_guard"
                break
            extra_body = {**_qwen_extra_body(enable_thinking=not finalizing, reasoning_effort=reasoning_effort),
                          "return_prompt_token_ids": True}
            started = time.time()
            response = client.chat.completions.create(
                model=model, messages=messages, tools=TOOLS, tool_choice="auto", parallel_tool_calls=True,
                temperature=temperature, top_p=top_p, presence_penalty=0.0, seed=seed, max_tokens=limit,
                timeout=min(request_timeout, max(0.001, deadline - time.time())), stream=False,
                extra_body=extra_body, **({"logit_bias": logit_bias} if logit_bias else {}))
            prompt_tokens, reported, accounted, accounting = _usage_tokens(response, limit)
            identity = _verify_prompt_ids(local_ids, _response_token_ids(response, ("prompt_token_ids",
                                                                                   "input_token_ids")),
                                          getattr(renderer, "image_token_id", None))
            if isinstance(prompt_tokens, int) and prompt_tokens != len(local_ids):
                raise PromptContractError(f"usage prompt_tokens={prompt_tokens} but rendered {len(local_ids)}")
            choice = response.choices[0]
            message = choice.message
            # Native calls only: Codex never recovers calls written as text.
            returned = _tool_call_dicts(message) if getattr(message, "tool_calls", None) else []
            truncated = choice.finish_reason == "length"
            calls = [] if truncated else returned  # Like Codex's truncation gate: a call cut short is never executed.
            stats["calls"] += 1
            stats["gen_tokens"] += accounted
            stats["completion_tokens_used"] += accounted
            snapshots.append(len(messages))
            assistant = {"role": "assistant", "content": getattr(message, "content", None) or ""}
            reasoning = _get_extra(message, "reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                assistant["reasoning_content"] = reasoning
            if calls:
                assistant["tool_calls"] = copy.deepcopy(calls)
            messages.append(assistant)
            turn = {"step": len(snapshots), "phase": "codex_lite", "request_message_count": snapshots[-1],
                    "request_max_tokens": limit, "thinking": not finalizing, "prompt_token_identity": identity,
                    "reasoning": assistant.get("reasoning_content", ""), "content": assistant["content"],
                    "finish_reason": choice.finish_reason, "prompt_tokens": prompt_tokens,
                    "completion_tokens": reported, "accounted_completion_tokens": accounted,
                    "accounting_source": accounting, "tool_calls": calls, "tool_results": [], "request_wall": round(time.time() - started, 3)}
            if truncated and returned:  # The trace keeps what the conversation omits.
                turn.update(unexecuted_tool_calls=copy.deepcopy(returned), unexecuted_reason="length")
            trace.append(turn)
            if calls:
                for call in calls:
                    stats["tool_calls"] += 1
                    observation, malformed = workspace.run_tool(call["function"]["name"],
                                                                call["function"].get("arguments") or "{}")
                    stats["failed_tool_calls"] += malformed
                    turn["tool_results"].append({"tool": call["function"]["name"], "tool_call_id": call["id"],
                                                 "response": observation})
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": observation})
                provisional.refresh()
                continue
            # A reply without a tool call is Codex's Stop event: check the saved files.
            check = inspect_completion(workspace, puzzle)
            stats["completion_checks"].append({k: check[k] for k in ("passed", "issues", "train_passed", "n_train")})
            remaining = cap - stats["completion_tokens_used"]
            if check["passed"]:
                stats["end_reason"] = "complete"
                break
            if remaining <= 0:
                stats["end_reason"] = "completion_token_budget"
                break
            if truncated and stats["truncation_nudges"] < STEERING["truncation_continuations"]:
                stats["truncation_nudges"] += 1
                text = (f"Your reasoning reached the {limit}-token limit for one response and was cut off. "
                        "Stop analyzing in prose. With exec_command, write solution.py now with your current "
                        "best transform(grid), run it on every training pair, and refine it from the mismatches; "
                        f"then write and validate predictions.json. You have at most {remaining} generated "
                        "tokens left in the original budget.")
            elif stats["completion_continuations"] >= COMPLETION_CONTINUATIONS:
                stats["end_reason"] = "continuation_limit"
                break
            else:
                stats["completion_continuations"] += 1
                text = (f"Completion check {stats['completion_continuations']}/{COMPLETION_CONTINUATIONS} failed. "
                        "This is visible-example and saved-file feedback only; no hidden answers were used.\n"
                        + "\n".join("- " + issue for issue in check["issues"])
                        + "\nContinue using exec_command and write_stdin. Repair the program, run visible "
                        "training checks, and write and validate predictions.json before finishing. "
                        f"You have at most {remaining} generated tokens left in the original budget. "
                        "If uncertain, save your best candidate now.")
            messages.append({"role": "user", "content": HOOK.format(text)})
        final = inspect_completion(workspace, puzzle)
        outputs = final["outputs"]
        source = "predictions.json" if outputs else None
        replayed = final["replay_test_outputs"]
        fits = len(final["train_passed"]) == len(train)
        if (isinstance(replayed, list) and len(replayed) == len(test_inputs) and all(map(valid_grid, replayed))
                and (outputs is None or (fits and replayed != outputs))):
            # Also over a saved file that disagrees with a program passing every visible training pair.
            outputs, source = replayed, "replay"
        if outputs is None and provisional.outputs is not None:
            # A later edit broke the files; keep the last good saved answer.
            outputs, source = provisional.outputs, "provisional_" + provisional.source
        if outputs is None and isinstance(replayed, list) and any(map(valid_grid, replayed)):
            # The test outputs the final program computes, None where it raised.
            outputs, source = [grid if valid_grid(grid) else None for grid in replayed], "replay_partial"
        missing = [i for i, grid in enumerate(outputs or [None] * len(test_inputs)) if grid is None]
        stats.update(outputs_source=source, final_check={k: final[k] for k in ("passed", "issues", "train_passed")},
                     commands=workspace.commands, peak_command_rss_kb=workspace.peak_rss_kb,
                     network_isolated=NETWORK_ISOLATED, pid_isolated=PID_ISOLATED,
                     best_n_passed=len(final["train_passed"]))
        if second is not None:
            # loop_driver.solve_puzzle offers these as the attempt's second submission candidates.
            stats["alt_outputs"], stats["alt_sources"] = second.collect(outputs)
        # Verified (kaggle_runner.primary_verified): every training pair passes and every test input has an output.
        if carry_out is not None and not (fits and not missing) and time.time() < deadline:
            try:
                recap = request_recap(client, model, renderer, messages, seed, stats=stats, trace=trace,
                                      timeout=min(request_timeout, max(0.001, deadline - time.time())),
                                      context_window=context_window, reasoning_effort=reasoning_effort,
                                      temperature=temperature, top_p=top_p, logit_bias=logit_bias)
                carry_out.mkdir(parents=True, exist_ok=True)
                shutil.copytree(workspace.work, carry_out / "work", symlinks=True, ignore=uncopied_files)
                shutil.copytree(workspace.tmp, carry_out / "tmp", symlinks=True, ignore=uncopied_files)
                (carry_out / "state.json").write_text(json.dumps({
                    "messages": messages, "recap": recap, "train_passed": final["train_passed"],
                    "tests_without_output": missing, "root": str(workspace.root),
                    "end_reason": stats.get("end_reason"), "recap_tokens": stats.get("recap_tokens", 0)}))
            except Exception as exc:  # noqa: BLE001 - the answer stands; the second attempt starts fresh.
                stats["carry_error"] = repr(exc)
                shutil.rmtree(carry_out, ignore_errors=True)
        return outputs
    finally:
        workspace.close()
        last = next((turn for turn in reversed(trace) if turn["phase"] == "codex_lite"), None)
        if last is not None:
            # The trace store projects the final request snapshot plus its response and results.
            # A carry recap turn keeps its own request.
            last["request_messages"] = copy.deepcopy(messages[:snapshots[-1]])
            last["request_tools"] = copy.deepcopy(TOOLS)
        if not os.environ.get("ARC_CODEX_LITE_KEEP"):
            shutil.rmtree(root, ignore_errors=True)
