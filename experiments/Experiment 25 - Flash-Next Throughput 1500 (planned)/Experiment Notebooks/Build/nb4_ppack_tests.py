"""CPU tests for nb4_ppack.py (E25 NB4 Q-pack / Q-defer). Quick: about 10-30 s, no GPU, no vllm install needed.

  python nb4_ppack_tests.py [--merlin-src D:/ARC AGI 2/_scratch/e23/p2/merlin-src]

T1 patch: applies to copies of the real Merlin ad42e37 files (pre/post sha, marker, py_compile), is idempotent,
   reverts to the exact pre-sha, refuses a modified copy and a mixed tree without writing anything.
T2 placement: in the patched model_runner the 'exec' flush sits after finish_forward() and before
   ExecuteModelState(...), the 'empty' flush inside the empty-step branch before its return.
T3 equivalence: the REAL original pp_utils.py and the REAL patched one (env off / NB4_PPACK / NB4_PPDEFER / both) are
   loaded with stubbed vllm imports, a fake torch.cuda (streams, events, wait_stream / wait_event) and a fake
   torch.distributed (a 2-rank NCCL broadcast simulator: per-group sequence matching, count + dtype must match,
   receive buffers start as a sentinel and are filled only when the sender posts). Two ranks (stage 0 receiver,
   stage 1 sender) run the model_runner call order over scripted steps: decode n = 1..9 (K = 3), partial masks
   (non-final prefill chunks), no-sample steps, empty steps, freed requests, s = 1 and s = K + 1 sampled widths.
   Every consumed slot (sampled tokens, counts, idx_mapping, deferred state, the draft table after the update, the
   receive() return) must equal the original's: values, dtype, shape, stride, storage offset. Run sequentially
   (sender first) and with 2 threads (sender and receiver concurrent; the receiver blocks in wait_event).
T4 env off: the patched file's per-rank op log (every broadcast with stream / shape / dtype, wait_stream, record_event,
   wait_event, record_stream) equals the original's exactly.
T5 ordering: original / Q-pack post receive(T) behind forward(T); Q-defer behind forward(T + 1) (or the empty-step
   marker); every wait_event at step S waits for an event recorded right after slot S-2's receive op(s) and before
   any later slot's; every broadcast runs on the broadcast stream; one message per step with Q-pack.
T6 fail closed: NB4_PPACK=1 with K = 0, a stale sender stash, drafts without a stash, flush during graph capture.
T7 readback_label / parse_log / server_env on synthetic log lines (rev 2 R1: per-rank last stats line, flush ratios,
   ENGAGED_FALLBACK).
rev 2 (fixes F1 / F2 / F3 of NB4_PR53948_COMPARISON_2026-11-01.md; the patch now also covers gpu_worker.py):
T2 also: GPUModelRunner.shutdown flushes ("shutdown") + logs the final stats line before torch.accelerator.synchronize();
   Worker.profile / Worker.shutdown start with the guarded "rpc" flush.
T8 F1 shutdown: a parked slot at GPUModelRunner.shutdown (the REAL patched function, extracted by AST, run against the
   simulator) is posted before the device sync, so the peer's outstanding send is matched; the final stats line is
   logged; the ORIGINAL function leaves the send unmatched (the hang H1 condition); env off adds no op.
T9 F2 syncing RPC: Worker.profile(is_start=False) (profiler stop = device sync) and Worker.shutdown (profiler / runner
   teardown syncs), REAL patched functions, with a parked slot: flushed first, every send matched at every sync; the
   original functions leave it unmatched (hang H2); the guards (no model_runner / no pp_handler / PP off) never raise.
T10 F3 draft-table race: the sender's broadcast stream lags (a read issued on it sees the table after the next
   propose() rewrote every row): Merlin's original sender ships the rewritten rows (negative control), Q-defer / Q-pack
   / both ship the snapshot taken on the main stream (receiver table == ground truth); record_stream is on the gathered
   [n, K] tensor, never on idx_mapping.
"""
import argparse
import ast
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
import types
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nb4_ppack as P  # noqa: E402

DEFAULT_SRC = "D:/ARC AGI 2/_scratch/e23/p2/merlin-src"
SENTINEL = -7777
RESULTS = []
EARLY = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    if not cond:
        print(f"FAIL {name}: {detail}"[:2000], flush=True)
    return bool(cond)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_tree(src, dst):
    for rel in P.PATCHES:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(src) / rel, dst / rel)


# ----------------------------------------------------------------------------------------------- T1 / T2 patch
def t1_patch(src, work):
    for rel, (pre, *_r) in P.PATCHES.items():
        check(f"T1 source {rel} is Merlin pre-sha", sha(Path(src) / rel) == pre, sha(Path(src) / rel))
    tree = work / "tree_apply"
    make_tree(src, tree)
    rec = P.apply_ppack_patch(tree)
    check("T1 apply ok", rec.get("ok") and rec["applied"] == "applied", json.dumps(rec)[:400])
    for rel, (pre, post, marker, _) in P.PATCHES.items():
        data = (tree / rel).read_bytes()
        check(f"T1 post-sha {rel}", sha(tree / rel) == post)
        check(f"T1 marker {rel}", marker.encode() in data)
        check(f"T1 LF only {rel}", b"\r" not in data)
    rec2 = P.apply_ppack_patch(tree)
    check("T1 re-apply = already_applied", rec2["applied"] == "already_applied")
    patched = {rel: (tree / rel).read_bytes() for rel in P.PATCHES}
    rv = P.revert_ppack_patch(tree)
    check("T1 revert ok", rv.get("ok") and rv["reverted"] == "reverted")
    for rel, (pre, *_r) in P.PATCHES.items():
        check(f"T1 revert pre-sha {rel}", sha(tree / rel) == pre)
    check("T1 revert again = not_applied", P.revert_ppack_patch(tree)["reverted"] == "not_applied")
    # a modified copy: one changed byte in a context line of pp_utils
    bad = work / "tree_bad"
    make_tree(src, bad)
    f = bad / P.PP_REL
    f.write_bytes(f.read_bytes().replace(b"self.queue.append(None)", b"self.queue.append(None) ", 1))
    before = {rel: sha(bad / rel) for rel in P.PATCHES}
    try:
        P.apply_ppack_patch(bad)
        check("T1 refuses a modified copy", False, "no exception")
    except RuntimeError as e:
        check("T1 refuses a modified copy", "PRE_SHA_MISMATCH" in str(e), str(e)[:200])
    check("T1 refusal wrote nothing", before == {rel: sha(bad / rel) for rel in P.PATCHES})
    try:
        P.revert_ppack_patch(bad)
        check("T1 revert refuses a modified copy", False, "no exception")
    except RuntimeError as e:
        check("T1 revert refuses a modified copy", "SHA_MISMATCH" in str(e), str(e)[:200])
    # mixed: pp_utils patched, model_runner original
    mixed = work / "tree_mixed"
    make_tree(src, mixed)
    (mixed / P.PP_REL).write_bytes(patched[P.PP_REL])
    check("T1 verify says mixed", P.verify_ppack_patch(mixed)["state"] == "mixed", P.verify_ppack_patch(mixed)["state"])
    try:
        P.apply_ppack_patch(mixed)
        check("T1 refuses a mixed tree", False, "no exception")
    except RuntimeError as e:
        check("T1 refuses a mixed tree", "PRE_SHA_MISMATCH" in str(e), str(e)[:200])
    # a context-line mismatch at the applier level also raises (no fuzz)
    try:
        P.apply_unified((Path(src) / P.MR_REL).read_text(encoding="utf-8").replace(
            "        self.kv_connector.finish_forward()\n", "        self.kv_connector.finish_forward()  # x\n", 1),
            P.MR_PATCH)
        check("T1 applier has no fuzz", False, "no exception")
    except ValueError:
        check("T1 applier has no fuzz", True)
    return patched


def t2_placement(patched_mr):
    text = patched_mr.decode("utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "execute_model":
            fn = node
            break
    check("T2 execute_model found", fn is not None)
    calls = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            name = node.func.attr
            if name == "flush_deferred_receive":
                calls.setdefault(name + ":" + node.args[0].value, []).append(node.lineno)
            elif name in ("finish_forward", "run_fullgraph"):
                calls.setdefault(name, []).append(node.lineno)
    ems = [i + 1 for i, l in enumerate(lines) if "self.execute_model_state = ExecuteModelState(" in l
           and fn.lineno <= i + 1 <= fn.end_lineno]
    empty_if = [i + 1 for i, l in enumerate(lines) if "if scheduler_output.total_num_scheduled_tokens == 0:" in l
                and fn.lineno <= i + 1 <= fn.end_lineno]
    ex, em = calls.get("flush_deferred_receive:exec", []), calls.get("flush_deferred_receive:empty", [])
    check("T2 one exec flush, one empty flush", len(ex) == 1 and len(em) == 1, str(calls))
    ff = calls.get("finish_forward", [])
    check("T2 exec flush after finish_forward, after run_fullgraph, before ExecuteModelState",
          len(ff) == 1 and len(ems) == 1 and ff[0] < ex[0] < ems[0] and all(r < ex[0] for r in calls.get("run_fullgraph", [])),
          f"ff={ff} ex={ex} ems={ems} rf={calls.get('run_fullgraph')}")
    ret = [i + 1 for i, l in enumerate(lines) if "return self._merge_ec_connector_no_forward(" in l
           and fn.lineno <= i + 1 <= fn.end_lineno]
    check("T2 empty flush inside the empty-step branch, before its return",
          len(empty_if) == 1 and ret and empty_if[0] < em[0] < ret[0], f"if={empty_if} em={em} ret={ret}")
    check("T2 exec flush skips dummy runs", "if self.pp_handler is not None and not dummy_run:" in lines[ex[0] - 4])
    # F1: GPUModelRunner.shutdown
    sd = find_method(tree, "GPUModelRunner", "shutdown")
    check("T2 GPUModelRunner.shutdown found", sd is not None)
    order = []
    for node in ast.walk(sd):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "flush_deferred_receive":
                order.append(("flush:" + node.args[0].value, node.lineno))
            elif node.func.attr in ("nb4_log_final", "synchronize"):
                order.append((node.func.attr, node.lineno))
    order.sort(key=lambda x: x[1])
    check("T2 shutdown: flush('shutdown') then the final stats line, both before the first synchronize()",
          [n for n, _ in order][:3] == ["flush:shutdown", "nb4_log_final", "synchronize"], order)


def find_method(tree, cls, name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == name:
                    return f
    return None


def t2_worker(patched_gw):
    tree = ast.parse(patched_gw.decode("utf-8"))
    for name in ("profile", "shutdown"):
        fn = find_method(tree, "Worker", name)
        ok = fn is not None and len(fn.body) >= 2
        if ok:
            a, b = fn.body[0], fn.body[1]
            ok = (isinstance(a, ast.Assign) and a.targets[0].id == "_nb4_flush"
                  and "getattr(getattr(getattr(self, 'model_runner', None), 'pp_handler', None), 'flush_deferred_receive', None)"
                  == ast.unparse(a.value)
                  and isinstance(b, ast.If) and ast.unparse(b.test) == "_nb4_flush is not None"
                  and "_nb4_flush('rpc')" in ast.unparse(b))
        check(f"T2 Worker.{name}: the guarded 'rpc' flush is the first statement", ok, ast.unparse(fn.body[0]) if fn else None)
    fn = find_method(tree, "Worker", "shutdown")
    check("T2 Worker.shutdown: a failing flush is logged, never stops the teardown",
          isinstance(fn.body[1].body[0], ast.Try) and "logger.exception" in ast.unparse(fn.body[1].body[0]))


# ----------------------------------------------------------------------------------------------- simulator
class Sim:
    """Two ranks: 0 = stage 0 (non-last, receiver), 1 = stage 1 (last rank, sender)."""

    def __init__(self):
        self.tl = threading.local()
        self.cv = threading.Condition()
        self.capturing = False
        self.logs = {0: [], 1: []}
        self.msgs = {}          # seq -> {"send": tensor clone, "recv": (tensor, op)}
        self.seq = {0: 0, 1: 0}
        self.errors = []
        self.mains = {}
        self.ev_id = 0

    @property
    def rank(self):
        return self.tl.rank

    def log(self, *rec):
        self.logs[self.rank].append(rec)

    def cur_stream(self):
        st = getattr(self.tl, "stack", None)
        return st[-1] if st else self.mains[self.rank]


class Op:
    def __init__(self, kind, cond=None, tag=None):
        self.kind, self.cond, self.tag = kind, cond, tag


class FakeStream:
    def __init__(self, sim, name):
        self.sim, self.name, self.ops = sim, name, []

    def complete_upto(self, upto):
        return all(op.cond is None or op.cond() for op in self.ops[:upto])

    def wait_stream(self, other):
        upto = len(other.ops)
        last = other.ops[upto - 1].tag if upto else None
        self.ops.append(Op("wait", lambda: other.complete_upto(upto), tag=("wait", other.name, upto, last)))
        self.sim.log("wait_stream", self.name, other.name)

    def record_event(self):
        self.sim.ev_id += 1
        ev = FakeEvent(self, len(self.ops), self.sim.ev_id)
        self.ops.append(Op("event", tag=("event", ev.id)))
        self.sim.log("record_event", self.name)
        return ev

    def wait_event(self, ev):
        self.sim.log("wait_event", self.name, ev.stream.name)
        self.sim.waits.setdefault(self.sim.rank, []).append((self.sim.tl.step, ev))
        deadline = time.time() + 20
        with self.sim.cv:
            while not ev.stream.complete_upto(ev.upto):
                if not self.sim.cv.wait(timeout=max(0.0, deadline - time.time())):
                    raise RuntimeError(f"SIM_DEADLOCK wait_event step {self.sim.tl.step}")
        self.ops.append(Op("waited", tag=("waited", ev.id)))

    def synchronize(self):
        raise AssertionError("synchronize is XPU-only")


class FakeEvent:
    def __init__(self, stream, upto, eid):
        self.stream, self.upto, self.id = stream, upto, eid


def build_fake_torch(sim):
    cuda = types.SimpleNamespace()

    def current_stream(device=None):
        return sim.mains[sim.rank]

    def Stream(device=None):  # noqa: N802
        return FakeStream(sim, f"bcast{sim.rank}")

    @contextmanager
    def stream(s):
        st = getattr(sim.tl, "stack", None)
        if st is None:
            st = sim.tl.stack = []
        st.append(s)
        try:
            yield
        finally:
            st.pop()

    cuda.current_stream, cuda.Stream, cuda.stream = current_stream, Stream, stream
    cuda.is_current_stream_capturing = lambda: sim.capturing
    cuda.Event = torch.cuda.Event

    def broadcast(tensor, src, group):
        r = sim.rank
        s = sim.cur_stream()
        k = sim.seq[r]
        sim.seq[r] += 1
        sim.log("broadcast", s.name, tuple(tensor.shape), str(tensor.dtype), "send" if r == src else "recv")
        with sim.cv:
            m = sim.msgs.setdefault(k, {})
            if r == src:
                m["send"] = tensor.detach().clone()
                s.ops.append(Op("bsend", tag=("bsend", k)))
            else:
                m["recv_first"] = "send" not in m
                tensor.fill_(SENTINEL)
                box = {"filled": False}
                m["recv"] = (tensor, box)
                s.ops.append(Op("brecv", cond=lambda: box["filled"], tag=("brecv", k, sim.tl.step)))
            if "send" in m and "recv" in m and not m["recv"][1]["filled"]:
                snd, (rcv, box) = m["send"], m["recv"]
                if snd.numel() != rcv.numel() or snd.dtype != rcv.dtype:
                    sim.errors.append(f"seq {k}: send {tuple(snd.shape)} {snd.dtype} vs recv {tuple(rcv.shape)} {rcv.dtype}")
                else:
                    rcv.view(-1).copy_(snd.reshape(-1))
                box["filled"] = True
                sim.cv.notify_all()

    dist = types.SimpleNamespace(broadcast=broadcast)

    class FT:
        def __getattr__(self, name):
            return getattr(torch, name)

    ft = FT()
    ft.cuda, ft.distributed = cuda, dist
    return ft


def install_fake_vllm(sim):
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    class FakePPGroup:
        def __init__(self, rank):
            self.rank_in_group, self.is_last_rank, self.last_rank, self.world_size = rank, rank == 1, 1, 2

        def make_sibling_device_group(self, group_desc):
            return ("group", group_desc)

    def get_pp_group():
        return FakePPGroup(sim.rank)

    class IntermediateTensors:  # noqa: D401 - stub
        pass

    def async_tensor_h2d(arr, device=None, **kw):
        return torch.as_tensor(np.asarray(arr), device="cpu")

    for n in ("vllm", "vllm.distributed", "vllm.utils", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu"):
        mod(n)
    mod("vllm.envs", VLLM_XPU_PP_MICROBATCH=True)
    mod("vllm.distributed.parallel_state", get_pp_group=get_pp_group)
    mod("vllm.platforms", current_platform=types.SimpleNamespace(is_xpu=lambda: False))
    mod("vllm.sequence", IntermediateTensors=IntermediateTensors)
    mod("vllm.utils.torch_utils", async_tensor_h2d=async_tensor_h2d)
    mod("vllm.v1.worker.gpu.input_batch", InputBatch=object)
    sys.modules["vllm.distributed"].parallel_state = sys.modules["vllm.distributed.parallel_state"]


def load_pp(text, sim, tag):
    install_fake_vllm(sim)
    m = types.ModuleType(f"nb4_pp_{tag}")
    m.__file__ = f"<{tag}>"
    sys.modules[m.__name__] = m       # dataclasses resolve the module by name
    exec(compile(text, f"pp_utils[{tag}]", "exec"), m.__dict__)  # noqa: S102
    m.torch = build_fake_torch(sim)   # functions look torch up at call time
    return m


class IB:
    """The InputBatch fields PPHandler reads."""

    def __init__(self, step):
        n = step["n"]
        self.num_reqs = n
        self.idx_mapping_np = np.asarray(step["rows"], dtype=np.intp)
        self.idx_mapping = torch.as_tensor(step["rows"], dtype=torch.int64)
        self.prefill_len_np = np.full(n, 10, dtype=np.int32)
        self.num_computed_tokens_np = np.where(step["final"], 100, 0).astype(np.int32)
        self.num_scheduled_tokens = np.where(step["final"], 4, 3).astype(np.int32)
        self.max_seq_len_np = np.full(n, 10 ** 6, dtype=np.int32)
        self.is_prefilling_np = ~np.asarray(step["final"])


def make_scenario(seed, k, max_reqs=24):
    rng = np.random.default_rng(seed)
    steps = []
    ns = list(range(1, 10)) * 3
    rng.shuffle(ns)
    t = 0
    for n in ns:
        kind = rng.choice(["decode"] * 6 + ["partial", "noneed", "empty"])
        rows = rng.choice(max_reqs, size=n, replace=False)
        final = np.ones(n, dtype=bool)
        if kind == "partial":
            final[rng.integers(0, n)] = False
            if n > 2:
                final[rng.integers(0, n)] = False
            if not final.any():
                final[0] = True
        elif kind == "noneed":
            final[:] = False
        s = int(rng.choice([k + 1, k + 1, 1, max(1, k)]))
        st = {"t": t, "kind": str(kind), "n": n, "rows": rows.tolist(), "final": final, "s": s,
              "sampled": torch.as_tensor(rng.integers(0, 151000, size=(n, s)), dtype=torch.int64),
              "num_sampled": torch.as_tensor(rng.integers(1, k + 2, size=n), dtype=torch.int32),
              "num_rejected": torch.as_tensor(rng.integers(0, k + 1, size=n), dtype=torch.int32),
              "drafts": torch.as_tensor(rng.integers(-1, 151000, size=(max_reqs, k)), dtype=torch.int64),
              "freed": rng.choice(max_reqs, size=int(rng.integers(0, 3)), replace=False).tolist(),
              "state": f"S{t}"}
        steps.append(st)
        t += 1
    for _ in range(3):  # drain: empty steps consume the last slots
        steps.append({"t": t, "kind": "empty", "n": 0, "freed": [], "state": None})
        t += 1
    return steps


def run_sim(text, env, scenario, k, mode, max_reqs=24, tag="x", flush=True, drafts_view=None):
    """drafts_view(sim, drafts) -> what the sender's broadcast_drafts() gets as the draft table (T10: LagTable)."""
    sim = Sim()
    sim.waits = {}
    old_rs = torch.Tensor.record_stream

    def rs(self, s):
        sim.log("record_stream", tuple(self.shape), str(self.dtype), getattr(s, "name", str(s)))

    for key in (P.ENV_PACK, P.ENV_DEFER):
        os.environ.pop(key, None)
    os.environ.update(env)
    torch.Tensor.record_stream = rs
    try:
        mod = load_pp(text, sim, tag)
        handlers = {}
        for r in (0, 1):
            sim.tl.rank = r
            sim.mains[r] = FakeStream(sim, f"main{r}")
            handlers[r] = mod.PPHandler(max_num_reqs=max_reqs, num_speculative_steps=k, device=torch.device("cpu"))
        out = {"recv": [], "ret": [], "sim": sim, "handlers": handlers, "mod": mod}

        def sender():
            sim.tl.rank, sim.tl.stack = 1, []
            h = handlers[1]
            for st in scenario:
                sim.tl.step = st["t"]
                for i in st["freed"]:
                    h.on_req_idx_freed(i)
                if st["kind"] == "empty":
                    continue
                if mode == "thread":
                    time.sleep(0.002)   # the receiver usually posts first (its buffers hold the sentinel)
                ib = IB(st)
                h.broadcast(st["sampled"], st["num_sampled"], st["num_rejected"], ib)
                if mode == "thread":
                    time.sleep(0.001)   # sampled sent, drafts not yet (the receiver may be waiting)
                h.broadcast_drafts(drafts_view(sim, st["drafts"]) if drafts_view else st["drafts"], ib)

        def receiver():
            sim.tl.rank, sim.tl.stack = 0, []
            h = handlers[0]
            main = sim.mains[0]
            table = torch.zeros(max_reqs, k, dtype=torch.int64)
            for st in scenario:
                sim.tl.step = st["t"]
                for i in st["freed"]:
                    h.on_req_idx_freed(i)
                o = h.get_prev_sampled_outputs(table)                       # update_pp_decode_requests
                rec = {"t": st["t"], "out": None, "table": None}
                if o is not None:
                    rec["out"] = {key: (v.clone(), tuple(v.stride()), v.storage_offset(), v.dtype, v.is_contiguous())
                                  if isinstance(v, torch.Tensor) else v for key, v in o.items()}
                rec["table"] = table.clone()
                out["recv"].append(rec)
                if st["kind"] == "empty":
                    main.ops.append(Op("marker", tag=("empty", st["t"])))
                    if flush and hasattr(h, "flush_deferred_receive"):
                        h.flush_deferred_receive("empty")
                    continue
                main.ops.append(Op("marker", tag=("forward", st["t"])))    # graph(T) launch
                if flush and hasattr(h, "flush_deferred_receive"):
                    h.flush_deferred_receive("exec")
                ib = IB(st)
                out["ret"].append(h.receive(ib))                            # sample_tokens
                h.attach_deferred_state(st["state"] if bool(np.asarray(st["final"]).any()) else None)

        if mode == "seq":
            sender()
            receiver()
        else:
            errs = []

            def wrap(fn):
                def w():
                    try:
                        fn()
                    except Exception:  # noqa: BLE001
                        errs.append(traceback.format_exc())
                return w
            th = [threading.Thread(target=wrap(f)) for f in (receiver, sender)]
            for x in th:
                x.start()
            for x in th:
                x.join(60)
            if errs:
                raise RuntimeError(errs[0])
        out["errors"] = list(sim.errors)
        out["early_recv"] = sum(1 for m in sim.msgs.values() if "recv" in m and m.get("recv_first"))
        return out
    finally:
        torch.Tensor.record_stream = old_rs
        for key in (P.ENV_PACK, P.ENV_DEFER):
            os.environ.pop(key, None)


def ground_truth_ok(res, scenario, k, max_reqs=24):
    """Original path vs the scenario itself (the simulator must be meaningful)."""
    table = torch.zeros(max_reqs, k, dtype=torch.int64)
    gen = np.zeros(max_reqs, dtype=np.int64)
    pending = {}
    bad = []
    for i, st in enumerate(scenario):
        for f in st["freed"]:
            gen[f] += 1
        exp = None
        src = pending.pop(st["t"] - 2, None)
        if src is not None:
            final = np.asarray(src["final"])
            freed = gen[src["rows"]] != src["gen"]
            excl = freed | ~final
            if not excl.all():
                pad = torch.nn.functional.pad(src["sampled"], (0, k + 1 - src["s"]))
                exp = (pad, src["num_sampled"], src["num_rejected"])
                keep = ~excl
                rows = np.asarray(src["rows"])[keep]
                table[torch.as_tensor(rows)] = src["drafts"][torch.as_tensor(src["rows"])][torch.as_tensor(keep)]
        got = res["recv"][i]["out"]
        if (exp is None) != (got is None):
            bad.append((st["t"], "presence"))
        elif exp is not None:
            if not (torch.equal(got["sampled_tokens"][0], exp[0]) and torch.equal(got["num_sampled"][0], exp[1])
                    and torch.equal(got["num_rejected"][0], exp[2])):
                bad.append((st["t"], "values"))
        if not torch.equal(res["recv"][i]["table"], table):
            bad.append((st["t"], "table"))
        if st["kind"] != "empty" and np.asarray(st["final"]).any():
            pending[st["t"]] = dict(st, gen=gen[st["rows"]].copy())
    return bad


def same_records(a, b):
    diffs = []
    if a["ret"] != b["ret"]:
        diffs.append("receive() returns differ")
    for ra, rb in zip(a["recv"], b["recv"]):
        if not torch.equal(ra["table"], rb["table"]):
            diffs.append(f"t{ra['t']} draft table")
        oa, ob = ra["out"], rb["out"]
        if (oa is None) != (ob is None):
            diffs.append(f"t{ra['t']} presence")
            continue
        if oa is None:
            continue
        if set(oa) != set(ob):
            diffs.append(f"t{ra['t']} keys {sorted(oa)} vs {sorted(ob)}")
            continue
        for key in oa:
            va, vb = oa[key], ob[key]
            if isinstance(va, tuple):
                ta, tb = va[0], vb[0]
                if not (ta.dtype == tb.dtype and ta.shape == tb.shape and torch.equal(ta, tb)):
                    diffs.append(f"t{ra['t']} {key} values/dtype/shape")
                if va[1:] != vb[1:]:
                    diffs.append(f"t{ra['t']} {key} layout {va[1:]} vs {vb[1:]}")
            elif va != vb:
                diffs.append(f"t{ra['t']} {key} {va!r} vs {vb!r}")
    if len(a["recv"]) != len(b["recv"]):
        diffs.append("record count")
    return diffs


def ordering_check(res, scenario, variant):
    """T5 on the receiver's streams."""
    sim = res["sim"]
    bc = res["handlers"][0].broadcast_stream
    problems = []
    # 1. the broadcast stream: group recv ops by the wait that precedes them
    groups, cur = [], None
    for idx, op in enumerate(bc.ops):
        if op.kind == "wait":
            cur = {"wait": op.tag, "recv": [], "event": None}
            groups.append(cur)
        elif op.kind == "brecv":
            cur["recv"].append((idx, op.tag[1]))
        elif op.kind == "event":
            cur["event"] = (idx, op.tag[1])
    posted_steps = [st["t"] for st in scenario if st["kind"] != "empty" and np.asarray(st["final"]).any()]
    if len(groups) != len(posted_steps):
        problems.append(f"{len(groups)} receive groups for {len(posted_steps)} sampling steps")
        return problems
    per = 1 if "pack" in variant else 3
    by_slot = {}
    for g, t in zip(groups, posted_steps):
        by_slot[t] = g
        if len(g["recv"]) != per:
            problems.append(f"slot {t}: {len(g['recv'])} receive ops (want {per})")
        if g["event"] is None or any(i > g["event"][0] for i, _ in g["recv"]):
            problems.append(f"slot {t}: event not after its receive ops")
        last = g["wait"][3]
        nxt = next((s for s in scenario if s["t"] > t), None)
        if "defer" in variant:
            if not (last is not None and last[1] == (nxt["t"] if nxt else None)):
                problems.append(f"slot {t}: deferred receive posted behind {last}, want step {nxt and nxt['t']}")
        else:
            if last != ("forward", t):
                problems.append(f"slot {t}: receive posted behind {last}, want ('forward', {t})")
    # 2. every wait_event at step S consumes slot S-2's event exactly
    evs = {g["event"][1]: (t, g["event"][0]) for t, g in by_slot.items()}
    for step, ev in sim.waits.get(0, []):
        t_ev, idx = evs.get(ev.id, (None, None))
        if t_ev != step - 2:
            problems.append(f"step {step}: waited on the event of slot {t_ev}, want {step - 2}")
            continue
        g = by_slot[t_ev]
        later = [i for t, gg in by_slot.items() if t > t_ev for i, _ in gg["recv"]]
        if ev.upto != idx or any(i < ev.upto for i in later):
            problems.append(f"step {step}: event span wrong")
    # 3. all broadcasts on a broadcast stream, both ranks
    for r in (0, 1):
        for rec in sim.logs[r]:
            if rec[0] == "broadcast" and not rec[1].startswith("bcast"):
                problems.append(f"rank {r}: broadcast on {rec[1]}")
    nsend = sum(1 for rec in sim.logs[1] if rec[0] == "broadcast")
    if nsend != per * len(posted_steps):
        problems.append(f"sender posted {nsend} broadcasts, want {per * len(posted_steps)}")
    return problems


def t3_t4_t5(orig_text, patched_text):
    variants = {"off": {}, "pack": {P.ENV_PACK: "1"}, "defer": {P.ENV_DEFER: "1"},
                "pack+defer": {P.ENV_PACK: "1", P.ENV_DEFER: "1"}}
    covered = set()
    for k in (3, 1, 2, 4):
        seeds = (11, 12, 13) if k == 3 else (21,)
        for seed in seeds:
            sc = make_scenario(seed, k)
            covered |= {(k, st["n"]) for st in sc if st["kind"] == "decode"}
            for mode in ("seq", "thread"):
                ref = run_sim(orig_text, {}, sc, k, mode, tag=f"orig_{k}_{seed}_{mode}")
                gt = ground_truth_ok(ref, sc, k)
                check(f"T3 sim ground truth k{k} s{seed} {mode}", not gt and not ref["errors"], f"{gt[:5]} {ref['errors'][:3]}")
                for vname, env in variants.items():
                    res = run_sim(patched_text, env, sc, k, mode, tag=f"p_{vname}_{k}_{seed}_{mode}")
                    d = same_records(ref, res)
                    if mode == "thread" and vname != "off":
                        EARLY.append(res["early_recv"])
                    check(f"T3 {vname} == original k{k} s{seed} {mode}", not d and not res["errors"],
                          f"{d[:6]} {res['errors'][:3]}")
                    if mode == "seq":
                        if vname == "off":
                            for r in (0, 1):
                                la, lb = ref["sim"].logs[r], res["sim"].logs[r]
                                check(f"T4 env-off op log identical rank {r} k{k} s{seed}", la == lb and len(la) > 20,
                                      f"{len(la)} vs {len(lb)} first diff "
                                      f"{next(((i, x, y) for i, (x, y) in enumerate(zip(la, lb)) if x != y), None)}")
                            check(f"T4 env-off flags False k{k} s{seed}",
                                  not any(h.nb4_ppack or h.nb4_ppdefer for h in res["handlers"].values()))
                            check(f"T4 env-off no stats k{k} s{seed}",
                                  not any(any(h.nb4_stats.values()) for h in res["handlers"].values()))
                        pr = ordering_check(res, sc, "orig" if vname == "off" else vname)
                        check(f"T5 ordering {vname} k{k} s{seed}", not pr, str(pr[:6]))
                        if vname != "off":
                            st0 = res["handlers"][0].nb4_stats
                            st1 = res["handlers"][1].nb4_stats
                            ok = True
                            if "pack" in vname:
                                ok &= st1["packed_send"] == st1["packed_stash"] == st0["packed_recv"] > 0
                            if "defer" in vname:
                                ok &= st0["defer_park"] > 0 and st0["flush_exec"] + st0["flush_empty"] \
                                    + st0["flush_recv"] + st0["flush_consume"] + st0["flush_rpc"] \
                                    + st0["flush_shutdown"] == st0["defer_park"]
                            check(f"T5 counters {vname} k{k} s{seed}", ok, f"{st0} {st1}")
                    if k == 3 and "defer" in vname:
                        # no model_runner flush at all: every post comes from the receive / consume fallbacks
                        nf = run_sim(patched_text, env, sc, k, mode, tag=f"nf_{vname}_{seed}_{mode}", flush=False)
                        st0 = nf["handlers"][0].nb4_stats
                        check(f"T3 {vname} fallbacks only == original s{seed} {mode}",
                              not same_records(ref, nf) and not nf["errors"] and st0["flush_exec"] == 0
                              and st0["flush_recv"] > 0 and st0["flush_consume"] > 0
                              and st0["flush_recv"] + st0["flush_consume"] == st0["defer_park"],
                              f"{same_records(ref, nf)[:4]} {st0}")
                    if k == 3 and seed == 11 and mode == "seq":
                        pr0 = ordering_check(ref, sc, "orig")
                        check("T5 ordering original", not pr0, str(pr0[:6]))
    check("T3 thread mode: receives posted before the send happened", sum(EARLY) > 0, str(EARLY))
    check("T3 decode n = 1..9 covered at K = 3", {n for kk, n in covered if kk == 3} >= set(range(1, 10)),
          str(sorted(covered)))


def t6_fail_closed(patched_text):
    sim = Sim()
    sim.waits = {}
    os.environ[P.ENV_PACK] = "1"
    try:
        mod = load_pp(patched_text, sim, "fc")
        sim.tl.rank = 1
        sim.mains[1] = FakeStream(sim, "main1")
        try:
            mod.PPHandler(max_num_reqs=8, num_speculative_steps=0, device=torch.device("cpu"))
            check("T6 PPACK with K=0 raises", False, "no exception")
        except RuntimeError as e:
            check("T6 PPACK with K=0 raises", "K > 0" in str(e))
        h = mod.PPHandler(max_num_reqs=8, num_speculative_steps=3, device=torch.device("cpu"))
        sc = {"n": 2, "rows": [1, 3], "final": np.ones(2, dtype=bool)}
        ib = IB(sc)
        s = torch.zeros(2, 4, dtype=torch.int64)
        c = torch.ones(2, dtype=torch.int32)
        old_rs = torch.Tensor.record_stream
        torch.Tensor.record_stream = lambda self, st: None
        try:
            h.broadcast(s, c, c, ib)
            try:
                h.broadcast(s, c, c, ib)
                check("T6 stale stash raises", False, "no exception")
            except RuntimeError as e:
                check("T6 stale stash raises", "never sent" in str(e))
            h.nb4_pack_stash = None
            try:
                h.broadcast_drafts(torch.zeros(8, 3, dtype=torch.int64), ib)
                check("T6 drafts without stash raises", False, "no exception")
            except RuntimeError as e:
                check("T6 drafts without stash raises", "stash=False" in str(e))
            h.broadcast(s, c, c, ib)
            try:
                h.broadcast_drafts(torch.zeros(8, 2, dtype=torch.int64), ib)
                check("T6 draft shape mismatch raises", False, "no exception")
            except RuntimeError as e:
                check("T6 draft shape mismatch raises", "draft shape" in str(e))
        finally:
            torch.Tensor.record_stream = old_rs
        os.environ.pop(P.ENV_PACK, None)
        os.environ[P.ENV_DEFER] = "1"
        sim.tl.rank = 0
        sim.mains[0] = FakeStream(sim, "main0")
        h0 = mod.PPHandler(max_num_reqs=8, num_speculative_steps=3, device=torch.device("cpu"))
        sim.tl.rank = 1
        h1d = mod.PPHandler(max_num_reqs=8, num_speculative_steps=3, device=torch.device("cpu"))
        sim.tl.rank = 0
        check("T6 PPDEFER is off on the last rank", not h1d.nb4_ppdefer and h0.nb4_ppdefer)
        torch.Tensor.record_stream = lambda self, st: None
        try:
            sim.tl.step = 0
            h0.get_prev_sampled_outputs(None)
            h0.receive(IB(sc))
            sim.capturing = True
            h0.flush_deferred_receive("exec")
            check("T6 no post during graph capture", h0.queue[-1].event is None and h0.nb4_stats["flush_exec"] == 0)
            sim.capturing = False
            h0.flush_deferred_receive("exec")
            check("T6 posts after capture", h0.queue[-1].event is not None and h0.nb4_stats["flush_exec"] == 1)
        finally:
            torch.Tensor.record_stream = old_rs
    finally:
        os.environ.pop(P.ENV_PACK, None)
        os.environ.pop(P.ENV_DEFER, None)


def t7_readback():
    def line(**kw):
        return "(Worker pid=1) NB4_PPQ " + json.dumps(kw)
    def pst(pid, send=0, recv=0, key=None):
        s = dict.fromkeys(("packed_stash", "defer_post", "defer_park", "flush_exec", "flush_empty", "flush_recv", "flush_consume",
                           "flush_rpc", "flush_shutdown"), 0)
        s.update(packed_send=send, packed_recv=recv)
        return line(event="stats", key=key or ("packed_send" if send else "packed_recv"), pid=pid, stats=s)
    lines = []
    for last, pid in ((False, 11), (False, 12), (True, 21), (True, 22)):
        lines.append(line(event="init", ppack=True, ppdefer=False, is_last_rank=last, pid=pid))
    lines += [pst(21, send=7), pst(22, send=7), pst(11, recv=7), pst(12, recv=7)]
    lab, d = P.readback_label("\n".join(lines), "Q-pack")
    check("T7 Q-pack ENGAGED (every rank shows its own packed counter)", lab == "ENGAGED" and len(d["pack_per_rank"]) == 4, str(d))
    # GPT audit 2026-11-01: 4 inits, but only one sender and one receiver packed -> NOT_ENGAGED (was ENGAGED)
    lab, d = P.readback_label("\n".join(lines[:4] + [pst(21, send=7), pst(11, recv=7)]), "Q-pack")
    check("T7 Q-pack: one sender + one receiver only -> NOT_ENGAGED", lab == "NOT_ENGAGED" and "22:packed_send" in d.get("why", "")
          and "12:packed_recv" in d.get("why", ""), str(d))
    lab, d = P.readback_label("\n".join(lines + [pst(12, recv=0, key="packed_stash")]), "Q-pack")
    check("T7 Q-pack: a receiver whose last line has packed_recv 0 -> NOT_ENGAGED", lab == "NOT_ENGAGED" and "12:packed_recv" in d.get("why", ""), str(d))
    nop = [line(event="init", ppack=True, ppdefer=False, is_last_rank=last) for last in (False, False, True, True)]
    lab, d = P.readback_label("\n".join(nop + [line(event="stats", key="packed_send"), line(event="stats", key="packed_recv")]), "Q-pack")
    check("T7 Q-pack: init lines without pids -> NOT_ENGAGED", lab == "NOT_ENGAGED", str(d))
    lab, d = P.readback_label("\n".join(lines[:3]), "Q-pack")
    check("T7 missing rank -> NOT_INTEGRATED", lab == "NOT_INTEGRATED", str(d))
    lab, d = P.readback_label("\n".join(lines), "Q-defer")
    check("T7 wrong flags -> NOT_INTEGRATED", lab == "NOT_INTEGRATED", str(d))
    dl = [line(event="init", ppack=False, ppdefer=not last, is_last_rank=last, pid=pid)
          for last, pid in ((False, 11), (False, 12), (True, 21), (True, 22))]
    lab, d = P.readback_label("\n".join(dl), "Q-defer")
    check("T7 Q-defer without stats -> NOT_ENGAGED", lab == "NOT_ENGAGED", str(d))

    def st(pid, park, ex, empty=0, recv=0, cons=0, key="flush_exec"):
        s = dict.fromkeys(("packed_stash", "packed_send", "packed_recv", "defer_post", "flush_rpc", "flush_shutdown"), 0)
        s.update(defer_park=park, flush_exec=ex, flush_empty=empty, flush_recv=recv, flush_consume=cons)
        return line(event="stats", key=key, pid=pid, stats=s)
    good = [st(11, 1, 0, key="defer_park"), st(11, 1000, 970, empty=29, recv=1), st(12, 1000, 980, empty=20)]
    lab, d = P.readback_label("\n".join(dl + good), "Q-defer")
    check("T7 Q-defer ENGAGED (exec >= 0.9 park, recv + consume <= 0.05 park, per rank)", lab == "ENGAGED"
          and set(d["defer_per_rank"]) == {"11", "12"} and d["defer_per_rank"]["11"]["clean"], str(d))
    lab, d = P.readback_label("\n".join(dl + good + [st(12, 2000, 1500, empty=480, recv=20, key="defer_park")]), "Q-defer")
    check("T7 Q-defer exec share < 0.9 on one rank -> ENGAGED_FALLBACK", lab == "ENGAGED_FALLBACK"
          and not d["defer_per_rank"]["12"]["clean"] and d["defer_per_rank"]["11"]["clean"], str(d))
    lab, d = P.readback_label("\n".join(dl + good + [st(11, 2000, 1850, recv=60, cons=50, key="flush_recv")]), "Q-defer")
    check("T7 Q-defer recv + consume > 0.05 park -> ENGAGED_FALLBACK", lab == "ENGAGED_FALLBACK", str(d))
    collapsed = [st(11, 1, 0, key="defer_park"), st(11, 1, 0, recv=1, key="flush_recv"), st(11, 500, 1, cons=499, key="flush_consume"),
                 st(12, 500, 1, cons=499, key="flush_consume"), st(11, 600, 1, cons=599, key="flush_exec")]
    lab, d = P.readback_label("\n".join(dl + collapsed), "Q-defer")
    check("T7 Q-defer that collapsed to post-at-consume (keys present) -> ENGAGED_FALLBACK", lab == "ENGAGED_FALLBACK", str(d))
    lab, d = P.readback_label("\n".join(dl + [st(11, 2000, 1500, empty=500, key="defer_park")] + good), "Q-defer")
    check("T7 the LAST stats line per rank decides", lab == "ENGAGED", str(d))
    lab, d = P.readback_label("\n".join(dl + [st(11, 1000, 990), line(event="stats", key="defer_park", pid=12)]), "Q-defer")
    check("T7 a non-last rank without a stats dict -> NOT_ENGAGED", lab == "NOT_ENGAGED" and "12" in d.get("why", ""), str(d))
    nopid = [line(event="init", ppack=False, ppdefer=not last, is_last_rank=last) for last in (False, False, True, True)]
    lab, d = P.readback_label("\n".join(nopid + good), "Q-defer")
    check("T7 init lines without pids -> NOT_ENGAGED", lab == "NOT_ENGAGED", str(d))
    fin = line(event="stats", key="final_shutdown", pid=12,
               stats=dict(json.loads(good[-1].split("NB4_PPQ ", 1)[1])["stats"], flush_shutdown=1, defer_park=1001, flush_exec=980))
    lab, d = P.readback_label("\n".join(dl + good + [fin]), "Q-defer")
    check("T7 the final shutdown line counts as the last line", lab == "ENGAGED" and d["defer_per_rank"]["12"]["flush_shutdown"] == 1, str(d))
    pk = [line(event="init", ppack=True, ppdefer=not last, is_last_rank=last, pid=pid)
          for last, pid in ((False, 11), (False, 12), (True, 21), (True, 22))]
    pk += [pst(21, send=7), pst(22, send=7), pst(11, recv=1), pst(12, recv=1)]

    def with_recv(ls, n=7):
        out = []
        for ln in ls:
            j = json.loads(ln.split("NB4_PPQ ", 1)[1])
            j["stats"]["packed_recv"] = n
            out.append(line(**j))
        return out
    lab, d = P.readback_label("\n".join(pk + with_recv(good)), "Q-pack+defer")
    check("T7 Q-pack+defer ENGAGED", lab == "ENGAGED", str(d))
    lab, d = P.readback_label("\n".join(pk + with_recv(collapsed)), "Q-pack+defer")
    check("T7 Q-pack+defer collapsed -> ENGAGED_FALLBACK", lab == "ENGAGED_FALLBACK", str(d))
    lab, d = P.readback_label("\n".join(pk + good), "Q-pack+defer")
    check("T7 Q-pack+defer with receivers at packed_recv 0 -> NOT_ENGAGED", lab == "NOT_ENGAGED", str(d))
    check("T7 server_env", P.server_env("Q-pack") == {"NB4_PPACK": "1"} and P.server_env("Q-defer") == {"NB4_PPDEFER": "1"})


# ----------------------------------------------------------------------------------------------- T8 / T9 / T10 (rev 2)
class StopAfterSync(Exception):
    pass


def all_sends_matched(sim):
    """The peer's device sync can finish: every broadcast the sender posted has a posted (and filled) receive."""
    return bool(sim.msgs) and all("send" in m and "recv" in m and m["recv"][1]["filled"] for m in sim.msgs.values())


@contextmanager
def no_record_stream():
    old = torch.Tensor.record_stream
    torch.Tensor.record_stream = lambda self, s: None
    try:
        yield
    finally:
        torch.Tensor.record_stream = old


def parked_pair(text, env, k=3):
    """One sampling step: the sender (stage 1) has sent step 0; the receiver (stage 0) ran receive(0) and (Q-defer)
    parked it: the state between steps the review's H1 / H2 describe."""
    for key in (P.ENV_PACK, P.ENV_DEFER):
        os.environ.pop(key, None)
    os.environ.update(env)
    sim = Sim()
    sim.waits = {}
    mod = load_pp(text, sim, "pair")
    hs = {}
    for r in (0, 1):
        sim.tl.rank, sim.tl.stack, sim.tl.step = r, [], 0
        sim.mains[r] = FakeStream(sim, f"main{r}")
        hs[r] = mod.PPHandler(max_num_reqs=8, num_speculative_steps=k, device=torch.device("cpu"))
    st = {"n": 3, "rows": [1, 4, 6], "final": np.ones(3, dtype=bool)}
    sampled = torch.arange(3 * (k + 1), dtype=torch.int64).view(3, k + 1) + 100
    ns_ = torch.tensor([k + 1, 2, 1], dtype=torch.int32)
    nr_ = torch.tensor([0, 2, 3], dtype=torch.int32)
    drafts = torch.arange(8 * k, dtype=torch.int64).view(8, k) + 7000
    sim.tl.rank = 1
    hs[1].broadcast(sampled, ns_, nr_, IB(st))
    hs[1].broadcast_drafts(drafts, IB(st))
    sim.tl.rank = 0
    hs[0].get_prev_sampled_outputs(torch.zeros(8, k, dtype=torch.int64))
    sim.mains[0].ops.append(Op("marker", tag=("forward", 0)))
    hs[0].receive(IB(st))
    for key in (P.ENV_PACK, P.ENV_DEFER):
        os.environ.pop(key, None)
    return sim, hs, sampled


def method_fn(text, cls, name, ns):
    """The REAL function `cls.name` of a source text, compiled alone into namespace `ns` (its globals are faked)."""
    fn = find_method(ast.parse(text), cls, name)
    mod = ast.Module(body=[fn], type_ignores=[])
    exec(compile(mod, f"<{cls}.{name}>", "exec"), ns)  # noqa: S102
    return ns[name]


def parked(h):
    return [i for i, s in enumerate(h.queue) if s is not None and s.event is None]


def t8_shutdown(orig_mr, patched_mr, patched_pp):
    seen = []

    def sync():
        seen.append({"matched": all_sends_matched(sim), "parked": parked(hs[0])})
        raise StopAfterSync()
    ns = {"torch": types.SimpleNamespace(accelerator=types.SimpleNamespace(synchronize=sync))}
    with no_record_stream():
        sim, hs, sampled = parked_pair(patched_pp, {P.ENV_DEFER: "1"})
        check("T8 setup: a parked slot, the peer's send unmatched", parked(hs[0]) == [len(hs[0].queue) - 1]
              and not all_sends_matched(sim))
        fn = method_fn(patched_mr, "GPUModelRunner", "shutdown", dict(ns))
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                fn(types.SimpleNamespace(pp_handler=hs[0]))
            except StopAfterSync:
                pass
        ev = P.parse_log(buf.getvalue())
        fin = [e for e in ev if e.get("key") == "final_shutdown"]
        check("T8 F1: at the device sync the parked slot is posted and every send is matched",
              seen == [{"matched": True, "parked": []}], seen)
        check("T8 F1: flush_shutdown counted once", hs[0].nb4_stats["flush_shutdown"] == 1, hs[0].nb4_stats)
        slot = hs[0].queue[-1]
        check("T8 F1: the flushed slot holds the sent tokens", torch.equal(slot.sampled_tokens, sampled), slot.sampled_tokens)
        check("T8 F1: the final stats line is logged (parse_log) with the shutdown flush",
              len(fin) == 1 and fin[0]["stats"]["flush_shutdown"] == 1 and fin[0]["stats"]["defer_park"] == 1, ev)
        # the original function: the slot stays parked at the sync (the hang H1 condition)
        seen.clear()
        sim, hs, _ = parked_pair(patched_pp, {P.ENV_DEFER: "1"})
        fo = method_fn(orig_mr, "GPUModelRunner", "shutdown", dict(ns))
        try:
            fo(types.SimpleNamespace(pp_handler=hs[0]))
        except StopAfterSync:
            pass
        check("T8 original shutdown syncs with the send unmatched (negative control)",
              len(seen) == 1 and not seen[0]["matched"] and seen[0]["parked"], seen)
        # env off: nothing parked, the patched shutdown adds no op and no log line
        seen.clear()
        sim, hs, _ = parked_pair(patched_pp, {})
        n0 = {r: len(sim.logs[r]) for r in (0, 1)}
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                fn(types.SimpleNamespace(pp_handler=hs[0]))
            except StopAfterSync:
                pass
        check("T8 env off: shutdown adds no op / no line, sync reached", seen == [{"matched": True, "parked": []}]
              and {r: len(sim.logs[r]) for r in (0, 1)} == n0 and "NB4_PPQ" not in buf.getvalue(), (seen, buf.getvalue()[:200]))
        seen.clear()
        try:
            fn(types.SimpleNamespace())                               # a runner without pp_handler (PP off / early)
        except StopAfterSync:
            pass
        try:
            fn(types.SimpleNamespace(pp_handler=None))
        except StopAfterSync:
            pass
        check("T8 no pp_handler / None: straight to the sync", len(seen) == 2)


class FakeLogger:
    def __init__(self):
        self.lines = []

    def __getattr__(self, name):
        return lambda *a, **k: self.lines.append((name, a))


def worker_ns(logger):
    gt = types.SimpleNamespace(stop_reader=lambda: True, drain=lambda **k: None, _MODE="log",
                               GuardFatalError=type("GuardFatalError", (Exception,), {}))
    return {"gc": types.SimpleNamespace(unfreeze=lambda: None), "guard_telemetry": gt, "logger": logger,
            "ensure_kv_transfer_shutdown": None, "ensure_ec_transfer_shutdown": None,
            "current_platform": types.SimpleNamespace(is_cuda_alike=lambda: False),
            "validate_profile_prefix": lambda p: None, "create_worker_profiler": None}


def t9_rpc(orig_gw, patched_gw, patched_pp):
    seen = []

    def sync(tag):
        def f(*a, **k):
            seen.append((tag, all_sends_matched(sim), parked(hs[0]) if hs.get(0) is not None else None))
        return f

    def worker(h, with_runner=True, **kw):
        prof = types.SimpleNamespace(stop=sync("profiler.stop"), shutdown=sync("profiler.shutdown"))
        w = types.SimpleNamespace(profiler_config=types.SimpleNamespace(profiler="torch"), profiler=prof,
                                  elastic_ep_executor=types.SimpleNamespace(shutdown=lambda: None),
                                  _ple_offload_worker_handle=None, **kw)
        if with_runner:
            w.model_runner = types.SimpleNamespace(pp_handler=h, shutdown=sync("model_runner.shutdown"))
        return w

    lg = FakeLogger()
    with no_record_stream():
        for which, text in (("patched", patched_gw), ("original", orig_gw)):
            prof = method_fn(text, "Worker", "profile", worker_ns(lg))
            shut = method_fn(text, "Worker", "shutdown", worker_ns(lg))
            # /stop_profile while the profiler is active: the stop synchronizes the device
            seen.clear()
            sim, hs, _ = parked_pair(patched_pp, {P.ENV_DEFER: "1"})
            prof(worker(hs[0]), is_start=False)
            if which == "patched":
                check("T9 F2: profile stop with a parked slot -> flushed first, the send matched at the sync",
                      seen == [("profiler.stop", True, [])] and hs[0].nb4_stats["flush_rpc"] == 1, (seen, hs[0].nb4_stats))
            else:
                check("T9 original profile stop syncs with the send unmatched (negative control)",
                      len(seen) == 1 and not seen[0][1] and seen[0][2], seen)
            # Worker.shutdown: profiler teardown and runner teardown both sync
            seen.clear()
            sim, hs, _ = parked_pair(patched_pp, {P.ENV_DEFER: "1"})
            shut(worker(hs[0]))
            tags = [t for t, _, _ in seen]
            if which == "patched":
                check("T9 F2: Worker.shutdown with a parked slot -> flushed first, matched at every sync",
                      tags == ["profiler.shutdown", "model_runner.shutdown"] and all(m and p == [] for _, m, p in seen)
                      and hs[0].nb4_stats["flush_rpc"] == 1, (seen, hs[0].nb4_stats))
            else:
                check("T9 original Worker.shutdown syncs with the send unmatched (negative control)",
                      tags == ["profiler.shutdown", "model_runner.shutdown"] and not seen[0][1], seen)
        prof = method_fn(patched_gw, "Worker", "profile", worker_ns(lg))
        shut = method_fn(patched_gw, "Worker", "shutdown", worker_ns(lg))
        # profile start (no sync): the flush also runs, harmless (posts the parked slot now, as the original would)
        sim, hs, _ = parked_pair(patched_pp, {P.ENV_DEFER: "1"})
        started = []
        w = worker(hs[0])
        w.profiler = types.SimpleNamespace(set_output_name=lambda n: None, start=lambda **k: started.append(k))
        ns_ = worker_ns(lg)
        sys.modules.setdefault("vllm.distributed.utils", types.ModuleType("vllm.distributed.utils"))
        sys.modules["vllm.distributed.utils"].get_worker_rank_suffix = lambda global_rank=None: "r0"
        w.rank, w.local_rank = 0, 0
        prof(w, is_start=True, profile_prefix=None, delay_iterations=20, max_iterations=40)
        check("T9 profile start: flushed, the profiler started", parked(hs[0]) == [] and started
              and started[0] == {"delay_iterations": 20, "max_iterations": 40} and hs[0].nb4_stats["flush_rpc"] == 1, started)
        # guards: no model_runner / no pp_handler attribute / pp_handler None / a handler without the method
        seen.clear()
        sim, hs = Sim(), {}
        bad = []
        for w in (worker(None, with_runner=False), worker(None), worker(object())):
            for f in (lambda: prof(w, is_start=False), lambda: shut(w)):
                try:
                    f()
                except Exception as e:  # noqa: BLE001
                    bad.append(repr(e))
        w = worker(None)
        del w.model_runner.pp_handler
        try:
            prof(w, is_start=False)
            shut(w)
        except Exception as e:  # noqa: BLE001
            bad.append(repr(e))
        check("T9 guards: no model_runner / no pp_handler / None / no method never raise", not bad, bad)
        # env off: the handler exists, nothing is parked, the RPC adds no op
        sim, hs, _ = parked_pair(patched_pp, {})
        n0 = {r: len(sim.logs[r]) for r in (0, 1)}
        seen.clear()
        prof(worker(hs[0]), is_start=False)
        shut(worker(hs[0]))
        check("T9 env off: profile / shutdown add no op", {r: len(sim.logs[r]) for r in (0, 1)} == n0
              and all(m for _, m, _ in seen) and hs[0].nb4_stats["flush_rpc"] == 0, seen)
        # a failing flush at Worker.shutdown is logged and the teardown continues
        sim, hs, _ = parked_pair(patched_pp, {P.ENV_DEFER: "1"})

        def boom(site):
            raise RuntimeError("flush failed")
        seen.clear()
        lg.lines.clear()
        shut(worker(types.SimpleNamespace(flush_deferred_receive=boom)))
        check("T9 a failing flush at Worker.shutdown is logged, teardown continues",
              [t for t, _, _ in seen] == ["profiler.shutdown", "model_runner.shutdown"]
              and any(n == "exception" for n, _ in lg.lines), (seen, lg.lines))


class LagTable:
    """The sender's persistent draft table while its broadcast stream lags: a read issued ON a broadcast stream runs
    late, after the main stream's next propose() rewrote every row (here: + 500000, so every id differs)."""

    def __init__(self, sim, now):
        self.sim, self.now, self.later, self.reads, self.rs = sim, now, now + 500000, [], []

    def __getitem__(self, idx):
        name = self.sim.cur_stream().name
        self.reads.append(name)
        out = (self.later if name.startswith("bcast") else self.now)[idx]
        out.record_stream = lambda s: self.rs.append(getattr(s, "name", str(s)))   # this gathered tensor only
        return out


def t10_draft_race(orig_text, patched_text):
    k = 3
    variants = {"original": (orig_text, {}), "off": (patched_text, {}), "defer": (patched_text, {P.ENV_DEFER: "1"}),
                "pack": (patched_text, {P.ENV_PACK: "1"}), "pack+defer": (patched_text, {P.ENV_PACK: "1", P.ENV_DEFER: "1"})}
    for seed in (31, 32):
        sc = make_scenario(seed, k)
        for mode in ("seq", "thread"):
            for vname, (text, env) in variants.items():
                tables = []

                def view(sim, d):
                    t = LagTable(sim, d)
                    tables.append(t)
                    return t
                res = run_sim(text, env, sc, k, mode, tag=f"lag_{vname}_{seed}_{mode}", drafts_view=view)
                bad = ground_truth_ok(res, sc, k)
                reads = {r for t in tables for r in t.reads}
                if vname in ("original", "off"):
                    check(f"T10 {vname}: Merlin's sender reads the table on the broadcast stream -> rewritten rows "
                          f"(negative control) s{seed} {mode}", any(b[1] == "table" for b in bad) and reads == {"bcast1"},
                          f"{bad[:4]} {reads}")
                    continue
                check(f"T10 F3 {vname}: the main-stream snapshot keeps the correct drafts s{seed} {mode}",
                      not bad and not res["errors"] and reads == {"main1"}, f"{bad[:4]} {res['errors'][:2]} {reads}")
                if mode == "seq":
                    rs = [rec for rec in res["sim"].logs[1] if rec[0] == "record_stream"]
                    sends = sum(1 for st in sc if st["kind"] != "empty" and np.asarray(st["final"]).any())
                    idx_rs = [rec for rec in rs if len(rec[1]) == 1 and rec[2] == "torch.int64"]   # = idx_mapping
                    gathered = [r for t in tables for r in t.rs]
                    check(f"T10 F3 {vname}: record_stream(broadcast) on each gathered draft tensor, never on idx_mapping s{seed}",
                          not idx_rs and gathered == ["bcast1"] * sends,
                          f"idx {idx_rs[:3]} gathered {gathered[:4]} tables {len(tables)} sends {sends}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merlin-src", default=DEFAULT_SRC)
    a = ap.parse_args()
    t0 = time.time()
    work = Path(tempfile.mkdtemp(prefix="nb4ppack_"))
    try:
        patched = t1_patch(a.merlin_src, work)
        t2_placement(patched[P.MR_REL])
        t2_worker(patched[P.GW_REL])
        orig_text = (Path(a.merlin_src) / P.PP_REL).read_text(encoding="utf-8")
        patched_text = patched[P.PP_REL].decode("utf-8")
        t3_t4_t5(orig_text, patched_text)
        t6_fail_closed(patched_text)
        t7_readback()
        for name, fn in (("T8", lambda: t8_shutdown((Path(a.merlin_src) / P.MR_REL).read_text(encoding="utf-8"),
                                                    patched[P.MR_REL].decode("utf-8"), patched_text)),
                         ("T9", lambda: t9_rpc((Path(a.merlin_src) / P.GW_REL).read_text(encoding="utf-8"),
                                               patched[P.GW_REL].decode("utf-8"), patched_text)),
                         ("T10", lambda: t10_draft_race(orig_text, patched_text))):
            try:
                fn()
            except Exception:  # noqa: BLE001
                check(f"{name} no exception", False, traceback.format_exc())
            finally:
                for key in (P.ENV_PACK, P.ENV_DEFER):
                    os.environ.pop(key, None)
    except Exception:  # noqa: BLE001
        check("no exception", False, traceback.format_exc())
    finally:
        shutil.rmtree(work, ignore_errors=True)
    n_ok = sum(ok for _, ok, _ in RESULTS)
    print(json.dumps({"checks": len(RESULTS), "pass": n_ok, "fail": len(RESULTS) - n_ok,
                      "secs": round(time.time() - t0, 1),
                      "shas": {"PP_PRE": P.PP_PRE, "PP_POST": P.PP_POST, "MR_PRE": P.MR_PRE, "MR_POST": P.MR_POST,
                               "GW_PRE": P.GW_PRE, "GW_POST": P.GW_POST}},
                     indent=1))
    print("ALL PASS" if n_ok == len(RESULTS) else "SOME FAILED")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
