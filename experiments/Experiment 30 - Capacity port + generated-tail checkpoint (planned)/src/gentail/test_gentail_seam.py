"""CPU unit tests for gentail_seam.py (E30 GENTAIL_CKPT, Option B prototype).

Run: python -I test_gentail_seam.py        (every test, no pytest needed)
 or: python -I -m pytest -q test_gentail_seam.py

What they cover:
  A. patch mechanics on a COPY of Merlin (tree_copy/, refreshed from the pristine tree first): pre / post shas,
     exact-once anchors, apply / verify / revert byte-identity, refusal of a foreign file, fail-safe restore,
     py_compile, every inserted hook guarded, default off.
  B. the controller logic, loaded from the PATCHED tree's vllm/v1/core/gentail_ckpt.py (the shipped bytes), driven
     by fakes of BlockPool / MambaManager / FullAttentionManager / KVCacheManager / Request and a two-rank PP model
     (PP1 commits at the step, PP0 commits at call T + pp_size from a ring of depth pp_size, skipped if the request
     was freed first; batch queue depth 2). Every Mamba key publish is checked at the moment it is written: both
     ranks must hold exactly that position's state in that block, and the block must not have been reused.
  C. data replays: the 677 K100Q turns (savings under the patch's rules) and the grace-pin pressure (G-3). These
     FAIL when the CSVs are absent (no silent skip).
The real classes (vLLM import) are not loaded here: no local venv has the full dependency set; that is the Kaggle
CPU kit's job (see GPU_TEST_SPEC_gentail.md, section 0).
"""

from __future__ import annotations

import csv
import importlib.util
import os
import random
import shutil
import sys
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gentail_seam as G  # noqa: E402

# the pristine Merlin tree (the folder holding vllm/) and the scratch copy the patch is applied to;
# env overrides for the Kaggle CPU kit. The copy is never under experiments/.
PRISTINE = Path(os.environ.get("GENTAIL_PRISTINE", "D:/ARC AGI 2/_scratch/e23/p2/merlin-src"))
TREE = Path(os.environ.get("GENTAIL_TREE_COPY", "D:/ARC AGI 2/_scratch/gentail_ckpt/tree_copy"))
DUCK = Path(os.environ.get(
    "GENTAIL_DUCK_TIMELINE",
    "D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/"
    "Experiment Results/prep_0gpu/duck_timeline",
))
TURNS_CSV = DUCK / "out_analysis2" / "reprefill_per_turn.csv"
GAPS_CSV = DUCK / "out_analysis1" / "gaps.csv"
BS = 4096
UNIT = 64
ENVS = G.ALL_ENVS + ("GENTAIL_CKPT_STATS_EVERY",)


# =========================================================================== helpers


def _refresh_tree():
    """tree_copy/vllm must exist (a copy of the pristine vllm/); put the three files back to pristine."""
    if not (TREE / "vllm").is_dir():
        shutil.copytree(PRISTINE / "vllm", TREE / "vllm")
    for rel in G.PATCHED_RELS:
        shutil.copyfile(PRISTINE / rel, TREE / rel)
        G._drop_pyc(TREE / rel)
    if (TREE / G.NEW_REL).exists():
        (TREE / G.NEW_REL).unlink()


def _applied_tree():
    _refresh_tree()
    G.apply_gentail_patch(TREE)
    return TREE


_MOD = None


def _mod():
    """The controller module from the patched tree (stdlib-only at import)."""
    global _MOD
    if _MOD is None:
        _applied_tree()
        spec = importlib.util.spec_from_file_location("gentail_ckpt_shipped", TREE / G.NEW_REL)
        m = importlib.util.module_from_spec(spec)
        sys.modules["gentail_ckpt_shipped"] = m  # dataclasses resolve the module by name
        spec.loader.exec_module(m)
        _MOD = m
    return _MOD


def _clear_env():
    for e in ENVS:
        os.environ.pop(e, None)


def _require(path):
    assert path.exists(), f"required data file is missing (the test fails, it does not skip): {path}"


# =========================================================================== fakes


class Status:
    def __init__(self, name):
        self.name = name


RUNNING = Status("RUNNING")
FINISHED_STOPPED = Status("FINISHED_STOPPED")
FINISHED_LENGTH = Status("FINISHED_LENGTH_CAPPED")
FINISHED_ABORTED = Status("FINISHED_ABORTED")
PREEMPTED = Status("PREEMPTED")


class FakeBlock:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_cnt = 0
        self.block_hash = None
        self.block_hash_num_tokens = None
        self.is_null = block_id == 0
        self.alloc_gen = 0
        self.content = {}  # rank -> position of the state the last commit stored here

    def __repr__(self):
        return f"B{self.block_id}(ref={self.ref_cnt},h={self.block_hash_num_tokens})"


class FakePool:
    """BlockPool semantics the controller relies on: touch / free_blocks / cache_partial_block, LRU free queue
    (unhashed first, hashed last), eviction of a hashed block on reuse."""

    def __init__(self, n=4000, hash_block_size=UNIT):
        self.num_gpu_blocks = n
        self.hash_block_size = hash_block_size
        self.blocks = [FakeBlock(i) for i in range(n)]
        self.null_block = self.blocks[0]
        self.free = deque(self.blocks[1:])
        self.cached = {}  # (hash, gid) -> block
        self.on_key = None  # test hook: called on every cache_partial_block

    def get_new_block(self):
        blk = self.free.popleft()
        assert blk.ref_cnt == 0
        if blk.block_hash is not None:
            self.cached.pop(blk.block_hash, None)
            blk.block_hash = None
            blk.block_hash_num_tokens = None
        blk.ref_cnt = 1
        blk.alloc_gen += 1
        blk.content = {}
        return blk

    def touch(self, blocks):
        for b in blocks:
            if b.ref_cnt == 0 and not b.is_null:
                self.free.remove(b)
            b.ref_cnt += 1

    def free_blocks(self, blocks):
        for b in blocks:
            b.ref_cnt -= 1
            assert b.ref_cnt >= 0, b
            if b.ref_cnt == 0 and not b.is_null:
                if b.block_hash is None:
                    self.free.appendleft(b)
                else:
                    self.free.append(b)

    def cache_partial_block(self, request, block, num_tokens, kv_cache_group_id, block_size):
        assert num_tokens % self.hash_block_size == 0 and num_tokens > 0
        h = (request.block_hashes[num_tokens // self.hash_block_size - 1], kv_cache_group_id)
        if self.on_key is not None:
            self.on_key(block, num_tokens, kv_cache_group_id)
        self.cached[h] = block
        block.block_hash = h
        block.block_hash_num_tokens = num_tokens
        return h

    def num_free(self):
        return len(self.free)


class FakeSpec:
    prefix_cacheable = True


class FakeMamba:
    def __init__(self, gid, pool):
        self.kv_cache_group_id = gid
        self.block_pool = pool
        self.block_size = BS
        self.req_to_blocks = {}
        self._retained_checkpoints = {}
        self.enable_caching = True
        self.mamba_cache_mode = "align"
        self.num_speculative_blocks = 0
        self.kv_cache_spec = FakeSpec()

    def cache_blocks(self, *a, **k):  # pragma: no cover - never used by the controller
        raise AssertionError


class FakeAttn:
    def __init__(self, gid, pool):
        self.kv_cache_group_id = gid
        self.block_pool = pool
        self.block_size = BS
        self.req_to_blocks = {}
        self.num_cached_block = {}
        self.enable_caching = True
        self.kv_cache_spec = FakeSpec()

    def cache_blocks(self, request, num_tokens, retention_interval=None, *, replay_boundaries=()):
        fake_base_cache_blocks(self, request, num_tokens, retention_interval, replay_boundaries=replay_boundaries)


class FakeRing(FakeAttn):
    """A CircularBuffer-like group: overrides cache_blocks, so the controller must leave it alone."""

    def cache_blocks(self, *a, **k):
        return None


def fake_base_cache_blocks(m, request, num_tokens, retention_interval=None, *, replay_boundaries=()):
    rid = request.request_id
    table = m.req_to_blocks.get(rid, [])
    num_full = num_tokens // m.block_size
    for i in range(m.num_cached_block.get(rid, 0), min(num_full, len(table))):
        blk = table[i]
        if blk.is_null:
            continue
        n = (i + 1) * m.block_size
        h = (request.block_hashes[n // UNIT - 1], m.kv_cache_group_id)
        m.block_pool.cached[h] = blk
        blk.block_hash = h
        blk.block_hash_num_tokens = n
    m.num_cached_block[rid] = max(m.num_cached_block.get(rid, 0), min(num_full, len(table)))


class FakeCoordinator:
    def __init__(self, managers, repref):
        self.single_type_managers = managers
        self.num_reprefillable_tokens = repref


class FakeKVM:
    """KVCacheManager's grace-pin surface (_grace_pins / _grace_block_ids / release_grace_pin / MAX_GRACE_PINS) and
    the order of the patched free(): on_free -> pop -> _take_grace_pin -> free table refs -> after_free."""

    MAX_GRACE_PINS = 16

    def __init__(self, pool, managers, repref):
        self.block_pool = pool
        self.coordinator = FakeCoordinator(managers, repref)
        self.enable_caching = True
        self.enable_grace_pins = True
        self._grace_pins = {}
        self._grace_block_ids = {}
        self._gentail = None

    def release_grace_pin(self, rid):
        blocks = self._grace_pins.pop(rid, None)
        self._grace_block_ids.pop(rid, None)
        if blocks:
            self.block_pool.free_blocks(blocks)
        return len(blocks or ())

    def _grace_eligible(self, r):
        return (self.enable_grace_pins and r.is_finished() and r.status.name not in
                ("FINISHED_ERROR", "FINISHED_IGNORED") and r.num_output_tokens > 0)

    def free(self, r):
        if self._gentail is not None:
            self._gentail.on_free(r)
        blocks = []
        for m in self.coordinator.single_type_managers:
            blocks += list(reversed(m.req_to_blocks.pop(r.request_id, [])))
            if hasattr(m, "num_cached_block"):
                m.num_cached_block.pop(r.request_id, None)
            held = getattr(m, "_retained_checkpoints", {}).pop(r.request_id, None)
            if held is not None:
                blocks.append(held[1])
        if self._grace_eligible(r):
            keep, seen = [], set()
            for b in blocks:
                if b.is_null or b.block_hash is None or id(b) in seen:
                    continue
                seen.add(id(b))
                keep.append(b)
            if keep:
                self.release_grace_pin(r.request_id)
                for b in keep:
                    b.ref_cnt += 1
                self._grace_pins[r.request_id] = keep
                self._grace_block_ids[r.request_id] = frozenset(seen)
                while len(self._grace_pins) > self.MAX_GRACE_PINS:
                    self.release_grace_pin(next(iter(self._grace_pins)))
        self.block_pool.free_blocks([b for b in blocks if not b.is_null])
        if self._gentail is not None:
            self._gentail.after_free()


class FakeRequest:
    def __init__(self, rid, prompt_len, out_len, rng):
        self.request_id = rid
        self.num_prompt_tokens = prompt_len
        self.target_out = out_len
        self.tokens = [rng.randrange(1, 50_000) for _ in range(prompt_len)]
        self.block_hashes = []
        self._rehash()
        self.num_computed_tokens = 0
        self.num_in_flight_tokens = 0
        self.last_sched_seq = 0
        self.status = RUNNING
        self.rng = rng

    def _rehash(self):
        h = self.block_hashes[-1] if self.block_hashes else 17
        for i in range(len(self.block_hashes) * UNIT, len(self.tokens) // UNIT * UNIT, UNIT):
            h = hash((h, tuple(self.tokens[i:i + UNIT])))
            self.block_hashes.append(h)

    def append(self, n):
        self.tokens += [self.rng.randrange(1, 50_000) for _ in range(n)]
        self._rehash()

    @property
    def num_tokens(self):
        return len(self.tokens)

    @property
    def num_output_tokens(self):
        return len(self.tokens) - self.num_prompt_tokens

    def is_finished(self):
        return self.status.name.startswith("FINISHED")


class Sim:
    """Scheduler + two PP ranks. Batch queue depth 2 (schedule while < 2 steps in flight, then process the oldest).
    A request has at most one window in flight. Finished ids ride the next scheduled call (PP0 frees the slot there,
    after popping the ring entry, before its forward): busy engine -> call L + 2, idle engine -> call L + 1."""

    def __init__(self, pp=2, k=3, repref=2, n_mamba=2, seed=1, env=None, pool_blocks=4000, accept=None,
                 truncate=False):
        _clear_env()
        for kk, vv in (env or {}).items():
            os.environ[kk] = vv
        self.mod = _mod()
        self.pool = FakePool(pool_blocks)
        self.mamba = [FakeMamba(i + 1, self.pool) for i in range(n_mamba)]
        self.attn = [FakeAttn(10, self.pool), FakeAttn(11, self.pool)]  # target attention + MTP draft
        self.ring_grp = FakeRing(12, self.pool)
        managers = [self.attn[0]] + self.mamba + [self.ring_grp, self.attn[1]]
        self.kvm = FakeKVM(self.pool, managers, repref)
        self.ctl = self.mod.GenTailController(
            self.kvm, pp_size=pp, spec_query_len=k + 1, max_num_seqs=21, mamba_cls=FakeMamba,
            attn_cls=FakeAttn, base_cache_blocks=fake_base_cache_blocks)
        self.kvm._gentail = self.ctl
        self.pp, self.k = pp, k
        self.rng = random.Random(seed)
        self.accept = accept  # None = random 0..k accepted drafts
        # truncate=False: the stop is the window's last sampled token (no trimmed tokens);
        # True: the stop can be an accepted draft followed by more accepted tokens (trimmed)
        self.truncate = truncate
        self.seq = 0
        self.cur = 0  # current_step: every schedule attempt, empty ones included
        self.inflight = deque()
        self.pp0_ring = deque([None] * pp)
        self.pp0_live = set()
        self.to_finish = []
        self.running = []
        self.hazards = []
        self.published = []
        self.forced = {}  # rid -> list of n_sampled to use next
        self.pool.on_key = self._on_key
        self.mamba_gids = {m.kv_cache_group_id for m in self.mamba}

    # --- the gate-3 check at publish time
    def _on_key(self, block, num_tokens, gid):
        if gid not in self.mamba_gids:
            return
        self.published.append((block.block_id, num_tokens))
        for rank in ("pp0", "pp1"):
            if block.content.get(rank) != num_tokens:
                self.hazards.append(("published_before_commit", rank, block.block_id, num_tokens,
                                     block.content.get(rank)))

    def add(self, req, offset=0):
        req.next_eligible = self.cur + offset
        self.running.append(req)
        return req

    def _ensure_tables(self, r, upto):
        cols = -(-upto // BS)
        for m in self.mamba + self.attn:
            t = m.req_to_blocks.setdefault(r.request_id, [])
            while len(t) < cols:
                t.append(self.pool.get_new_block())

    def _retire(self, r):
        committed = r.num_computed_tokens - r.num_in_flight_tokens
        keep_from = max(0, -(-committed // BS) - 1)
        for m in self.mamba:
            t = m.req_to_blocks.get(r.request_id, [])
            for i in range(min(keep_from, len(t))):
                if not t[i].is_null:
                    blk = t[i]
                    t[i] = self.pool.null_block
                    self.pool.free_blocks([blk])

    def schedule(self, spacing):
        self.cur += 1
        cands = [r for r in self.running if r.num_in_flight_tokens == 0
                 and (spacing is None or self.cur >= r.next_eligible)]
        if not cands:
            return False
        self.seq += 1
        s = self.seq
        wins = []
        for r in cands:
            self._retire(r)
            before = r.num_computed_tokens
            if before == 0:
                qlen = r.num_prompt_tokens
                n_sampled = 1
                decode = False
            else:
                qlen = 1 + self.k
                forced = self.forced.get(r.request_id)
                if forced:
                    n_sampled = forced.pop(0)
                elif self.accept is not None:
                    n_sampled = 1 + self.accept
                else:
                    n_sampled = 1 + self.rng.randint(0, self.k)
                decode = True
                if not self.truncate:
                    n_sampled = max(1, min(n_sampled, r.target_out - r.num_output_tokens))
            self._ensure_tables(r, before + qlen)
            self.ctl.on_schedule(r, qlen, s)
            r.num_computed_tokens += qlen
            r.num_in_flight_tokens += qlen
            r.last_sched_seq = s
            r.next_eligible = self.cur + (spacing or 0)
            wins.append((r, before, qlen, n_sampled, decode))
        self._execute(s, wins)
        self.inflight.append((s, wins))
        return True

    def _execute(self, s, wins):
        # PP0, call s: pop the ring (commit of step s - pp), free finished slots, then forward.
        slot = self.pp0_ring.popleft()
        self.pp0_ring.append(None)
        if slot is not None:
            for rid, blk, gen, pos in slot[1]:
                if rid in self.pp0_live:
                    if blk.alloc_gen != gen:
                        self.hazards.append(("late_write_into_reused_block", blk.block_id, pos))
                    blk.content["pp0"] = pos
        for rid in self.to_finish:
            self.pp0_live.discard(rid)
        self.to_finish = []
        commits = []
        for r, before, qlen, n_sampled, decode in wins:
            self.pp0_live.add(r.request_id)
            if not decode:
                continue
            final = before + min(n_sampled, qlen, self.k + 1)
            b = (before // BS + 1) * BS
            if final >= b:
                for m in self.mamba:
                    blk = m.req_to_blocks[r.request_id][b // BS - 1]
                    blk.content["pp1"] = b  # last rank: same step
                    commits.append((r.request_id, blk, blk.alloc_gen, b))
        self.pp0_ring[-1] = (s, commits)

    def process_one(self):
        s, wins = self.inflight.popleft()
        self.ctl.drain(s)
        for r, before, qlen, n_sampled, decode in wins:
            r.num_in_flight_tokens -= qlen
            if r.is_finished() or r not in self.running:
                continue
            if decode:
                r.num_computed_tokens -= qlen - n_sampled
            self.ctl.on_step_output(r, qlen, n_sampled, s, False)
            room = r.target_out - r.num_output_tokens
            r.append(min(n_sampled, room))  # stop: later accepted tokens are trimmed
            if r.num_output_tokens >= r.target_out:
                r.status = FINISHED_STOPPED
                self.finish(r)

    def finish(self, r):
        self.inflight_at_finish = getattr(self, "inflight_at_finish", {})
        self.inflight_at_finish[r.request_id] = len(self.inflight)  # other microbatches still in flight
        self.running.remove(r)
        self.kvm.free(r)
        self.to_finish.append(r.request_id)

    def abort(self, r):
        assert r.num_in_flight_tokens == 0
        r.status = FINISHED_ABORTED
        self.finish(r)

    def run(self, spacing=2, until=None, max_steps=100_000):
        steps = 0
        while steps < max_steps:
            if until is not None and until():
                return
            while len(self.inflight) < 2 and self.schedule(spacing):
                pass
            if not self.inflight:
                if not self.running:
                    return
                steps += 1
                continue  # an empty step: current_step advanced, no seq
            self.process_one()
            steps += 1

    def flush(self, n=None):
        """The next turn's steps: run a dummy request for n more steps (default pp + 2) so the PP0 ring drains and
        the fence passes; then abort it (its blocks go back, nothing is published for it)."""
        d = self.add(FakeRequest(f"flush{self.seq}", 64, 10 ** 9, random.Random(5)))
        for _ in range(n or self.pp + 2):
            if not self.schedule(None):
                break
            self.process_one()
        while self.inflight:
            self.process_one()
        self.abort(d)


# =========================================================================== A. patch mechanics


def test_patch_apply_verify_revert_roundtrip():
    _clear_env()
    _refresh_tree()
    assert G.verify_gentail_patch(TREE)["state"] == "not_applied"
    assert G.tree_state(TREE) == "ORIGINAL"
    rec = G.apply_gentail_patch(TREE)
    assert rec["ok"] and rec["applied"] == "applied" and rec["py_compile"]["rc"] == 0
    assert G.tree_state(TREE) == "FULLY_PATCHED"
    for rel in G.PATCHED_RELS:
        assert G.sha256_bytes((TREE / rel).read_bytes()) == G.POST[rel]
    assert G.sha256_bytes((TREE / G.NEW_REL).read_bytes()) == G.NEW_SHA
    assert G.apply_gentail_patch(TREE)["applied"] == "already_applied"
    rec = G.revert_gentail_patch(TREE)
    assert rec["ok"] and rec["reverted"] == "reverted"
    for rel in G.PATCHED_RELS:
        assert (TREE / rel).read_bytes() == (PRISTINE / rel).read_bytes()
    assert not (TREE / G.NEW_REL).exists()
    assert G.revert_gentail_patch(TREE)["reverted"] == "not_applied"


def test_patch_refuses_foreign_and_mixed():
    _refresh_tree()
    f = TREE / G.KVM_REL
    f.write_bytes(f.read_bytes() + b"\n# local edit\n")
    before = {rel: (TREE / rel).read_bytes() for rel in G.PATCHED_RELS}
    try:
        G.apply_gentail_patch(TREE)
        raise AssertionError("foreign file accepted")
    except RuntimeError as e:
        assert "PRE_SHA_MISMATCH" in str(e)
    assert {rel: (TREE / rel).read_bytes() for rel in G.PATCHED_RELS} == before
    assert not (TREE / G.NEW_REL).exists()
    _refresh_tree()
    (TREE / G.NEW_REL).write_text("x = 1\n")
    assert G.verify_gentail_patch(TREE)["state"] in ("foreign", "mixed")
    _refresh_tree()


def test_patch_fail_safe_restores_on_compile_failure():
    _refresh_tree()
    orig = G._py_compile
    G._py_compile = lambda py, files: {"rc": 1, "err": "forced"}
    try:
        try:
            G.apply_gentail_patch(TREE)
            raise AssertionError("no failure")
        except RuntimeError as e:
            assert "VERIFY_FAILED" in str(e)
    finally:
        G._py_compile = orig
    for rel in G.PATCHED_RELS:
        assert (TREE / rel).read_bytes() == (PRISTINE / rel).read_bytes()
    assert not (TREE / G.NEW_REL).exists()


def test_anchors_exactly_once_in_pristine():
    for rel, edits in G.EDITS.items():
        text = (PRISTINE / rel).read_text(encoding="utf-8")
        for anchor, repl in edits:
            assert text.count(anchor) == 1, (rel, anchor[:60])
            assert text.count(repl) == 0, (rel, repl[:60])


def test_every_hook_is_guarded_and_default_off():
    """Every inserted executable line is either a class attribute `_gentail = None`, the install pair, or sits
    under `if self._gentail is not None:`. With the env unset, maybe_install_gentail returns None without touching
    the scheduler (an object that raises on any attribute access)."""
    _applied_tree()
    for rel, edits in G.EDITS.items():
        for anchor, repl in edits:
            added = [ln for ln in repl.splitlines() if ln not in anchor.splitlines()]
            code = [ln.strip() for ln in added if ln.strip() and not ln.strip().startswith("#")]
            guarded = False
            for ln in code:
                if ln.startswith("if self._gentail is not None:"):
                    guarded = True
                    continue
                ok = (ln.startswith("_gentail = None") or ln.startswith("from vllm.v1.core.gentail_ckpt import")
                      or ln == "self._gentail = maybe_install_gentail(self)" or guarded)
                assert ok, (rel, ln)
                assert "GENTAIL_CKPT" in repl
    _clear_env()
    mod = _mod()

    class Boom:
        def __getattr__(self, name):
            raise AssertionError("touched " + name)

    assert mod.maybe_install_gentail(Boom()) is None
    os.environ["GENTAIL_CKPT"] = "0"
    assert mod.maybe_install_gentail(Boom()) is None
    _clear_env()


def test_patched_scheduler_compiles_and_hook_order():
    """The hooks sit where the design needs them (text order inside the patched files)."""
    _applied_tree()
    s = (TREE / G.SCH_REL).read_text(encoding="utf-8")
    i_sched = s.index("self._gentail.on_schedule(")
    assert s.index("request.num_computed_tokens += num_scheduled_token") > i_sched
    i_out = s.index("self._gentail.on_step_output(")
    assert s.index("request.num_computed_tokens -= num_rejected") < i_out
    assert s.index("self._drain_deferred_frees()\n            if self._gentail") > 0
    k = (TREE / G.KVM_REL).read_text(encoding="utf-8")
    assert k.count("self._gentail.on_free(request)") == 2 and k.count("self._gentail.after_free()") == 2
    free_body = k[k.index("    def free(self, request: Request)"):k.index("    MAX_GRACE_PINS = 16")]
    assert free_body.index("on_free") < free_body.index("pop_blocks_for_free") < free_body.index("after_free")


# =========================================================================== B. logic


def test_commit_plan_port():
    cp = _mod().commit_plan_boundary
    assert cp(4093, 4, 4, 4, BS) == (4, 4097, 4096)  # crosses
    assert cp(4092, 4, 4, 4, BS) == (4, 4096, 4096)  # ends exactly on the boundary: still stored
    assert cp(4091, 4, 4, 4, BS) == (4, 4095, 0)
    assert cp(4093, 0, 4, 4, BS) == (0, 4093, 0)  # nothing committed, nothing stored
    assert cp(4094, 7, 4, 4, BS) == (4, 4098, 4096)  # capped by query_len
    assert cp(4094, 4, 6, 4, BS) == (4, 4098, 4096)  # capped by SPEC_QUERY_LEN
    assert cp(8190, 3, 4, 4, BS) == (3, 8193, 8192)


def test_position_math_limit_and_reprefillable():
    """b must be <= floor(valid/64)*64 - 64 with valid = min(committed, num_tokens) - reprefillable."""
    mod = _mod()
    for repref in (0, 2):
        sim = Sim(repref=repref)
        r = FakeRequest("R", 1000, 0, random.Random(3))
        r.tokens += [1] * (4200 - 1000)
        r._rehash()
        r.num_computed_tokens = 4199
        r.status = FINISHED_STOPPED
        r.last_sched_seq = 50
        pend = mod.PendingTail("R", 1, 4096, (1, 2), 10)
        sim.ctl._gen["R"] = 1
        assert sim.ctl._finish_reason(r, pend, 1) is None
        valid = 4199 - repref
        assert pend.valid_tokens == valid and pend.limit == valid // 64 * 64 - 64
        # limit >= 4096 <=> floor(valid/64) >= 65 <=> committed - repref >= 4160 <=> total >= 4161 + repref
        for total, ok in ((4161 + repref, True), (4160 + repref, False)):
            r2 = FakeRequest("R2", 1000, 0, random.Random(3))
            r2.tokens += [1] * (total - 1000)
            r2._rehash()
            r2.num_computed_tokens = total - 1
            r2.status = FINISHED_STOPPED
            r2.last_sched_seq = 50
            p2 = mod.PendingTail("R2", 1, 4096, (1, 2), 10)
            assert (sim.ctl._finish_reason(r2, p2, 1) is None) == ok, (repref, total)


def test_g1_attention_key_at_block_ends():
    """G-1: A % 4096 == 0 -> no partial key, the full block A/4096 - 1 must be hashed at A; else the partial key goes
    on block A // 4096. A = 4032 / 4096 / 4160 / 8192."""
    for a in (4032, 4096, 4160, 8192):
        sim = Sim()
        r = FakeRequest("R", 100, 0, random.Random(4))
        r.tokens += [2] * (a + 300 - 100)
        r._rehash()
        sim._ensure_tables(r, a + 300)
        targets = sim.ctl._attention_targets(r, a)
        assert len(targets) == 2  # target attention + MTP draft; the ring group is left alone
        for m, blk in targets:
            t = m.req_to_blocks["R"]
            if a % BS == 0:
                assert blk is None
                assert t[a // BS - 1].block_hash_num_tokens == a
                assert t[a // BS].block_hash is None  # the NEXT block is never keyed
            else:
                assert blk is t[a // BS]
                assert a // BS == 0 or t[a // BS - 1].block_hash_num_tokens == (a // BS) * BS
        assert not sim.ring_grp.req_to_blocks.get("R")


def _turn(sim, rid, prompt, out, spacing, rng_seed=11, fillers=0):
    r = sim.add(FakeRequest(rid, prompt, out, random.Random(rng_seed)))
    # fillers start one step later: with spacing 2 they run in the other microbatch (busy engine)
    fill = [sim.add(FakeRequest(f"{rid}_f{i}", 200, 10 ** 9, random.Random(rng_seed + i + 1)), offset=2)
            for i in range(fillers)]
    sim.run(spacing=spacing, until=lambda: r.is_finished())
    while sim.inflight:
        sim.process_one()
    for f in fill:
        if f in sim.running:
            sim.abort(f)  # fillers never publish (skip_abort)
    return r


def test_busy_two_microbatch_publishes_safely():
    sim = Sim(seed=2)
    r = _turn(sim, "R", 3000, 9000, spacing=2, fillers=2)
    assert sim.inflight_at_finish["R"] >= 1  # busy: a second microbatch was in flight at R's finish
    sim.flush()
    st = sim.ctl.stats
    assert st["published"] == 1, dict(st)
    assert not sim.hazards, sim.hazards
    # position: the newest grid state at or below the limit
    committed_end = 3000 + 9000 - 1
    assert sim.published[0][1] == 8192 and 8192 <= committed_end - 2 - 64
    # the attention partial key at A and the Mamba key at b are in the pool, the blocks under the grace pin
    pins = sim.kvm._grace_pins["R"]
    assert all(any(b.block_hash_num_tokens == 8192 and b in pins for b in [blk]) for blk in
               [sim.pool.blocks[bid] for bid, _ in sim.published])


def test_idle_engine_final_window_crossing_is_skipped():
    """Idle engine: R alone, its LAST window crosses 8192 -> producer rule skips it (PP0 never commits it: the finish
    rides call L + 1, the slot is gone at call L + 2)."""
    sim = Sim(seed=3)
    out = 8192 - 3000 + 2  # the stop lands just after the crossing
    sim.forced["R"] = []
    r = sim.add(FakeRequest("R", 3000, out, random.Random(5)))

    def until():
        return r.is_finished()

    sim.run(spacing=None, until=until)
    while sim.inflight:
        sim.process_one()
    sim.flush()
    st = sim.ctl.stats
    assert st["published"] == 0, dict(st)
    assert not [h for h in sim.hazards if h[0] == "published_before_commit"], sim.hazards
    assert st["skip_final_window"] + st["skip_past_limit"] + st["skip_within_prompt"] >= 1, dict(st)


def test_idle_engine_earlier_crossing_publishes_after_fence():
    """pp_size 2: b <= limit puts the producer >= 16 windows before the end, so the fence (T + 2) has always passed
    at finish (published at once, deferred 0). The deferral needs T + pp_size == L + 1 with b <= limit, i.e. >= 17
    windows of 4 tokens after b: pp_size 24, every window commits 4 tokens, prompt 3000 (a window ends exactly on
    8192 at step T), 23 more windows (L = T + 23): the producer rule passes (T + 24 <= L + 1), the fence T + 24 is
    one step ahead at finish -> queued, deferred; the next turn's first step passes it (PP0 popped T's commit in
    that call before it freed R's slot) -> published with both ranks committed."""
    sim = Sim(seed=4)
    _turn(sim, "R", 3000, 6000, spacing=None)
    assert sim.inflight_at_finish["R"] == 0  # idle: nothing behind R's last step
    st = sim.ctl.stats
    assert st["published"] == 1 and st["deferred"] == 0 and not sim.hazards, dict(st)
    sim = Sim(seed=4, pp=24, accept=3)
    _turn(sim, "R", 3000, 1 + 4 * (1298 + 23), spacing=None)
    st = sim.ctl.stats
    assert st["queued"] == 1 and st["deferred"] == 1 and st["published"] == 0, dict(st)
    sim.flush()
    assert sim.ctl.stats["published"] == 1 and not sim.hazards, (dict(sim.ctl.stats), sim.hazards)
    assert sim.published[0][1] == 8192


def test_negative_control_nofence_shows_the_hazard():
    """GENTAIL_CKPT_NOFENCE=1 publishes at finish without fence and producer rule: in the idle final-window case the
    published block has no PP0 state -> the gate-3 check must see it (else the test could not see the hazard)."""
    found = 0
    for seed in range(20):
        for env, expect in (({"GENTAIL_CKPT_NOFENCE": "1"}, True), ({}, False)):
            sim = Sim(seed=seed, env=env)
            out = 8192 - 3000 + 2
            r = sim.add(FakeRequest("R", 3000, out, random.Random(seed)))
            sim.run(spacing=None, until=lambda: r.is_finished())
            while sim.inflight:
                sim.process_one()
            sim.flush()
            pp0 = [h for h in sim.hazards if h[0] == "published_before_commit" and h[1] == "pp0"]
            if not expect:
                assert not sim.hazards, (seed, sim.hazards)  # the same run with the fence: clean
            elif pp0:
                found += 1
                assert all(h[2] for h in pp0) and not [h for h in sim.hazards if h[1] == "pp1"]
    assert found >= 5, f"the negative control exposed the PP0 hazard in only {found} / 20 runs"
    _clear_env()


def test_abort_at_boundary_publishes_nothing_and_leaks_nothing():
    sim = Sim(seed=6)
    free0 = sim.pool.num_free()
    r = sim.add(FakeRequest("R", 3000, 10 ** 9, random.Random(7)))  # noqa: F841
    # run until the step that crossed 4096 was processed
    sim.run(spacing=None, until=lambda: "R" in sim.ctl._pending)
    while sim.inflight:
        sim.process_one()
    assert sim.ctl._pending["R"].position == 4096
    sim.abort(r)
    sim.flush()
    st = sim.ctl.stats
    assert st["published"] == 0 and st["skip_abort"] == 1, dict(st)
    assert not sim.ctl._queue and not sim.ctl._pending
    assert all(b.ref_cnt == 0 for b in sim.pool.blocks[1:]), [b for b in sim.pool.blocks[1:] if b.ref_cnt]
    assert sim.pool.num_free() == free0


def test_preemption_releases_hold_behind_fence():
    sim = Sim(seed=8)
    r = sim.add(FakeRequest("R", 3000, 10 ** 9, random.Random(9)))
    sim.run(spacing=None, until=lambda: "R" in sim.ctl._pending)
    while sim.inflight:
        sim.process_one()
    pend = sim.ctl._pending["R"]
    held = [sim.pool.blocks[i] for i in pend.block_ids]
    # preemption = free while not finished
    sim.running.remove(r)
    r.status = PREEMPTED
    sim.kvm.free(r)
    assert sim.ctl.stats["skip_not_finished"] == 1
    if pend.producer_step + sim.pp > sim.ctl.processed_seq:
        assert all(b.ref_cnt == 1 for b in held)  # still held until the fence
    sim.flush()
    assert all(b.ref_cnt == 0 for b in held)


def test_index_reuse_before_fence_discards():
    """R finishes (queued, fence not passed: idle), then a NEW request under the same id is scheduled before the
    fence: the drain must discard R's checkpoint (generation changed)."""
    sim = Sim(seed=10, pp=24, accept=3)  # the deferral setup of the previous test: the fence is one step ahead
    _turn(sim, "R", 3000, 1 + 4 * (1298 + 23), spacing=None)
    assert sim.ctl.stats["queued"] == 1 and sim.ctl.stats["published"] == 0
    r2 = sim.add(FakeRequest("R", 500, 50, random.Random(99)))
    sim.run(spacing=None, until=lambda: r2.is_finished())
    while sim.inflight:
        sim.process_one()
    sim.flush()
    st = sim.ctl.stats
    assert st["published"] == 0 and st["skip_generation_at_drain"] == 1, dict(st)
    assert not sim.hazards


def test_index_reuse_while_pending_releases_old_hold():
    sim = Sim(seed=12)
    r = sim.add(FakeRequest("R", 3000, 10 ** 9, random.Random(13)))
    sim.run(spacing=None, until=lambda: "R" in sim.ctl._pending)
    while sim.inflight:
        sim.process_one()
    old = [sim.pool.blocks[i] for i in sim.ctl._pending["R"].block_ids]
    r_new = FakeRequest("R", 100, 10, random.Random(14))
    sim.ctl.on_schedule(r_new, 100, sim.seq + 1)  # same id, new object
    assert sim.ctl.stats["index_reuse"] == 1 and "R" not in sim.ctl._pending
    sim.running.remove(r)
    r.status = FINISHED_STOPPED
    sim.kvm.free(r)
    sim.flush()
    assert all(b.ref_cnt == 0 for b in old)


def test_stop_truncation_default_and_strict():
    """The stop is an accepted draft and later drafts were accepted (committed >= num_tokens). Default (strict, the
    audit rule): discarded. GENTAIL_CKPT_STRICT_TRUNC=0: the producer rule + the limit decide (the grid state was
    produced >= 16 windows earlier) and it is published, still with both ranks committed."""
    for strict, expect in ((True, 0), (False, 1)):
        env = {} if strict else {"GENTAIL_CKPT_STRICT_TRUNC": "0"}
        sim = Sim(seed=15, env=env, accept=3, truncate=True)  # every window commits 4 tokens
        r = _turn(sim, "R", 3001, 6002, spacing=2, fillers=1)
        assert r.num_computed_tokens - r.num_in_flight_tokens >= r.num_tokens, "no truncation happened"
        sim.flush()
        st = sim.ctl.stats
        assert st["stop_truncated"] == 1 and st["published"] == expect, (strict, dict(st))
        assert not sim.hazards
    _clear_env()


def test_finish_status_rule_pr50551():
    """Only FINISHED_STOPPED publishes; FINISHED_LENGTH_CAPPED (and abort, tested above) are discarded."""
    for status, expect in ((FINISHED_STOPPED, 1), (FINISHED_LENGTH, 0), (Status("FINISHED_REPETITION"), 0)):
        sim = Sim(seed=22)
        r = sim.add(FakeRequest("R", 3000, 6000, random.Random(23)))
        sim.run(spacing=None, until=lambda: r.num_output_tokens >= 6000 - 8)
        while sim.inflight:
            sim.process_one()
        r.status = status
        sim.finish(r)
        sim.flush()
        st = sim.ctl.stats
        assert st["published"] == expect and not sim.hazards, (status.name, dict(st))
        if not expect:
            assert st["skip_status_" + status.name.lower()] == 1
        for rid in list(sim.kvm._grace_pins):
            sim.kvm.release_grace_pin(rid)
        assert all(b.ref_cnt == 0 for b in sim.pool.blocks[1:])


def test_mtp_all_accepted_and_all_rejected():
    for accept in (0, 3):
        sim = Sim(seed=24, accept=accept)
        _turn(sim, "R", 3000, 6000, spacing=2, fillers=1)
        sim.flush()
        st = sim.ctl.stats
        assert st["published"] == 1 and st["plan_mismatch"] == 0 and not sim.hazards, (accept, dict(st))
        assert sim.published[0][1] == 8192


def test_per_step_guard_untouched():
    """configure_recoverssm_pp_checkpoints stays in the patched scheduler, and the controller never writes
    publish_decode_checkpoints (the finish-time path is separate)."""
    _applied_tree()
    sch = (TREE / G.SCH_REL).read_text(encoding="utf-8")
    assert sch.count("manager.publish_decode_checkpoints = False") == 1
    assert sch.index("            configure_recoverssm_pp_checkpoints(") < sch.index("maybe_install_gentail(self)")
    mod_src = (TREE / G.NEW_REL).read_text(encoding="utf-8")
    assert ".publish_decode_checkpoints =" not in mod_src  # no attribute write (the docstring names it)


def test_superseded_and_retired_holds():
    """Two crossings (4096, 8192): the 4096 hold is released when 8192 is recorded (behind the fence); a retired
    pending block stays out of the free queue (table ref gone, hold ref left)."""
    sim = Sim(seed=16)
    r = sim.add(FakeRequest("R", 3000, 10 ** 9, random.Random(17)))
    sim.run(spacing=2, until=lambda: "R" in sim.ctl._pending)
    while sim.inflight:
        sim.process_one()
    first = [sim.pool.blocks[i] for i in sim.ctl._pending["R"].block_ids]
    sim.run(spacing=2, until=lambda: r.num_computed_tokens > 4096 + 200)
    while sim.inflight:
        sim.process_one()
    sim.schedule(2)  # allocate -> retire column 0
    sim.process_one()
    assert all(m.req_to_blocks["R"][0].is_null for m in sim.mamba)
    assert all(b.ref_cnt == 1 and b not in sim.pool.free for b in first)  # retired, still held
    sim.run(spacing=2, until=lambda: sim.ctl._pending["R"].position == 8192)
    while sim.inflight:
        sim.process_one()
    sim.flush()
    assert sim.ctl.stats["superseded"] == 1 and all(b.ref_cnt == 0 for b in first)
    r.status = FINISHED_STOPPED
    sim.running.remove(r)
    sim.kvm.free(r)
    sim.flush()


def test_grace_pin_handoff_lru_and_adoption():
    sim = Sim(seed=18)
    _turn(sim, "R", 3000, 6000, spacing=2, fillers=1)
    sim.flush()
    assert sim.ctl.stats["published"] == 1 and sim.ctl.stats["published_pinned"] == 1
    pins = sim.kvm._grace_pins["R"]
    pub = [sim.pool.blocks[i] for i, _ in sim.published]
    assert all(b in pins and id(b) in sim.kvm._grace_block_ids["R"] for b in pub)
    assert all(b.ref_cnt == 1 for b in pub)
    # next turn adopts: a request whose Mamba table resumes at 8192 from the published blocks
    nxt = FakeRequest("N", 9500, 10, random.Random(19))
    nxt.num_computed_tokens = 8192
    for m, b in zip(sim.mamba, pub):
        m.req_to_blocks["N"] = [sim.pool.null_block, b]
    sim.ctl.on_schedule(nxt, 9500 - 8192, sim.seq + 1)
    assert sim.ctl.stats["adopted"] == 1 and sim.ctl.stats["adopted_tokens"] == 8192
    for m in sim.mamba:
        m.req_to_blocks.pop("N")
    # release the grace pin: every block back exactly once, hashed -> evict-last end of the free queue
    sim.kvm.release_grace_pin("R")
    assert all(b.ref_cnt == 0 and b.block_hash is not None for b in pub)
    assert all(list(sim.pool.free).index(b) > len(sim.pool.free) - 50 for b in pub)
    # no grace pin -> LRU
    sim2 = Sim(seed=18)
    sim2.kvm.enable_grace_pins = False
    _turn(sim2, "R", 3000, 6000, spacing=2, fillers=1)
    sim2.flush()
    assert sim2.ctl.stats["published_lru"] == 1 and not sim2.hazards


def test_evicted_counter():
    sim = Sim(seed=20, pool_blocks=120)
    _turn(sim, "R", 3000, 6000, spacing=2, fillers=1)
    sim.flush()
    sim.kvm.release_grace_pin("R")
    # allocate every free block: the published (hashed) ones are evicted last but evicted
    taken = [sim.pool.get_new_block() for _ in range(sim.pool.num_free())]
    sim.ctl._sweep_evicted()
    assert sim.ctl.stats["evicted"] == 2, dict(sim.ctl.stats)
    sim.pool.free_blocks(taken)


def test_pin_budget_caps_pinned_blocks():
    sim = Sim(seed=21, env={"GENTAIL_CKPT_PIN_BUDGET_FRAC": "0.01"})  # 40 blocks
    assert sim.ctl.pin_budget == 40 and sim.kvm.MAX_GRACE_PINS == 21
    for i in range(6):
        _turn(sim, f"R{i}", 3000, 6000, spacing=2, rng_seed=30 + i, fillers=1)
        sim.flush()
    pinned = sum(len(v) for v in sim.kvm._grace_pins.values())
    assert pinned <= 40 or len(sim.kvm._grace_pins) == 1, pinned
    assert sim.ctl.stats["pin_budget_released"] >= 1
    assert f"R5" in sim.kvm._grace_pins  # the newest stays
    _clear_env()


def test_install_gates():
    mod = _mod()
    _clear_env()
    os.environ["GENTAIL_CKPT"] = "1"

    class Cfg:
        use_recoverssm = True

    class Par:
        pipeline_parallel_size = 2

    class SCfg:
        max_num_seqs = 21

    class Sched:
        def __init__(self, defer, recover=True):
            pool = FakePool(100)
            self.kv_cache_manager = FakeKVM(pool, [FakeAttn(0, pool), FakeMamba(1, pool)], 2)
            self.cache_config = Cfg()
            self.cache_config.use_recoverssm = recover
            self.parallel_config = Par()
            self.scheduler_config = SCfg()
            self.num_spec_tokens = 3
            self.defer_block_free = defer

    # the real install imports vllm's manager classes: give it the fakes under those names
    fake_mod = type(sys)("vllm.v1.core.single_type_kv_cache_manager")
    fake_mod.MambaManager = FakeMamba
    fake_mod.FullAttentionManager = FakeAttn
    fake_mod.SingleTypeKVCacheManager = type("S", (), {"cache_blocks": staticmethod(fake_base_cache_blocks)})
    saved = {k: sys.modules.get(k) for k in ("vllm", "vllm.v1", "vllm.v1.core",
                                             "vllm.v1.core.single_type_kv_cache_manager")}
    for k in ("vllm", "vllm.v1", "vllm.v1.core"):
        sys.modules[k] = type(sys)(k)
    sys.modules["vllm.v1.core.single_type_kv_cache_manager"] = fake_mod
    try:
        s = Sched(defer=True)
        ctl = mod.maybe_install_gentail(s)
        assert ctl is not None and s.kv_cache_manager._gentail is ctl and ctl.pp_size == 2
        assert ctl.spec_query_len == 4 and s.kv_cache_manager.MAX_GRACE_PINS == 21
        try:
            mod.maybe_install_gentail(Sched(defer=False))
            raise AssertionError("defer_block_free off accepted")
        except RuntimeError as e:
            assert "defer_block_free" in str(e)
        assert mod.maybe_install_gentail(Sched(defer=True, recover=False)) is None  # inactive, no change
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        _clear_env()


def test_long_random_soak_no_hazard_no_leak():
    """Many turns, random acceptance, busy and idle mixes: no published state is ever missing on a rank or written
    into a reused block, and every ref returns."""
    for seed in range(6):
        sim = Sim(seed=100 + seed, truncate=bool(seed % 2))  # odd seeds: stop-as-accepted-draft truncations
        rng = random.Random(seed)
        for t in range(8):
            _turn(sim, f"T{seed}_{t}", rng.randint(500, 9000), rng.randint(100, 14_000),
                  spacing=rng.choice([None, 2]), rng_seed=seed * 100 + t, fillers=rng.choice([0, 1, 2]))
        sim.flush(6)
        assert not sim.hazards, (seed, sim.hazards[:3])
        for rid in list(sim.kvm._grace_pins):
            sim.kvm.release_grace_pin(rid)
        assert not sim.ctl._queue and not sim.ctl._pending, (seed, sim.ctl._queue)
        assert all(b.ref_cnt == 0 for b in sim.pool.blocks[1:]), seed
        assert sim.ctl.stats["published"] >= 1, (seed, dict(sim.ctl.stats))


# =========================================================================== C. data replays


def _turn_rows():
    _require(TURNS_CSV)
    with open(TURNS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 677, len(rows)
    return rows


def test_k100q_savings_under_patch_rules():
    """The 677 K100Q later turns under the patch's own rules: valid = P' + G' - 1 - 2 (reprefillable 2),
    b = the newest 4096 grid state <= floor(valid/64)*64 - 64 (and below the next prompt), b > the previous prompt,
    only FINISHED_STOPPED finishes (prev_finish 'length' = length-capped: discarded). The producer rule never binds
    here (b <= limit puts the crossing >= 16 windows before the end); the check is kept for clarity. Strict stop
    truncation is not in the table (measured on the GPU)."""
    rows = _turn_rows()
    perfect = sum(int(r["Gp"]) for r in rows)
    assert perfect == 2_430_527
    saved = helped = within_prompt = length_capped = 0
    for r in rows:
        p, c, pp_, gp = int(r["P"]), int(r["C"]), int(r["Pp"]), int(r["Gp"])
        if r["prev_finish"] == "length":  # FINISHED_LENGTH_CAPPED: discarded (PR #50551 rule)
            length_capped += 1
            continue
        committed = pp_ + gp - 1
        valid = committed - 2
        limit = valid // UNIT * UNIT - UNIT
        b = min(limit, (p - 1) // UNIT * UNIT) // BS * BS
        if b <= pp_:
            within_prompt += int(b > c)
            continue
        if committed - b < 4:  # crossing inside the final window: producer rule
            continue
        if b > c:
            saved += b - c
            helped += 1
    share = saved / perfect
    print(f"K100Q under patch rules: saved {saved} of {perfect} ({share:.2%}), {helped} turns helped, "
          f"{within_prompt} turns lose a <=127-token gain to the b > prompt rule, "
          f"{length_capped} turns follow a length-capped finish (discarded)")
    assert 0.65 < share < 0.75, share


def test_g3_grace_pin_pressure_simulation():
    """G-3: how many tokens sit under grace pins at once (Duck K100Q gaps: finish -> next request). A pin holds ALL
    hashed blocks of the finished turn (~ its next prompt). Pool = 1,652,197 tokens (K100Q boot line)."""
    _require(GAPS_CSV)
    with open(GAPS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 600
    pool_tokens = 1_652_197
    out = {}
    for extra in (0.0, 2.0):  # +2 s: the next turn waits for admission
        for cap in (16, 21):
            ev = []
            for r in rows:
                t0 = float(r["t_rel"])
                tok = float(r["next_prompt"] or 0)
                ev.append((t0, t0 + float(r["gap_sec"]) + extra, tok))
            ev.sort()
            live = []  # (start, end, tok)
            peak_tok = peak_n = 0
            for t0, t1, tok in ev:
                live = [x for x in live if x[1] > t0]
                live.append((t0, t1, tok))
                while len(live) > cap:
                    live.pop(0)
                peak_n = max(peak_n, len(live))
                peak_tok = max(peak_tok, sum(x[2] for x in live))
            out[(extra, cap)] = (peak_n, peak_tok, peak_tok / pool_tokens)
    for k, v in out.items():
        print(f"grace pins extra_wait={k[0]}s cap={k[1]}: peak pins {v[0]}, peak pinned tokens {v[1]:.0f} "
              f"({v[2]:.1%} of the pool)")
    assert out[(0.0, 21)][0] <= 21


def test_publish_log_and_readback():
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sim = Sim(seed=25, env={"GENTAIL_CKPT_LOG_PUBLISH": "1", "GENTAIL_CKPT_STATS_EVERY": "1"})
        _turn(sim, "R", 3000, 6000, spacing=2, fillers=1)
        sim.flush()
    recs = G.parse_log(buf.getvalue())
    pubs = [r for r in recs if r["event"] == "publish"]
    assert len(pubs) == 1 and pubs[0]["position"] == 8192 and pubs[0]["request_id"] == "R"
    assert pubs[0]["processed_seq"] >= pubs[0]["fence"] and len(pubs[0]["block_ids"]) == 2
    rb = G.readback(buf.getvalue())
    assert rb["label"] == "ABSENT" and rb["last_stats"]["stats"]["published"] == 1  # no install line in the fakes
    _clear_env()


def test_missing_csv_fails_not_skips():
    global TURNS_CSV, GAPS_CSV
    saved = TURNS_CSV, GAPS_CSV
    TURNS_CSV = GAPS_CSV = HERE / "no_such_dir" / "missing.csv"
    try:
        for fn in (test_k100q_savings_under_patch_rules, test_g3_grace_pin_pressure_simulation):
            try:
                fn()
                raise RuntimeError("passed without its CSV")
            except AssertionError as e:
                assert "missing" in str(e)
    finally:
        TURNS_CSV, GAPS_CSV = saved


# =========================================================================== runner

if __name__ == "__main__":
    fns = [(k, v) for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print("PASS", name)
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback

            traceback.print_exc()
            print("FAIL", name, repr(e)[:300])
    _clear_env()
    print(f"{len(fns) - failed} / {len(fns)} passed")
    sys.exit(1 if failed else 0)
