"""CPU unit tests for gentail_logic.py. Run: python -I -m pytest -q test_gentail_logic.py
(or python -I test_gentail_logic.py, which runs every test function without pytest).

The data-replay tests read the K100Q per-turn re-prefill table
(Experiment 25 .../prep_0gpu/duck_timeline/out_analysis2/reprefill_per_turn.csv)
and FAIL if it is not there (G-4: a missing file must never pass silently)."""

from __future__ import annotations

import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gentail_logic import (  # noqa: E402
    PendingPublishQueue,
    ProducedState,
    Rank0CommitModel,
    current_prompt_hit,
    generated_tail_positions,
    grid_checkpoint,
    pp_fence_seq,
    ring_slot,
    state_is_pp_safe,
)

UNIT = 64
BLOCK = 4096
DATA = os.path.join(os.environ.get(
    "GENTAIL_DUCK_TIMELINE",
    "D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/"
    "Experiment Results/prep_0gpu/duck_timeline"), "out_analysis2", "reprefill_per_turn.csv")


def _rows():
    assert os.path.exists(DATA), f"required data file is missing (fail, not skip): {DATA}"
    with open(DATA, newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- 1


def test_current_hit_is_64_to_127_short():
    for p in range(130, 200_000, 37):
        h = current_prompt_hit(p)
        assert h % UNIT == 0
        assert 64 <= p - h <= 127, (p, h)


def test_current_hit_matches_k100q_exactly():
    rows = _rows()
    mismatches = [
        r for r in rows if current_prompt_hit(int(r["Pp"])) != int(r["C"])
    ]
    # The source reading predicts every one of the 677 cached counts exactly.
    assert not mismatches, mismatches[:3]
    assert len(rows) == 677


# --------------------------------------------------------------------------- 2


def test_tail_positions_normal_finish():
    # last sampled token (e.g. <|im_end|>) is never fed: num_computed = num_tokens - 1
    for n_tokens in range(300, 20_000, 113):
        t = generated_tail_positions(n_tokens - 1, n_tokens)
        assert t.attn_key % UNIT == 0 and t.mamba_target % UNIT == 0
        assert t.mamba_target == t.attn_key - UNIT
        assert t.valid_tokens == n_tokens - 3
        # the hit stays 66..129 tokens short of the generation end
        assert 66 <= n_tokens - t.hit <= 66 + 63 + 64


def test_tail_positions_stop_was_an_accepted_draft():
    # window [t0,d1,d2,d3] all accepted, d1 is the stop token: output keeps d1 only,
    # num_computed counts d2,d3 too (scheduler.py:2800-2817 trims tokens, not counts).
    c = 10_000
    num_tokens = c + 2
    num_computed = c + 4
    t = generated_tail_positions(num_computed, num_tokens)
    assert t.valid_tokens == num_tokens - 2  # capped by the true token count
    assert t.hit <= num_tokens - 66


def test_grid_checkpoint():
    assert grid_checkpoint(4095, BLOCK) == 0
    assert grid_checkpoint(4096, BLOCK) == 4096
    assert grid_checkpoint(20_000, BLOCK) == 16_384


# --------------------------------------------------------------------------- 3


def _simulate_decode(rng, start, n_steps, k=3):
    """Decode windows of 1..k+1 committed tokens; return (crossings, final_pos)
    where crossings = [(unit_position, step_index)]."""
    pos = start
    crossings = []
    for step in range(n_steps):
        adv = rng.randint(1, k + 1)
        nxt = pos + adv
        u = (pos // UNIT + 1) * UNIT
        if nxt >= u:
            crossings.append((u, step))
        pos = nxt
    return crossings, pos


def test_ring_coverage():
    """Ring of R tail pages, written at every unit crossing; a finish-time
    checkpoint is usable if the target position is still in the ring and was
    produced before the final step (PP rule). Expected: R=2 ~94-97 %, R=3 ~100 %."""
    rng = random.Random(7)
    cover = {2: 0, 3: 0, 4: 0}
    trials = 20_000
    for _ in range(trials):
        start = rng.randint(1_000, 60_000)
        crossings, final = _simulate_decode(rng, start, rng.randint(40, 4_000))
        num_tokens = final + 1  # last sampled token not fed
        t = generated_tail_positions(final, num_tokens)
        for r in cover:
            ring = {}
            for u, step in crossings:
                ring[ring_slot(u, UNIT, r)] = (u, step)
            slot = ring.get(ring_slot(t.mamba_target, UNIT, r))
            if slot is not None and slot[0] == t.mamba_target:
                cover[r] += 1
    frac = {r: cover[r] / trials for r in cover}
    # The ~0.1 % misses at R=3 are generations that never crossed the target unit
    # (target below the decode start): those fall back to the prompt checkpoint.
    assert frac[3] >= 0.998 and frac[4] == frac[3], frac
    assert 0.90 < frac[2] < 0.995, frac


def test_target_never_produced_in_final_window():
    """mamba_target <= valid - 64 and a window commits <= 4 tokens, so the target's
    crossing is always >= 15 steps before the end: the PP 'produced before the last
    step' rule never costs a hit for option C."""
    for final in range(200, 50_000, 7):
        t = generated_tail_positions(final, final + 1)
        assert final - t.mamba_target >= 64 + 2


# --------------------------------------------------------------------------- 4


def test_pp_fence_model_busy_engine():
    """Busy engine, batch queue depth 2: request R decodes at seqs 10, 12, ...;
    finishes at its step L=20; its id reaches the worker in call 22's finished set.
    Every R commit, the last one included, is applied on the first rank."""
    pp = 2
    m = Rank0CommitModel(pp)
    m.execute(1, [], [], ["R", "X"])
    for seq in range(2, 30):
        sched = []
        if seq % 2 == 0 and seq <= 20:
            sched.append("R")
        if seq % 2 == 1:
            sched.append("X")
        finished = ["R"] if seq == 22 else []
        m.execute(seq, sched, finished, [])
    r_steps = [s for s in range(2, 21, 2)]
    for s in r_steps:
        assert ("R", s) in m.applied, s
    assert not any(rid == "R" for rid, _ in m.skipped)


def test_pp_fence_model_idle_engine_skips_last_commit():
    """Idle engine: only R runs; nothing was in flight behind step L=20, so the
    finish rides in call 21. The first rank pops step 19's (empty) slot, frees R,
    and step 20's commit is later SKIPPED (req freed). A state produced in step 20
    must therefore never be published; states from steps < 20 were applied."""
    pp = 2
    m = Rank0CommitModel(pp)
    m.execute(1, [], [], ["R"])
    for seq in range(2, 21):
        m.execute(seq, ["R"], [], [])
    m.execute(21, ["Y"], ["R"], ["Y"])  # next call carries R's finish
    m.execute(22, ["Y"], [], [])
    m.execute(23, ["Y"], [], [])
    assert ("R", 20) in m.skipped
    for s in range(2, 20):
        assert ("R", s) in m.applied
    assert not state_is_pp_safe(ProducedState("R", 4096, 20), request_last_sched_seq=20)
    assert state_is_pp_safe(ProducedState("R", 4096, 19), request_last_sched_seq=20)


def test_fence_queue():
    q = PendingPublishQueue()
    q.push(pp_fence_seq(10, 2), "a")
    q.push(pp_fence_seq(11, 2), "b")
    assert q.drain(11) == []
    assert q.drain(12) == ["a"]
    assert q.drain(20) == ["b"]
    assert q.drain(30) == []


# --------------------------------------------------------------------------- 5


def _savings(rows, mode, ring=3):
    total = 0
    helped = 0
    for r in rows:
        p, c, pp_, gp = int(r["P"]), int(r["C"]), int(r["Pp"]), int(r["Gp"])
        fed = pp_ + gp - 1
        t = generated_tail_positions(fed, pp_ + gp)
        h = min(t.hit, (p - 1) // UNIT * UNIT)
        if mode == "B":
            h = grid_checkpoint(h, BLOCK)
        elif mode == "session":
            h = min(fed, p - 1)
        if h > c:
            total += h - c
            helped += 1
    return total, helped


def test_k100q_savings():
    rows = _rows()
    a = sum(int(r["Gp"]) for r in rows)
    assert a == 2_430_527
    sc, _ = _savings(rows, "C")
    sb, nb = _savings(rows, "B")
    ss, _ = _savings(rows, "session")
    print(f"perfect(a)={a}  C(64-unit ring)={sc} ({sc / a:.3%})  "
          f"B(4096 grid)={sb} ({sb / a:.3%}, {nb} turns)  session={ss} ({ss / a:.3%})")
    assert sc / a > 0.995
    assert 0.65 < sb / a < 0.75
    assert ss >= sc


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
