"""Randomized interleaving model of the P5 flag/epoch protocol (host_ar.cu), run on CPU.

Model: W ranks, each runs a stream of K calls (kernels) one after another (stream order). Every call has B blocks
(B may change between calls, identical on all ranks, like the real `blocks` argument). A block is a small state
machine with the kernel's steps:
    read seq -> P1 write stage[r][b] := epoch -> flag0[r][b] := epoch -> wait flag0[j][b] == epoch (j != r)
    -> P2 read stage[j][b] (must hold this epoch's data) -> write red[r][b] := epoch -> flag1[r][b] := epoch
    -> wait flag1[j][b] == epoch -> P3 read red[j][b] (must hold this epoch) -> arrive (last block: seq := epoch)
Each step is atomic and stores are seen in program order (what fence.sc.sys + acquire give the real kernel). A random
scheduler picks any runnable block of any rank; up to `resident` blocks of a rank run at once (a block occupies a slot
until it finishes, a free slot takes a random not-started block; the next kernel of a rank starts only after the previous one finished all blocks).
Checks: (1) every P2/P3 read sees data tagged with the reader's epoch (no overwrite by a faster rank, no stale read),
(2) a wait never sees flag > epoch (the kernel's E_DESYNC never fires in a correct run), (3) no deadlock while
resident >= B, (4) all ranks finish with seq == K. With resident < B the model reports whether a deadlock was
reachable (the reason the binding caps the grid at the co-resident block count).
Prints SIM_JSON {...}; exit 0 when every resident >= B trial passes.
"""
import json
import random
import sys


def run_trial(rng, W, K, blist, resident):
    seq = [0] * W
    call = [0] * W                      # index of the call each rank is executing
    flags = {}                          # (phase, rank, block) -> epoch
    stage = {}                          # (rank, block) -> epoch tag of the data
    red = {}
    blocks = {}                         # rank -> list of block states for the current call
    errors = []

    def start_call(r):
        if call[r] >= K:
            blocks[r] = None
            return
        B = blist[call[r]]
        blocks[r] = [{"b": b, "pc": 0, "e": None, "B": B} for b in range(B)]

    for r in range(W):
        start_call(r)

    def runnable(r, st):
        """Return True if the block can make progress now."""
        pc, e, b = st["pc"], st["e"], st["b"]
        if pc == 3:   # wait flag0
            return all(flags.get((0, j, b), 0) >= e for j in range(W) if j != r)
        if pc == 7:   # wait flag1
            return all(flags.get((1, j, b), 0) >= e for j in range(W) if j != r)
        return pc < 10

    def step(r, st):
        pc, b = st["pc"], st["b"]
        if pc == 0:
            st["e"] = seq[r] + 1
        elif pc == 1:
            stage[(r, b)] = st["e"]
        elif pc == 2:
            flags[(0, r, b)] = st["e"]
        elif pc == 3:
            for j in range(W):
                if j != r and flags.get((0, j, b), 0) > st["e"]:
                    errors.append(f"desync flag0 r{r} b{b} e{st['e']}")
        elif pc == 4:
            for j in range(W):
                if j != r and stage.get((j, b)) != st["e"]:
                    errors.append(f"P2 stale/overwritten stage[{j}][{b}]={stage.get((j, b))} want {st['e']}")
        elif pc == 5:
            red[(r, b)] = st["e"]
        elif pc == 6:
            flags[(1, r, b)] = st["e"]
        elif pc == 7:
            for j in range(W):
                if j != r and flags.get((1, j, b), 0) > st["e"]:
                    errors.append(f"desync flag1 r{r} b{b} e{st['e']}")
        elif pc == 8:
            for j in range(W):
                if j != r and red.get((j, b)) != st["e"]:
                    errors.append(f"P3 stale/overwritten red[{j}][{b}]={red.get((j, b))} want {st['e']}")
        elif pc == 9:
            st["arrived"] = True
        st["pc"] = pc + 1

    steps = 0
    while True:
        cands = []
        for r in range(W):
            bl = blocks.get(r)
            if not bl:
                continue
            # residency: at most `resident` started-and-unfinished blocks; a free slot takes a RANDOM not-started
            # block (launch order is not guaranteed under SM contention), started blocks stay until they finish
            started = [st for st in bl if st.get("started") and st["pc"] < 10]
            waiting = [st for st in bl if not st.get("started")]
            while len(started) < resident and waiting:
                st = waiting.pop(rng.randrange(len(waiting)))
                st["started"] = True
                started.append(st)
            live = started
            for st in live:
                if runnable(r, st):
                    cands.append((r, st))
        if not cands:
            break
        r, st = rng.choice(cands)
        step(r, st)
        steps += 1
        bl = blocks[r]
        if all(s["pc"] >= 10 for s in bl):
            seq[r] = bl[0]["e"]         # last block stores seq = epoch
            assert all(s["e"] == bl[0]["e"] for s in bl), "blocks of one call disagree on the epoch"
            call[r] += 1
            start_call(r)
    finished = all(blocks.get(r) is None for r in range(W))
    return {"finished": finished, "errors": errors[:5], "n_errors": len(errors), "steps": steps,
            "seq": seq, "deadlock": not finished}


def main():
    rng = random.Random(20261029)
    res = {"trials": 0, "ok": 0, "fail": [], "partial_residency": {"trials": 0, "deadlocks": 0}}
    for t in range(400):
        W = rng.choice([2, 4, 4, 4, 8])
        K = rng.randint(1, 12)
        blist = [rng.randint(1, 6) for _ in range(K)]
        out = run_trial(rng, W, K, blist, resident=max(blist))
        res["trials"] += 1
        good = out["finished"] and out["n_errors"] == 0 and out["seq"] == [K] * W
        res["ok"] += int(good)
        if not good and len(res["fail"]) < 5:
            res["fail"].append({"W": W, "K": K, "blist": blist, **out})
    for t in range(200):   # residency below the grid: can it deadlock? (documented requirement)
        W = 4
        K = rng.randint(1, 6)
        blist = [rng.randint(3, 6) for _ in range(K)]
        out = run_trial(rng, W, K, blist, resident=2)
        res["partial_residency"]["trials"] += 1
        res["partial_residency"]["deadlocks"] += int(out["deadlock"])
        if out["n_errors"]:
            res["fail"].append({"partial": True, **out})
    res["all_ok"] = res["ok"] == res["trials"] and not res["fail"]
    print("SIM_JSON " + json.dumps(res))
    sys.exit(0 if res["all_ok"] else 1)


if __name__ == "__main__":
    main()
