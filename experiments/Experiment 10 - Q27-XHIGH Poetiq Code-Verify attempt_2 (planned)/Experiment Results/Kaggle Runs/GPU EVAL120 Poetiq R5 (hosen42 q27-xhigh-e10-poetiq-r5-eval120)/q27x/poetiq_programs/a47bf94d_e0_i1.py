import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid by routing colored 3x3 pieces through the 8/5/9 frame.

    Rule inferred from the examples:
    - The grid contains a "machine" made of pixels 8, 5, and 9.
    - On opposite sides of this machine are 3x3 slots (ports).
    - A piece in a slot is one of:
        * solid 3x3,
        * diamond: 0 1 0 / 1 0 1 / 0 1 0,
        * X: 1 0 1 / 0 1 0 / 1 0 1.
    - In the output, each connected port pair has:
        * a diamond on the top/left side,
        * an X on the bottom/right side,
        with the same color.
    - Solid inputs become diamonds, and the missing opposite-side piece is created.
    - The route between ports is traced through the 8/5/9 structure.  At ambiguous
      turns, directions are preferred if they contain nearby 5 pixels.
    """
    grid = np.array(grid, dtype=int)
    if grid.ndim != 2:
        return grid

    h, w = grid.shape
    if h < 3 or w < 3:
        return grid

    # Machine colors: 8 and 5 are wires, 9 is traversable fill inside the wires.
    machine = np.array([8, 5, 9])
    W = np.isin(grid, machine)

    # Directions: 0=up, 1=right, 2=down, 3=left
    dr = [-1, 0, 1, 0]
    dc = [0, 1, 0, -1]

    def in_bounds(r, c):
        return 0 <= r < h and 0 <= c < w

    # ------------------------------------------------------------------
    # 1. Find port endpoints: degree-1 8/5 cells that border a 3x3 slot.
    # ------------------------------------------------------------------
    endpoints = []

    for r in range(h):
        for c in range(w):
            if grid[r, c] not in (8, 5):
                continue

            neigh = []
            for d in range(4):
                nr, nc = r + dr[d], c + dc[d]
                if in_bounds(nr, nc) and W[nr, nc]:
                    neigh.append(d)

            if len(neigh) != 1:
                continue

            d = neigh[0]  # direction into the machine

            # The 3x3 slot is outside the endpoint.
            if d == 2:      # into machine down -> slot is above
                r1, r2, c1, c2 = r - 3, r + 1, c - 1, c + 2
                side = "top"
            elif d == 0:    # into machine up -> slot is below
                r1, r2, c1, c2 = r + 1, r + 4, c - 1, c + 2
                side = "bottom"
            elif d == 1:    # into machine right -> slot is left
                r1, r2, c1, c2 = r - 1, r + 2, c - 3, c + 1
                side = "left"
            else:           # into machine left -> slot is right
                r1, r2, c1, c2 = r - 1, r + 2, c + 1, c + 4
                side = "right"

            if not (0 <= r1 < r2 <= h and 0 <= c1 < c2 <= w):
                continue

            sub = grid[r1:r2, c1:c2]
            # A valid slot should not contain machine colors.
            if np.isin(sub, machine).any():
                continue

            endpoints.append({
                "id": len(endpoints),
                "r": r,
                "c": c,
                "dir_in": d,
                "side": side,
                "rect": (r1, r2, c1, c2)
            })

    if not endpoints:
        return grid.copy()

    # ------------------------------------------------------------------
    # 2. Keep only aligned endpoints on each side (remove accidental dead ends).
    # ------------------------------------------------------------------
    from collections import defaultdict, Counter

    by_side = defaultdict(list)
    for ep in endpoints:
        by_side[ep["side"]].append(ep)

    def filter_side(side, eps):
        if not eps:
            return []
        if side in ("top", "bottom"):
            coords = [ep["r"] for ep in eps]
        else:
            coords = [ep["c"] for ep in eps]

        counts = Counter(coords)
        if not counts:
            return eps

        best = max(counts.values())
        # If there is a clear dominant row/col, keep only those endpoints.
        if best >= 2 and best > len(eps) / 2:
            good = set(k for k, v in counts.items() if v == best)
            if side in ("top", "bottom"):
                return [ep for ep in eps if ep["r"] in good]
            else:
                return [ep for ep in eps if ep["c"] in good]
        return eps

    for side in list(by_side.keys()):
        by_side[side] = filter_side(side, by_side[side])

    # Choose the active opposite-side pair.
    candidates = []
    if by_side.get("top") and by_side.get("bottom"):
        candidates.append(("top", "bottom"))
    if by_side.get("left") and by_side.get("right"):
        candidates.append(("left", "right"))

    if not candidates:
        return grid.copy()

    # Prefer the pair with more ports.
    candidates.sort(key=lambda ab: min(len(by_side[ab[0]]), len(by_side[ab[1]])), reverse=True)
    d_side, x_side = candidates[0]

    # ------------------------------------------------------------------
    # 3. Read the color currently present in each port (if any).
    # ------------------------------------------------------------------
    for ep in endpoints:
        r1, r2, c1, c2 = ep["rect"]
        sub = grid[r1:r2, c1:c2]
        nz = sub[sub != 0]
        if len(nz) == 0:
            ep["color"] = None
        else:
            vals, cnts = np.unique(nz, return_counts=True)
            ep["color"] = int(vals[np.argmax(cnts)])

    # ------------------------------------------------------------------
    # 4. Trace wires from D-side ports to X-side ports.
    # ------------------------------------------------------------------
    ep_by_pos = {(ep["r"], ep["c"]): ep for ep in endpoints}

    def score_dir(r, c, d):
        """
        Heuristic score for turning/continuing in direction d.
        Prefer directions that have a 5 soon ahead.
        """
        best = 0
        for k in range(1, 9):
            nr = r + dr[d] * k
            nc = c + dc[d] * k
            if not in_bounds(nr, nc) or not W[nr, nc]:
                break
            if grid[nr, nc] == 5:
                best = max(best, 8 - k)
        return best

    def make_tracer(target_side):
        target_pos = {(ep["r"], ep["c"]): ep for ep in endpoints if ep["side"] == target_side}
        non_target_pos = {(ep["r"], ep["c"]): ep for ep in endpoints if ep["side"] != target_side}

        memo = {}
        visiting = set()

        def next_states(state):
            r, c, d = state
            cands = []

            # Straight first.
            nr, nc = r + dr[d], c + dc[d]
            if in_bounds(nr, nc) and W[nr, nc]:
                cands.append((nr, nc, d))

            # Perpendiculars, sorted by 5-proximity heuristic.
            p1 = (d + 1) % 4
            p2 = (d + 3) % 4
            perps = sorted([p1, p2], key=lambda p: score_dir(r, c, p), reverse=True)

            for p in perps:
                nr, nc = r + dr[p], c + dc[p]
                if in_bounds(nr, nc) and W[nr, nc]:
                    cands.append((nr, nc, p))

            return cands

        def dfs(state):
            if state in memo:
                return memo[state]
            if state in visiting:
                return None

            r, c, d = state

            ep_at = ep_by_pos.get((r, c))
            if ep_at is not None:
                if ep_at["side"] == target_side:
                    memo[state] = ep_at["id"]
                    return ep_at["id"]
                else:
                    memo[state] = None
                    return None

            visiting.add(state)
            ans = None
            for ns in next_states(state):
                ans = dfs(ns)
                if ans is not None:
                    break
            visiting.remove(state)
            memo[state] = ans
            return ans

        return dfs

    # Trace from the D-side (top/left) to the X-side (bottom/right).
    pairs = []
    x_used = set()
    d_tracer = make_tracer(x_side)

    for ep in by_side[d_side]:
        d = ep["dir_in"]
        sr, sc = ep["r"] + dr[d], ep["c"] + dc[d]
        if not (in_bounds(sr, sc) and W[sr, sc]):
            continue
        target_id = d_tracer((sr, sc, d))
        if target_id is not None and target_id not in x_used:
            x_used.add(target_id)
            pairs.append((ep["id"], target_id))

    # Fallback: trace from unpaired X-side ports back to D-side.
    x_tracer = make_tracer(d_side)
    d_used = {d_id for d_id, _ in pairs}

    for ep in by_side[x_side]:
        if ep["id"] in x_used:
            continue
        d = ep["dir_in"]
        sr, sc = ep["r"] + dr[d], ep["c"] + dc[d]
        if not (in_bounds(sr, sc) and W[sr, sc]):
            continue
        target_id = x_tracer((sr, sc, d))
        if target_id is not None and target_id not in d_used:
            d_used.add(target_id)
            pairs.append((target_id, ep["id"]))
            x_used.add(ep["id"])

    # ------------------------------------------------------------------
    # 5. Write the output patterns.
    # ------------------------------------------------------------------
    out = grid.copy()

    def set_diamond(color, rect):
        r1, r2, c1, c2 = rect
        out[r1:r2, c1:c2] = 0
        # top-middle, left-middle, right-middle, bottom-middle
        out[r1 + 0, c1 + 1] = color
        out[r1 + 1, c1 + 0] = color
        out[r1 + 1, c1 + 2] = color
        out[r1 + 2, c1 + 1] = color

    def set_x(color, rect):
        r1, r2, c1, c2 = rect
        out[r1:r2, c1:c2] = 0
        # corners + center
        out[r1 + 0, c1 + 0] = color
        out[r1 + 0, c1 + 2] = color
        out[r1 + 1, c1 + 1] = color
        out[r1 + 2, c1 + 0] = color
        out[r1 + 2, c1 + 2] = color

    ep_by_id = {ep["id"]: ep for ep in endpoints}
    paired_ids = set()

    for d_id, x_id in pairs:
        de = ep_by_id[d_id]
        xe = ep_by_id[x_id]

        if de["side"] != d_side or xe["side"] != x_side:
            continue

        color = de["color"]
        if color is None:
            color = xe["color"]
        if color is None:
            continue

        set_diamond(color, de["rect"])
        set_x(color, xe["rect"])
        paired_ids.add(d_id)
        paired_ids.add(x_id)

    # If any port was not routed, at least normalize its own side pattern.
    for ep in endpoints:
        if ep["id"] in paired_ids or ep["color"] is None:
            continue
        if ep["side"] == d_side:
            set_diamond(ep["color"], ep["rect"])
        else:
            set_x(ep["color"], ep["rect"])

    return out