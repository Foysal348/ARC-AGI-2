import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform ARC "active 3x3 icons through a static 5/8/9 wiring diagram".

    Rule
    ----
    The grid contains:
      * static wiring/background cells: 5, 8, 9
      * active 3x3 icons in three forms:
          F = filled 3x3 block
          P = plus / edge-centre icon
             0 c 0
             c 0 c
             0 c 0
          B = diamond / centre+corners icon
             c 0 c
             0 c 0
             c 0 c

    Each active icon sits in a 3x3 "port".  A port is detected by a 3x3 box
    containing no static cells and having exactly one 8 immediately outside
    the middle of one side.

    The static wiring pairs ports.  For a paired port containing:
      * F and the other is empty   -> source becomes P, destination becomes B
      * B and the other is empty   -> source stays B, destination becomes P
      * P and the other is empty   -> source stays P, destination becomes B
      * both already active with the same colour -> left unchanged
      * otherwise -> leave unchanged (robust fallback)

    Pairing is found by a shortest-path search over the static wiring with
    costs that strongly prefer continuing straight and prefer 4-neighbor
    movement over diagonal 5-cell bridges.  True "crossings" where both a
    horizontal and vertical wire continue through an 8/5 cell are not treated
    as turnable junctions; only T-junctions may turn.
    """
    grid = np.array(grid, dtype=np.int8, copy=True)
    if grid.ndim != 2:
        return grid

    H, W = grid.shape
    if H < 3 or W < 3:
        return grid

    out = np.array(grid, dtype=np.int8, copy=True)

    STATIC = {5, 8, 9}
    WIRE = {5, 8, 9}

    # 8-neighbor directions: N, NE, E, SE, S, SW, W, NW
    DR8 = (-1, -1, 0, 1, 1, 1, 0, -1)
    DC8 = (0, 1, 1, 1, 0, -1, -1, -1)
    H_DIRS = {2, 6}
    V_DIRS = {0, 4}
    DIAG_DIRS = {1, 3, 5, 7}

    def inb(r, c):
        return 0 <= r < H and 0 <= c < W

    def is_wire_val(v):
        return int(v) in WIRE

    def is_static_val(v):
        return int(v) in STATIC

    # ------------------------------------------------------------------
    # Detect active 3x3 port boxes.
    # ------------------------------------------------------------------
    ports = []

    for r in range(1, H - 1):
        for c in range(1, W - 1):
            sub = grid[r - 1:r + 2, c - 1:c + 2]

            # Port boxes never contain static wiring cells.
            if np.any(np.isin(sub, [5, 8, 9])):
                continue

            # Exactly one side has the port's 8 stub.
            candidates = []
            if inb(r - 2, c) and grid[r - 2, c] == 8:
                candidates.append(("top", r - 2, c, 0))       # away direction N
            if inb(r + 2, c) and grid[r + 2, c] == 8:
                candidates.append(("bottom", r + 2, c, 4))    # away direction S
            if inb(r, c - 2) and grid[r, c - 2] == 8:
                candidates.append(("left", r, c - 2, 6))      # away direction W
            if inb(r, c + 2) and grid[r, c + 2] == 8:
                candidates.append(("right", r, c + 2, 2))     # away direction E

            if len(candidates) != 1:
                continue

            side, sr, sc, away_dir = candidates[0]

            # Classify the 3x3 icon.
            vals = set(int(x) for x in sub.flat)
            kind = None
            color = 0

            if len(vals) == 1 and 0 not in vals and not (vals & STATIC):
                kind = "F"
                color = int(sub[1, 1])
            else:
                c00 = int(sub[0, 0]); c01 = int(sub[0, 1]); c02 = int(sub[0, 2])
                c10 = int(sub[1, 0]); c11 = int(sub[1, 1]); c12 = int(sub[1, 2])
                c20 = int(sub[2, 0]); c21 = int(sub[2, 1]); c22 = int(sub[2, 2])

                # P: edge centres, empty centre and corners
                if (c00 == c02 == c11 == c20 == c22 == 0 and
                    c01 == c10 == c12 == c21 and
                    c01 not in STATIC and c01 != 0):
                    kind = "P"
                    color = c01
                # B: corners + centre, empty edge centres
                elif (c01 == c10 == c12 == c21 == 0 and
                      c00 == c02 == c11 == c20 == c22 and
                      c00 not in STATIC and c00 != 0):
                    kind = "B"
                    color = c00
                elif sub.min() == 0 and sub.max() == 0:
                    kind = None
                    color = 0
                else:
                    # Unknown active shape: ignore as a transformable port.
                    continue

            ports.append({
                "center": (r, c),
                "stub": (sr, sc),
                "away_dir": away_dir,
                "kind": kind,
                "color": color,
                "side": side,
            })

    nports = len(ports)
    if nports == 0:
        return out

    # ------------------------------------------------------------------
    # Wire graph helpers.
    # ------------------------------------------------------------------
    stub_to_port = {}
    for i, p in enumerate(ports):
        stub_to_port[p["stub"]] = i

    wire = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            wire[r, c] = is_wire_val(grid[r, c])

    def four_neighbors_wire(r, c):
        """
        Return (up, down, left, right) booleans for 4-neighbor wire cells.
        """
        u = inb(r - 1, c) and wire[r - 1, c]
        d = inb(r + 1, c) and wire[r + 1, c]
        l = inb(r, c - 1) and wire[r, c - 1]
        rt = inb(r, c + 1) and wire[r, c + 1]
        return bool(u), bool(d), bool(l), bool(rt)

    def is_true_cross(r, c):
        """
        A true crossing is an 8/5 cell where both horizontal and vertical
        wires continue through the cell.  Diagonal 5-bridges do not make a
        cell a true crossing.
        """
        if not wire[r, c] or grid[r, c] == 9:
            return False
        u, d, l, rt = four_neighbors_wire(r, c)
        return (l and rt) and (u and d)

    def allowed_next(r, c, prev_dir, ndir, prev_cell):
        nr = r + DR8[ndir]
        nc = c + DC8[ndir]
        if not inb(nr, nc):
            return False
        if (nr, nc) == prev_cell:
            return False
        if not wire[nr, nc]:
            return False

        # Reverse is forbidden.
        diff = (ndir - prev_dir) % 8
        if diff == 4:
            return False

        # Diagonal moves are only allowed when a 5 participates, and 9 is
        # not used diagonally.
        if ndir in DIAG_DIRS:
            if grid[r, c] == 9 or grid[nr, nc] == 9:
                return False
            if grid[r, c] != 5 and grid[nr, nc] != 5:
                return False

        # True 8/5 crossings must go straight through; they do not turn.
        if is_true_cross(r, c):
            prev_axis_h = prev_dir in H_DIRS
            next_axis_h = ndir in H_DIRS
            if prev_axis_h != next_axis_h:
                return False

        return True

    def move_cost(prev_dir, ndir, nr, nc):
        diff = (ndir - prev_dir) % 8
        diff = min(diff, 8 - diff)

        # Base movement cost.
        cost = 1

        # Turn penalty.
        if diff == 0:
            cost += 0
        elif diff == 1:
            cost += 4
        elif diff == 2:
            cost += 7
        else:  # diff == 3
            cost += 10

        # Diagonal penalty.
        if ndir in DIAG_DIRS:
            cost += 8

        # Slight penalty for entering/exiting 9, to prefer straight 8/5
        # corridors when equivalent paths exist.
        if grid[nr, nc] == 9:
            cost += 1

        return cost

    # ------------------------------------------------------------------
    # Dijkstra from each port to all other port stubs.
    # ------------------------------------------------------------------
    INF = 10**9
    cost_matrix = [[INF] * nports for _ in range(nports)]

    for start_i, start_port in enumerate(ports):
        sr, sc = start_port["stub"]
        start_dir = start_port["away_dir"]

        dist = {}
        pq = [(0, sr, sc, start_dir, None)]
        found = [False] * nports
        found[start_i] = True
        found_costs = {}

        while pq and len(found_costs) < nports - 1:
            cost, r, c, d, _ = pq.pop()
            state = (r, c, d)
            if dist.get(state, INF) < cost:
                continue

            # If this cell is another port stub, record it and do not expand.
            if (r, c) in stub_to_port:
                pi = stub_to_port[(r, c)]
                if not found[pi]:
                    found[pi] = True
                    found_costs[pi] = cost
                    continue

            for ndir in range(8):
                if not allowed_next(r, c, d, ndir, (r + DR8[d], c + DC8[d])):
                    continue
                nr = r + DR8[ndir]
                nc = c + DC8[ndir]
                ncost = cost + move_cost(d, ndir, nr, nc)
                nstate = (nr, nc, ndir)
                if ncost < dist.get(nstate, INF):
                    dist[nstate] = ncost
                    pq.append((ncost, nr, nc, ndir, (r, c)))

        for j, cval in found_costs.items():
            cost_matrix[start_i][j] = cval
            cost_matrix[j][i] = min(cost_matrix[j][i], cval)

    # Symmetrize and make diagonal zero.
    for i in range(nports):
        cost_matrix[i][i] = 0
    for i in range(nports):
        for j in range(nports):
            cost_matrix[i][j] = min(cost_matrix[i][j], cost_matrix[j][i])

    # ------------------------------------------------------------------
    # Minimum-weight perfect matching of ports.
    # ------------------------------------------------------------------
    import sys
    from functools import lru_cache

    if nports % 2 != 0:
        # Should not happen for valid cases; fall back to greedy nearest.
        pair = [-1] * nports
        used = [False] * nports
        for i in range(nports):
            if used[i]:
                continue
            best = -1
            best_c = INF
            for j in range(nports):
                if not used[j] and j != i:
                    if cost_matrix[i][j] < best_c:
                        best_c = cost_matrix[i][j]
                        best = j
            if best != -1:
                pair[i] = best
                pair[best] = i
                used[i] = True
                used[best] = True
            else:
                pair[i] = i
    else:
        pair = [-1] * nports

        sys.setrecursionlimit(10000)

        @lru_cache(None)
        def dp(mask):
            # mask has bits set for ports already matched.
            # Find first unmatched port.
            i = -1
            for k in range(nports):
                if not (mask >> k) & 1:
                    i = k
                    break
            if i == -1:
                return 0

            best = INF
            for j in range(nports):
                if (mask >> j) & 1:
                    continue
                if j == i:
                    continue
                nmask = mask | (1 << i) | (1 << j)
                val = cost_matrix[i][j] + dp(nmask)
                if val < best:
                    best = val
            return best

        # Reconstruct matching.
        def reconstruct(mask, i):
            for j in range(nports):
                if (mask >> j) & 1 or j == i:
                    continue
                nmask = mask | (1 << i) | (1 << j)
                if cost_matrix[i][j] + dp(nmask) == dp(mask):
                    pair[i] = j
                    pair[j] = i
                    reconstruct(nmask, -1)
                    return

        reconstruct(0, -1)

        # In case of a recursion issue, greedy fill any -1.
        used = [False] * nports
        for i in range(nports):
            if pair[i] != -1:
                used[i] = True
        for i in range(nports):
            if not used[i]:
                best = -1
                best_c = INF
                for j in range(nports):
                    if not used[j] and j != i and pair[j] == -1:
                        if cost_matrix[i][j] < best_c:
                            best_c = cost_matrix[i][j]
                            best = j
                if best != -1:
                    pair[i] = best
                    pair[best] = i
                    used[i] = True
                    used[best] = True

    # ------------------------------------------------------------------
    # Draw icons.
    # ------------------------------------------------------------------
    def clear_box(r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if inb(rr, cc):
                    out[rr, cc] = 0

    def draw_icon(r, c, kind, color):
        clear_box(r, c)
        if kind is None or color == 0:
            return
        if kind == "F":
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    out[r + dr, c + dc] = color
        elif kind == "P":
            out[r - 1, c] = color
            out[r + 1, c] = color
            out[r, c - 1] = color
            out[r, c + 1] = color
        elif kind == "B":
            out[r - 1, c - 1] = color
            out[r - 1, c + 1] = color
            out[r, c] = color
            out[r + 1, c - 1] = color
            out[r + 1, c + 1] = color

    # First, clear all active port boxes so stale cells do not remain.
    for p in ports:
        if p["kind"] is not None:
            r, c = p["center"]
            clear_box(r, c)

    # Apply pair transformations.
    processed = set()
    for i in range(nports):
        j = pair[i]
        if j == i or (i, j) in processed or (j, i) in processed:
            continue
        processed.add((i, j))
        processed.add((j, i))

        ki, ci = ports[i]["kind"], ports[i]["color"]
        kj, cj = ports[j]["kind"], ports[j]["color"]

        # If both active:
        #  - same colour: leave as-is (e.g. two P4 endpoints in Example 2)
        #  - different colour: conservative leave-as-is
        if ki is not None and kj is not None:
            if ci == cj:
                draw_icon(*ports[i]["center"], ki, ci)
                draw_icon(*ports[j]["center"], kj, cj)
            else:
                draw_icon(*ports[i]["center"], ki, ci)
                draw_icon(*ports[j]["center"], kj, cj)
            continue

        # One active, one empty.
        if ki is not None and kj is None:
            src, dst = i, j
            src_kind, src_color = ki, ci
        elif kj is not None and ki is None:
            src, dst = j, i
            src_kind, src_color = kj, cj
        else:
            continue

        # Source normalisation:
        #   F -> P
        #   P -> P
        #   B -> B
        if src_kind == "F":
            src_out = "P"
        elif src_kind == "B":
            src_out = "B"
        else:
            src_out = "P"

        # Destination is the complementary shape.
        dst_out = "B" if src_out == "P" else "P"

        draw_icon(*ports[src]["center"], src_out, src_color)
        draw_icon(*ports[dst]["center"], dst_out, src_color)

    return out