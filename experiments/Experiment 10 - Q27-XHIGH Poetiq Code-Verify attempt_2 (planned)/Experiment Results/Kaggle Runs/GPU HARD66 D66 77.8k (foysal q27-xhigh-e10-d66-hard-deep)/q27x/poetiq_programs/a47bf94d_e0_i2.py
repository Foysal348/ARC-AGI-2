import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform ARC circuit grids.

    Rule inferred from the examples:
    - The 5/8/9 cells form a static pipe/router diagram.
    - Active 3x3 blocks use colors {1, 2, 3, 4, 6}.
      Color 5 is NOT active; it is part of the pipe diagram.
    - Active blocks can appear in three 3x3 forms:
        F: full 3x3 block
        A: edge-center cross / open ring
             0 c 0
             c 0 c
             0 c 0
        B: corner+center diamond
             c 0 c
             0 c 0
             c 0 c
    - The pipe diagram pairs port boxes (3x3 positions) together.
    - If a paired port contains F of color c and the other port is empty:
        source F -> A, destination -> B.
    - If a paired port contains B of color c and the other port is empty:
        source B stays B, destination -> A.
    - Existing A ports are preserved and do not create new blocks.
    - The 5/8/9 diagram is preserved unchanged.

    The implementation traces the pipe network by:
      1. detecting 3x3 active port boxes and empty 3x3 port boxes,
      2. protecting the wire cells adjacent to those port boxes,
      3. pruning non-protected dead-end pipe cells,
      4. tracing straight-priority pipe paths (9 blocks are straight-through),
      5. using the traced endpoint pairs to apply the F/A/B state rule.
    """
    out = np.array(grid, copy=True)
    H, W = out.shape
    if H < 3 or W < 3:
        return out

    ACTIVE = {1, 2, 3, 4, 6}
    PIPE = {5, 8}
    WIRE = {5, 8, 9}

    DR = (-1, 1, 0, 0)
    DC = (0, 0, -1, 1)

    def inb(r, c):
        return 0 <= r < H and 0 <= c < W

    def is_pipe(v):
        return v in PIPE

    def is_wire(v):
        return v in WIRE

    # ------------------------------------------------------------------
    # 1. Detect existing active 3x3 patterns.
    # ------------------------------------------------------------------
    active = {}

    for r in range(1, H - 1):
        for c in range(1, W - 1):
            sub = grid[r - 1:r + 2, c - 1:c + 2]
            if np.any(np.isin(sub, [5, 8, 9])):
                continue

            vals = set(int(x) for x in sub.flat)

            # Full block.
            if len(vals) == 1 and vals.pop() in ACTIVE:
                active[(r, c)] = ("F", int(grid[r, c]))
                continue

            z = 0

            # A pattern: edge centers only.
            col_a = int(sub[0, 1])
            ok_a = (
                col_a in ACTIVE
                and sub[1, 0] == col_a
                and sub[1, 2] == col_a
                and sub[2, 1] == col_a
                and sub[0, 0] == 0
                and sub[0, 2] == 0
                and sub[1, 1] == 0
                and sub[2, 0] == 0
                and sub[2, 2] == 0
            )
            if ok_a:
                active[(r, c)] = ("A", col_a)
                continue

            # B pattern: corners + center.
            col_b = int(sub[0, 0])
            ok_b = (
                col_b in ACTIVE
                and sub[0, 2] == col_b
                and sub[1, 1] == col_b
                and sub[2, 0] == col_b
                and sub[2, 2] == col_b
                and sub[0, 1] == 0
                and sub[1, 0] == 0
                and sub[1, 2] == 0
                and sub[2, 1] == 0
            )
            if ok_b:
                active[(r, c)] = ("B", col_b)
                continue

    active_centers = set(active.keys())

    def box_is_empty(center):
        r, c = center
        if not (1 <= r < H - 1 and 1 <= c < W - 1):
            return False
        sub = grid[r - 1:r + 2, c - 1:c + 2]
        if np.any(np.isin(sub, [5, 8, 9])):
            return False
        if np.any(sub != 0):
            return False
        return True

    def box_is_valid_port(center):
        return center in active_centers or box_is_empty(center)

    # ------------------------------------------------------------------
    # 2. Initial pipe degrees, used to protect real port stubs.
    # ------------------------------------------------------------------
    def initial_degree(r, c):
        deg = 0
        for d in range(4):
            nr, nc = r + DR[d], c + DC[d]
            if inb(nr, nc) and is_wire(grid[nr, nc]):
                deg += 1
        return deg

    # protected endpoint wire cell -> (port center, direction from port to endpoint)
    protected = {}

    def try_protect_endpoint(center, d):
        r, c = center
        er, ec = r + 2 * DR[d], c + 2 * DC[d]
        mr, mc = r + DR[d], c + DC[d]
        if not inb(er, ec):
            return
        if not is_pipe(grid[er, ec]):
            return
        if is_wire(grid[mr, mc]):
            return
        # The cell behind the endpoint toward the port must not be pipe.
        br, bc = er - DR[d], ec - DC[d]
        if inb(br, bc) and is_wire(grid[br, bc]):
            return

        # Accept if it is a dead-end pipe cell, or if it visibly continues
        # into the pipe network in the endpoint direction.
        ar, ac = er + DR[d], ec + DC[d]
        continues = inb(ar, ac) and is_wire(grid[ar, ac])
        if initial_degree(er, ec) <= 1 or continues:
            protected[(er, ec)] = (center, d)

    # Endpoints for active ports.
    for center in active_centers:
        for d in range(4):
            try_protect_endpoint(center, d)

    # Endpoints for empty 3x3 port boxes.
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if box_is_empty((r, c)):
                for d in range(4):
                    try_protect_endpoint((r, c), d)

    # ------------------------------------------------------------------
    # 3. Prune non-protected dead-end pipe cells.
    # ------------------------------------------------------------------
    alive = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if is_pipe(grid[r, c]):
                alive[r, c] = True

    protected_set = set(protected.keys())
    pipe_cells = [(r, c) for r in range(H) for c in range(W) if alive[r, c]]

    changed = True
    while changed:
        changed = False
        for r, c in pipe_cells:
            if not alive[r, c] or (r, c) in protected_set:
                continue

            deg = 0
            for d in range(4):
                nr, nc = r + DR[d], c + DC[d]
                if not inb(nr, nc):
                    continue
                if grid[nr, nc] == 9:
                    deg += 1
                elif alive[nr, nc] and is_pipe(grid[nr, nc]):
                    deg += 1

            if deg <= 1:
                alive[r, c] = False
                changed = True

    # ------------------------------------------------------------------
    # 4. Straight-priority pipe tracing with 9 as straight-through blocks.
    # ------------------------------------------------------------------
    def advance_straight(r, c, d):
        """Move one step in direction d, jumping straight through 9 blocks."""
        nr, nc = r + DR[d], c + DC[d]
        if not inb(nr, nc):
            return None

        v = grid[nr, nc]
        if is_pipe(v):
            if alive[nr, nc]:
                return (nr, nc)
            return None

        if v == 9:
            r2, c2 = nr, nc
            while inb(r2, c2) and grid[r2, c2] == 9:
                r2 += DR[d]
                c2 += DC[d]

            if inb(r2, c2):
                v2 = grid[r2, c2]
                if is_pipe(v2) and alive[r2, c2]:
                    return (r2, c2)
            return None

        return None

    def perpendiculars(d):
        # Prefer left, then right relative to the current direction.
        return [(d + 1) % 4, (d + 3) % 4]

    def open_turns(r, c, d):
        return [td for td in perpendiculars(d) if advance_straight(r, c, td) is not None]

    def trace_path(start_cell, start_dir):
        """
        Trace from a protected endpoint into the pipe network.
        Returns (start_cell, start_dir, end_cell, end_dir) or None.
        """
        r, c = start_cell
        d = start_dir
        visited = set()
        max_steps = max(1000, 20 * H * W * 4)

        for _ in range(max_steps):
            state = (r, c, d)
            if state in visited:
                return None
            visited.add(state)

            nxt = advance_straight(r, c, d)
            if nxt is not None:
                r, c = nxt
                continue

            turns = open_turns(r, c, d)
            if not turns:
                return (start_cell, start_dir, (r, c), d)

            if len(turns) == 1:
                d = turns[0]
                continue

            # Rare junction: choose a turn by short lookahead to a protected endpoint.
            best = None
            best_score = -1
            for td in turns:
                rr, cc = r, c
                dd = td
                seen = set()
                score = 0
                found = False
                for _ in range(500):
                    st = (rr, cc, dd)
                    if st in seen:
                        break
                    seen.add(st)

                    nn = advance_straight(rr, cc, dd)
                    if nn is not None:
                        rr, cc = nn
                        continue

                    ts = open_turns(rr, cc, dd)
                    if not ts:
                        if (rr, cc) in protected_set:
                            score = 1000
                            found = True
                        break

                    dd = ts[0]

                if found:
                    break
                if score > best_score:
                    best_score = score
                    best = td

            if best is None:
                # Fallback: prefer a turn whose immediate pipe cell is 5,
                # otherwise use the first (left-biased) perpendicular.
                scored = []
                for td in turns:
                    nr, nc = r + DR[td], c + DC[td]
                    pref = 1 if inb(nr, nc) and grid[nr, nc] == 5 else 0
                    scored.append((-pref, td))
                scored.sort()
                best = scored[0][1]

            d = best

        return None

    # ------------------------------------------------------------------
    # 5. Build port-to-port pairs from traced endpoints.
    # ------------------------------------------------------------------
    pairing = {}
    used_endpoints = set()

    def valid_center(center):
        r, c = center
        return 1 <= r < H - 1 and 1 <= c < W - 1 and box_is_valid_port(center)

    for endpoint in sorted(protected.keys()):
        if endpoint in used_endpoints:
            continue
        if not alive[endpoint]:
            continue

        center, sdir = protected[endpoint]
        if not valid_center(center):
            continue

        # A real endpoint should have an open move into the network.
        if advance_straight(endpoint[0], endpoint[1], sdir) is None:
            continue

        path = trace_path(endpoint, sdir)
        if path is None:
            continue

        start_cell, start_dir, end_cell, end_dir = path

        start_center = protected.get(start_cell, (None, None))[0]
        if start_center is None:
            start_center = (start_cell[0] - 2 * start_dir and 0, 0)  # placeholder
            start_center = (start_cell[0] - 2 * DR[start_dir],
                            start_cell[1] - 2 * DC[start_dir])

        end_center = protected.get(end_cell, (None, None))[0]
        if end_center is None:
            # When arriving at the far endpoint, current direction points toward the port.
            end_center = (end_cell[0] + 2 * DR[end_dir],
                          end_cell[1] + 2 * DC[end_dir])

        if not valid_center(start_center) or not valid_center(end_center):
            continue
        if start_center == end_center:
            continue

        used_endpoints.add(start_cell)
        used_endpoints.add(end_cell)

        # Store a single consistent pairing.  The generated puzzles use
        # disjoint pairs, so conflicts should not occur.
        if start_center not in pairing and end_center not in pairing:
            pairing[start_center] = end_center
            pairing[end_center] = start_center
        else:
            # If a conflict appears, keep the first pair.
            pass

    # ------------------------------------------------------------------
    # 6. Apply active color state changes.
    # ------------------------------------------------------------------
    def can_draw_empty(center):
        return center not in active_centers and box_is_empty(center)

    def draw_pattern(center, kind, color):
        r, c = center
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if inb(rr, cc):
                    v = out[rr, cc]
                    if v not in (5, 8, 9):
                        out[rr, cc] = 0

        if kind == "A":
            out[r - 1, c] = color
            out[r, c - 1] = color
            out[r, c + 1] = color
            out[r + 1, c] = color
        elif kind == "B":
            out[r - 1, c - 1] = color
            out[r - 1, c + 1] = color
            out[r, c] = color
            out[r + 1, c - 1] = color
            out[r + 1, c + 1] = color

    for center, (kind, color) in active.items():
        other = pairing.get(center)

        if kind == "F":
            draw_pattern(center, "A", color)
            if other is not None and can_draw_empty(other):
                draw_pattern(other, "B", color)

        elif kind == "B":
            # Normalize the source B, then create the paired A if empty.
            draw_pattern(center, "B", color)
            if other is not None and can_draw_empty(other):
                draw_pattern(other, "A", color)

        elif kind == "A":
            # Existing A ports are already in the completed state.
            draw_pattern(center, "A", color)

    return out