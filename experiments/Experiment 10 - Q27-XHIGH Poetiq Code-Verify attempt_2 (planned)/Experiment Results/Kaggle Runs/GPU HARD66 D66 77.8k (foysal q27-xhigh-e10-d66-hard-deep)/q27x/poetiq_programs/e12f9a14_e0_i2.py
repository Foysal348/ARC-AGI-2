import numpy as np
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform framed 2x2 objects by letting their interior colour leak out of
    missing frame cells.

    Implemented rule
    ----------------
    1. Background is the dominant colour on the outer border.
    2. Each object is a 2x2 interior block surrounded by a 4x4 frame of another
       colour.  Missing frame cells are background cells on the 4x4 ring.
    3. Each missing ring cell emits the interior colour in the outward normal
       direction:
         - side hole: straight out
         - corner hole: diagonally out
    4. The emitted paths are routed to the grid border.  They are allowed to
       move in the 8 directions, but they strongly prefer:
         - continuing straight,
         - then turning 45 degrees,
         - then turning 90 degrees.
       Turns larger than 90 degrees are avoided.
    5. Routes avoid:
         - non-background input cells,
         - already placed routes of different colours.
       Same-colour routes may merge.
    6. The route is chosen by Dijkstra over (cell, direction) states with a
       cost that makes "stay as straight as possible" dominate over "be short".
       A small target penalty keeps a route aiming toward the natural exit
       border when several exits are possible.

    This matches the visible behaviour in the examples: straight leaks when
    unobstructed, and 45-degree / staircase detours when two leaks would
    otherwise intersect.
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape
    out = np.zeros_like(grid, dtype=grid.dtype)

    # ------------------------------------------------------------------
    # Background: dominant outer-border colour, fallback to global majority.
    # ------------------------------------------------------------------
    border_vals = np.concatenate([
        grid[0, :].ravel(),
        grid[-1, :].ravel(),
        grid[:, 0].ravel(),
        grid[:, -1].ravel(),
    ])
    vals, counts = np.unique(border_vals, return_counts=True)
    background = int(vals[np.argmax(counts)])

    # If the border is not informative, use the global majority.
    all_vals, all_counts = np.unique(grid, return_counts=True)
    if counts[np.argmax(counts)] <= max(H, W):
        background = int(all_vals[np.argmax(all_counts)])

    # ------------------------------------------------------------------
    # Find all 2x2 interior blocks with a surrounding frame.
    # Each detected object gives interior colour and frame colour.
    # ------------------------------------------------------------------
    objects = []
    used_interior = set()

    for r in range(H - 1):
        for c in range(W - 1):
            col = int(grid[r, c])
            if col == background:
                continue
            if not (
                grid[r, c + 1] == col and
                grid[r + 1, c] == col and
                grid[r + 1, c + 1] == col
            ):
                continue

            pts = {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}
            if any(p in used_interior for p in pts):
                continue

            # 4x4 bounding box: one-cell ring around the 2x2 interior.
            r0, c0 = r - 1, c - 1
            r1, c1 = r + 2, c + 2

            ring_vals = []
            ring_positions = []
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    if rr < 0 or rr >= H or cc < 0 or cc >= W:
                        continue
                    if (r <= rr <= r + 1 and c <= cc <= c + 1):
                        continue
                    v = int(grid[rr, cc])
                    ring_vals.append(v)
                    ring_positions.append((rr, cc))

            # Frame colour: most frequent non-background, non-interior colour
            # in the ring, with at least 3 occurrences.
            frame_candidates = [v for v in ring_vals if v != background and v != col]
            if not frame_candidates:
                continue
            fv, fc = np.unique(np.array(frame_candidates), return_counts=True)
            frame = int(fv[np.argmax(fc)])
            if max(fc) < 3:
                continue

            # Require a reasonably complete frame: at least two opposite sides.
            side_touch = {
                "top": any(rr == r0 and int(grid[rr, cc]) == frame for rr, cc in ring_positions),
                "bottom": any(rr == r1 and int(grid[rr, cc]) == frame for rr, cc in ring_positions),
                "left": any(cc == c0 and int(grid[rr, cc]) == frame for rr, cc in ring_positions),
                "right": any(cc == c1 and int(grid[rr, cc]) == frame for rr, cc in ring_positions),
            }
            if not ((side_touch["top"] and side_touch["bottom"]) or
                    (side_touch["left"] and side_touch["right"])):
                continue

            used_interior.update(pts)
            objects.append({
                "interior": col,
                "frame": frame,
                "r": r,
                "c": c,
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
            })

    if not objects:
        return grid.copy()

    # ------------------------------------------------------------------
    # Create leak sources: background cells on the 4x4 ring.
    # Direction is the outward normal of the missing ring cell.
    # ------------------------------------------------------------------
    sources = []

    for obj in objects:
        r, c = obj["r"], obj["c"]
        r0, c0, r1, c1 = obj["r0"], obj["c0"], obj["r1"], obj["c1"]

        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if rr < 0 or rr >= H or cc < 0 or cc >= W:
                    continue
                if r <= rr <= r + 1 and c <= cc <= c + 1:
                    continue
                if grid[rr, cc] != background:
                    continue

                # Determine outward direction from the position on the ring.
                top = (rr == r0)
                bottom = (rr == r1)
                left = (cc == c0)
                right = (cc == c1)

                if top and left:
                    d = (-1, -1)
                elif top and right:
                    d = (-1, 1)
                elif bottom and left:
                    d = (1, -1)
                elif bottom and right:
                    d = (1, 1)
                elif top:
                    d = (-1, 0)
                elif bottom:
                    d = (1, 0)
                elif left:
                    d = (0, -1)
                elif right:
                    d = (0, 1)
                else:
                    continue

                sources.append({
                    "color": int(obj["interior"]),
                    "start": (rr, cc),
                    "dir": d,
                })

    if not sources:
        return grid.copy()

    # ------------------------------------------------------------------
    # 8 directions.  Index 0..7 is used for turn-cost computation.
    # ------------------------------------------------------------------
    D8 = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ]

    def dir_index(dr, dc):
        for i, (x, y) in enumerate(D8):
            if x == dr and y == dc:
                return i
        return -1

    # Turn costs.  The important idea: step cost is huge, so the router does
    # not shortcut by turning early.  It only turns when straight movement is
    # blocked or clearly unhelpful, and 45-degree turns are much cheaper than
    # 90-degree turns.
    STEP = 1000
    TURN45 = 30
    TURN90 = 90
    TURN135 = 200
    TURN180 = 500
    TARGET = 1

    def turn_cost(a, b):
        if a == b:
            return 0
        x1, y1 = D8[a]
        x2, y2 = D8[b]
        dot = x1 * x2 + y1 * y2
        cross = x1 * y2 - y1 * x2
        if dot < 0:
            return TURN180 if cross == 0 else TURN135
        if dot == 0:
            return TURN90
        return TURN45

    # Static walls: every non-background input cell.
    wall = (grid != background)

    # ------------------------------------------------------------------
    # Straight exit: where the initial ray would hit the border if nothing
    # blocked it.  Used only as a small tie-breaker.
    # ------------------------------------------------------------------
    def ideal_exit(source):
        sr, sc = source["start"]
        dr, dc = source["dir"]
        r, c = sr, sc
        steps = 0
        while 0 <= r < H and 0 <= c < W and steps < H + W:
            r += dr
            c += dc
            if 0 <= r < H and 0 <= c < W:
                pass
            steps += 1
        # last valid cell
        r -= dr
        c -= dc
        if not (0 <= r < H and 0 <= c < W):
            return (sr, sc)
        return (r, c)

    # A target is acceptable if it is in the outward half-plane of the source.
    def allowed_target(source, r, c):
        sr, sc = source["start"]
        dr, dc = source["dir"]
        if dr < 0 and r > sr:
            return False
        if dr > 0 and r < sr:
            return False
        if dc < 0 and c > sc:
            return False
        if dc > 0 and c < sc:
            return False
        return (r - sr) * dr + (c - sc) * dc >= 0

    def target_penalty(source, r, c):
        ir, ic = ideal_exit(source)
        return abs(r - ir) + abs(c - ic)

    # ------------------------------------------------------------------
    # Dijkstra for one source, given cells blocked by other colours.
    # Returns list of cells to paint.
    # ------------------------------------------------------------------
    import heapq

    def route_source(source, blocked):
        sr, sc = source["start"]
        dr, dc = source["dir"]

        # The source cell itself is always painted.
        if sr == 0 or sr == H - 1 or sc == 0 or sc == W - 1:
            if allowed_target(source, sr, sc):
                return [(sr, sc)]

        def is_blocked(r, c):
            if r < 0 or r >= H or c < 0 or c >= W:
                return True
            if wall[r, c]:
                return True
            if (r, c) in blocked:
                return True
            return False

        # Mandatory first step in the source direction, if possible.
        nr, nc = sr + dr, sc + dc
        if not is_blocked(nr, nc):
            start_state = (nr, nc, dir_index(dr, dc))
        else:
            # Fallback: start at the source and let Dijkstra choose a legal
            # first move biased by turn cost.
            start_state = (sr, sc, dir_index(dr, dc))

        sr_state, sc_state, sd = start_state
        INF = 10**18
        dist = {(sr_state, sc_state, sd): 0}
        parent = {}
        heap = [(0, sr_state, sc_state, sd)]

        best_border = None
        best_border_cost = INF

        while heap:
            cost, r, c, d = heapq.heappop(heap)
            cur = (r, c, d)
            if dist.get(cur, INF) < cost:
                continue

            is_border = (r == 0 or r == H - 1 or c == 0 or c == W - 1)
            if is_border and allowed_target(source, r, c):
                total = cost + TARGET * target_penalty(source, r, c)
                if total < best_border_cost:
                    best_border_cost = total
                    best_border = cur

            for nd in range(8):
                ndr, ndc = D8[nd]
                nr2, nc2 = r + ndr, c + ndc
                if is_blocked(nr2, nc2):
                    continue

                add = STEP + turn_cost(d, nd)
                ncost = cost + add
                nkey = (nr2, nc2, nd)
                if ncost < dist.get(nkey, INF):
                    dist[nkey] = ncost
                    parent[nkey] = cur
                    heapq.heappush(heap, (ncost, nr2, nc2, nd))

        if best_border is None:
            return [(sr, sc)]

        path = []
        cur = best_border
        while cur is not None:
            path.append((cur[0], cur[1]))
            cur = parent.get(cur)
        path.reverse()

        # Ensure the original source cell is included.
        if not path or path[0] != (sr, sc):
            # If the source was not the start state, it is adjacent to the
            # first state; prepend it.
            if path and (abs(path[0][0] - sr) <= 1 and abs(path[0][1] - sc) <= 1):
                path.insert(0, (sr, sc))
            else:
                path = [(sr, sc)] + path

        # Remove consecutive duplicates.
        clean = []
        for p in path:
            if not clean or clean[-1] != p:
                clean.append(p)
        return clean

    # ------------------------------------------------------------------
    # Route sources iteratively.  We use several rounds with other colours'
    # current paths as soft obstacles.  This gives the visible "bypass"
    # behaviour when two leaks would otherwise cross.
    # ------------------------------------------------------------------
    # Deterministic order: by colour, then source position.  The iterative
    # update reduces order bias.
    order = sorted(range(len(sources)),
                   key=lambda i: (sources[i]["color"], sources[i]["start"]))

    initial_paths = []
    for i in order:
        p = route_source(sources[i], set())
        initial_paths.append(p)

    paths = [None] * len(sources)
    for idx, i in enumerate(order):
        paths[i] = initial_paths[idx]

    for _ in range(12):
        blocked_by_color = defaultdict(set)
        for i, p in enumerate(paths):
            col = sources[i]["color"]
            for cell in p:
                blocked_by_color[col].add(cell)

        new_paths = [None] * len(sources)
        changed = False

        for i in order:
            blocked = set()
            col = sources[i]["color"]
            for ocol, cells in blocked_by_color.items():
                if ocol != col:
                    blocked.update(cells)

            p = route_source(sources[i], blocked)
            new_paths[i] = p
            if p != paths[i]:
                changed = True

        paths = new_paths
        if not changed:
            break

    # ------------------------------------------------------------------
    # Paint.  Input cells are preserved; background cells are filled by leak
    # paths.  If two different colours ever claim the same cell, the smaller
    # colour wins for determinism, but this should not happen after routing.
    # ------------------------------------------------------------------
    out = grid.copy()

    # Paint in colour order, then by path start, for determinism.
    paint_order = sorted(range(len(sources)),
                         key=lambda i: (sources[i]["color"], paths[i][0]))
    for i in paint_order:
        col = sources[i]["color"]
        for r, c in paths[i]:
            if 0 <= r < H and 0 <= c < W and out[r, c] == background:
                out[r, c] = col

    return out