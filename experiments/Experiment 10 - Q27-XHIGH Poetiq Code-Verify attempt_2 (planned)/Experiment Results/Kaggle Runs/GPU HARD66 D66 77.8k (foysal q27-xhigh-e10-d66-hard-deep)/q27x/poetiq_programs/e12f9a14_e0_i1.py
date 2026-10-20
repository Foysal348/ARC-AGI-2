import numpy as np
import heapq
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform hollow framed objects by "leaking" their interior colour out of
    missing frame cells.

    Rule implemented
    ----------------
    Each object is a 2x2 block of one colour, surrounded by a mostly-complete
    4x4 frame of another colour.  Missing frame cells are treated as holes.
    From each hole, the interior colour leaks outward in the hole's normal
    direction:
      - top/bottom/side holes leak N/S/E/W
      - corner holes leak diagonally

    The leak is routed to the nearest allowed border while avoiding:
      - input walls/frame/other interiors
      - leaks from different colours

    The routing strongly prefers straight motion and small turns, which yields
    the diagonal detours seen in the examples when two leaks would otherwise
    cross.
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape

    # ------------------------------------------------------------------
    # 1. Determine background colour.
    #    The border of the grid is almost always background in ARC grids.
    # ------------------------------------------------------------------
    border_vals = []
    border_vals.extend(grid[0, :].tolist())
    border_vals.extend(grid[-1, :].tolist())
    border_vals.extend(grid[:, 0].tolist())
    border_vals.extend(grid[:, -1].tolist())
    vals, counts = np.unique(np.array(border_vals), return_counts=True)
    background = int(vals[np.argmax(counts)])

    # ------------------------------------------------------------------
    # 2. Detect 2x2 interior blocks and their surrounding frame colour.
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

            # Skip if this 2x2 block overlaps a previously accepted object.
            pts = {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}
            if any(p in used_interior for p in pts):
                continue

            # Inspect the 12 border cells of the 4x4 bounding box.
            ring = []
            r0, c0 = r - 1, c - 1
            r1, c1 = r + 2, c + 2
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    if rr < 0 or rr >= H or cc < 0 or cc >= W:
                        continue
                    if (r0 <= rr <= r1 and c0 <= cc <= c1) and not (r <= rr <= r + 1 and c <= cc <= c + 1):
                        ring.append(int(grid[rr, cc]))

            # The frame colour is the most common non-background, non-interior
            # colour in the ring.
            frame_candidates = [x for x in ring if x != background and x != col]
            if not frame_candidates:
                continue

            fv, fc = np.unique(np.array(frame_candidates), return_counts=True)
            frame = int(fv[np.argmax(fc)])
            if max(fc) < 3:
                continue

            # Require the frame to touch at least two opposite sides of the
            # 4x4 box, to avoid random 2x2 blobs.
            top = any(int(grid[rr, cc]) == frame for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)
                      if rr == r0 and 0 <= rr < H and 0 <= cc < W)
            bottom = any(int(grid[rr, cc]) == frame for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)
                         if rr == r1 and 0 <= rr < H and 0 <= cc < W)
            left = any(int(grid[rr, cc]) == frame for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)
                       if cc == c0 and 0 <= rr < H and 0 <= cc < W)
            right = any(int(grid[rr, cc]) == frame for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)
                        if cc == c1 and 0 <= rr < H and 0 <= cc < W)

            if not ((top and bottom) or (left and right)):
                continue

            used_interior.update(pts)
            objects.append({
                "color": col,
                "frame": frame,
                "r": r,
                "c": c
            })

    # ------------------------------------------------------------------
    # 3. Create initial leak seeds from missing frame cells.
    # ------------------------------------------------------------------
    D8 = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)
    ]

    seeds = []
    for obj in objects:
        col = obj["color"]
        r, c = obj["r"], obj["c"]
        r0, c0 = r - 1, c - 1
        r1, c1 = r + 2, c + 2

        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if rr < 0 or rr >= H or cc < 0 or cc >= W:
                    continue
                if r <= rr <= r + 1 and c <= cc <= c + 1:
                    continue

                # Only background cells in the 4x4 border are holes.
                if grid[rr, cc] != background:
                    continue

                # Determine outward direction.
                if rr == r0 and cc == c0:
                    d = (-1, -1)
                elif rr == r0 and cc == c1:
                    d = (-1, 1)
                elif rr == r1 and cc == c0:
                    d = (1, -1)
                elif rr == r1 and cc == c1:
                    d = (1, 1)
                elif rr == r0:
                    d = (-1, 0)
                elif rr == r1:
                    d = (1, 0)
                elif cc == c0:
                    d = (0, -1)
                elif cc == c1:
                    d = (0, 1)
                else:
                    continue

                seeds.append({
                    "color": col,
                    "start": (rr, cc),
                    "dir": d
                })

    # If no objects/holes are detected, return unchanged.
    if not seeds:
        return grid.copy()

    # ------------------------------------------------------------------
    # 4. Static input walls: everything not background.
    # ------------------------------------------------------------------
    input_wall = (grid != background)

    # ------------------------------------------------------------------
    # 5. Helper: initial straight leak path for seeding the router.
    # ------------------------------------------------------------------
    def straight_path(seed):
        r, c = seed["start"]
        dr, dc = seed["dir"]
        path = [(r, c)]

        nr, nc = r + dr, c + dc
        steps = 0
        while 0 <= nr < H and 0 <= nc < W and not input_wall[nr, nc]:
            path.append((nr, nc))
            nr += dr
            nc += dc
            steps += 1
            if steps > H + W:
                break

        return path

    # ------------------------------------------------------------------
    # 6. Dijkstra router.
    # ------------------------------------------------------------------
    # Cost constants:
    #   step cost dominates
    #   turning cost encourages straight/45-degree motion
    #   target tie-breaker prefers exits near the original straight exit
    STEP = 100
    TURN45 = 10
    TURN90 = 25
    TURN135 = 45
    TURN180 = 80
    TARGET = 1

    def dir_index(dr, dc):
        for i, (x, y) in enumerate(D8):
            if x == dr and y == dc:
                return i
        return -1

    def turn_cost(a, b):
        if a == b:
            return 0
        da = (D8[a][0] + D8[b][0], D8[a][1] + D8[b][1])
        # angle by dot/cross
        x1, y1 = D8[a]
        x2, y2 = D8[b]
        dot = x1 * x2 + y1 * y2
        cross = x1 * y2 - y1 * x2
        if dot == 0 and cross == 0:
            return TURN180
        if dot < 0:
            if cross == 0:
                return TURN180
            return TURN135
        if dot == 0:
            return TURN90
        return TURN45

    def ideal_exit(seed):
        r, c = seed["start"]
        dr, dc = seed["dir"]
        last = (r, c)
        steps = 0
        while 0 <= r < H and 0 <= c < W:
            last = (r, c)
            r += dr
            c += dc
            steps += 1
            if steps > H + W:
                break
        return last

    def allowed_target(seed, r, c):
        sr, sc = seed["start"]
        dr, dc = seed["dir"]
        drt = r - sr
        dct = c - sc

        # Forward half-plane / quadrant.
        if dr < 0 and r > sr:
            return False
        if dr > 0 and r < sr:
            return False
        if dc < 0 and c > sc:
            return False
        if dc > 0 and c < sc:
            return False

        dot = drt * dr + dct * dc
        return dot > 0

    def best_target_penalty(seed, r, c):
        ir, ic = ideal_exit(seed)
        return abs(r - ir) + abs(c - ic)

    def route_seed(seed, blocked_other_colors):
        """
        Route one leak to a border.  Returns list of (r,c) cells.
        """
        sr, sc = seed["start"]
        dr, dc = seed["dir"]

        # If start is already border and allowed, just use it.
        if sr == 0 or sr == H - 1 or sc == 0 or sc == W - 1:
            if allowed_target(seed, sr, sc):
                return [(sr, sc)]

        # Prefer the mandatory first step in the hole direction.
        nr, nc = sr + dr, sc + dc
        start_state = None

        def blocked(rr, cc):
            if rr < 0 or rr >= H or cc < 0 or cc >= W:
                return True
            if input_wall[rr, cc]:
                return True
            if (rr, cc) in blocked_other_colors:
                return True
            return False

        if 0 <= nr < H and 0 <= nc < W and not blocked(nr, nc):
            start_state = (nr, nc, dir_index(dr, dc))
        else:
            # Fallback: begin at the hole, but still bias toward d.
            start_state = (sr, sc, dir_index(dr, dc))

        start_r, start_c, start_dir = start_state
        init_cost = 0

        # Dijkstra over (r,c,dir)
        INF = 10**18
        dist = {}
        parent = {}
        key = (start_r, start_c, start_dir)
        dist[key] = init_cost
        heap = [(init_cost, 0, start_r, start_c, start_dir)]

        best_border = None
        best_border_cost = INF
        best_border_state = None

        while heap:
            cost, steps, r, c, d = heapq.heappop(heap)
            cur = (r, c, d)
            if dist.get(cur, INF) < cost:
                continue

            is_border = (r == 0 or r == H - 1 or c == 0 or c == W - 1)
            if is_border and allowed_target(seed, r, c):
                total = cost + TARGET * best_target_penalty(seed, r, c)
                if total < best_border_cost:
                    best_border_cost = total
                    best_border_state = cur

            # Move to neighbours.
            for nd in range(8):
                ndr, ndc = D8[nd]
                nr2, nc2 = r + ndr, c + ndc
                if blocked(nr2, nc2):
                    continue

                add = STEP + turn_cost(d, nd)
                ncost = cost + add
                nkey = (nr2, nc2, nd)
                if ncost < dist.get(nkey, INF):
                    dist[nkey] = ncost
                    parent[nkey] = cur
                    heapq.heappush(heap, (ncost, steps + 1, nr2, nc2, nd))

        if best_border_state is None:
            # Could not route; just keep the hole.
            return [(sr, sc)]

        # Reconstruct path.
        path = []
        cur = best_border_state
        while cur is not None:
            path.append((cur[0], cur[1]))
            cur = parent.get(cur)

        # Add the seed start if reconstruction did not include it
        # (can happen when start_state was the seed itself).
        if path[-1] != (sr, sc):
            # Find where the path begins; prepend seed if adjacent/needed.
            path.append((sr, sc))
            # Reverse order: reconstruction is target->source
            pass

        path = list(reversed(path))

        # If the source cell is missing, insert it at the front.
        if path[0] != (sr, sc):
            path.insert(0, (sr, sc))

        # Remove consecutive duplicates.
        clean = []
        for p in path:
            if not clean or clean[-1] != p:
                clean.append(p)
        return clean

    # ------------------------------------------------------------------
    # 7. Iteratively route all leaks using each other's previous paths as
    #    soft obstacles.  Simultaneous update reduces order bias.
    # ------------------------------------------------------------------
    paths = [straight_path(s) for s in seeds]

    for _ in range(8):
        blocked_by_color = defaultdict(set)
        for i, p in enumerate(paths):
            col = seeds[i]["color"]
            for cell in p:
                blocked_by_color[col].add(cell)

        new_paths = []
        changed = False

        for i, seed in enumerate(seeds):
            # Block cells of other colours only.
            blocked = set()
            for col, cells in blocked_by_color.items():
                if col != seed["color"]:
                    blocked.update(cells)

            p = route_seed(seed, blocked)
            new_paths.append(p)

            if p != paths[i]:
                changed = True

        paths = new_paths
        if not changed:
            break

    # ------------------------------------------------------------------
    # 8. Paint final output.
    # ------------------------------------------------------------------
    out = grid.copy()

    # Sort by colour for deterministic conflict handling: smaller colour first.
    order = sorted(range(len(seeds)), key=lambda i: (seeds[i]["color"], paths[i][0][0], paths[i][0][1]))
    for i in order:
        col = seeds[i]["color"]
        for r, c in paths[i]:
            if 0 <= r < H and 0 <= c < W:
                # Only fill background cells; do not overwrite input objects.
                if out[r, c] == background:
                    out[r, c] = col

    return out