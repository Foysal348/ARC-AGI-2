import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform 4s by moving them according to directions encoded in colored frames.

    For each color C that forms a closed rectangular frame containing both 1s and 2s,
    the movement direction for color C is the 8-neighbor direction from the centroid
    of the 1s to the centroid of the 2s.

    Each 4 is assigned to the relevant color/maze and moved step-by-step in that
    direction until the next cell is blocked. Original 4s are removed and final 4s
    are placed in the output grid.
    """
    grid = np.array(grid, dtype=int)
    H, W = grid.shape

    if H == 0 or W == 0:
        return grid.copy()

    fours = np.argwhere(grid == 4)
    if len(fours) == 0:
        return grid.copy()

    directions = {}
    scores = {}

    # 0 is background, 2 is the "head" marker, 4 is the moving dot.
    # Color 1 can also be a frame color.
    candidate_colors = {int(x) for x in np.unique(grid) if int(x) not in (0, 2, 4)}

    # ------------------------------------------------------------------
    # 1) Detect instruction frames and compute one direction per color.
    # ------------------------------------------------------------------
    for C in candidate_colors:
        mask = (grid == C)
        visited = np.zeros((H, W), dtype=bool)

        for i in range(H):
            for j in range(W):
                if not mask[i, j] or visited[i, j]:
                    continue

                # BFS for this connected component of color C.
                comp = np.zeros((H, W), dtype=bool)
                stack = [(i, j)]
                visited[i, j] = True
                comp[i, j] = True

                rmin = rmax = i
                cmin = cmax = j

                while stack:
                    r, c = stack.pop()

                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < H
                            and 0 <= nc < W
                            and mask[nr, nc]
                            and not visited[nr, nc]
                        ):
                            visited[nr, nc] = True
                            comp[nr, nc] = True
                            stack.append((nr, nc))

                            if nr < rmin:
                                rmin = nr
                            elif nr > rmax:
                                rmax = nr
                            if nc < cmin:
                                cmin = nc
                            elif nc > cmax:
                                cmax = nc

                # Need a real frame with interior.
                if rmax - rmin < 2 or cmax - cmin < 2:
                    continue

                # The component must form the entire outer border of its bbox.
                if not comp[rmin, cmin:cmax + 1].all():
                    continue
                if not comp[rmax, cmin:cmax + 1].all():
                    continue
                if not comp[rmin:rmax + 1, cmin].all():
                    continue
                if not comp[rmin:rmax + 1, cmax].all():
                    continue

                interior = grid[rmin + 1:rmax, cmin + 1:cmax]

                # Locate 2s.
                twos_rel = np.argwhere(interior == 2)
                twos_off = np.array([rmin + 1, cmin + 1])
                if len(twos_rel) == 0:
                    twos_rel = np.argwhere(grid[rmin:rmax + 1, cmin:cmax + 1] == 2)
                    twos_off = np.array([rmin, cmin])
                    if len(twos_rel) == 0:
                        continue

                # Locate 1s. If color is 1 and there are no interior 1s,
                # fall back to other non-background/non-2 interior cells
                # or to the 1-frame itself.
                ones_rel = np.argwhere(interior == 1)
                ones_off = np.array([rmin + 1, cmin + 1])

                if len(ones_rel) == 0:
                    if C == 1:
                        tail_rel = np.argwhere((interior != 0) & (interior != 2))
                        if len(tail_rel) > 0:
                            ones_rel = tail_rel
                            ones_off = np.array([rmin + 1, cmin + 1])
                        else:
                            ones_rel = np.argwhere(grid[rmin:rmax + 1, cmin:cmax + 1] == 1)
                            ones_off = np.array([rmin, cmin])
                    else:
                        continue

                if len(ones_rel) == 0:
                    continue

                ones_abs = ones_rel + ones_off
                twos_abs = twos_rel + twos_off

                d = twos_abs.mean(axis=0) - ones_abs.mean(axis=0)
                dr = int(np.sign(d[0]))
                dc = int(np.sign(d[1]))

                if dr == 0 and dc == 0:
                    continue

                score = len(ones_rel) + len(twos_rel)
                if C not in scores or score > scores[C]:
                    directions[C] = (dr, dc)
                    scores[C] = score

    if not directions:
        return grid.copy()

    # Base grid with all moving dots temporarily removed.
    base = grid.copy()
    base[base == 4] = 0

    # Coordinates for each direction color, used for nearest-color assignment.
    color_coords = {}
    for C in directions:
        coords = np.argwhere(grid == C)
        if len(coords) > 0:
            color_coords[C] = coords

    def move_one(r, c, dr, dc, obstacles):
        """Move from (r,c) in direction (dr,dc) until blocked."""
        while True:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and obstacles[nr, nc] == 0:
                r, c = nr, nc
            else:
                break

        br, bc = r + dr, c + dc
        if 0 <= br < H and 0 <= bc < W:
            block = int(obstacles[br, bc])
        else:
            block = -1

        return r, c, block

    def min_dist_to_color(r, c, C):
        """Minimum Manhattan distance from (r,c) to any cell of color C."""
        coords = color_coords.get(C)
        if coords is None or len(coords) == 0:
            return 10**9
        return int(np.min(np.abs(coords[:, 0] - r) + np.abs(coords[:, 1] - c)))

    # ------------------------------------------------------------------
    # 2) Assign each 4 to a direction/color.
    # ------------------------------------------------------------------
    assignments = []

    for r, c in fours:
        r = int(r)
        c = int(c)

        candidates = []

        # A candidate color C is one whose encoded direction stops at a wall of C.
        for C, (dr, dc) in directions.items():
            if dr == 0 and dc == 0:
                continue

            tr, tc, block = move_one(r, c, dr, dc, base)
            if block == C:
                stop_dist = abs(tr - r) + abs(tc - c)
                nearest_dist = min_dist_to_color(r, c, C)
                candidates.append((nearest_dist, stop_dist, C))

        if candidates:
            # Prefer the closer surrounding color, then the shorter move.
            candidates.sort(key=lambda x: (x[0], x[1], x[2]))
            C = candidates[0][2]
        else:
            # Fallback: use the nearest known direction color.
            if color_coords:
                C = min(color_coords.keys(), key=lambda cc: min_dist_to_color(r, c, cc))
            else:
                C = None

        assignments.append((C, (r, c)))

    # ------------------------------------------------------------------
    # 3) Move 4s, avoiding overlaps when multiple 4s are in the same stream.
    # ------------------------------------------------------------------
    obstacles = base.copy()
    final_positions = []

    groups = {}
    for C, start in assignments:
        if C is None or C not in directions:
            key = (None, (0, 0))
        else:
            key = (C, directions[C])
        groups.setdefault(key, []).append(start)

    for (C, (dr, dc)), starts in groups.items():
        if C is None or (dr == 0 and dc == 0):
            for r, c in starts:
                final_positions.append((r, c))
            continue

        # Move front-most dots first so later dots stop before them if needed.
        starts_sorted = sorted(
            starts,
            key=lambda pos: pos[0] * dr + pos[1] * dc,
            reverse=True
        )

        for r, c in starts_sorted:
            tr, tc, _ = move_one(r, c, dr, dc, obstacles)
            final_positions.append((tr, tc))
            obstacles[tr, tc] = 4

    # ------------------------------------------------------------------
    # 4) Build output.
    # ------------------------------------------------------------------
    out = base.copy()
    for r, c in final_positions:
        out[r, c] = 4

    return out