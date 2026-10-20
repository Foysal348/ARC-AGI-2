import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    H, W = grid.shape
    arrow_color = 1

    # Identify payload-arrow centers: non-arrow cells with exactly three arrow
    # neighbours forming a T.  Their own colour is the payload colour.
    payload = {}
    for r in range(H):
        for c in range(W):
            if grid[r, c] == arrow_color:
                continue
            arms = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == arrow_color:
                    arms.append((dr, dc))
            if len(arms) == 3:
                payload[(r, c)] = (int(grid[r, c]), arms)

    # Cells belonging to arrow shapes / payloads are removed before building
    # host regions.
    blocked = np.zeros((H, W), dtype=bool)
    for r, c in range(H):
        if grid[r, c] == arrow_color:
            blocked[r, c] = True
    for (r, c) in payload:
        blocked[r, c] = True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == arrow_color:
                blocked[nr, nc] = True

    # Connected host regions in the unblocked grid.
    region = -np.ones((H, W), dtype=int)
    region_color = []
    region_cells = []

    for r in range(H):
        for c in range(W):
            if blocked[r, c] or region[r, c] != -1:
                continue
            rid = len(region_color)
            color = int(grid[r, c])
            region_color.append(color)
            cells = []
            q = deque([(r, c)])
            region[r, c] = rid
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if not blocked[nr, nc] and region[nr, nc] == -1:
                            region[nr, nc] = rid
                            q.append((nr, nc))
            region_cells.append(cells)

    def stem_direction(arms):
        for a in arms:
            others = [b for b in arms if b != a]
            if len(others) == 2:
                b1, b2 = others
                if b1[0] == -b2[0] and b1[1] == -b2[1]:
                    return a
        return None

    # Build edges: target_region -> source_color.
    edge_color = {}
    for (r, c), (src_color, arms) in payload.items():
        d = stem_direction(arms)
        if d is None:
            continue
        dr, dc = d
        rr, cc = r, c
        host = -1
        # The host is the region that surrounds the arrow.  Find it by looking
        # at unblocked neighbours of the payload cell / arm cells.
        candidates = {}
        for sr, sc in ((r, c), (r + dr, c + dc)):
            for a in arms:
                ar, ac = r + a[0], c + a[1]
                for x, y in ((ar, ac), (ar - a[0], ac - a[1])):
                    if 0 <= x < H and 0 <= y < W and not blocked[x, y]:
                        rid = region[x, y]
                        if rid >= 0:
                            candidates[rid] = candidates.get(rid, 0) + 1
        if candidates:
            host = max(candidates, key=lambda k: candidates[k])
        else:
            host = -1

        while True:
            rr += dr
            cc += dc
            if not (0 <= rr < H and 0 <= cc < W):
                break
            if blocked[rr, cc]:
                continue
            rid = region[rr, cc]
            if rid >= 0 and rid != host:
                edge_color[rid] = src_color
                break

    # Pure arrow clusters made only of arrow cells.
    seen = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if grid[r, c] != arrow_color or seen[r, c]:
                continue
            q = deque([(r, c)])
            cells = []
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and grid[nr, nc] == arrow_color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if len(cells) != 4:
                continue
            s = set(cells)
            center = None
            for rr, cc in cells:
                cnt = sum(1 for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                          if (rr + dr, cc + dc) in s)
                if cnt == 3:
                    center = (rr, cc)
                    break
            if center is None:
                continue
            cr, cc = center
            arms = [(dr, dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    if (cr + dr, cc + dc) in s]
            d = stem_direction(arms)
            if d is None:
                continue
            dr, dc = d
            host = -1
            candidates = {}
            for a in arms:
                ar, ac = cr + a[0], cc + a[1]
                for x, y in ((ar, ac), (ar - a[0], ac - a[1])):
                    if 0 <= x < H and 0 <= y < W and not blocked[x, y]:
                        rid = region[x, y]
                        if rid >= 0:
                            candidates[rid] = candidates.get(rid, 0) + 1
            if candidates:
                host = max(candidates, key=lambda k: candidates[k])
            src_color = region_color[host] if host >= 0 else None
            if src_color is None:
                continue
            rr, cc = cr, cc
            while True:
                rr += dr
                cc += dc
                if not (0 <= rr < H and 0 <= cc < W):
                    break
                if blocked[rr, cc]:
                    continue
                rid = region[rr, cc]
                if rid >= 0 and rid != host:
                    edge_color[rid] = src_color
                    break

    # Final region colours.
    final_color = list(region_color)
    for rid, col in edge_color.items():
        if rid >= 0:
            final_color[rid] = col

    # Paint the full input-sized solved map.
    solved = np.zeros((H, W), dtype=int)
    for r in range(H):
        for c in range(W):
            if not blocked[r, c] and region[r, c] >= 0:
                solved[r, c] = final_color[region[r, c]]

    # Fill arrow/payload holes by the nearest surrounding solved region.
    for r in range(H):
        for c in range(W):
            if solved[r, c] != 0:
                continue
            best = 0
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                rr, cc = r, c
                while 0 <= rr < H and 0 <= cc < W:
                    if solved[rr, cc] != 0:
                        best = solved[rr, cc]
                        break
                    rr += dr
                    cc += dc
                if best != 0:
                    break
            solved[r, c] = best

    # For non-square training cases the solved map is presented transposed.
    if H != W:
        solved = solved.T

    return solved