import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Recover the thin centerline of the thick colored "band", color that
    centerline with the repeating legend pattern, and mark dead-end /
    omitted-turn cells with the endpoint color.

    Inferred rule:
    1. The top-left corner is the background color.
    2. The most common non-background color is the thick band.
    3. Small 2x2 blocks in the top-left give the repeating color pattern.
       A small 2x2 block in the bottom-right gives the endpoint color.
       These legend blocks are erased from the output.
    4. Single left-edge pixels matching the first pattern color are path
       starts.
    5. Skeletonize the thick band (Zhang-Suen thinning).  The skeleton plus
       the start pixels forms the thin path graph.
    6. Multi-source BFS from the starts gives a distance for each path cell.
       Each path cell is painted pattern[distance % len(pattern)].
    7. For each directed path node, if the straight continuation of the
       incoming direction is missing, paint that continuation cell with the
       endpoint color.  This marks dead ends and the spurs created by bends.
    """
    def zs_skeleton(mask):
        """Vectorized Zhang-Suen thinning for a boolean mask."""
        img = np.pad(mask.astype(np.uint8), 1, constant_values=0)

        while True:
            changed = False

            for step in (1, 2):
                center = img[1:-1, 1:-1]
                if not center.any():
                    break

                p2 = img[:-2, 1:-1]      # up
                p3 = img[:-2, 2:]        # up-right
                p4 = img[1:-1, 2:]       # right
                p5 = img[2:, 2:]         # down-right
                p6 = img[2:, 1:-1]       # down
                p7 = img[2:, :-2]        # down-left
                p8 = img[1:-1, :-2]      # left
                p9 = img[:-2, :-2]       # up-left

                B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9

                A = (
                    ((p2 == 0) & (p3 == 1)).astype(np.uint8) +
                    ((p3 == 0) & (p4 == 1)).astype(np.uint8) +
                    ((p4 == 0) & (p5 == 1)).astype(np.uint8) +
                    ((p5 == 0) & (p6 == 1)).astype(np.uint8) +
                    ((p6 == 0) & (p7 == 1)).astype(np.uint8) +
                    ((p7 == 0) & (p8 == 1)).astype(np.uint8) +
                    ((p8 == 0) & (p9 == 1)).astype(np.uint8) +
                    ((p9 == 0) & (p2 == 1)).astype(np.uint8)
                )

                if step == 1:
                    cond = (
                        (B >= 2) & (B <= 6) & (A == 1) &
                        ((p2 & p4 & p6) == 0) &
                        ((p4 & p6 & p8) == 0)
                    )
                else:
                    cond = (
                        (B >= 2) & (B <= 6) & (A == 1) &
                        ((p2 & p4 & p8) == 0) &
                        ((p2 & p6 & p8) == 0)
                    )

                delete = (center == 1) & cond
                if delete.any():
                    img[1:-1, 1:-1] = np.where(delete, 0, img[1:-1, 1:-1])
                    changed = True

            if not changed:
                break

        return img[1:-1, 1:-1].astype(bool)

    g = np.array(grid, dtype=int, copy=True)
    if g.size == 0:
        return g

    H, W = g.shape
    out = g.copy()

    if H < 2 or W < 2:
        return out

    dirs4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    dirs8 = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    bg = int(g[0, 0])

    colors, counts_arr = np.unique(g, return_counts=True)
    counts = {int(c): int(n) for c, n in zip(colors, counts_arr)}

    non_bg_colors = [c for c in counts if c != bg]
    if not non_bg_colors:
        return out

    band_color = max(non_bg_colors, key=lambda c: counts[c])

    def in_2x2(r, c, col):
        """True if (r,c) belongs to any solid 2x2 block of color col."""
        for rr in (r - 1, r):
            for cc in (c - 1, c):
                if 0 <= rr < H - 1 and 0 <= cc < W - 1:
                    if (g[rr, cc] == col and
                        g[rr + 1, cc] == col and
                        g[rr, cc + 1] == col and
                        g[rr + 1, cc + 1] == col):
                        return True
        return False

    def bfs_distance(path_set, starts, dirs):
        """Unweighted BFS distances from starts over path_set."""
        dist = {}
        q = deque()

        for s in starts:
            if s in path_set and s not in dist:
                dist[s] = 0
                q.append(s)

        while q:
            v = q.popleft()
            r, c = v
            for dr, dc in dirs:
                u = (r + dr, c + dc)
                if u in path_set and u not in dist:
                    dist[u] = dist[v] + 1
                    q.append(u)

        return dist

    def mask_components(mask):
        """4-connected component labels for the band mask."""
        ids = -np.ones((H, W), dtype=int)
        comp = 0

        for r in range(H):
            for c in range(W):
                if mask[r, c] and ids[r, c] == -1:
                    ids[r, c] = comp
                    q = deque([(r, c)])

                    while q:
                        cr, cc = q.popleft()
                        for dr, dc in dirs4:
                            nr, nc = cr + dr, cc + dc
                            if (0 <= nr < H and 0 <= nc < W and
                                mask[nr, nc] and ids[nr, nc] == -1):
                                ids[nr, nc] = comp
                                q.append((nr, nc))

                    comp += 1

        return ids

    # ------------------------------------------------------------------
    # Find 2x2 legend blocks.  The thick band itself is excluded.
    # ------------------------------------------------------------------
    blocks = []
    used = np.zeros((H, W), dtype=bool)

    for r in range(H - 1):
        for c in range(W - 1):
            col = int(g[r, c])
            if col == bg or col == band_color:
                continue

            if (g[r, c + 1] == col and
                g[r + 1, c] == col and
                g[r + 1, c + 1] == col):

                if used[r, c] or used[r, c + 1] or used[r + 1, c] or used[r + 1, c + 1]:
                    continue

                blocks.append((r, c, col))
                used[r, c] = True
                used[r, c + 1] = True
                used[r + 1, c] = True
                used[r + 1, c + 1] = True

    # Pattern blocks are the top-left legend blocks.
    pattern_blocks = [b for b in blocks if b[0] < 10 and b[1] < 15]
    if not pattern_blocks:
        pattern_blocks = [b for b in blocks if b[0] < H // 2 and b[1] < W // 2]

    pattern_blocks.sort(key=lambda b: (b[1], b[0]))
    pattern = [b[2] for b in pattern_blocks]

    # Endpoint block is the bottom-right legend block.
    endpoint_blocks = [b for b in blocks if b[0] >= H - 8 and b[1] >= W - 8]
    if not endpoint_blocks:
        endpoint_blocks = [b for b in blocks if b[0] + b[1] >= H / 2 + W / 2]

    endpoint_block = None
    endpoint_color = None

    if endpoint_blocks:
        endpoint_block = max(endpoint_blocks, key=lambda b: (b[0] + b[1], b[0], b[1]))
        endpoint_color = endpoint_block[2]

    # Fallbacks for unusual cases.
    if not pattern:
        for r in range(H):
            col = int(g[r, 0])
            if col == bg or col == band_color:
                continue
            if (r + 1 < H and g[r + 1, 0] == band_color) or \
               (r > 0 and g[r - 1, 0] == band_color) or \
               (g[r, 1] == band_color):
                pattern = [col]
                break

        if not pattern:
            seen = []
            for r in range(min(H, 10)):
                for c in range(min(W, 15)):
                    col = int(g[r, c])
                    if col != bg and col != band_color and col not in seen:
                        seen.append(col)
            pattern = seen

    if not pattern:
        return out

    # ------------------------------------------------------------------
    # Find start pixels: isolated left-edge pixels of the first pattern
    # color adjacent to the thick band.
    # ------------------------------------------------------------------
    starts = []
    first_pattern_color = pattern[0]

    for r in range(H):
        col = int(g[r, 0])
        if col == bg or col == band_color or col != first_pattern_color:
            continue

        # Starts are single isolated edge pixels, not part of a 2x2 block.
        if (r > 0 and g[r - 1, 0] == col) or (r + 1 < H and g[r + 1, 0] == col):
            continue
        if r + 1 < H and g[r, 1] == col and g[r + 1, 1] == col:
            continue
        if r - 1 >= 0 and g[r - 1, 1] == col and g[r - 1, 0] == col:
            continue

        if (g[r, 1] == band_color) or \
           (r + 1 < H and g[r + 1, 0] == band_color) or \
           (r > 0 and g[r - 1, 0] == band_color):
            starts.append((r, 0))

    # Fallback: any matching single pixel adjacent to the band.
    if not starts:
        for r in range(H):
            for c in range(W):
                col = int(g[r, c])
                if col != first_pattern_color or col == band_color:
                    continue
                if in_2x2(r, c, col):
                    continue

                if (r > 0 and g[r - 1, c] == band_color) or \
                   (r + 1 < H and g[r + 1, c] == band_color) or \
                   (c > 0 and g[r, c - 1] == band_color) or \
                   (c + 1 < W and g[r, c + 1] == band_color):
                    starts.append((r, c))

    # ------------------------------------------------------------------
    # Skeletonize the thick band and build the thin path graph.
    # ------------------------------------------------------------------
    mask = (g == band_color)

    if mask.any() and starts:
        skel = zs_skeleton(mask)

        # If thinning produced nothing (should be rare), use a simple
        # one-pixel erosion as a fallback.
        if not skel.any():
            eroded = np.zeros_like(mask)
            eroded[1:-1, 1:-1] = (
                mask[1:-1, 1:-1] &
                mask[:-2, 1:-1] &
                mask[2:, 1:-1] &
                mask[1:-1, :-2] &
                mask[1:-1, 2:]
            )
            if eroded.any():
                skel = eroded
            else:
                skel = mask

        skel_coords = set(map(tuple, np.argwhere(skel)))
        path_set = set(skel_coords) | set(starts)

        # Ensure each start is connected to the skeleton.  Normally it is
        # directly adjacent, but add a short bridge if necessary.
        for s in starts:
            r, c = s

            connected = False
            for dr, dc in dirs4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) in path_set and (nr, nc) != s:
                    connected = True
                    break

            if connected:
                continue

            q = deque([(s, 0)])
            parent = {s: None}
            found = None

            while q:
                v, d = q.popleft()
                if d >= 10:
                    continue

                vr, vc = v
                for dr, dc in dirs4:
                    nr, nc = vr + dr, vc + dc
                    nb = (nr, nc)

                    if not (0 <= nr < H and 0 <= nc < W) or nb in parent:
                        continue

                    if mask[nr, nc] or nb in skel_coords:
                        parent[nb] = v
                        if nb in skel_coords:
                            found = nb
                            break
                        q.append((nb, d + 1))

                if found is not None:
                    break

            if found is not None:
                cur = found
                while cur != s:
                    path_set.add(cur)
                    cur = parent[cur]
            else:
                # Last-resort bridge: add immediate band neighbors.
                for dr, dc in dirs4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc]:
                        path_set.add((nr, nc))

        # First try 4-connected traversal.
        dist = bfs_distance(path_set, starts, dirs4)
        used_dirs = dirs4

        # If 4-connectivity is insufficient, allow 8-connectivity only for
        # skeleton cells in the same band component as a start.
        if len(dist) < len(path_set):
            comp_ids = mask_components(mask)
            start_comps = set()

            for s in starts:
                r, c = s
                for dr, dc in dirs4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc]:
                        if comp_ids[nr, nc] >= 0:
                            start_comps.add(int(comp_ids[nr, nc]))

            need8 = False
            if start_comps:
                for v in skel_coords:
                    if v not in dist:
                        cr, cc = v
                        if 0 <= cr < H and 0 <= cc < W and int(comp_ids[cr, cc]) in start_comps:
                            need8 = True
                            break

            if need8:
                dist8 = bfs_distance(path_set, starts, dirs8)
                start_set = set(starts)
                filtered = {}

                for v, d in dist8.items():
                    if v in start_set:
                        filtered[v] = d
                    else:
                        cr, cc = v
                        if (0 <= cr < H and 0 <= cc < W and
                            mask[cr, cc] and
                            int(comp_ids[cr, cc]) in start_comps):
                            filtered[v] = d

                if len(filtered) > len(dist):
                    dist = filtered
                    used_dirs = dirs8

        # Paint the thin path with the repeating pattern.
        L = len(pattern)
        for v, d in dist.items():
            out[v[0], v[1]] = int(pattern[d % L])

        # Paint endpoint color where the straight continuation is missing.
        if endpoint_color is not None:
            for v, d in dist.items():
                r, c = v

                neigh = []
                for dr, dc in used_dirs:
                    u = (r + dr, c + dc)
                    if u in path_set:
                        neigh.append(u)

                if not neigh:
                    continue

                preds = [u for u in neigh if dist.get(u) is not None and dist[u] == d - 1]

                # A directed turn/dead-end has exactly one immediate predecessor.
                if len(preds) != 1:
                    continue

                u = preds[0]
                dr = r - u[0]
                dc = c - u[1]
                target = (r + dr, c + dc)

                # If the path continues straight, no endpoint here.
                if target in path_set:
                    continue

                tr, tc = target
                if 0 <= tr < H and 0 <= tc < W:
                    out[tr, tc] = int(endpoint_color)

    # Erase the legend blocks in the final grid.
    for r, c, _ in pattern_blocks:
        out[r:r + 2, c:c + 2] = int(bg)

    if endpoint_block is not None:
        r, c, _ = endpoint_block
        out[r:r + 2, c:c + 2] = int(bg)

    return out