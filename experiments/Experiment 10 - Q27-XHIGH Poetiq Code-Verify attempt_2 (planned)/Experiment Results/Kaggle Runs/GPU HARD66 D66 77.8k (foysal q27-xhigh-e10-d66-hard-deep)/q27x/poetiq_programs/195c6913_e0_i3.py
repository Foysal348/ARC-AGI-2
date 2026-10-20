import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Recover the thin monotone path hidden inside the thick band.

    Rule inferred from the examples:
    * The top-left color is the background.
    * The most frequent non-background color is the thick band.
    * Small 2x2 blocks in the top-left are the repeating color legend.
    * A small 2x2 block in the bottom-right is the endpoint / dead-end color.
    * Single left-edge cells with the first legend color are path starts.
    * The hidden path is a 1-pixel centerline of the band, moving only up
      and right from the starts.  It is obtained by skeletonizing the band,
      converting the skeleton to a 4-connected monotone path, pruning very
      short spurs, and keeping only cells reachable by up/right moves.
    * The recovered path is painted with the legend pattern by BFS distance
      from the starts.  The endpoint color is painted in the straight-ahead
      cell at every turn / dead end where the path does not continue.
    """
    if grid.size == 0:
        return grid

    g = np.array(grid, dtype=int, copy=True)
    H, W = g.shape
    out = g.copy()

    if H < 2 or W < 2:
        return out

    # ------------------------------------------------------------------
    # Basic color identification.
    # ------------------------------------------------------------------
    bg = int(g[0, 0])
    colors, counts_arr = np.unique(g, return_counts=True)
    counts = {int(c): int(n) for c, n in zip(colors, counts_arr)}

    non_bg = [c for c in counts if c != bg]
    if not non_bg:
        return out

    # The thick band is the most frequent non-background color.
    band_color = max(non_bg, key=lambda c: counts[c])

    # ------------------------------------------------------------------
    # Find 2x2 legend blocks.
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
                if (used[r, c] or used[r, c + 1] or
                    used[r + 1, c] or used[r + 1, c + 1]):
                    continue
                blocks.append((r, c, col))
                used[r, c] = True
                used[r, c + 1] = True
                used[r + 1, c] = True
                used[r + 1, c + 1] = True

    # Top-left legend blocks are the pattern, read left-to-right/top-to-bottom.
    pattern_blocks = [b for b in blocks if b[0] < H // 2 and b[1] < W // 2]
    if not pattern_blocks:
        pattern_blocks = [b for b in blocks if b[0] < 10 and b[1] < 12]
    if not pattern_blocks:
        pattern_blocks = [b for b in blocks if b[0] + b[1] < (H + W) // 2]

    pattern_blocks.sort(key=lambda b: (b[1], b[0]))
    pattern = [b[2] for b in pattern_blocks]

    # Bottom-right legend block is the endpoint color.
    endpoint_blocks = [b for b in blocks if b[0] >= H // 2 and b[1] >= W // 2]
    if not endpoint_blocks:
        endpoint_blocks = [b for b in blocks if b[0] >= H - 8 and b[1] >= W - 8]
    if not endpoint_blocks:
        endpoint_blocks = [b for b in blocks if b[0] + b[1] >= (H + W) // 2]

    endpoint_block = None
    endpoint_color = None
    if endpoint_blocks:
        endpoint_block = max(endpoint_blocks, key=lambda b: (b[0] + b[1], b[0], b[1]))
        endpoint_color = endpoint_block[2]

    if not pattern:
        # Fallback: use the most common color that is neither bg nor band.
        fallback = [c for c in counts if c != bg and c != band_color]
        if fallback:
            pattern = [max(fallback, key=lambda c: counts[c])]
        else:
            return out

    # Erase legend blocks in the output.
    for r, c, _ in pattern_blocks:
        out[r:r + 2, c:c + 2] = bg
    if endpoint_block is not None:
        r, c, _ = endpoint_block
        out[r:r + 2, c:c + 2] = bg

    # ------------------------------------------------------------------
    # Starts: single left-edge pixels of the first pattern color adjacent
    # to the thick band.
    # ------------------------------------------------------------------
    first_color = pattern[0]
    starts = []

    for r in range(H):
        col = int(g[r, 0])
        if col != first_color:
            continue

        # Ignore cells that are part of a 2x2 legend block.
        in_2x2 = False
        for rr in (r - 1, r):
            for cc in (0, 1):
                if 0 <= rr < H - 1 and 0 <= cc < W - 1:
                    if (g[rr, cc] == col and
                        g[rr + 1, cc] == col and
                        g[rr, cc + 1] == col and
                        g[rr + 1, cc + 1] == col):
                        in_2x2 = True
        if in_2x2:
            continue

        # It should touch the band.
        touches = False
        for dr, dc in ((0, 1), (1, 0), (-1, 0)):
            nr, nc = r + dr, c = 0 + dc
            if 0 <= nr < H and 0 <= nc < W and int(g[nr, nc]) == band_color:
                touches = True
                break

        if touches:
            starts.append((r, 0))

    if not starts:
        # Fallback: any isolated first-color pixel adjacent to band.
        for r in range(H):
            for c in range(W):
                if int(g[r, c]) != first_color:
                    continue
                touches = False
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and int(g[nr, nc]) == band_color:
                        touches = True
                        break
                if touches:
                    starts.append((r, c))

    if not starts:
        return out

    # ------------------------------------------------------------------
    # Thin the thick band to a 1-pixel skeleton.
    # ------------------------------------------------------------------
    def zs_skeleton(mask: np.ndarray) -> np.ndarray:
        """Zhang-Suen thinning for a boolean mask."""
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

    band_mask = (g == band_color)
    skel = zs_skeleton(band_mask)

    # If thinning failed completely, use a very thin erosion fallback.
    if not skel.any():
        eroded = np.zeros((H, W), dtype=bool)
        if H >= 3 and W >= 3:
            eroded[1:-1, 1:-1] = (
                band_mask[1:-1, 1:-1] &
                band_mask[:-2, 1:-1] &
                band_mask[2:, 1:-1] &
                band_mask[1:-1, :-2] &
                band_mask[1:-1, 2:]
            )
        if eroded.any():
            skel = eroded
        else:
            skel = band_mask.copy()

    # Keep only skeleton cells that are 4-connected to the band component(s)
    # adjacent to starts.
    def component_ids(mask: np.ndarray):
        ids = -np.ones((H, W), dtype=int)
        comp = 0
        for r in range(H):
            for c in range(W):
                if mask[r, c] and ids[r, c] == -1:
                    ids[r, c] = comp
                    q = deque([(r, c)])
                    while q:
                        cr, cc = q.popleft()
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = cr + dr, cc + dc
                            if (0 <= nr < H and 0 <= nc < W and
                                mask[nr, nc] and ids[nr, nc] == -1):
                                ids[nr, nc] = comp
                                q.append((nr, nc))
                    comp += 1
        return ids

    band_ids = component_ids(band_mask)
    start_comps = set()

    for r, c in starts:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and band_mask[nr, nc]:
                start_comps.add(int(band_ids[nr, nc]))

    # If the start is itself in band (unusual), include its component too.
    for r, c in starts:
        if band_mask[r, c]:
            start_comps.add(int(band_ids[r, c]))

    if start_comps:
        keep = np.isin(band_ids, list(start_comps))
        skel = skel & keep

    if not skel.any():
        # Last resort: use the band itself as the path mask.
        skel = band_mask.copy()

    # ------------------------------------------------------------------
    # Convert the possibly 8-connected skeleton into a 4-connected mask.
    # ------------------------------------------------------------------
    path_mask = skel.copy()

    def band4(r, c):
        return 0 <= r < H and 0 <= c < W and band_mask[r, c]

    def score_candidate(r, c):
        """
        Local score for choosing an intermediate cell when resolving a
        diagonal skeleton connection.  Higher is preferred.
        """
        if not (0 <= r < H and 0 <= c < W):
            return -1000
        if not band_mask[r, c]:
            return -1000

        s = 0
        # Prefer cells with support in the two forward directions.
        if 0 <= r - 1 < H and 0 <= c < W and band_mask[r - 1, c]:
            s += 3
        if 0 <= r < H and 0 <= c + 1 < W and band_mask[r, c + 1]:
            s += 3

        # Prefer cells that can continue at least one more step forward.
        if 0 <= r - 2 < H and 0 <= c < W and band_mask[r - 2, c]:
            s += 2
        if 0 <= r < H and 0 <= c + 2 < W and band_mask[r, c + 2]:
            s += 2

        # Slight preference for cells already in the skeleton.
        if path_mask[r, c]:
            s += 10

        return s

    # Resolve diagonal connections in 2x2 windows.
    for r in range(H - 1):
        for c in range(W - 1):
            # Main diagonal: (r,c) and (r+1,c+1)
            if path_mask[r, c] and path_mask[r + 1, c + 1]:
                # Already 4-connected if one orthogonal cell exists.
                if path_mask[r, c + 1] or path_mask[r + 1, c]:
                    continue
                a = (r, c + 1)
                b = (r + 1, c)
                sa = score_candidate(*a)
                sb = score_candidate(*b)
                if sa >= sb and band4(*a):
                    path_mask[a] = True
                elif sb > sa and band4(*b):
                    path_mask[b] = True

            # Anti-diagonal: (r,c+1) and (r+1,c)
            if path_mask[r, c + 1] and path_mask[r + 1, c]:
                if path_mask[r, c] or path_mask[r + 1, c + 1]:
                    continue
                a = (r, c)
                b = (r + 1, c + 1)
                sa = score_candidate(*a)
                sb = score_candidate(*b)
                if sa >= sb and band4(*a):
                    path_mask[a] = True
                elif sb > sa and band4(*b):
                    path_mask[b] = True

    # Ensure each start is connected to the 4-connected path mask.
    def bridge_start_to_path(sr, sc):
        """
        Add a short 4-connected bridge from a start to the nearest path cell
        if necessary.
        """
        if path_mask[sr, sc]:
            return

        # Already adjacent?
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = sr + dr, sc + dc
            if 0 <= nr < H and 0 <= nc < W and path_mask[nr, nc]:
                return

        # BFS through band cells up to a small distance.
        best = None
        q = deque([(sr, sc, 0)])
        seen = {(sr, sc): None}

        while q:
            r, c, d = q.popleft()
            if d > 12:
                continue
            if path_mask[r, c]:
                best = (r, c)
                break
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if (0 <= nr < H and 0 <= nc < W and
                    (nr, nc) not in seen and
                    (band_mask[nr, nc] or path_mask[nr, nc])):
                    seen[(nr, nc)] = (r, c)
                    q.append((nr, nc, d + 1))

        if best is not None:
            cur = best
            while cur != (sr, sc):
                path_mask[cur] = True
                cur = seen[cur]
        else:
            # Fallback: add immediate band neighbors.
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = sr + dr, sc + dc
                if 0 <= nr < H and 0 <= nc < W and band_mask[nr, nc]:
                    path_mask[nr, nc] = True

    for s in starts:
        bridge_start_to_path(*s)

    # Add starts themselves to the path mask.
    for sr, sc in starts:
        path_mask[sr, sc] = True

    # ------------------------------------------------------------------
    # Keep only cells reachable from starts using only up/right moves.
    # This enforces the monotone direction seen in all examples.
    # ------------------------------------------------------------------
    reachable = np.zeros((H, W), dtype=bool)
    q = deque()

    for sr, sc in starts:
        if path_mask[sr, sc]:
            reachable[sr, sc] = True
            q.append((sr, sc))

    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (0, 1)):  # up, right
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and path_mask[nr, nc] and not reachable[nr, nc]:
                reachable[nr, nc] = True
                q.append((nr, nc))

    # If the strict up/right reachability removed too much, fallback to
    # 4-connectivity from starts (should rarely be needed).
    if not reachable.any():
        q = deque()
        for sr, sc in starts:
            if path_mask[sr, sc]:
                reachable[sr, sc] = True
                q.append((sr, sc))
        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and path_mask[nr, nc] and not reachable[nr, nc]:
                    reachable[nr, nc] = True
                    q.append((nr, nc))

    path_mask = path_mask & reachable

    # ------------------------------------------------------------------
    # Prune very short spurs from the 4-connected monotone path graph.
    # We only prune 1-cell spurs; terminal cells on the top/right grid
    # boundary are protected.
    # ------------------------------------------------------------------
    def neighbors4(r, c):
        res = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and path_mask[nr, nc]:
                res.append((nr, nc))
        return res

    def prune_short_spurs():
        changed = True
        while changed:
            changed = False

            # Compute leaves.
            leaves = []
            for r in range(H):
                for c in range(W):
                    if not path_mask[r, c]:
                        continue
                    # Do not prune starts or boundary terminal cells.
                    if (r, c) in starts:
                        continue
                    if r == 0 or c == W - 1:
                        continue

                    nb = neighbors4(r, c)
                    if len(nb) == 1:
                        leaves.append((r, c))

            # For each leaf, check branch length.  If the branch is only
            # one cell long before hitting a junction/start, remove it.
            for r, c in leaves:
                nb = neighbors4(r, c)
                if len(nb) != 1:
                    continue
                pr, pc = nb[0]

                # The branch consists of the leaf plus maybe a chain of
                # degree-1 cells.  Stop at a node with degree != 1 or a start.
                length = 1
                cur = (r, c)
                prev = None

                while True:
                    nbs = neighbors4(*cur)
                    # Move away from prev.
                    nexts = [x for x in nbs if x != prev]
                    if len(nexts) == 0:
                        break
                    nxt = nexts[0]

                    if nxt in starts or (nxt[0] == 0 or nxt[1] == W - 1):
                        break

                    nbs_nxt = neighbors4(*nxt)
                    if len(nbs_nxt) != 1:
                        break

                    length += 1
                    prev = cur
                    cur = nxt

                    if length > 1:
                        break

                if length <= 1:
                    path_mask[r, c] = False
                    changed = True

    prune_short_spurs()

    # Reapply monotone reachability after pruning.
    reachable2 = np.zeros((H, W), dtype=bool)
    q = deque()
    for sr, sc in starts:
        if path_mask[sr, sc]:
            reachable2[sr, sc] = True
            q.append((sr, sc))

    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and path_mask[nr, nc] and not reachable2[nr, nc]:
                reachable2[nr, nc] = True
                q.append((nr, nc))

    if reachable2.any():
        path_mask = path_mask & reachable2

    if not path_mask.any():
        return out

    # ------------------------------------------------------------------
    # Color the recovered path with the legend pattern using BFS distance
    # from the starts, moving only up/right along the path graph.
    # ------------------------------------------------------------------
    dist = {}
    q = deque()

    for sr, sc in starts:
        if path_mask[sr, sc]:
            dist[(sr, sc)] = 0
            q.append((sr, sc))

    while q:
        r, c = q.popleft()
        d = dist[(r, c)]
        for dr, dc in ((-1, 0), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and path_mask[nr, nc]:
                if (nr, nc) not in dist:
                    dist[(nr, nc)] = d + 1
                    q.append((nr, nc))

    if not dist:
        return out

    L = len(pattern)
    for (r, c), d in dist.items():
        out[r, c] = int(pattern[d % L])

    # Preserve the original start marker colors.
    for sr, sc in starts:
        out[sr, sc] = int(g[sr, sc])

    # ------------------------------------------------------------------
    # Paint endpoint color at missing straight continuations.
    # ------------------------------------------------------------------
    if endpoint_color is not None:
        path_set = set(dist.keys())

        for (r, c), d in dist.items():
            preds = []
            # Predecessors are below (down) or left.
            for dr, dc in ((1, 0), (0, -1)):
                pr, pc = r + dr, c + dc
                if (pr, pc) in path_set and dist[(pr, pc)] == d - 1:
                    preds.append((pr, pc))

            # Only use a unique incoming direction.
            if len(preds) != 1:
                continue

            pr, pc = preds[0]
            dr = r - pr
            dc = c - pc

            sr, sc = r + dr, c + dc
            if 0 <= sr < H and 0 <= sc < W:
                # Do not overwrite a cell that is already part of the path.
                if (sr, sc) in path_set:
                    continue
                # Do not overwrite starts.
                if (sr, sc) in starts:
                    continue
                out[sr, sc] = int(endpoint_color)

    # Final safety: legend blocks must remain background unless overwritten
    # by endpoint/path cells.
    for r, c, _ in pattern_blocks:
        for rr in (r, r + 1):
            for cc in (c, c + 1):
                if (rr, cc) not in dist and (rr, cc) not in starts:
                    out[rr, cc] = bg

    if endpoint_block is not None:
        r, c, _ = endpoint_block
        for rr in (r, r + 1):
            for cc in (c, c + 1):
                if (rr, cc) not in dist and (rr, cc) not in starts:
                    out[rr, cc] = bg

    return out