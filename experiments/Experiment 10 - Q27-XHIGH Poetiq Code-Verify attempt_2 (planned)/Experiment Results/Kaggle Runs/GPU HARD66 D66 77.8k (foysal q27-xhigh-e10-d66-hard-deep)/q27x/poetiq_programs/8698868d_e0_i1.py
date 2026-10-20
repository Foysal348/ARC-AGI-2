import numpy as np
from collections import defaultdict, deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a grid containing several rectangular "frame" objects into a compact
    grid of combined objects.

    Rule implemented
    ----------------
    1.  The background color is taken to be the most frequent color.
    2.  Every connected component of non-background pixels is treated as one object.
        Each object is represented by its bounding box and its main color.
    3.  Larger objects are used as outer frames. Smaller objects are used as inner
        contents when they can be centered inside a larger one.
        The size difference in each dimension must be even and at most 2, which
        matches the examples:
            - square 6x6 outer with 4x4 inner
            - square 8x8 outer with 6x6 inner
            - rectangular 8x7 outer with 6x7 inner
    4.  Outer/inner pairs are selected by a spatial scoring function that prefers
        close, aligned rectangles.  This makes the solution robust for the given
        challenge and still behaves sensibly on the examples.
    5.  For each selected pair:
         - create a rectangle of the outer object's bounding box filled with the
           outer color;
         - paste the inner object centered inside it;
         - wherever the inner object has background holes, keep the outer color.
    6.  The resulting combined rectangles are arranged in the same row/column layout
        as their outer objects, with no background gaps, and returned.
    """

    if grid.size == 0:
        return np.zeros_like(grid)

    # ------------------------------------------------------------------
    # 1. Background color
    # ------------------------------------------------------------------
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[np.argmax(counts)])

    H, W = grid.shape
    non_bg = grid != background

    # ------------------------------------------------------------------
    # 2. Connected components of non-background pixels
    # ------------------------------------------------------------------
    visited = np.zeros((H, W), dtype=bool)
    objs = []

    for r in range(H):
        for c in range(W):
            if non_bg[r, c] and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                cells = []
                while q:
                    y, x = q.popleft()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            if non_bg[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                q.append((ny, nx))

                ys = [p[0] for p in cells]
                xs = [p[1] for p in cells]
                top, bottom = min(ys), max(ys)
                left, right = min(xs), max(xs)

                # Main color of the component.
                comp_colors, comp_counts = np.unique(
                    grid[top : bottom + 1, left : right + 1][
                        np.array([grid[y, x] for y, x in cells], dtype=int)
                    ],
                    return_counts=True,
                )
                main_color = int(comp_colors[np.argmax(comp_counts)])

                objs.append(
                    {
                        "top": int(top),
                        "left": int(left),
                        "bottom": int(bottom),
                        "right": int(right),
                        "h": int(bottom - top + 1),
                        "w": int(right - left + 1),
                        "color": int(main_color),
                        "mask": non_bg[top : bottom + 1, left : right + 1],
                        "color_mask": (grid[top : bottom + 1, left : right + 1] == main_color),
                    }
                )

    # If no non-background objects were found, return an empty grid.
    if not objs:
        return np.zeros_like(grid)

    # Give objects a stable reading order.
    for i, obj in enumerate(objs):
        obj["idx"] = i
        obj["area"] = obj["h"] * obj["w"]

    # ------------------------------------------------------------------
    # 3. Candidate outer / inner relationships
    # ------------------------------------------------------------------
    n = len(objs)

    def can_contain(outer, inner):
        """
        Inner can be centered inside outer if:
          - outer is at least as tall/wide,
          - differences are even,
          - the maximum difference is at most 2.
        """
        dh = outer["h"] - inner["h"]
        dw = outer["w"] - inner["w"]
        if dh < 0 or dw < 0:
            return False
        if dh % 2 != 0 or dw % 2 != 0:
            return False
        if max(dh, dw) == 0:
            return False
        if max(dh, dw) > 2:
            return False
        return True

    # Build valid directed edges outer -> inner.
    valid = [[False] * n for _ in range(n)]
    for i, a in enumerate(objs):
        for j, b in enumerate(objs):
            if i != j and can_contain(a, b):
                valid[i][j] = True

    # Determine which objects are best treated as outer frames:
    # largest first, and an object is not allowed to be both outer and inner
    # in the final one-to-one matching.
    order_by_size = sorted(range(n), key=lambda i: -objs[i]["area"])

    outer_set = set()
    inner_used = set()
    for i in order_by_size:
        if i in inner_used:
            continue
        if any(valid[i][j] for j in range(n) if j not in inner_used):
            outer_set.add(i)

    if not outer_set:
        # Fallback: if the above heuristic finds nothing, use the largest objects.
        for i in order_by_size:
            if any(valid[i][j] for j in range(n)):
                outer_set.add(i)
                break

    outer_ids = sorted(outer_set)
    m = len(outer_ids)

    # Inner candidates are all objects not selected as outer.
    inner_candidates = [j for j in range(n) if j not in outer_set]

    # If the counts do not match, keep only the largest inner candidates.
    if len(inner_candidates) > m:
        inner_candidates.sort(key=lambda j: -objs[j]["area"])
        inner_candidates = inner_candidates[:m]
    if len(inner_candidates) < m:
        # In rare irregular cases, allow any unused object as inner.
        extra = [j for j in range(n) if j not in outer_set and j not in inner_candidates]
        inner_candidates.extend(extra[: m - len(inner_candidates)])

    # ------------------------------------------------------------------
    # 4. Pairing score
    # ------------------------------------------------------------------
    def overlap_width(a, b):
        return max(0, min(a["right"], b["right"]) - max(a["left"], b["left"]) + 1)

    def overlap_height(a, b):
        return max(0, min(a["bottom"], b["bottom"]) - max(a["top"], b["top"]) + 1)

    def gap(a, b, axis):
        if axis == "v":
            if a["bottom"] < b["top"]:
                return b["top"] - a["bottom"]
            if b["bottom"] < a["top"]:
                return a["top"] - b["bottom"]
            return 0
        else:
            if a["right"] < b["left"]:
                return b["left"] - a["right"]
            if b["right"] < a["left"]:
                return a["left"] - b["right"]
            return 0

    def center_dist(a, b):
        ar = (a["top"] + a["bottom"]) / 2.0
        ac = (a["left"] + a["right"]) / 2.0
        br = (b["top"] + b["bottom"]) / 2.0
        bc = (b["left"] + b["right"]) / 2.0
        return abs(ar - br) + abs(ac - bc)

    def pair_score(oi, ii):
        a = objs[oi]
        b = objs[ii]
        if not valid[oi][ii]:
            return -10**12

        vg = gap(a, b, "v")
        hg = gap(a, b, "h")
        cd = center_dist(a, b)
        ovw = overlap_width(a, b)
        ovh = overlap_height(a, b)

        # Strong preference for close, aligned rectangles.
        score = -1000.0 * vg - 400.0 * hg - 50.0 * cd
        score += 10.0 * ovw
        score += 10.0 * ovh

        # Tiny tie-breakers for deterministic behavior.
        score += 0.01 * (n - oi)
        score += 0.001 * (n - ii)
        return score

    # Brute-force / bitmask DP is sufficient for ARC-sized inputs.
    # inner_idx: k -> inner_candidates[k]
    K = len(inner_candidates)

    if K < m:
        # Should not happen after fallback, but keep safe.
        return np.zeros_like(grid)

    # dp[mask] = best score using first popcount(mask) outers?
    # Simpler: recursive DP over outer index and mask of used inners.
    import functools

    sys.setrecursionlimit(10000)

    @functools.lru_cache(None)
    def dp(k, mask):
        if k == m:
            return 0.0, 0

        best = None
        for t in range(K):
            bit = 1 << t
            if mask & bit:
                continue
            s = pair_score(outer_ids[k], inner_candidates[t])
            if s == -10**12:
                continue
            sub, nxt = dp(k + 1, mask | bit)
            total = s + sub
            if best is None or total > best[0] or (total == best[0] and nxt < best[1]):
                best = (total, nxt | bit)
        if best is None:
            return -10**18, 0
        return best

    total, final_mask = dp(0, 0)

    if final_mask == 0:
        # No valid matching: return the largest object unchanged.
        big = max(objs, key=lambda x: x["area"])
        return big["color_mask"].astype(np.uint8) * big["color"]

    pairs = []
    for k, oi in enumerate(outer_ids):
        # Find which inner was used for this outer by replaying DP.
        mask = final_mask
        for kk in range(k + 1):
            if kk < k:
                # reconstruct previous choices by recomputing; small size, so easy
                pass
        # Simpler reconstruction:
    # Reconstruct explicitly.
    chosen = {}
    mask = final_mask
    for k in range(m):
        for t in range(K):
            bit = 1 << t
            if mask & bit:
                s = pair_score(outer_ids[k], inner_candidates[t])
                if s == -10**12:
                    continue
                # Check if remaining DP works
                rem, _ = dp(k + 1, mask ^ bit)
                if rem != -10**18:
                    chosen[outer_ids[k]] = inner_candidates[t]
                    mask ^= bit
                    break

    # ------------------------------------------------------------------
    # 5. Build combined object for each pair
    # ------------------------------------------------------------------
    def combine(outer_id, inner_id):
        a = objs[outer_id]
        b = objs[inner_id]

        out = np.full((a["h"], a["w"]), a["color"], dtype=np.uint8)

        off_r = (a["h"] - b["h"]) // 2
        off_c = (a["w"] - b["w"]) // 2

        # Paste inner object's main color; holes remain outer color.
        for r in range(b["h"]):
            for c in range(b["w"]):
                if b["color_mask"][r, c]:
                    out[r + off_r, c + off_c] = b["color"]
        return out

    combined = {}
    for oi, ii in chosen.items():
        combined[oi] = combine(oi, ii)

    # ------------------------------------------------------------------
    # 6. Arrange combined blocks according to outer-object layout
    # ------------------------------------------------------------------
    # Group outer objects into rows by similar top coordinate.
    outers = [objs[i] for i in chosen.keys()]
    outers.sort(key=lambda x: (x["top"], x["left"]))

    # Simple clustering of top coordinates.
    row_groups = []
    for o in outers:
        placed = False
        for grp in row_groups:
            rep = grp[0]
            if abs(o["top"] - rep["top"]) <= max(1, o["h"] // 2):
                grp.append(o)
                placed = True
                break
        if not placed:
            row_groups.append([o])

    row_groups.sort(key=lambda g: min(x["top"] for x in g))
    for g in row_groups:
        g.sort(key=lambda x: x["left"])

    # For each row, cluster by left coordinate if needed.  In practice the
    # outers are already in left order, so each row list defines the columns.
    max_h = max(o["h"] for o in outers)
    max_w = max(o["w"] for o in outers)

    # Output height is number of row groups * max_h;
    # output width is max columns in a row * max_w.
    num_rows = len(row_groups)
    num_cols = max(len(g) for g in row_groups) if row_groups else 0
    if num_cols == 0:
        return np.zeros_like(grid)

    output = np.zeros((num_rows * max_h, num_cols * max_w), dtype=np.uint8)

    for ri, grp in enumerate(row_groups):
        # Center this row's blocks vertically if their heights differ.
        for ci, o in enumerate(grp):
            if o["idx"] not in combined:
                continue
            block = combined[o["idx"]]
            bh, bw = block.shape

            y0 = ri * max_h + (max_h - bh) // 2
            x0 = ci * max_w + (max_w - bw) // 2
            output[y0 : y0 + bh, x0 : x0 + bw] = block

    # Crop to non-zero bounding box.  If output is all zero, return a 1x1 zero.
    nz = np.argwhere(output != 0)
    if len(nz) == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    return output[r0 : r1 + 1, c0 : c1 + 1]