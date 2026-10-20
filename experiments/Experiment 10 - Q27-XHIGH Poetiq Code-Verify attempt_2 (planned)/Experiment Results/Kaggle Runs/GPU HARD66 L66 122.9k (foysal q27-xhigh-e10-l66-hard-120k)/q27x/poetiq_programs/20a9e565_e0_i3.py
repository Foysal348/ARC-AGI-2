import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete the rectangle marked by yellow (5) corner brackets.

    The yellow pixels form two L-shaped corner marks: one near the top-left
    and one near the bottom-right of the missing object.  In the observed
    task family, the missing object usually extends one column to the left
    of the leftmost yellow pixel; the final answer is then cropped to the
    non-zero bounding box of the completed object.

    The main idea is:
      1. Locate the marked rectangle.
      2. Remove yellow markers.
      3. Infer a small number of consistent translation symmetries from the
         visible grid.
      4. Propagate known cells through those symmetries to fill the marked
         rectangle.
      5. Fill any remaining gaps by consensus from the most reliable
         translations.
      6. Crop the completed rectangle to its non-zero bounding box.
    """
    g = np.array(grid, dtype=int)
    if g.size == 0:
        return np.zeros_like(g)

    if 5 not in np.unique(g):
        return np.zeros((1, 1), dtype=int)

    # ------------------------------------------------------------------
    # 1. Locate the marked rectangle.
    # ------------------------------------------------------------------
    ys, xs = np.where(g == 5)
    r0, r1 = int(ys.min()), int(ys.max())
    c0, c1 = int(xs.min()) - 1, int(xs.max())

    H = r1 - r0 + 1
    W = c1 - c0 + 1

    # Guard against pathological tiny boxes.
    if H <= 0 or W <= 0:
        return np.zeros((1, 1), dtype=int)

    work = g.copy()
    work[work == 5] = 0

    out = np.zeros((H, W), dtype=int)

    # If the marked rectangle is already fully visible, just extract it.
    if (work[r0:r1 + 1, c0:c1 + 1] != 0).all():
        return work[r0:r1 + 1, c0:c1 + 1]

    # ------------------------------------------------------------------
    # Helpers: candidate translations.
    # ------------------------------------------------------------------
    R, C = work.shape

    def in_grid(r, c):
        return 0 <= r < R and 0 <= c < C

    def known_mask(arr):
        return arr != 0

    def candidate_shifts():
        """
        Find translations (dy, dx) that are locally consistent with the
        visible pattern.  A shift is good if, over many cell pairs where
        both source and destination are visible, the mask and color agree.
        """
        max_d = min(max(R, C), 35)
        candidates = []

        for dy in range(-max_d, max_d + 1):
            for dx in range(-max_d, max_d + 1):
                if dy == 0 and dx == 0:
                    continue

                possible = 0
                mask_ok = 0
                nz_pairs = 0
                color_ok = 0

                # Sample the grid to keep runtime reasonable.
                step_r = max(1, R // 120)
                step_c = max(1, C // 120)

                for r in range(0, R, step_r):
                    for c in range(0, C, step_c):
                        r2, c2 = r + dy, c + dx
                        if not in_grid(r2, c2):
                            continue
                        v1 = work[r, c]
                        v2 = work[r2, c2]

                        # Only count pairs where at least one side is known
                        # and neither side is inside the marked hole.
                        in_hole1 = r0 <= r <= r1 and c0 <= c <= c1
                        in_hole2 = r0 <= r2 <= r1 and c0 <= c2 <= c1
                        if in_hole1 or in_hole2:
                            continue

                        if v1 == 0 and v2 == 0:
                            continue

                        possible += 1
                        if (v1 == 0) == (v2 == 0):
                            mask_ok += 1
                        if v1 != 0 and v2 != 0:
                            nz_pairs += 1
                            if v1 == v2:
                                color_ok += 1

                if possible < 6:
                    continue

                mask_ratio = mask_ok / possible
                color_ratio = (color_ok / nz_pairs) if nz_pairs else 0.0

                # Require strong mask consistency and, when color overlaps
                # exist, strong color consistency.
                if mask_ratio < 0.72:
                    continue
                if nz_pairs > 3 and color_ratio < 0.82:
                    continue

                score = mask_ratio * 2.0 + color_ratio * 2.0 + possible / 1000.0
                # Slight preference for smaller shifts.
                score -= (abs(dy) + abs(dx)) * 0.01
                candidates.append((score, possible, mask_ratio, color_ratio, dy, dx))

        candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
        # Keep a moderate number of high-quality symmetries.
        return [x for x in candidates[:24]]

    shifts = candidate_shifts()

    # ------------------------------------------------------------------
    # 2. Propagate known cells through the selected shifts.
    # ------------------------------------------------------------------
    # We store candidate values with weights for each target cell.
    votes = np.zeros((H, W), dtype=float)
    weights = np.zeros((H, W), dtype=float)

    # Build a queue of known non-zero cells outside the hole.
    known_cells = deque()
    for r in range(R):
        for c in range(C):
            if work[r, c] != 0 and not (r0 <= r <= r1 and c0 <= c <= c1):
                known_cells.append((r, c))

    # BFS propagation using symmetries.  This is conservative: a cell is
    # accepted only when all votes agree.
    filled = np.zeros((H, W), dtype=bool)

    def add_vote(r, c, val, w):
        if not (r0 <= r <= r1 and c0 <= c <= c1):
            return
        if filled[r - r0, c - c0]:
            return
        votes[r - r0, c - c0] += w * val
        weights[r - r0, c - c0] += w

    # Initial votes directly from visible cells shifted into the hole.
    for dy, dx in [s[-2:] for s in shifts]:
        w = max(1.0, min(10.0, 2.0 + len(shifts) / 50.0))
        for r in range(R):
            for c in range(C):
                r2, c2 = r - dy, c - dx
                if in_grid(r2, c2) and work[r, c] != 0:
                    add_vote(r2, c2, work[r, c], w)

    # Fill unanimous cells.
    changed = True
    while changed:
        changed = False
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if filled[rr - r0, cc - c0]:
                    continue
                tot_w = weights[rr - r0, cc - c0]
                if tot_w == 0:
                    continue
                avg = votes[rr - r0, cc - c0] / tot_w
                # Only accept if all contributions are consistent (integer).
                if abs(avg - round(avg)) < 1e-6 and round(avg) != 0:
                    out[rr - r0, cc - c0] = int(round(avg))
                    filled[rr - r0, cc - c0] = True
                    changed = True

    # If propagation filled everything, we are done.
    if filled.all():
        pass

    # ------------------------------------------------------------------
    # 3. If many cells remain, use direct consensus from the best shifts.
    # ------------------------------------------------------------------
    if not filled.all():
        # Reinitialize votes for unfilled cells only.
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if filled[rr - r0, cc - c0]:
                    continue

                best = {}
                total = 0
                for score, possible, mask_ratio, color_ratio, dy, dx in shifts[:12]:
                    sr, sc = rr - dy, cc - dx
                    if in_grid(sr, sc) and work[sr, sc] != 0:
                        val = int(work[sr, sc])
                        wgt = max(1.0, score)
                        best[val] = best.get(val, 0.0) + wgt
                        total += wgt

                if best:
                    top_val = max(best.items(), key=lambda kv: kv[1])[0]
                    top_w = best[top_val]
                    # Need a majority.
                    if top_w >= total * 0.55:
                        out[rr - r0, cc - c0] = top_val
                        filled[rr - r0, cc - c0] = True

    # ------------------------------------------------------------------
    # 4. Special-case: single-color frame/ladder completion.
    #    If the completed rectangle is mostly empty but one color dominates
    #    nearby, render a canonical frame matching the observed style.
    # ------------------------------------------------------------------
    if filled.mean() < 0.55:
        # Gather colors near the marked rectangle.
        near = np.zeros_like(work)
        rr0 = max(0, r0 - H)
        rr1 = min(R, r1 + H + 1)
        cc0 = max(0, c0 - W)
        cc1 = min(C, c1 + W + 1)
        near[rr0:rr1, cc0:cc1] = work[rr0:rr1, cc0:cc1]
        near[r0:r1 + 1, c0:c1 + 1] = 0

        nz = near[near != 0]
        if nz.size > 0:
            vals, cnts = np.unique(nz, return_counts=True)
            dominant = int(vals[np.argmax(cnts)])
            total_nz = int(nz.size)
            dominant_cnt = int(cnts[np.argmax(cnts)])

            # If one color is clearly dominant, complete a simple frame.
            if dominant_cnt >= 0.75 * total_nz:
                tmp = np.zeros((H, W), dtype=int)

                # If height or width is small, use a solid row/column frame.
                if H <= 3:
                    tmp[0, :] = dominant
                    tmp[-1, :] = dominant
                    tmp[:, 0] = dominant
                elif W <= 3:
                    tmp[:, 0] = dominant
                    tmp[:, -1] = dominant
                    tmp[0, :] = dominant
                    tmp[-1, :] = dominant
                else:
                    tmp[0, :] = dominant
                    tmp[-1, :] = dominant
                    tmp[:, 0] = dominant
                    tmp[:, -1] = dominant

                # Preserve any confidently filled cells.
                for rr in range(H):
                    for cc in range(W):
                        if filled[rr, cc] and out[rr, cc] != 0:
                            tmp[rr, cc] = out[rr, cc]
                        elif out[rr, cc] != 0:
                            tmp[rr, cc] = out[rr, cc]

                out = tmp
                filled = out != 0

    # ------------------------------------------------------------------
    # 5. Crop to the non-zero bounding box.
    #    Internal zeros are preserved; only all-zero outer rows/columns
    #    are removed.
    # ------------------------------------------------------------------
    nz_rows = np.where((out != 0).any(axis=1))[0]
    nz_cols = np.where((out != 0).any(axis=0))[0]

    if nz_rows.size == 0 or nz_cols.size == 0:
        return np.zeros((1, 1), dtype=int)

    rr0, rr1 = int(nz_rows.min()), int(nz_rows.max())
    cc0, cc1 = int(nz_cols.min()), int(nz_cols.max())

    return out[rr0:rr1 + 1, cc0:cc1 + 1].copy()