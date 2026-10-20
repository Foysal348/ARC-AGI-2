import numpy as np
from math import gcd

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule:
    1. The full row and full column of 1s divide the grid into four quadrants.
    2. One quadrant contains the reference multi-color object.
       In the given examples this is the top-left quadrant.
    3. For every other quadrant that contains a block of 2s:
       - take the reference object,
       - reflect it vertically and/or horizontally according to the
         relative position of the two quadrants,
       - scale it uniformly so that its 2-block fits the target 2-block
         (using the smaller of the horizontal/vertical scale ratios),
       - place it so the top-left corner of its 2-block starts at the
         top-left corner of the target 2-block,
       - if the placed object would extend outside its quadrant, shift it
         minimally so that it fits; if it is still too large, crop it.
    4. Only non-background colors are painted.
    """
    grid = np.array(grid, dtype=np.int64, copy=True)
    if grid.ndim != 2:
        return grid

    H, W = grid.shape
    if H < 3 or W < 3:
        return grid

    # Detect the central cross of 1s.
    row_ones = (grid == 1).sum(axis=1)
    col_ones = (grid == 1).sum(axis=0)
    cross_row = int(np.argmax(row_ones))
    cross_col = int(np.argmax(col_ones))

    if row_ones[cross_row] <= H // 2 or col_ones[cross_col] <= W // 2:
        return grid

    # Background is the most frequent color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    def quadrant_slice(r_id: int, c_id: int):
        """r_id: 0=top, 1=bottom; c_id: 0=left, 1=right."""
        rs = slice(0, cross_row) if r_id == 0 else slice(cross_row + 1, H)
        cs = slice(0, cross_col) if c_id == 0 else slice(cross_col + 1, W)
        return rs, cs

    def quadrant_grid(r_id: int, c_id: int):
        rs, cs = quadrant_slice(r_id, c_id)
        return grid[rs, cs]

    def bbox(mask: np.ndarray):
        """Return (r0, r1, c0, c1) with exclusive r1/c1, or None."""
        if not mask.any():
            return None
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        r0 = int(np.argmax(rows))
        r1 = int(mask.shape[0] - 1 - np.argmax(rows[::-1])) + 1
        c0 = int(np.argmax(cols))
        c1 = int(mask.shape[1] - 1 - np.argmax(cols[::-1])) + 1
        return r0, r1, c0, c1

    # Identify the reference/source quadrant.
    # It should contain non-background, non-2, non-1 colors.
    source_id = 0
    best_score = -1
    for qid in range(4):
        rid, cid = qid // 2, qid % 2
        q = quadrant_grid(rid, cid)
        if q.size == 0:
            score = -1
        else:
            score = int(np.sum((q != bg) & (q != 2) & (q != 1)))
        if score > best_score:
            best_score = score
            source_id = qid

    # Fallback: choose the quadrant with the most non-background pixels.
    if best_score <= 0:
        best_score = -1
        for qid in range(4):
            rid, cid = qid // 2, qid % 2
            q = quadrant_grid(rid, cid)
            if q.size == 0:
                score = -1
            else:
                score = int(np.sum(q != bg))
            if score > best_score:
                best_score = score
                source_id = qid

    # Extract source object.
    src_r, src_c = source_id // 2, source_id % 2
    src_q = quadrant_grid(src_r, src_c)
    obj = src_q != bg
    src_bbox = bbox(obj)
    if src_bbox is None:
        return grid

    sr0, sr1, sc0, sc1 = src_bbox
    template = src_q[sr0:sr1, sc0:sc1].copy()

    two_box = bbox(template == 2)
    if two_box is None:
        return grid

    out = grid.copy()

    def ceil_mul(n: int, num: int, den: int) -> int:
        """ceil(n * num / den) for n >= 0."""
        return (n * num + den - 1) // den

    def fit_shift(min_pos: int, max_pos: int, bound0: int, bound1: int) -> int:
        """
        min_pos/max_pos are inclusive.
        bound0/bound1 are exclusive quadrant bounds.
        Return translation to make the object fit, if possible.
        """
        if min_pos < bound0 and max_pos >= bound1:
            # Object is larger than the quadrant: align to the near edge
            # and let the far part be cropped.
            return bound0 - min_pos
        if min_pos < bound0:
            return bound0 - min_pos
        if max_pos >= bound1:
            return (bound1 - 1) - max_pos
        return 0

    for tid in range(4):
        if tid == source_id:
            continue

        rid, cid = tid // 2, tid % 2
        q = quadrant_grid(rid, cid)

        two_mask = q == 2
        target_box = bbox(two_mask)
        if target_box is None:
            continue

        tr0_local, tr1_local, tc0_local, tc1_local = target_box
        base_r = 0 if rid == 0 else cross_row + 1
        base_c = 0 if cid == 0 else cross_col + 1

        tr0 = base_r + tr0_local
        tr1 = base_r + tr1_local
        tc0 = base_c + tc0_local
        tc1 = base_c + tc1_local

        target_h = tr1 - tr0
        target_w = tc1 - tc0
        if target_h <= 0 or target_w <= 0:
            continue

        # Reflect the source template according to quadrant relative position.
        T = template
        flip_v = (src_r != rid)
        flip_h = (src_c != cid)

        if flip_v:
            T = np.flipud(T)
        if flip_h:
            T = np.fliplr(T)

        T_two_box = bbox(T == 2)
        if T_two_box is None:
            continue

        a, b, c, d = T_two_box
        h2 = b - a
        w2 = d - c
        if h2 <= 0 or w2 <= 0:
            continue

        # Uniform scale: min(target_h / h2, target_w / w2).
        num1, den1 = target_h, h2
        num2, den2 = target_w, w2

        if num1 * den2 <= num2 * den1:
            s_num, s_den = num1, den1
        else:
            s_num, s_den = num2, den2

        g = gcd(s_num, s_den)
        s_num //= g
        s_den //= g

        if s_num <= 0:
            continue

        TH, TW = T.shape
        denom = 2 * s_num

        # Relative output coordinates around the scaled 2-block:
        # (0, 0) corresponds to the target 2-block top-left.
        y_min = -ceil_mul(a, s_num, s_den)
        y_max = ceil_mul(TH - a, s_num, s_den)
        x_min = -ceil_mul(c, s_num, s_den)
        x_max = ceil_mul(TW - c, s_num, s_den)

        cells = []
        min_y = max_y = None
        min_x = max_x = None

        for y in range(y_min, y_max):
            src_r = a + ((2 * y + 1) * s_den) // denom
            if src_r < 0 or src_r >= TH:
                continue
            for x in range(x_min, x_max):
                src_c = c + ((2 * x + 1) * s_den) // denom
                if src_c < 0 or src_c >= TW:
                    continue

                color = int(T[src_r, src_c])
                if color == bg:
                    continue

                cells.append((y, x, color))

                if min_y is None or y < min_y:
                    min_y = y
                if max_y is None or y > max_y:
                    max_y = y
                if min_x is None or x < min_x:
                    min_x = x
                if max_x is None or x > max_x:
                    max_x = x

        if not cells:
            continue

        # Initial placement: 2-block top-left at target 2-block top-left.
        min_r = tr0 + min_y
        max_r = tr0 + max_y
        min_c = tc0 + min_x
        max_c = tc0 + max_x

        # Quadrant bounds, exclusive.
        q_r0 = 0 if rid == 0 else cross_row + 1
        q_r1 = cross_row if rid == 0 else H
        q_c0 = 0 if cid == 0 else cross_col + 1
        q_c1 = cross_col if cid == 0 else W

        dr = fit_shift(min_r, max_r, q_r0, q_r1)
        dc = fit_shift(min_c, max_c, q_c0, q_c1)

        for y, x, color in cells:
            fr = tr0 + y + dr
            fc = tc0 + x + dc

            if q_r0 <= fr < q_r1 and q_c0 <= fc < q_c1:
                out[fr, fc] = color

    return out