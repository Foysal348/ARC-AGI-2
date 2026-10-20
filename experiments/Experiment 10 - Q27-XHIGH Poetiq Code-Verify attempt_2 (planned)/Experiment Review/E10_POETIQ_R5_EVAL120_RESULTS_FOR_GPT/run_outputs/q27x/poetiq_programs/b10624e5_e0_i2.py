import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule:
    - The grid is split into four quadrants by a full row and full column of 1s.
    - One quadrant contains the original multi-colour object; in the supplied
      examples this is the top-left quadrant.
    - Every other quadrant that contains a region of 2s is a target quadrant.
      The 2-region in that quadrant tells where the copied object's 2-region
      should appear and how large it should become.
    - The source object is reflected horizontally/vertically according to the
      target quadrant's position relative to the source quadrant.
    - The reflected object is scaled uniformly so that its 2-region matches the
      target 2-region as closely as possible.  The 2-region of the transformed
      object is aligned to the top-left corner of the target 2-region.
    - Non-background pixels of the transformed object are painted inside the
      target quadrant, avoiding the separator row/column.
    """
    grid = np.array(grid, dtype=np.int64, copy=True)
    if grid.ndim != 2:
        return grid

    H, W = grid.shape
    if H < 3 or W < 3:
        return grid

    # Find the separator cross: the row and column with the most 1s.
    row_ones = (grid == 1).sum(axis=1)
    col_ones = (grid == 1).sum(axis=0)
    cross_r = int(np.argmax(row_ones))
    cross_c = int(np.argmax(col_ones))

    # If no convincing cross exists, return the grid unchanged.
    if row_ones[cross_r] < H // 2 or col_ones[cross_c] < W // 2:
        return grid

    # Background is the most common colour.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    def quadrant_bounds(rid: int, cid: int):
        """
        Return (r0, r1, c0, c1) for quadrant rid/cid,
        where rid=0 top, rid=1 bottom; cid=0 left, cid=1 right.
        Bounds are half-open: r0 <= r < r1, c0 <= c < c1.
        """
        r0 = 0 if rid == 0 else cross_r + 1
        r1 = cross_r if rid == 0 else H
        c0 = 0 if cid == 0 else cross_c + 1
        c1 = cross_c if cid == 0 else W
        return r0, r1, c0, c1

    def bbox(mask: np.ndarray):
        """
        Return bounding box (r0, r1, c0, c1), half-open, or None.
        """
        if not mask.any():
            return None
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        r0 = int(np.argmax(rows))
        r1 = int(mask.shape[0] - 1 - np.argmax(rows[::-1])) + 1
        c0 = int(np.argmax(cols))
        c1 = int(mask.shape[1] - 1 - np.argmax(cols[::-1])) + 1
        return r0, r1, c0, c1

    # Choose the source quadrant.
    # Prefer quadrants that contain non-background colours other than 2 and 1.
    # If none exists, use the quadrant with the most non-background pixels.
    source_id = -1
    best = -1
    for qid in range(4):
        rid, cid = qid // 2, qid % 2
        r0, r1, c0, c1 = quadrant_bounds(rid, cid)
        q = grid[r0:r1, c0:c1]
        if q.size == 0:
            continue
        score = int(np.sum((q != bg) & (q != 2) & (q != 1)))
        if score > best:
            best = score
            source_id = qid

    if source_id == -1:
        best = -1
        for qid in range(4):
            rid, cid = qid // 2, qid % 2
            r0, r1, c0, c1 = quadrant_bounds(rid, cid)
            q = grid[r0:r1, c0:c1]
            if q.size == 0:
                continue
            score = int(np.sum(q != bg))
            if score > best:
                best = score
                source_id = qid

    if source_id == -1:
        return grid

    src_r, src_c = source_id // 2, source_id % 2
    sr0, sr1, sc0, sc1 = quadrant_bounds(src_r, src_c)
    src_q = grid[sr0:sr1, sc0:sc1]

    # Extract the source object bounding box (all non-background pixels).
    obj_mask = src_q != bg
    obj_box = bbox(obj_mask)
    if obj_box is None:
        return grid

    or0, or1, oc0, oc1 = obj_box
    template = src_q[or0:or1, oc0:oc1].copy()

    # The reference 2-region inside the source object.
    two_box = bbox(template == 2)
    if two_box is None:
        return grid

    out = grid.copy()

    for tid in range(4):
        if tid == source_id:
            continue

        rid, cid = tid // 2, tid % 2
        r0, r1, c0, c1 = quadrant_bounds(rid, cid)
        q = grid[r0:r1, c0:c1]
        if q.size == 0:
            continue

        # Target 2-region in this quadrant.
        target_mask = q == 2
        target_box = bbox(target_mask)
        if target_box is None:
            continue

        tr0, tr1, tc0, tc1 = target_box
        target_h = tr1 - tr0
        target_w = tc1 - tc0
        if target_h <= 0 or target_w <= 0:
            continue

        # Reflect source according to relative quadrant position.
        T = template
        if src_r != rid:
            T = np.flipud(T)
        if src_c != cid:
            T = np.fliplr(T)

        # 2-region in the reflected template.
        ref_box = bbox(T == 2)
        if ref_box is None:
            continue

        a, b, c, d = ref_box
        ref_h = b - a
        ref_w = d - c
        if ref_h <= 0 or ref_w <= 0:
            continue

        # Uniform scale: as close as possible while fitting the target 2-region.
        # Use min(target_h / ref_h, target_w / ref_w), implemented exactly.
        if target_h * ref_w <= target_w * ref_h:
            num, den = target_h, ref_h
        else:
            num, den = target_w, ref_w

        if num <= 0 or den <= 0:
            continue

        TH, TW = T.shape

        # For each output cell in a tight box around the scaled 2-region,
        # find the corresponding source cell using nearest-neighbour sampling.
        # Coordinate (0,0) is the top-left cell of the target 2-region.
        y_min = int(np.floor(-a * num / den))
        y_max = int(np.ceil((TH - a) * num / den))
        x_min = int(np.floor(-c * num / den))
        x_max = int(np.ceil((TW - c) * num / den))

        # First pass: compute all non-background cells and their extents.
        cells = []
        min_y = max_y = None
        min_x = max_x = None

        for y in range(y_min, y_max):
            # Nearest-neighbour source coordinate.
            # The centre of output y maps to source coordinate a + (y + 0.5) * den / num.
            sr = a + ((2 * y + 1) * den) // (2 * num)
            if sr < 0 or sr >= TH:
                continue

            for x in range(x_min, x_max):
                sc = c + ((2 * x + 1) * den) // (2 * num)
                if sc < 0 or sc >= TW:
                    continue

                color = int(T[sr, sc])
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

        # Initial placement: align top-left of transformed 2-region with
        # top-left of target 2-region.
        base_r = r0 + tr0
        base_c = c0 + tc0

        min_r = base_r + min_y
        max_r = base_r + max_y
        min_c = base_c + min_x
        max_c = base_c + max_x

        # Shift minimally so the painted cells stay inside this quadrant.
        # Do not shift if already inside.
        dr = 0
        if min_r < r0:
            dr = r0 - min_r
        elif max_r >= r1:
            dr = (r1 - 1) - max_r

        dc = 0
        if min_c < c0:
            dc = c0 - min_c
        elif max_c >= c1:
            dc = (c1 - 1) - max_c

        # Paint.
        for y, x, color in cells:
            rr = base_r + y + dr
            cc = base_c + x + dc
            if r0 <= rr < r1 and c0 <= cc < c1:
                out[rr, cc] = color

    return out