import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input ARC grid.

    Rule:
    - The vertical line of 6s separates instruction cells on the left from the canvas.
    - The top-left non-background pattern is the source motif. Its two non-zero colors
      are swapped to make the stamp motif.
    - A lower single-color pattern is a binary mask.
    - A 4 in the bottom controls determines a rotation and an anchor corner.
    - If the 7 markers are horizontally adjacent, the transformed mask is flipped
      horizontally.
    - Each 1 in the transformed mask places one motif tile on the canvas.
    """
    g = np.array(grid, dtype=np.int32, copy=True)
    if g.ndim != 2:
        return g

    H, W = g.shape
    if H == 0 or W == 0:
        return g

    def component_bbox(region, r, c):
        """Return bounding box of the connected non-False component containing (r,c)."""
        h, w = region.shape
        if r < 0 or r >= h or c < 0 or c >= w or not region[r, c]:
            return None

        seen = np.zeros((h, w), dtype=bool)
        stack = [(r, c)]
        seen[r, c] = True

        min_r = max_r = r
        min_c = max_c = c

        while stack:
            x, y = stack.pop()
            min_r = min(min_r, x)
            max_r = max(max_r, x)
            min_c = min(min_c, y)
            max_c = max(max_c, y)

            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and region[nx, ny] and not seen[nx, ny]:
                    seen[nx, ny] = True
                    stack.append((nx, ny))

        return min_r, max_r, min_c, max_c

    # ------------------------------------------------------------------
    # 1. Find the vertical separator column of 6s.
    # ------------------------------------------------------------------
    sep = -1
    for c in range(W):
        if np.all(g[:, c] == 6):
            sep = c
            break

    if sep == -1:
        counts6 = np.sum(g == 6, axis=0)
        cand = np.where(counts6 >= max(1, int(0.8 * H)))[0]
        if len(cand):
            sep = int(cand[0])
        else:
            sep = 0

    if sep >= W - 1:
        return g

    canvas = g[:, sep + 1:].copy()
    left = g[:, :sep]
    left_w = left.shape[1]

    if left_w == 0:
        return canvas

    # ------------------------------------------------------------------
    # 2. Find the horizontal 6 separator in the instruction strip.
    # ------------------------------------------------------------------
    all6_rows = np.all(left == 6, axis=1)
    control_row = H
    for r in range(H):
        if all6_rows[r]:
            if r > 0 and np.any((left[:r] != 0) & (left[:r] != 6)):
                control_row = r
                break

    # ------------------------------------------------------------------
    # 3. Extract the source motif from the top of the instruction strip.
    # ------------------------------------------------------------------
    top_region = left[:control_row]
    nonbg = (top_region != 0) & (top_region != 6)
    row_has = np.any(nonbg, axis=1)

    if not np.any(row_has):
        return canvas

    r0 = int(np.argmax(row_has))
    r1 = r0
    while r1 + 1 < len(row_has) and row_has[r1 + 1]:
        r1 += 1

    band = nonbg[r0:r1 + 1]
    col_has = np.any(band, axis=0)

    if not np.any(col_has):
        return canvas

    c0 = int(np.argmax(col_has))
    c1 = int(left_w - 1 - np.argmax(col_has[::-1]))

    motif = top_region[r0:r1 + 1, c0:c1 + 1]

    # Swap the two main non-zero colors of the source motif.
    motif_swap = motif.copy()
    vals, counts = np.unique(motif[(motif != 0) & (motif != 6)], return_counts=True)

    if len(vals) == 2:
        a, b = int(vals[0]), int(vals[1])
        motif_swap = np.where(motif == a, b, motif_swap)
        motif_swap = np.where(motif == b, a, motif_swap)
    elif len(vals) > 2:
        order = np.argsort(counts)[::-1]
        a, b = int(vals[order[0]]), int(vals[order[1]])
        motif_swap = np.where(motif == a, b, motif_swap)
        motif_swap = np.where(motif == b, a, motif_swap)

    src_colors = set(int(v) for v in vals)
    motif_h, motif_w = motif_swap.shape

    if motif_h == 0 or motif_w == 0:
        return canvas

    # ------------------------------------------------------------------
    # 4. Extract the mask pattern below the source motif.
    # ------------------------------------------------------------------
    source_bottom = r1 + 1
    if source_bottom < control_row:
        mask_region = left[source_bottom:control_row]
    else:
        mask_region = np.zeros((0, left_w), dtype=g.dtype)

    allowed = (mask_region != 0) & (mask_region != 6)
    mask_color = None

    if np.any(allowed):
        unique_vals, unique_counts = np.unique(mask_region[allowed], return_counts=True)

        # Prefer colors that are not source colors.
        filtered = [
            (int(v), int(cnt))
            for v, cnt in zip(unique_vals, unique_counts)
            if int(v) not in src_colors
        ]

        if not filtered:
            filtered = [(int(v), int(cnt)) for v, cnt in zip(unique_vals, unique_counts)]

        if filtered:
            mask_color = max(filtered, key=lambda x: x[1])[0]

    if mask_color is None:
        return canvas

    mask = (mask_region == mask_color)

    if not np.any(mask):
        return canvas

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    mr0 = int(np.argmax(rows))
    mr1 = int(mask.shape[0] - 1 - np.argmax(rows[::-1]))
    mc0 = int(np.argmax(cols))
    mc1 = int(mask.shape[1] - 1 - np.argmax(cols[::-1]))

    mask = mask[mr0:mr1 + 1, mc0:mc1 + 1].astype(np.uint8)

    # ------------------------------------------------------------------
    # 5. Read 4/7 controls: rotation, anchor, optional horizontal flip.
    # ------------------------------------------------------------------
    rot = 0
    flip_h = False
    row_align = 0  # 0=top, 1=middle, 2=bottom
    col_align = 0  # 0=left, 1=middle, 2=right

    if control_row + 1 < H:
        ctrl = left[control_row + 1:]
    else:
        ctrl = np.zeros((0, left_w), dtype=g.dtype)

    if ctrl.size > 0:
        fours = np.argwhere(ctrl == 4)
        sevens = np.argwhere(ctrl == 7)

        if fours.size > 0:
            fr, fc = int(fours[0][0]), int(fours[0][1])
            bbox = component_bbox(ctrl != 6, fr, fc)

            if bbox is not None:
                br0, br1, bc0, bc1 = bbox

                row_pos = 0 if fr == br0 else (2 if fr == br1 else 1)
                col_pos = 0 if fc == bc0 else (2 if fc == bc1 else 1)

                row_align = row_pos
                col_align = col_pos

                if row_pos == 0 and col_pos == 2:
                    rot = 1          # 90 degrees clockwise
                elif row_pos == 2 and col_pos == 2:
                    rot = 2          # 180 degrees
                elif row_pos == 2 and col_pos == 0:
                    rot = 3          # 270 degrees clockwise
                else:
                    rot = 0

        if sevens.size >= 2:
            # Horizontal 7s cause a horizontal flip of the transformed mask.
            row_var = int(np.max(sevens[:, 0]) - np.min(sevens[:, 0]))
            col_var = int(np.max(sevens[:, 1]) - np.min(sevens[:, 1]))
            flip_h = (row_var == 0) or (row_var < col_var)

    # ------------------------------------------------------------------
    # 6. Transform the mask.
    # ------------------------------------------------------------------
    tmask = mask.copy()

    if rot == 1:
        tmask = np.rot90(tmask, k=3)   # clockwise
    elif rot == 2:
        tmask = np.rot90(tmask, k=2)
    elif rot == 3:
        tmask = np.rot90(tmask, k=1)   # counterclockwise

    if flip_h:
        tmask = tmask[:, ::-1]

    rows = np.any(tmask, axis=1)
    cols = np.any(tmask, axis=0)

    if not np.any(rows):
        return canvas

    tr0 = int(np.argmax(rows))
    tr1 = int(tmask.shape[0] - 1 - np.argmax(rows[::-1]))
    tc0 = int(np.argmax(cols))
    tc1 = int(tmask.shape[1] - 1 - np.argmax(cols[::-1]))

    tmask = tmask[tr0:tr1 + 1, tc0:tc1 + 1].astype(bool)

    # ------------------------------------------------------------------
    # 7. Determine the block grid size and place the transformed mask.
    # ------------------------------------------------------------------
    Hc, Wc = canvas.shape

    nb_rows = (Hc - motif_h) // motif_h + 1 if Hc >= motif_h else 0
    nb_cols = (Wc - motif_w) // motif_w + 1 if Wc >= motif_w else 0

    if nb_rows <= 0 or nb_cols <= 0:
        return canvas

    th, tw = tmask.shape

    # Crop the transformed mask if it is too large for the block grid.
    if th > nb_rows:
        if row_align == 0:
            tmask = tmask[:nb_rows]
        elif row_align == 2:
            tmask = tmask[-nb_rows:]
        else:
            start = (th - nb_rows) // 2
            tmask = tmask[start:start + nb_rows]

    if tw > nb_cols:
        if col_align == 0:
            tmask = tmask[:, :nb_cols]
        elif col_align == 2:
            tmask = tmask[:, -nb_cols:]
        else:
            start = (tw - nb_cols) // 2
            tmask = tmask[:, start:start + nb_cols]

    th, tw = tmask.shape

    if row_align == 0:
        place_r = 0
    elif row_align == 2:
        place_r = nb_rows - th
    else:
        place_r = (nb_rows - th) // 2

    if col_align == 0:
        place_c = 0
    elif col_align == 2:
        place_c = nb_cols - tw
    else:
        place_c = (nb_cols - tw) // 2

    place_r = max(0, min(place_r, nb_rows - th))
    place_c = max(0, min(place_c, nb_cols - tw))

    block_mask = np.zeros((nb_rows, nb_cols), dtype=bool)
    block_mask[place_r:place_r + th, place_c:place_c + tw] = tmask

    # ------------------------------------------------------------------
    # 8. Stamp the swapped motif into the canvas.
    # ------------------------------------------------------------------
    out = canvas.copy()
    motif_pixels = motif_swap != 0

    if np.any(motif_pixels):
        for i in range(nb_rows):
            r_start = i * motif_h
            for j in range(nb_cols):
                if block_mask[i, j]:
                    c_start = j * motif_w
                    region = out[r_start:r_start + motif_h, c_start:c_start + motif_w]
                    region[motif_pixels] = motif_swap[motif_pixels]

    return out