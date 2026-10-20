import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform an ARC grid.

    The grid contains separator columns made of 6s. The largest region between
    those separators is the canvas. Other regions are instruction panels.

    Each instruction panel contains:
      - a small two-color motif in the top part,
      - a single-color mask below the motif,
      - a 4 in a control box that determines rotation and placement corner.

    The motif's two colors are swapped. The mask is rotated according to the
    4's corner, then tiled with the swapped motif onto the canvas using the
    same corner alignment.
    """
    g = np.array(grid, dtype=int, copy=False)
    if g.ndim != 2:
        return g

    H, W = g.shape
    if H == 0 or W == 0:
        return g.copy()

    def bbox_crop(bool_region: np.ndarray):
        """Crop a boolean array to the bounding box of True cells."""
        if bool_region is None or bool_region.size == 0:
            return None

        rows = np.any(bool_region, axis=1)
        cols = np.any(bool_region, axis=0)

        if not np.any(rows) or not np.any(cols):
            return None

        r0 = int(np.argmax(rows))
        r1 = int(bool_region.shape[0] - 1 - int(np.argmax(rows[::-1])))
        c0 = int(np.argmax(cols))
        c1 = int(bool_region.shape[1] - 1 - int(np.argmax(cols[::-1])))

        return bool_region[r0:r1 + 1, c0:c1 + 1].copy()

    def extract_instruction(panel: np.ndarray):
        """Extract motif, mask, rotation, and anchor from an instruction panel."""
        Hp, Wp = panel.shape
        if Hp == 0 or Wp == 0:
            return None

        # Horizontal separator rows are rows made entirely of 6s inside the panel.
        hsep = [r for r in range(Hp) if np.all(panel[r] == 6)]
        if not hsep:
            return None

        # First meaningful horizontal separator: there must be non-background content above it.
        sep1 = None
        for r in hsep:
            if r > 0 and np.any((panel[:r] != 0) & (panel[:r] != 6)):
                sep1 = r
                break
        if sep1 is None:
            sep1 = hsep[0]

        sep2 = None
        for r in hsep:
            if r > sep1:
                sep2 = r
                break
        if sep2 is None:
            sep2 = Hp

        top = panel[:sep1]
        if top.size == 0:
            return None

        nonbg = (top != 0) & (top != 6)
        if not np.any(nonbg):
            return None

        # Find row bands containing non-background pixels.
        row_has = np.any(nonbg, axis=1)
        runs = []
        r = 0
        while r < row_has.size:
            if row_has[r]:
                start = r
                while r < row_has.size and row_has[r]:
                    r += 1
                runs.append((start, r - 1))
            else:
                r += 1

        if not runs:
            return None

        # First row band is the motif.
        mr0, mr1 = runs[0]
        motif_region = nonbg[mr0:mr1 + 1, :]
        col_has = np.any(motif_region, axis=0)
        if not np.any(col_has):
            return None

        mc0 = int(np.argmax(col_has))
        mc1 = int(motif_region.shape[1] - 1 - int(np.argmax(col_has[::-1])))
        motif = top[mr0:mr1 + 1, mc0:mc1 + 1]

        # Swap the two main non-background colors in the motif.
        motif_vals, motif_counts = np.unique(
            motif[(motif != 0) & (motif != 6)],
            return_counts=True
        )

        motif_swap = motif.copy()
        if len(motif_vals) >= 2:
            if len(motif_vals) == 2:
                a = int(motif_vals[0])
                b = int(motif_vals[1])
            else:
                order = np.argsort(motif_counts)[::-1]
                a = int(motif_vals[order[0]])
                b = int(motif_vals[order[1]])

            if a != b:
                motif_swap = motif.copy()
                motif_swap[motif == a] = b
                motif_swap[motif == b] = a
        elif len(motif_vals) == 0:
            return None

        # Remaining non-background rows below the motif form the mask.
        below_rows = [i for i in range(mr1 + 1, top.shape[0]) if row_has[i]]
        if not below_rows:
            return None

        m_start = below_rows[0]
        m_end = below_rows[-1]

        mask_region = top[m_start:m_end + 1, :]
        mask_nonbg = (mask_region != 0) & (mask_region != 6)
        if not np.any(mask_nonbg):
            return None

        mask_vals, mask_counts = np.unique(mask_region[mask_nonbg], return_counts=True)
        if len(mask_vals) == 0:
            return None

        motif_val_set = set(int(v) for v in motif_vals)
        candidates = [
            (int(v), int(c))
            for v, c in zip(mask_vals, mask_counts)
            if int(v) not in motif_val_set
        ]
        if not candidates:
            candidates = [(int(v), int(c)) for v, c in zip(mask_vals, mask_counts)]

        # Choose the most frequent candidate color as the mask color.
        mask_color = max(candidates, key=lambda x: (x[1], -x[0]))[0]
        mask_bool = (mask_region == mask_color)
        mask = bbox_crop(mask_bool)
        if mask is None:
            return None

        # Read the 4 control to determine rotation and placement corner.
        rot = 0
        row_align = 0  # 0 top, 1 middle, 2 bottom
        col_align = 0  # 0 left, 1 middle, 2 right

        if sep2 > sep1 + 1:
            ctrl4 = panel[sep1 + 1:sep2, :]
        else:
            ctrl4 = panel[sep1 + 1:, :]

        fours = np.argwhere(ctrl4 == 4)
        if len(fours) > 0 and ctrl4.size > 0:
            r, c = int(fours[0][0]), int(fours[0][1])

            top_row = 0
            bottom_row = ctrl4.shape[0] - 1

            if r == top_row:
                row_align = 0
            elif r == bottom_row:
                row_align = 2
            else:
                row_align = 1

            # Find vertical 6 borders on the same row as the 4.
            border_left = None
            for cc in range(c - 1, -1, -1):
                if ctrl4[r, cc] == 6:
                    border_left = cc
                    break
            inner_left = border_left + 1 if border_left is not None else 0

            border_right = None
            for cc in range(c + 1, ctrl4.shape[1]):
                if ctrl4[r, cc] == 6:
                    border_right = cc
                    break
            inner_right = border_right - 1 if border_right is not None else ctrl4.shape[1] - 1

            if inner_left <= inner_right:
                if c == inner_left:
                    col_align = 0
                elif c == inner_right:
                    col_align = 2
                else:
                    col_align = 1
            else:
                col_align = 1

            if row_align == 0 and col_align == 2:
                rot = 1      # 90 degrees clockwise
            elif row_align == 2 and col_align == 2:
                rot = 2      # 180 degrees
            elif row_align == 2 and col_align == 0:
                rot = 3      # 270 degrees clockwise
            else:
                rot = 0

        return {
            "motif": motif_swap,
            "mask": mask,
            "rot": rot,
            "row_align": row_align,
            "col_align": col_align,
        }

    def apply_instruction(out: np.ndarray, instr: dict):
        """Stamp one instruction panel onto the output canvas."""
        motif = instr["motif"]
        mask = instr["mask"].astype(bool)
        rot = instr["rot"]

        th, tw = motif.shape
        if th == 0 or tw == 0 or mask.size == 0:
            return

        if rot == 1:
            mask = np.rot90(mask, k=3)   # 90 clockwise
        elif rot == 2:
            mask = np.rot90(mask, k=2)   # 180
        elif rot == 3:
            mask = np.rot90(mask, k=1)   # 90 counterclockwise / 270 clockwise

        mask = bbox_crop(mask)
        if mask is None:
            return

        Hc, Wc = out.shape
        max_rows = Hc // th if th > 0 else 0
        max_cols = Wc // tw if tw > 0 else 0

        if max_rows <= 0 or max_cols <= 0:
            return

        row_align = instr["row_align"]
        col_align = instr["col_align"]

        # If the mask is larger than the canvas grid, crop it according to the anchor.
        if mask.shape[0] > max_rows:
            if row_align == 0:
                mask = mask[:max_rows]
            elif row_align == 2:
                mask = mask[-max_rows:]
            else:
                start = (mask.shape[0] - max_rows) // 2
                mask = mask[start:start + max_rows]

        if mask.shape[1] > max_cols:
            if col_align == 0:
                mask = mask[:, :max_cols]
            elif col_align == 2:
                mask = mask[:, -max_cols:]
            else:
                start = (mask.shape[1] - max_cols) // 2
                mask = mask[:, start:start + max_cols]

        mh, mw = mask.shape
        if mh == 0 or mw == 0:
            return

        block_h = mh * th
        block_w = mw * tw
        if block_h <= 0 or block_w <= 0:
            return

        if row_align == 0:
            off_r = 0
        elif row_align == 2:
            off_r = Hc - block_h
        else:
            off_r = (Hc - block_h) // 2

        if col_align == 0:
            off_c = 0
        elif col_align == 2:
            off_c = Wc - block_w
        else:
            off_c = (Wc - block_w) // 2

        off_r = int(max(0, min(off_r, Hc - block_h)))
        off_c = int(max(0, min(off_c, Wc - block_w)))

        motif_px = (motif != 0) & (motif != 6)
        if not np.any(motif_px):
            return

        for i in range(mh):
            for j in range(mw):
                if not mask[i, j]:
                    continue

                r0 = off_r + i * th
                c0 = off_c + j * tw

                if r0 < 0 or c0 < 0 or r0 + th > Hc or c0 + tw > Wc:
                    continue

                region = out[r0:r0 + th, c0:c0 + tw]
                region[motif_px] = motif[motif_px]

    # -------------------------------------------------------------------------
    # Identify vertical separator columns made of 6.
    # -------------------------------------------------------------------------
    sep_cols = [c for c in range(W) if np.all(g[:, c] == 6)]

    if not sep_cols:
        counts6 = np.sum(g == 6, axis=0)
        sep_cols = [c for c in range(W) if counts6[c] >= max(1, int(0.95 * H))]

    if not sep_cols:
        return g.copy()

    # Segments between separator columns.
    bounds = []
    prev = 0
    for c in sep_cols:
        if c > prev:
            bounds.append((prev, c - 1))
        prev = c + 1
    if prev < W:
        bounds.append((prev, W - 1))

    bounds = [(a, b) for a, b in bounds if b >= a]
    if not bounds:
        return g.copy()

    # Choose the canvas: prefer largest width, then fewer 6s, then middle/right.
    def seg_score(i):
        a, b = bounds[i]
        width = b - a + 1
        six_count = int(np.sum(g[:, a:b + 1] == 6))
        middle_bonus = 1 if len(bounds) > 2 and 0 < i < len(bounds) - 1 else 0
        return (width, -six_count, middle_bonus, i)

    canvas_idx = max(range(len(bounds)), key=seg_score)
    canvas_a, canvas_b = bounds[canvas_idx]
    out = g[:, canvas_a:canvas_b + 1].copy()

    # Apply every instruction panel, left to right.
    for i, (a, b) in enumerate(bounds):
        if i == canvas_idx:
            continue

        panel = g[:, a:b + 1]
        instr = extract_instruction(panel)
        if instr is not None:
            apply_instruction(out, instr)

    return out