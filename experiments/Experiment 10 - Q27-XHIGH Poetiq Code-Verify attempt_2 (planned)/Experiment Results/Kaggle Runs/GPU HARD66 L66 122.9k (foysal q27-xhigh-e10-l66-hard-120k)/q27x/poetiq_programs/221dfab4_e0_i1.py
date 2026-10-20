import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a grid by projecting the unique 4-colored marker line
    perpendicular to itself.

    Rule:
      * The marker color is 4.
      * The background is the border mode color.
      * The main color is the most frequent non-background, non-4 color.
      * Starting from the marker, place copies every 2 cells in the
        perpendicular direction.
      * The copy color pattern is 4, 4, 3, 4, 4, 3, ...
      * On an even step:
          - color 4: paint only the marker footprint with 4
          - color 3: paint all main-color pixels in that row/column with 3,
                     and also paint the marker footprint with 3
      * On an odd step:
          - normally erase main-color pixels inside the marker footprint
          - if the first 3-position is directly supported by the main color
            in its footprint, the first odd steps become 4-stamps, and the
            step just before the first 3 trims unsupported main pixels.
    """
    arr = np.array(grid, dtype=int)
    out = arr.copy()

    if arr.ndim != 2:
        return out

    H, W = arr.shape
    if H == 0 or W == 0:
        return out

    # Background: most common color on the border.
    border = np.concatenate((arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]))
    bvals, bcounts = np.unique(border, return_counts=True)
    bg = int(bvals[np.argmax(bcounts)])

    # Marker: cells of color 4.
    marker = arr == 4
    if not marker.any():
        return out

    mr, mc = np.where(marker)
    r0, r1 = int(mr.min()), int(mr.max())
    c0, c1 = int(mc.min()), int(mc.max())

    uniq_r = np.unique(mr)
    uniq_c = np.unique(mc)

    # Determine marker orientation and span.
    if len(uniq_r) == 1 and len(uniq_c) == 1:
        # Single-cell marker: choose orientation from edge location if possible.
        if mr[0] == 0 or mr[0] == H - 1:
            orient = "h"
            m_row = int(mr[0])
            c_min, c_max = int(c0), int(c1)
        elif mc[0] == 0 or mc[0] == W - 1:
            orient = "v"
            m_col = int(mc[0])
            r_min, r_max = int(r0), int(r1)
        else:
            orient = "h"
            m_row = int(mr[0])
            c_min, c_max = int(c0), int(c1)
    elif len(uniq_r) == 1:
        orient = "h"
        m_row = int(uniq_r[0])
        c_min, c_max = c0, c1
    elif len(uniq_c) == 1:
        orient = "v"
        m_col = int(uniq_c[0])
        r_min, r_max = r0, r1
    else:
        # Fallback for non-ideal marker: use bounding-box orientation.
        if (r1 - r0) <= (c1 - c0):
            orient = "h"
            m_row = int((r0 + r1) // 2)
            c_min, c_max = c0, c1
        else:
            orient = "v"
            m_col = int((c0 + c1) // 2)
            r_min, r_max = r0, r1

    # Main color: most frequent color that is neither background nor marker.
    candidate = (arr != bg) & (arr != 4)
    if candidate.any():
        mvals, mcounts = np.unique(arr[candidate], return_counts=True)
        main_color = int(mvals[np.argmax(mcounts)])
        main_mask = arr == main_color
    else:
        main_mask = np.zeros_like(arr, dtype=bool)

    # Always keep/paint the marker footprint with 4.
    if orient == "h":
        out[m_row, c_min:c_max + 1] = 4
    else:
        out[r_min:r_max + 1, m_col] = 4

    # Horizontal marker: process upward/downward column footprint.
    if orient == "h":
        def process_direction(direction: int) -> None:
            if direction == 1:
                max_d = H - 1 - m_row
            else:
                max_d = m_row

            if max_d <= 0:
                return

            # Base solidification if the first 3-position is supported.
            base = False
            if max_d >= 4:
                p4 = m_row + direction * 4
                if 0 <= p4 < H and np.any(main_mask[p4, c_min:c_max + 1]):
                    base = True

            for d in range(1, max_d + 1):
                p = m_row + direction * d

                if d % 2 == 0:
                    k = d // 2
                    color = 3 if k % 3 == 2 else 4
                    stamp = True
                else:
                    if base and d <= 3:
                        color = 4
                        stamp = True
                    else:
                        color = 4
                        stamp = False

                if stamp:
                    if color == 3:
                        out[p, main_mask[p]] = 3
                        out[p, c_min:c_max + 1] = 3
                    else:
                        out[p, c_min:c_max + 1] = 4
                else:
                    cols = np.arange(c_min, c_max + 1)
                    m = main_mask[p, c_min:c_max + 1]
                    if m.any():
                        out[p, cols[m]] = bg

                # Trim unsupported pixels just before the first 3 row.
                if base and d == 3 and stamp and color == 4:
                    pt = p - direction
                    if 0 <= pt < H:
                        m = main_mask[p].copy()
                        m[c_min:c_max + 1] = False
                        m &= ~main_mask[pt]
                        if m.any():
                            out[p, m] = bg

        if m_row == 0:
            directions = [1]
        elif m_row == H - 1:
            directions = [-1]
        else:
            directions = [1, -1]

        for d in directions:
            process_direction(d)

    # Vertical marker: process leftward/rightward row footprint.
    else:
        def process_direction(direction: int) -> None:
            if direction == 1:
                max_d = W - 1 - m_col
            else:
                max_d = m_col

            if max_d <= 0:
                return

            # Base solidification if the first 3-position is supported.
            base = False
            if max_d >= 4:
                p4 = m_col + direction * 4
                if 0 <= p4 < W and np.any(main_mask[r_min:r_max + 1, p4]):
                    base = True

            for d in range(1, max_d + 1):
                p = m_col + direction * d

                if d % 2 == 0:
                    k = d // 2
                    color = 3 if k % 3 == 2 else 4
                    stamp = True
                else:
                    if base and d <= 3:
                        color = 4
                        stamp = True
                    else:
                        color = 4
                        stamp = False

                if stamp:
                    if color == 3:
                        rows = np.where(main_mask[:, p])[0]
                        out[rows, p] = 3
                        out[r_min:r_max + 1, p] = 3
                    else:
                        out[r_min:r_max + 1, p] = 4
                else:
                    rows = np.arange(r_min, r_max + 1)
                    m = main_mask[r_min:r_max + 1, p]
                    if m.any():
                        out[rows[m], p] = bg

                # Trim unsupported pixels just before the first 3 column.
                if base and d == 3 and stamp and color == 4:
                    pt = p - direction
                    if 0 <= pt < W:
                        m = main_mask[:, p].copy()
                        m[r_min:r_max + 1] = False
                        m &= ~main_mask[:, pt]
                        if m.any():
                            out[m, p] = bg

        if m_col == 0:
            directions = [1]
        elif m_col == W - 1:
            directions = [-1]
        else:
            directions = [1, -1]

        for d in directions:
            process_direction(d)

    return out