import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Project a straight 4-colored marker line perpendicular to itself.

    Rule:
      * Background is the most common border color other than 4.
      * Main object color is the most common color other than background and 4.
      * For each perpendicular line at distance d from the marker line:
          - odd d: erase main-color pixels inside the marker footprint.
          - even d:
              * if d % 6 == 4:
                  - change all main-color pixels on that perpendicular line to 3
                  - set the marker footprint to 3
              * otherwise:
                  - set the marker footprint to 4
      * The original marker footprint remains 4.
    """
    arr = np.asarray(grid, dtype=int)

    if arr.ndim != 2:
        return arr.copy()

    out = arr.copy()
    H, W = arr.shape

    if H == 0 or W == 0:
        return out

    marker = (arr == 4)
    if not marker.any():
        return out

    mr, mc = np.where(marker)
    uniq_r = np.unique(mr)
    uniq_c = np.unique(mc)

    # Background: most common border color other than 4.
    border = np.concatenate((arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]))
    border = border[border != 4]

    non4 = arr[arr != 4]
    all_counts = {}
    if non4.size:
        all_vals, all_counts_arr = np.unique(non4, return_counts=True)
        all_counts = dict(zip(all_vals.tolist(), all_counts_arr.tolist()))

    if border.size:
        b_vals, b_counts = np.unique(border, return_counts=True)
        bg = int(max(
            zip(b_vals.tolist(), b_counts.tolist()),
            key=lambda x: (x[1], all_counts.get(x[0], 0))
        )[0])
    elif non4.size:
        vals, counts = np.unique(non4, return_counts=True)
        bg = int(vals[np.argmax(counts)])
    else:
        bg = 0

    # Main color: most frequent non-background, non-4 color.
    candidate = (arr != bg) & (arr != 4)
    if candidate.any():
        vals, counts = np.unique(arr[candidate], return_counts=True)
        main_color = int(vals[np.argmax(counts)])
        main_mask = (arr == main_color)
    else:
        main_mask = np.zeros((H, W), dtype=bool)

    # Determine marker orientation.
    if len(uniq_r) == 1 and len(uniq_c) == 1:
        r = int(mr[0])
        c = int(mc[0])
        if r == 0 or r == H - 1:
            orient = "h"
        elif c == 0 or c == W - 1:
            orient = "v"
        else:
            orient = "h"
    elif len(uniq_r) == 1:
        orient = "h"
    elif len(uniq_c) == 1:
        orient = "v"
    else:
        orient = "h" if len(uniq_r) <= len(uniq_c) else "v"

    if orient == "h":
        # Horizontal marker line.
        row_counts = np.bincount(mr, minlength=H)
        m_row = int(np.argmax(row_counts))
        foot_cols = np.where(marker[m_row, :])[0]
        if foot_cols.size == 0:
            foot_cols = uniq_c

        out[m_row, foot_cols] = 4

        for direction in (-1, 1):
            if direction == -1 and m_row == 0:
                continue
            if direction == 1 and m_row == H - 1:
                continue

            max_d = m_row if direction == -1 else H - 1 - m_row

            for d in range(1, max_d + 1):
                p = m_row + direction * d

                if d % 2 == 1:
                    # Odd distance: erase main-color pixels inside footprint.
                    if foot_cols.size:
                        m = main_mask[p, foot_cols]
                        if m.any():
                            out[p, foot_cols[m]] = bg
                else:
                    # Even distance: 4 or 3 depending on the repeating pattern.
                    if (d // 2) % 3 == 2:
                        # 3-line: convert main-color pixels in this row to 3.
                        if main_mask[p].any():
                            out[p, main_mask[p]] = 3
                        out[p, foot_cols] = 3
                    else:
                        # 4-line: stamp the footprint with 4.
                        out[p, foot_cols] = 4

    else:
        # Vertical marker line.
        col_counts = np.bincount(mc, minlength=W)
        m_col = int(np.argmax(col_counts))
        foot_rows = np.where(marker[:, m_col])[0]
        if foot_rows.size == 0:
            foot_rows = uniq_r

        out[foot_rows, m_col] = 4

        for direction in (-1, 1):
            if direction == -1 and m_col == 0:
                continue
            if direction == 1 and m_col == W - 1:
                continue

            max_d = m_col if direction == -1 else W - 1 - m_col

            for d in range(1, max_d + 1):
                p = m_col + direction * d

                if d % 2 == 1:
                    # Odd distance: erase main-color pixels inside footprint.
                    if foot_rows.size:
                        m = main_mask[foot_rows, p]
                        if m.any():
                            out[foot_rows[m], p] = bg
                else:
                    # Even distance: 4 or 3 depending on the repeating pattern.
                    if (d // 2) % 3 == 2:
                        # 3-column: convert main-color pixels in this column to 3.
                        col_main = main_mask[:, p]
                        if col_main.any():
                            out[col_main, p] = 3
                        out[foot_rows, p] = 3
                    else:
                        # 4-column: stamp the footprint with 4.
                        out[foot_rows, p] = 4

    return out