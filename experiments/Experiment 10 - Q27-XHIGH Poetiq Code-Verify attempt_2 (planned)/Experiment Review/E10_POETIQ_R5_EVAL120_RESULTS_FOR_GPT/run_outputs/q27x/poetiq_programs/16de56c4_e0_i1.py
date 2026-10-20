import numpy as np
from math import gcd


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform sparse colored pixels into extended periodic lines.

    Rule summary:
    1. For each color, look for that color appearing at least twice in a row
       or at least twice in a column.
    2. Choose the color's active orientation:
       - horizontal if the smallest horizontal same-color gap is smaller than
         the smallest vertical same-color gap,
       - vertical if the reverse is true,
       - on ties, use the orientation with more aligned lines; if still tied,
         use the smaller period; if still tied, use both orientations.
    3. For every active row/column line containing that color at least twice:
       - Compute the period as the gcd of the coordinate differences.
       - Collect all non-zero pixels in that line lying on that periodic line.
       - If all collected pixels are the original color, fill the entire
         periodic line to the grid edge with that color.
       - Otherwise, fill from the first to last collected non-zero pixel with
         the endpoint color that is not the original color. If both endpoints
         are different, use the endpoint farther from the original color's
         pixels.
    4. Apply fills with a deterministic priority: full/unblocked fills first,
       then blocked/terminal fills; larger periods override smaller periods.
    """
    out = np.array(grid, copy=True, dtype=int)

    if out.ndim != 2:
        return out

    H, W = out.shape
    if H == 0 or W == 0:
        return out

    def calc_period(pos_list):
        """Return the gcd of consecutive differences in a sorted position list."""
        p = 0
        for a, b in zip(pos_list, pos_list[1:]):
            p = gcd(p, b - a)
        return p

    fills = []
    INF = 10**9

    for c in np.unique(out[out != 0]):
        c = int(c)
        rows, cols = np.where(out == c)

        if len(rows) < 2:
            continue

        h_pos = {}
        v_pos = {}

        for r, col in zip(rows, cols):
            r = int(r)
            col = int(col)
            h_pos.setdefault(r, []).append(col)
            v_pos.setdefault(col, []).append(r)

        h_gaps = []
        h_periods = []
        h_lines = 0
        for pos_list in h_pos.values():
            if len(pos_list) >= 2:
                pos_list.sort()
                h_lines += 1
                h_gaps.extend(b - a for a, b in zip(pos_list, pos_list[1:]))
                h_periods.append(calc_period(pos_list))

        v_gaps = []
        v_periods = []
        v_lines = 0
        for pos_list in v_pos.values():
            if len(pos_list) >= 2:
                pos_list.sort()
                v_lines += 1
                v_gaps.extend(b - a for a, b in zip(pos_list, pos_list[1:]))
                v_periods.append(calc_period(pos_list))

        min_h = min(h_gaps) if h_gaps else INF
        min_v = min(v_gaps) if v_gaps else INF

        if min_h < min_v:
            orientations = ["h"]
        elif min_v < min_h:
            orientations = ["v"]
        else:
            if h_lines != v_lines:
                orientations = ["h"] if h_lines > v_lines else ["v"]
            else:
                min_p_h = min(h_periods) if h_periods else INF
                min_p_v = min(v_periods) if v_periods else INF
                if min_p_h < min_p_v:
                    orientations = ["h"]
                elif min_p_v < min_p_h:
                    orientations = ["v"]
                else:
                    orientations = ["h", "v"] if (h_gaps or v_gaps) else []

        for orient in orientations:
            if orient == "h":
                length = W
                line_items = [
                    (r, pos_list)
                    for r, pos_list in h_pos.items()
                    if len(pos_list) >= 2
                ]
            else:
                length = H
                line_items = [
                    (col, pos_list)
                    for col, pos_list in v_pos.items()
                    if len(pos_list) >= 2
                ]

            for line_idx, pos_list in line_items:
                pos_list.sort()
                p = calc_period(pos_list)
                if p <= 0:
                    continue

                residue = pos_list[0] % p

                s_positions = []
                s_colors = []

                if orient == "h":
                    for pos in range(residue, length, p):
                        val = int(out[line_idx, pos])
                        if val != 0:
                            s_positions.append(pos)
                            s_colors.append(val)
                else:
                    for pos in range(residue, length, p):
                        val = int(out[pos, line_idx])
                        if val != 0:
                            s_positions.append(pos)
                            s_colors.append(val)

                if not s_positions:
                    continue

                min_pos = min(s_positions)
                max_pos = max(s_positions)
                all_same = all(val == c for val in s_colors)

                if all_same:
                    fill_color = c
                    cells = list(range(residue, length, p))
                    blocked = False
                else:
                    if orient == "h":
                        min_val = int(out[line_idx, min_pos])
                        max_val = int(out[line_idx, max_pos])
                    else:
                        min_val = int(out[min_pos, line_idx])
                        max_val = int(out[max_pos, line_idx])

                    if min_val == c and max_val == c:
                        fill_color = c
                    elif min_val != c and max_val == c:
                        fill_color = min_val
                    elif min_val == c and max_val != c:
                        fill_color = max_val
                    else:
                        # Both endpoints are non-original colors:
                        # choose the endpoint farther from the seed color pixels.
                        def dist_to_seed(pos):
                            return min(abs(pos - q) for q in pos_list)

                        d_min = dist_to_seed(min_pos)
                        d_max = dist_to_seed(max_pos)

                        if d_max > d_min:
                            fill_color = max_val
                        else:
                            fill_color = min_val

                    cells = list(range(min_pos, max_pos + 1, p))
                    blocked = True

                priority = (
                    1 if blocked else 0,
                    p,
                    0 if orient == "h" else 1,
                    line_idx,
                    min_pos,
                    fill_color,
                )

                fills.append((priority, orient, line_idx, fill_color, cells))

    for _, orient, line_idx, fill_color, cells in sorted(fills, key=lambda x: x[0]):
        if orient == "h":
            for pos in cells:
                out[line_idx, pos] = fill_color
        else:
            for pos in cells:
                out[pos, line_idx] = fill_color

    return out