import numpy as np
from collections import Counter, defaultdict


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Repair damaged small framed cells.

    The input is treated as a collection of rectangular cells separated by a
    dominant frame color. Each cell contains a small periodic pattern using the
    frame color and a foreground color. Some interior pixels may be corrupted.

    The transform:
      1. Copies the grid.
      2. Finds the dominant non-zero color, assumed to be the frame color.
      3. Finds connected components of non-frame pixels; each component is the
         interior of a cell.
      4. For each cell interior, infers the most consistent row and column
         values and replaces inconsistent interior pixels with the inferred
         value.
      5. Returns the repaired grid.

    This matches the examples, where corrupted small patterns inside framed
    rectangles are restored to their clean periodic forms.
    """
    if grid is None or grid.size == 0:
        return grid

    out = grid.copy()
    h, w = out.shape

    # If trivially small, return unchanged.
    if h < 3 or w < 3:
        return out

    # Choose frame color: most frequent nonzero color.
    colors, counts = np.unique(out.reshape(-1), return_counts=True)
    nonzero_counts = {int(c): int(n) for c, n in zip(colors, counts) if c != 0}
    if not nonzero_counts:
        return out
    frame_color = max(nonzero_counts, key=lambda c: nonzero_counts[c])

    # Mask of interior cells: not frame, not background zero.
    interior_mask = (out != frame_color) & (out != 0)

    # Connected components of interior pixels using 4-connectivity.
    comp_id = np.zeros((h, w), dtype=int)
    comp_count = 0
    for r in range(h):
        for c in range(w):
            if interior_mask[r, c] and comp_id[r, c] == 0:
                comp_count += 1
                stack = [(r, c)]
                comp_id[r, c] = comp_count
                while stack:
                    cr, cc = stack.pop()
                    for nr, nc in ((cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)):
                        if 0 <= nr < h and 0 <= nc < w:
                            if interior_mask[nr, nc] and comp_id[nr, nc] == 0:
                                comp_id[nr, nc] = comp_count
                                stack.append((nr, nc))

    # Process each component as one cell interior.
    for comp in range(1, comp_count + 1):
        rs, cs = np.where(comp_id == comp)
        if len(rs) == 0:
            continue

        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())
        cell_h = r1 - r0 + 1
        cell_w = c1 - c0 + 1

        # Very small components are usually noise; leave unchanged unless they
        # can be trivially repaired.
        if cell_h <= 1 or cell_w <= 1:
            continue

        sub = out[r0:r1 + 1, c0:c1 + 1]

        # In each row, the dominant color is the expected clean row value.
        row_expected = np.array([
            Counter(sub[r, :]).most_common(1)[0][0]
            for r in range(cell_h)
        ])

        # In each column, the dominant color is the expected clean column value.
        col_expected = np.array([
            Counter(sub[:, c]).most_common(1)[0][0]
            for c in range(cell_w)
        ])

        # A pixel is consistent if it matches both the dominant row value and
        # the dominant column value. Otherwise replace it by the majority of
        # the two inferred values.
        for r in range(cell_h):
            for c in range(cell_w):
                current = sub[r, c]
                matches_row = current == row_expected[r]
                matches_col = current == col_expected[c]

                if matches_row and matches_col:
                    continue

                # Use row inference if it is more common among consistent pixels
                # in that row, otherwise column inference.
                row_score = int(np.sum(sub[r, :] == row_expected[r]))
                col_score = int(np.sum(sub[:, c] == col_expected[c]))

                if row_score >= col_score:
                    expected = row_expected[r]
                else:
                    expected = col_expected[c]

                if current != expected:
                    out[r0 + r, c0 + c] = expected

    return out