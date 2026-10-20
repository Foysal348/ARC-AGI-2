import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the interior of selected complete vertical bars.

    A complete vertical bar is a maximal non-zero vertical run where:
      - the top and bottom pixels have the same color,
      - the interior pixels all have one uniform different color.

    Selection rule:
      - If there is only one endpoint color among complete bars,
        select the rightmost bar of that color.
      - Otherwise, select:
          * the leftmost bar of the smallest endpoint color,
          * the rightmost bar of the largest endpoint color.

    The interior of each selected bar is replaced by its endpoint color.
    """
    if grid is None or grid.ndim != 2:
        return np.array(grid, copy=True)

    out = grid.copy()
    h, w = out.shape

    if h == 0 or w == 0:
        return out

    # bars_by_color[color] = list of (col, top_row, bottom_row, interior_length)
    bars_by_color = {}

    for col in range(w):
        row = 0
        while row < h:
            if out[row, col] == 0:
                row += 1
                continue

            top = row
            while row < h and out[row, col] != 0:
                row += 1
            bottom = row - 1

            length = bottom - top + 1
            if length < 3:
                continue

            top_color = int(out[top, col])
            bottom_color = int(out[bottom, col])

            if top_color != bottom_color:
                continue

            interior = out[top + 1:bottom, col]
            if interior.size == 0:
                continue

            body_color = int(interior[0])

            if body_color == top_color:
                continue

            if not np.all(interior == body_color):
                continue

            bars_by_color.setdefault(top_color, []).append(
                (col, top, bottom, bottom - top - 1)
            )

    if not bars_by_color:
        return out

    def choose_left(bars):
        # Smallest column; tie: longer interior; tie: topmost.
        return min(bars, key=lambda b: (b[0], -b[3], b[1]))

    def choose_right(bars):
        # Largest column; tie: longer interior; tie: topmost.
        return max(bars, key=lambda b: (b[0], b[3], -b[1]))

    colors = sorted(bars_by_color.keys())

    if len(colors) == 1:
        color = colors[0]
        col, top, bottom, _ = choose_right(bars_by_color[color])
        out[top + 1:bottom, col] = color
    else:
        min_color = colors[0]
        max_color = colors[-1]

        col, top, bottom, _ = choose_left(bars_by_color[min_color])
        out[top + 1:bottom, col] = min_color

        col, top, bottom, _ = choose_right(bars_by_color[max_color])
        out[top + 1:bottom, col] = max_color

    return out