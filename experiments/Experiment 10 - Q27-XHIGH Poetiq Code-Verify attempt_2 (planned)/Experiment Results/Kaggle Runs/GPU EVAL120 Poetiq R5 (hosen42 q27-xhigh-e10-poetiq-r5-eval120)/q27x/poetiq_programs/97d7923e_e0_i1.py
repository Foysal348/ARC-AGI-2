import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    For each valid vertical bar, if its endpoint color is 1, 2, 3, or 4,
    select the bar of that color whose center is closest to the corresponding
    corner:
        1 -> top-left
        2 -> top-right
        3 -> bottom-right
        4 -> bottom-left
    Then recolor the selected bar's interior cells to the endpoint color.
    """
    arr = np.array(grid, dtype=int, copy=True)

    if arr.ndim != 2:
        return arr

    H, W = arr.shape
    if H == 0 or W == 0:
        return arr

    # Corner mapping for endpoint colors 1-4.
    corners = {
        1: (0, 0),
        2: (0, W - 1),
        3: (H - 1, W - 1),
        4: (H - 1, 0),
    }

    # Fallback for unexpected colors: cycle through the four corners.
    corner_list = [(0, 0), (0, W - 1), (H - 1, W - 1), (H - 1, 0)]

    def corner_for(color: int):
        if color in corners:
            return corners[color]
        return corner_list[(int(color) - 1) % 4]

    candidates = []

    # Find all vertical bars.
    for col in range(W):
        row = 0
        while row < H:
            if arr[row, col] == 0:
                row += 1
                continue

            top_row = row
            while row + 1 < H and arr[row + 1, col] != 0:
                row += 1
            bottom_row = row

            # A valid bar must have length at least 3, same top/bottom color,
            # and some interior pixel different from that endpoint color.
            if bottom_row - top_row + 1 >= 3:
                cap = int(arr[top_row, col])

                if cap == int(arr[bottom_row, col]):
                    interior = arr[top_row + 1:bottom_row, col]

                    if interior.size > 0 and np.any(interior != cap):
                        center_r = (top_row + bottom_row) / 2.0
                        center_c = float(col)

                        cr, cc = corner_for(cap)

                        # Manhattan distance from bar center to assigned corner.
                        dist = abs(center_r - cr) + abs(center_c - cc)
                        dist_sq = (center_r - cr) ** 2 + (center_c - cc) ** 2

                        # Tie-breakers: prefer the direction of the corner.
                        c_pref = center_c if cc == 0 else -center_c
                        r_pref = center_r if cr == 0 else -center_r

                        key = (dist, dist_sq, c_pref, r_pref, center_r, center_c)
                        candidates.append((cap, key, top_row, bottom_row, col))

            row = bottom_row + 1

    # Choose the closest valid bar for each endpoint color.
    best = {}
    for cap, key, top_row, bottom_row, col in candidates:
        if cap not in best or key < best[cap][0]:
            best[cap] = (key, top_row, bottom_row, col)

    # Recolor selected bar interiors.
    for cap, (_, top_row, bottom_row, col) in best.items():
        for r in range(top_row + 1, bottom_row):
            arr[r, col] = cap

    return arr