import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    For every 0 pixel that has at least three orthogonal neighbors of the same
    non-background color C, paint the 3x3 block around that pixel with the
    other main non-background color, and set the pixel itself to 8.
    """
    arr = np.array(grid, dtype=int, copy=True)

    if arr.ndim != 2:
        return arr

    h, w = arr.shape
    if h == 0 or w == 0:
        return arr

    # Determine the main non-background colors, ignoring 8 if present.
    main_mask = (arr != 0) & (arr != 8)
    if not np.any(main_mask):
        return arr

    colors, counts = np.unique(arr[main_mask], return_counts=True)
    order = np.argsort(-counts)
    main_colors = colors[order].tolist()

    def other_color(base_color: int) -> int:
        """Return the other main color different from base_color."""
        for c in main_colors:
            if c != base_color:
                return int(c)
        return 0

    centers = []
    center_set = set()

    # Find all zero pixels with at least three orthogonal neighbors of one color.
    for r in range(h):
        for c in range(w):
            if arr[r, c] != 0:
                continue

            neighbor_counts = {}
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    v = int(arr[nr, nc])
                    if v != 0 and v != 8:
                        neighbor_counts[v] = neighbor_counts.get(v, 0) + 1

            if not neighbor_counts:
                continue

            base_color = max(neighbor_counts, key=lambda k: neighbor_counts[k])
            if neighbor_counts[base_color] >= 3:
                if (r, c) not in center_set:
                    center_set.add((r, c))
                    centers.append((r, c, other_color(base_color)))

    out = arr.copy()

    # Accumulate proposed stamp colors so overlapping same-color stamps combine
    # naturally. Center pixels are excluded and will be forced to 8 later.
    proposed = {}
    for r, c, stamp_color in centers:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue

                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in center_set:
                    cell_counts = proposed.setdefault((nr, nc), {})
                    cell_counts[stamp_color] = cell_counts.get(stamp_color, 0) + 1

    # Apply the chosen stamp color to non-center cells.
    for (nr, nc), cell_counts in proposed.items():
        chosen = max(cell_counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        out[nr, nc] = int(chosen)

    # Mark all centers with 8.
    for r, c, _ in centers:
        out[r, c] = 8

    return out