import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the intended nested object from scattered fragments.

    Strategy:
    1. Identify the center marker as the rarest non-zero color.
    2. Identify the middle layer as the most frequent non-zero color near that center.
    3. Identify the outer layer as the remaining non-zero color.
    4. Reconstruct layers relative to the center, ignoring small far-away decoys.
    5. Crop to the tight bounding box of the reconstructed object.
    """
    if grid.size == 0:
        return np.zeros_like(grid)

    colors = np.unique(grid)
    nonzero_colors = [int(c) for c in colors if c != 0]

    if not nonzero_colors:
        return np.zeros_like(grid)

    # Count pixels per non-zero color.
    counts = {}
    for c in nonzero_colors:
        counts[c] = int((grid == c).sum())

    # Center marker: rarest color. If tie, choose smallest color.
    center_color = min(nonzero_colors, key=lambda c: (counts[c], c))

    center_positions = np.argwhere(grid == center_color)
    if len(center_positions) == 0:
        return np.zeros_like(grid)

    # Use the densest center cluster if there are multiple rare-color decoys.
    # Choose the center position closest to the centroid of the most frequent non-center color.
    other_counts = {c: cnt for c, cnt in counts.items() if c != center_color}
    if other_counts:
        anchor_color = max(other_counts, key=lambda c: other_counts[c])
        anchor_pos = np.argwhere(grid == anchor_color)
        if len(anchor_pos) > 0:
            centroid = anchor_pos.mean(axis=0)
            center_pos = center_positions[np.argmin(
                np.sum((center_positions - centroid) ** 2, axis=1)
            )].astype(int)
        else:
            center_pos = center_positions[0].astype(int)
    else:
        center_pos = center_positions[0].astype(int)

    cy, cx = center_pos

    # Middle color: most frequent non-center color within radius 8.
    def color_count_near_color(color, center_color, radius=8):
        ys, xs = np.where(grid == color)
        if len(ys) == 0:
            return 0
        mask = (np.abs(ys - cy) <= radius) & (np.abs(xs - cx) <= radius)
        return int(mask.sum())

    middle_color = None
    best = -1
    for c in other_counts:
        cnt = color_count_near_color(c, center_color)
        if cnt > best:
            best = cnt
            middle_color = c

    if middle_color is None:
        # Degenerate case: only one color present.
        return grid.copy()

    outer_color = None
    for c in other_counts:
        if c != middle_color:
            outer_color = c
            break

    if outer_color is None:
        outer_color = center_color  # fallback, should not happen

    # Collect relative positions for each layer.
    def rel_positions(color, min_count=1):
        ys, xs = np.where(grid == color)
        pts = [(y - cy, x - cx) for y, x in zip(ys, xs)]
        return pts

    center_rel = rel_positions(center_color)
    middle_rel = rel_positions(middle_color)
    outer_rel = rel_positions(outer_color)

    # Remove far-away outer decoys: keep outer pixels within radius 12.
    outer_rel_main = [(dy, dx) for dy, dx in outer_rel if max(abs(dy), abs(dx)) <= 12]

    # Fallback if filtering removed too much.
    if len(outer_rel_main) == 0:
        outer_rel_main = outer_rel

    # Build a sparse map of output cells relative to center.
    cells = {}

    # Middle layer first.
    for dy, dx in middle_rel:
        if -50 <= dy <= 50 and -50 <= dx <= 50:
            cells[(dy, dx)] = middle_color

    # Center layer (may overwrite middle if same spot, but normally distinct).
    for dy, dx in center_rel:
        cells[(dy, dx)] = center_color

    # Outer layer last, but do not overwrite center/middle pixels in the core region.
    for dy, dx in outer_rel_main:
        if -50 <= dy <= 50 and -50 <= dx <= 50:
            # Avoid painting outer decoys through the middle/center core.
            if max(abs(dy), abs(dx)) <= 4 and (dy, dx) in cells:
                continue
            cells[(dy, dx)] = outer_color

    if not cells:
        return np.zeros_like(grid)

    ys = [dy for dy, dx in cells]
    xs = [dx for dy, dx in cells]
    min_y = min(ys)
    max_y = max(ys)
    min_x = min(xs)
    max_x = max(xs)

    height = max_y - min_y + 1
    width = max_x - min_x + 1

    if height <= 0 or width <= 0:
        return np.zeros_like(grid)

    out = np.zeros((height, width), dtype=int)
    for (dy, dx), color in cells.items():
        r = dy - min_y
        c = dx - min_x
        if 0 <= r < height and 0 <= c < width:
            out[r, c] = color

    return out