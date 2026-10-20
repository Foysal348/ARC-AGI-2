import numpy as np
from collections import Counter, deque
from typing import List, Tuple

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid according to the examples.

    High-level rule
    ---------------
    1. The most frequent colour is the background.
    2. The second most frequent colour is the frame/wall colour.
    3. All other colours are "content" colours.
    4. Crop the minimal bounding box of all content pixels, expanded by:
         - 1 row above,
         - 2 rows below,
         - 1 column left,
         - 1 column right.
       This gives the working canvas.
    5. For every connected component of every content colour, normalise it
       vertically using the following observed behaviour:
         - Let the component have height H.
         - A "full row" is a row of the component with maximum number of pixels.
         - A "cap row" is a row containing exactly one pixel, occurring before
           the first full row.
         - If H == 1:
             * if the component touches the top or bottom edge of the content
               bounding box, duplicate the row once;
             * otherwise keep it as is.
         - If H >= 2:
             * remove non-cap rows above the first full row;
             * if the only row above the full row is a cap and H == 2,
               output cap / full / cap;
             * otherwise output kept caps followed by full rows, and fill any
               remaining lower rows with the full-row pattern;
             * if there are multiple full rows, keep the component height and
               fill non-full rows with the full-row pattern.
    6. Place the normalised component back into the output canvas, preserving
       relative horizontal position and the same number of vertical slots.

    This reproduces both provided examples and handles the same construction
    style expected for the challenges.
    """
    if grid.size == 0:
        return grid.copy()

    # Flatten and count colours
    flat = grid.flatten()
    counts = Counter(flat.tolist())

    if not counts:
        return grid.copy()

    # Background and wall/frame colours
    background = counts.most_common(1)[0][0]
    if len(counts) == 1:
        return grid.copy()
    wall = counts.most_common(2)[1][0]

    # Content mask: any colour that is neither background nor wall
    content_mask = (grid != background) & (grid != wall)
    content_positions = np.argwhere(content_mask)

    if content_positions.size == 0:
        return grid.copy()

    # Bounding box of all content
    rmin, cmin = content_positions.min(axis=0)
    rmax, cmax = content_positions.max(axis=0)

    # Expanded crop: 1 above, 2 below, 1 left, 1 right
    top = max(0, rmin - 1)
    bottom = min(grid.shape[0] - 1, rmax + 2)
    left = max(0, cmin - 1)
    right = min(grid.shape[1] - 1, cmax + 1)

    crop = grid[top:bottom + 1, left:right + 1].copy()
    out = crop.copy()

    # Content colours present in the crop
    content_colours = set(np.unique(crop).tolist())
    content_colours.discard(background)
    content_colours.discard(wall)

    H, W = crop.shape

    # Global content min/max rows inside crop, for edge detection
    crop_content_mask = (crop != background) & (crop != wall)
    if not np.any(crop_content_mask):
        return out

    global_min_row = int(np.argwhere(crop_content_mask).min(axis=0)[0])
    global_max_row = int(np.argwhere(crop_content_mask).max(axis=0)[0])

    def get_component(mask: np.ndarray, r: int, c: int) -> List[List[int]]:
        """4-connected component collection."""
        if not mask[r, c]:
            return []
        h, w = mask.shape
        seen = np.zeros_like(mask, dtype=bool)
        q = deque([(r, c)])
        seen[r, c] = True
        comp = []
        while q:
            x, y = q.popleft()
            comp.append((x, y))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                    seen[nx, ny] = True
                    q.append((nx, ny))
        return comp

    def normalize_component(cells: List[Tuple[int, int]], top_global_row: int,
                            left_global_col: int) -> np.ndarray:
        """
        Given cells of one connected component in crop coordinates,
        return its normalised mask in local component coordinates.
        """
        if not cells:
            return np.zeros((0, 0), dtype=bool)

        local_cells = [(r - top_global_row, c - left_global_col) for r, c in cells]
        rmin = min(r for r, _ in local_cells)
        rmax = max(r for r, _ in local_cells)
        cmin = min(c for _, c in local_cells)
        cmax = max(c for _, c in local_cells)

        h = rmax - rmin + 1
        w = cmax - cmin + 1
        local_mask = np.zeros((h, w), dtype=bool)
        for r, c in local_cells:
            local_mask[r, c] = True

        # Single-row component
        if h == 1:
            # Edge of global content bounding box?
            is_top_edge = (top_global_row == global_min_row)
            is_bottom_edge = (top_global_row == global_max_row)
            if is_top_edge or is_bottom_edge:
                # Duplicate row downward
                return np.vstack([local_mask, local_mask])
            return local_mask

        # Multi-row component
        row_counts = local_mask.sum(axis=1)
        max_count = int(row_counts.max())
        full_rows = [i for i, v in enumerate(row_counts) if v == max_count]

        # If there are multiple full rows, keep the whole height and fill
        # non-full rows with the full-row pattern.
        if len(full_rows) > 1:
            res = np.zeros_like(local_mask)
            full_pattern = local_mask[full_rows[-1]]
            for i in range(h):
                res[i] = full_pattern
            # Keep original full rows as they are (they already equal full_pattern)
            return res

        # Single full row case
        full_row = full_rows[0]

        # Cap rows are strictly above the full row and contain exactly one pixel
        cap_rows = []
        for i in range(full_row):
            if row_counts[i] == 1:
                cap_rows.append(i)
            else:
                # Non-cap row above the full row is removed
                break

        # Special case observed for 2-row components with one cap above a full row
        if h == 2 and full_row == 1 and cap_rows == [0]:
            cap = local_mask[0]
            full = local_mask[1]
            return np.vstack([cap, full, cap])

        # If no cap rows are kept and original height was 2, collapse to one row
        if h == 2 and not cap_rows:
            return local_mask[full_row:full_row + 1]

        # General case:
        # Keep at most (h - 2) cap rows so that at least two full rows remain.
        keep_caps = cap_rows[:max(0, h - 2)]
        res_rows = [local_mask[i] for i in keep_caps]
        # Add enough full rows to reach original height
        while len(res_rows) < h:
            res_rows.append(local_mask[full_row])
        # If we kept fewer caps than h - 2, fill the rest with full rows
        res = np.vstack(res_rows[:h])
        return res

    # Process each content colour and its connected components
    for colour in content_colours:
        colour_mask = (crop == colour)
        processed = np.zeros_like(colour_mask, dtype=bool)

        for r in range(H):
            for c in range(W):
                if colour_mask[r, c] and not processed[r, c]:
                    cells = get_component(colour_mask, r, c)
                    for rr, cc in cells:
                        processed[rr, cc] = True

                    if not cells:
                        continue

                    # Bounding box of the component
                    rr_min = min(rr for rr, cc in cells)
                    rr_max = max(rr for rr, cc in cells)
                    cc_min = min(cc for rr, cc in cells)
                    cc_max = max(cc for rr, cc in cells)

                    old_h = rr_max - rr_min + 1
                    old_w = cc_max - cc_min + 1

                    norm = normalize_component(cells, rr_min, cc_min)
                    new_h, new_w = norm.shape

                    # We only allow vertical expansion in this solution.
                    # Horizontal size should not change; if it does, keep old width.
                    if new_w != old_w:
                        # Fallback: crop/pad horizontally to old width
                        tmp = np.zeros((new_h, old_w), dtype=bool)
                        tmp[:, :min(new_w, old_w)] = norm[:, :min(new_w, old_w)]
                        norm = tmp
                        new_w = old_w

                    # If the component expanded vertically, place it starting at
                    # the original top row. This matches the examples where
                    # expanded rows overwrite rows below the original component.
                    start_r = rr_min
                    start_c = cc_min

                    # Clear old component pixels in out, then write normalised mask
                    for rr, cc in cells:
                        if out[rr, cc] == colour:
                            out[rr, cc] = background

                    # Write normalised component
                    for i in range(new_h):
                        j0 = start_c
                        j1 = start_c + new_w
                        r0 = start_r + i
                        if r0 >= H:
                            break
                        row_mask = norm[i]
                        for j in range(new_w):
                            if row_mask[j] and 0 <= j0 + j < W:
                                out[r0, j0 + j] = colour

    return out