import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by:
    1. Detecting 3x5 plus/cross objects.
    2. Using bottom-row 1x5 markers:
       - color 2 = source columns
       - color 3 = target columns
    3. Recoloring all plus objects in source columns to color 5.
    4. Adding that many plus objects above the topmost plus in each target column,
       using the target column's existing color.
    5. Removing the bottom-row marker lines.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2 or grid.size == 0:
        return grid.copy()

    out = grid.copy()
    H, W = grid.shape

    # Background is the most frequent color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # Relative coordinates of the plus shape, centered at (0, 0).
    PLUS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -2), (0, -1), (0, 0), (0, 1), (0, 2),
        (1, -1), (1, 0), (1, 1),
    ]
    PLUS_SET = set(PLUS)

    # Detect all plus objects.
    # Store as (top_row, center_col, color).
    plus_tokens = []
    for r in range(1, H - 1):
        for c in range(2, W - 2):
            color = int(grid[r, c])
            if color == bg:
                continue

            ok = True

            # All plus cells must be the same non-background color.
            for dr, dc in PLUS:
                if grid[r + dr, c + dc] != color:
                    ok = False
                    break
            if not ok:
                continue

            # All other cells inside the 3x5 bounding box must be background.
            for dr in range(-1, 2):
                for dc in range(-2, 3):
                    if (dr, dc) not in PLUS_SET and grid[r + dr, c + dc] != bg:
                        ok = False
                        break
                if not ok:
                    break

            if ok:
                plus_tokens.append((r - 1, c, color))

    # Group plus objects by center column.
    col_pluses = {}
    for top, c, color in plus_tokens:
        col_pluses.setdefault(c, []).append((top, color))

    # Detect bottom-row marker lines.
    # Markers are usually exact length 5 runs.
    candidates = []
    if H > 0:
        r = H - 1
        c = 0
        while c < W:
            if grid[r, c] == bg:
                c += 1
                continue

            start = c
            color = int(grid[r, c])
            while c < W and grid[r, c] == color:
                c += 1

            length = c - start
            if length == 5:
                candidates.append((start, start + 2, length, color))
            elif color in (2, 3) and length > 0:
                # Fallback for slightly malformed markers.
                candidates.append((start, start + length // 2, length, color))

    # Identify source/target marker colors.
    # Prefer fixed labels 2 and 3, with a small fallback for other cases.
    present_colors = sorted({m[3] for m in candidates})
    source_colors = set()
    target_colors = set()

    if present_colors:
        if 2 in present_colors:
            source_colors.add(2)
        if 3 in present_colors:
            target_colors.add(3)

        if not source_colors and not target_colors:
            source_colors.add(present_colors[0])
            target_colors.add(present_colors[1] if len(present_colors) > 1 else present_colors[0])
        elif not target_colors:
            for col in present_colors:
                if col not in source_colors:
                    target_colors.add(col)
                    break
        elif not source_colors:
            for col in present_colors:
                if col not in target_colors:
                    source_colors.add(col)
                    break

    source_centers = {m[1] for m in candidates if m[3] in source_colors}
    target_centers = {m[1] for m in candidates if m[3] in target_colors}

    # Total number of plus objects in all source columns.
    source_total = sum(len(col_pluses.get(c, [])) for c in source_centers)

    SPACING = 4

    def draw_plus(top, center, color, safe=False):
        """Draw a 3x5 plus with top row `top` and center column `center`."""
        if top < 0 or top + 2 >= H:
            return
        if center - 2 < 0 or center + 2 >= W:
            return

        center_row = top + 1
        for dr, dc in PLUS:
            rr = center_row + dr
            cc = center + dc

            if safe:
                # Only overwrite background or cells already of this color.
                if out[rr, cc] != bg and out[rr, cc] != color:
                    continue

            out[rr, cc] = color

    # Recolor all source-column plus objects to 5.
    for c in source_centers:
        for top, _ in col_pluses.get(c, []):
            draw_plus(top, c, 5, safe=False)

    # Extend each target column upward by source_total plus objects.
    for c in target_centers:
        if source_total <= 0:
            continue

        shapes = col_pluses.get(c, [])

        if shapes:
            top_top = min(top for top, _ in shapes)

            # Use the target column's current color after source recoloring.
            target_color = int(out[top_top + 1, c])

            # Fallback if the sampled cell became background for some reason.
            if target_color == bg:
                color_counts = {}
                for _, colr in shapes:
                    color_counts[colr] = color_counts.get(colr, 0) + 1
                target_color = max(color_counts.items(), key=lambda kv: kv[1])[0]

            for i in range(1, source_total + 1):
                new_top = top_top - i * SPACING
                if new_top < 0 or new_top + 2 >= H:
                    break
                draw_plus(new_top, c, target_color, safe=True)

        else:
            # If a target column has no existing plus objects, start from the
            # bottom-most standard position and use color 5 as the new color.
            target_color = 5
            base_top = H - 5
            if base_top < 0:
                base_top = max(0, H - 3)

            for i in range(source_total):
                new_top = base_top - i * SPACING
                if new_top < 0 or new_top + 2 >= H:
                    break
                draw_plus(new_top, c, target_color, safe=True)

    # Erase bottom-row markers.
    removal_colors = source_colors | target_colors
    for start, _, length, color in candidates:
        if color in removal_colors:
            out[H - 1, start:start + length] = bg

    return out