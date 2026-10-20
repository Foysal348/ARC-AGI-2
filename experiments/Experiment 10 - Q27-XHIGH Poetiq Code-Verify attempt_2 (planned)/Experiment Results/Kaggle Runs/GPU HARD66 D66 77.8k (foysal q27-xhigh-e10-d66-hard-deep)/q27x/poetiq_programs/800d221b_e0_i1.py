import numpy as np
from collections import deque, defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule inferred from the examples:

    1. The most frequent color is treated as the background.
    2. There is a "line" color that forms thin 1-pixel lines and one
       solid 3x3 junction.  The junction is detected as a 3x3 block of
       a non-background color whose immediate 5x5 neighborhood contains
       only that line color and the background color.
    3. All other non-background colors are "paint" colors.
    4. Remove the 3x3 junction and label 4-connected components of the
       line color.  Each component is painted with the most common paint
       color adjacent to it.  Ties are broken by global paint frequency.
    5. The 3x3 junction border remains the line color, while its center
       pixel is painted with the color that was used for the largest
       number of line pixels.  Ties are broken by the local surrounding
       ring and then by global paint frequency.
    """
    grid = np.array(grid, dtype=int, copy=True)
    if grid.ndim != 2 or grid.size == 0:
        return grid

    H, W = grid.shape
    if H < 3 or W < 3:
        return grid

    flat = grid.ravel()
    max_color = int(flat.max()) if flat.size else 9
    counts = np.bincount(flat, minlength=max(10, max_color + 1))
    colors = [i for i in range(len(counts)) if counts[i] > 0]

    if len(colors) <= 1:
        return grid

    # Background is the most frequent color.
    bg = int(np.argmax(counts))

    # ------------------------------------------------------------
    # Find the line color and the central 3x3 junction.
    # ------------------------------------------------------------
    line_color = None
    r0 = c0 = None
    best_score = None

    for c in colors:
        if c == bg:
            continue

        mask = (grid == c)

        for r in range(H - 2):
            for col in range(W - 2):
                if not mask[r:r + 3, col:col + 3].all():
                    continue

                r1 = max(0, r - 1)
                r2 = min(H, r + 3)
                c1 = max(0, col - 1)
                c2 = min(W, col + 3)

                area = grid[r1:r2, c1:c2]
                ring_same = int(np.sum(area == c)) - 9

                # The central junction has some line cells around it.
                if ring_same <= 0:
                    continue

                uniq = set(np.unique(area).tolist())
                subset = 1 if uniq <= {c, bg} else 0
                bg_count = int(np.sum(area == bg))

                # Prefer a fully enclosed-looking junction, with background
                # visible around it and not on the grid border.
                border_bonus = 0 if (
                    r > 0 and col > 0 and
                    r + 2 < H - 1 and col + 2 < W - 1
                ) else -1000

                score = (subset, bg_count, border_bonus, int(counts[c]))
                if best_score is None or score > best_score:
                    best_score = score
                    line_color = int(c)
                    r0, c0 = r, col

    # Fallback: if the strict test failed, choose any solid 3x3 block of a
    # non-background color, preferring one with more background around it.
    if line_color is None:
        fb_best = None
        for c in colors:
            if c == bg:
                continue
            mask = (grid == c)
            for r in range(H - 2):
                for col in range(W - 2):
                    if mask[r:r + 3, col:col + 3].all():
                        r1 = max(0, r - 1)
                        r2 = min(H, r + 3)
                        c1 = max(0, col - 1)
                        c2 = min(W, col + 3)
                        area = grid[r1:r2, c1:c2]
                        bg_count = int(np.sum(area == bg))
                        score = (bg_count, int(counts[c]))
                        if fb_best is None or score > fb_best[0]:
                            fb_best = (score, r, col, c)
        if fb_best is not None:
            _, r0, c0, line_color = fb_best

    if line_color is None:
        return grid

    out = grid.copy()

    # Paint colors are all colors except background and line.
    paint_colors = [int(c) for c in colors if c != bg and c != line_color]
    if not paint_colors:
        return out

    paint_set = set(paint_colors)
    paint_counts = {p: int(counts[p]) for p in paint_colors}

    # ------------------------------------------------------------
    # Label line components after removing the central 3x3 block.
    # ------------------------------------------------------------
    comp_mask = (grid == line_color)
    comp_mask[r0:r0 + 3, c0:c0 + 3] = False

    visited = np.zeros((H, W), dtype=bool)
    line_totals = defaultdict(int)

    def choose_target(adj_counts):
        """Choose paint color for a component from adjacent paint counts."""
        best_p = None
        best_key = None
        for p in paint_colors:
            key = (adj_counts.get(p, 0), paint_counts[p], -p)
            if best_key is None or key > best_key:
                best_key = key
                best_p = p
        return best_p

    for i in range(H):
        for j in range(W):
            if not comp_mask[i, j] or visited[i, j]:
                continue

            q = deque([(i, j)])
            visited[i, j] = True
            rows = []
            cols = []
            adj_counts = defaultdict(int)

            while q:
                r, c = q.popleft()
                rows.append(r)
                cols.append(c)

                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if comp_mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                        elif (not comp_mask[nr, nc]) and (grid[nr, nc] in paint_set):
                            adj_counts[int(grid[nr, nc])] += 1

            target = choose_target(adj_counts)
            line_totals[target] += len(rows)

            if rows:
                out[np.array(rows, dtype=int), np.array(cols, dtype=int)] = target

    # ------------------------------------------------------------
    # Paint the center of the 3x3 junction.
    # ------------------------------------------------------------
    # Count paint colors in the immediate 5x5 ring around the junction,
    # excluding the junction itself.
    r1 = max(0, r0 - 1)
    r2 = min(H, r0 + 3)
    c1 = max(0, c0 - 1)
    c2 = min(W, c0 + 3)

    local_counts = {p: 0 for p in paint_colors}
    for r in range(r1, r2):
        for c in range(c1, c2):
            if r0 <= r < r0 + 3 and c0 <= c < c0 + 3:
                continue
            val = int(out[r, c])
            if val in paint_set:
                local_counts[val] += 1

    center_color = max(
        paint_colors,
        key=lambda p: (
            line_totals.get(p, 0),
            local_counts.get(p, 0),
            paint_counts[p],
            -p
        )
    )

    out[r0 + 1, c0 + 1] = center_color
    return out