import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid into a small bar-chart-like output.

    Rule inferred from the examples:
      - The grid contains several rectangular "container" walls, each in its
        own color.
      - One color is scattered "noise/filler" and is used as the padding
        color in the output.
      - For every non-noise container color:
          * Reconstruct its rectangular wall network.
          * Repair small gaps in horizontal/vertical wall lines.
          * Count the enclosed empty rooms inside that container.
      - The output has one row per container color, sorted by:
          1. room count, ascending
          2. color, ascending
        Each row contains the container color repeated `count` times, followed
        by the noise/filler color.

    The implementation is deliberately conservative:
      - noise is identified as the color with the least long linear structure;
      - walls are reconstructed from the container color itself, with small
        gaps filled only when there is strong horizontal/vertical evidence;
      - narrow one-pixel rooms are preserved by not filling gaps between two
        single-pixel vertical/horizontal wall supports.
    """
    grid = np.asarray(grid, dtype=int)

    if grid.size == 0:
        return np.empty((0, 0), dtype=int)

    h, w = grid.shape
    unique_colors = [int(c) for c in np.unique(grid) if c != 0]

    if not unique_colors:
        return np.zeros((1, 1), dtype=int)

    # ------------------------------------------------------------------
    # Helper: largest 4-connected component size for one color mask.
    # ------------------------------------------------------------------
    def max_component_size(mask: np.ndarray) -> int:
        seen = np.zeros_like(mask, dtype=bool)
        best = 0

        for i in range(h):
            for j in range(w):
                if mask[i, j] and not seen[i, j]:
                    stack = [(i, j)]
                    seen[i, j] = True
                    size = 0

                    while stack:
                        y, x = stack.pop()
                        size += 1

                        for ny, nx in ((y + 1, x), (y - 1, x),
                                       (y, x + 1), (y, x - 1)):
                            if 0 <= ny < h and 0 <= nx < w:
                                if mask[ny, nx] and not seen[ny, nx]:
                                    seen[ny, nx] = True
                                    stack.append((ny, nx))

                    if size > best:
                        best = size

        return best

    # ------------------------------------------------------------------
    # Helper: structural statistics for identifying the noise/filler color.
    # Noise tends to have short runs and few pixels belonging to long runs.
    # ------------------------------------------------------------------
    def color_stats(color: int):
        mask = (grid == color)
        total = int(mask.sum())
        if total == 0:
            return 0, 0.0, 0, -1, color

        max_run = 0
        long_pixels = 0

        # Horizontal runs.
        for i in range(h):
            j = 0
            while j < w:
                if mask[i, j]:
                    k = j
                    while k + 1 < w and mask[i, k + 1]:
                        k += 1
                    run_len = k - j + 1
                    if run_len > max_run:
                        max_run = run_len
                    if run_len >= 3:
                        long_pixels += run_len
                    j = k + 1
                else:
                    j += 1

        # Vertical runs.
        for j in range(w):
            i = 0
            while i < h:
                if mask[i, j]:
                    k = i
                    while k + 1 < h and mask[k + 1, j]:
                        k += 1
                    run_len = k - i + 1
                    if run_len > max_run:
                        max_run = run_len
                    if run_len >= 3:
                        long_pixels += run_len
                    i = k + 1
                else:
                    i += 1

        long_frac = float(long_pixels) / float(total)
        largest_comp = max_component_size(mask)

        # Pick the least structurally meaningful color as noise.
        return (max_run, long_frac, largest_comp, -total, color)

    stats = {c: color_stats(c) for c in unique_colors}

    if len(unique_colors) == 1:
        noise_color = 0
        container_colors = unique_colors
    else:
        noise_color = min(stats.values())[-1]
        container_colors = [c for c in unique_colors if c != noise_color]

        # Extremely degenerate fallback.
        if not container_colors:
            noise_color = 0
            container_colors = unique_colors

    # ------------------------------------------------------------------
    # Helper: find a plausible bounding rectangle for one container color.
    # Use pixels that belong to short horizontal/vertical runs, which filters
    # out isolated single-pixel strays while still keeping broken borders.
    # ------------------------------------------------------------------
    def container_bbox(color: int):
        mask = (grid == color)

        if not mask.any():
            return None

        cand = np.zeros((h, w), dtype=bool)

        # Pixels in horizontal runs of length >= 2.
        for i in range(h):
            j = 0
            while j < w:
                if mask[i, j]:
                    k = j
                    while k + 1 < w and mask[i, k + 1]:
                        k += 1
                    if k - j + 1 >= 2:
                        cand[i, j:k + 1] = True
                    j = k + 1
                else:
                    j += 1

        # Pixels in vertical runs of length >= 2.
        for j in range(w):
            i = 0
            while i < h:
                if mask[i, j]:
                    k = i
                    while k + 1 < h and mask[k + 1, j]:
                        k += 1
                    if k - i + 1 >= 2:
                        cand[i:k + 1, j] = True
                    i = k + 1
                else:
                    i += 1

        if cand.any():
            rows = np.where(cand.any(axis=1))[0]
            cols = np.where(cand.any(axis=0))[0]
            r0, r1 = int(rows.min()), int(rows.max())
            c0, c1 = int(cols.min()), int(cols.max())
        else:
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            if rows.size == 0 or cols.size == 0:
                return None
            r0, r1 = int(rows.min()), int(rows.max())
            c0, c1 = int(cols.min()), int(cols.max())

        if r1 <= r0 or c1 <= c0:
            return None

        return r0, r1, c0, c1

    # ------------------------------------------------------------------
    # Count enclosed rooms for one container color.
    # ------------------------------------------------------------------
    def count_rooms(color: int) -> int:
        bbox = container_bbox(color)
        if bbox is None:
            return 0

        r0, r1, c0, c1 = bbox
        H = r1 - r0 + 1
        W = c1 - c0 + 1

        if H < 3 or W < 3:
            # A meaningful enclosed room needs at least a 3x3 frame.
            return 0

        wall = np.zeros((H, W), dtype=bool)

        # The reconstructed outer border is assumed to be closed.
        wall[0, :] = True
        wall[-1, :] = True
        wall[:, 0] = True
        wall[:, -1] = True

        # All visible pixels of this container are wall evidence.
        wall |= (grid[r0:r1 + 1, c0:c1 + 1] == color)

        def vertical_support(x: int, y: int) -> bool:
            """Is (y, x) supported vertically by an adjacent wall pixel?"""
            if x < 0 or x >= W or y < 0 or y >= H:
                return False
            if not wall[y, x]:
                return False
            if y > 0 and wall[y - 1, x]:
                return True
            if y + 1 < H and wall[y + 1, x]:
                return True
            return False

        def horizontal_support(y: int, x: int) -> bool:
            """Is (y, x) supported horizontally by an adjacent wall pixel?"""
            if x < 0 or x >= W or y < 0 or y >= H:
                return False
            if not wall[y, x]:
                return False
            if x > 0 and wall[y, x - 1]:
                return True
            if x + 1 < W and wall[y, x + 1]:
                return True
            return False

        def fill_rows() -> bool:
            """Repair small horizontal gaps in wall lines."""
            changed = False

            for y in range(H):
                x = 0
                while x < W:
                    if not wall[y, x]:
                        x += 1
                        continue

                    # Current wall run: [start, end]
                    start = x
                    while x < W and wall[y, x]:
                        x += 1
                    end = x - 1
                    L = end - start + 1

                    # Gap after this run.
                    if x < W and not wall[y, x]:
                        gap_start = x
                        while x < W and not wall[y, x]:
                            x += 1
                        gap_len = x - gap_start

                        # Wall run after the gap: [rstart, rend]
                        if x < W:
                            rstart = x
                            while x < W and wall[y, x]:
                                x += 1
                            rend = x - 1
                            R = rend - rstart + 1

                            left_ok = (L >= 2) or vertical_support(end, y)
                            right_ok = (R >= 2) or vertical_support(rstart, y)

                            if gap_len == 1:
                                # Do not fill a one-pixel room between two
                                # single vertical supports.
                                if left_ok and right_ok and (L >= 2 or R >= 2):
                                    wall[y, gap_start:rstart] = True
                                    changed = True

                            elif gap_len == 2:
                                if L >= 2 and R >= 2:
                                    wall[y, gap_start:rstart] = True
                                    changed = True

                            elif gap_len == 3:
                                if L >= 3 and R >= 3:
                                    wall[y, gap_start:rstart] = True
                                    changed = True

            return changed

        def fill_cols() -> bool:
            """Repair small vertical gaps in wall lines."""
            changed = False

            for x in range(W):
                y = 0
                while y < H:
                    if not wall[y, x]:
                        y += 1
                        continue

                    # Current wall run: [start, end]
                    start = y
                    while y < H and wall[y, x]:
                        y += 1
                    end = y - 1
                    L = end - start + 1

                    # Gap after this run.
                    if y < H and not wall[y, x]:
                        gap_start = y
                        while y < H and not wall[y, x]:
                            y += 1
                        gap_len = y - gap_start

                        # Wall run after the gap: [rstart, rend]
                        if y < H:
                            rstart = y
                            while y < H and wall[y, x]:
                                y += 1
                            rend = y - 1
                            R = rend - rstart + 1

                            top_ok = (L >= 2) or horizontal_support(end, x)
                            bottom_ok = (R >= 2) or horizontal_support(rstart, x)

                            if gap_len == 1:
                                if top_ok and bottom_ok and (L >= 2 or R >= 2):
                                    wall[gap_start:rstart, x] = True
                                    changed = True

                            elif gap_len == 2:
                                if L >= 2 and R >= 2:
                                    wall[gap_start:rstart, x] = True
                                    changed = True

                            elif gap_len == 3:
                                if L >= 3 and R >= 3:
                                    wall[gap_start:rstart, x] = True
                                    changed = True

            return changed

        # Iterate until the wall network stabilizes.
        for _ in range(12):
            if not (fill_rows() or fill_cols()):
                break

        # Count connected non-wall components inside the reconstructed frame.
        visited = np.zeros((H, W), dtype=bool)
        rooms = 0

        for i in range(H):
            for j in range(W):
                if not wall[i, j] and not visited[i, j]:
                    rooms += 1
                    stack = [(i, j)]
                    visited[i, j] = True

                    while stack:
                        y, x = stack.pop()
                        for ny, nx in ((y + 1, x), (y - 1, x),
                                       (y, x + 1), (y, x - 1)):
                            if 0 <= ny < H and 0 <= nx < W:
                                if not wall[ny, nx] and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    stack.append((ny, nx))

        return rooms

    # ------------------------------------------------------------------
    # Build the final histogram.
    # ------------------------------------------------------------------
    items = []
    for color in container_colors:
        cnt = count_rooms(color)
        if cnt > 0:
            items.append((int(cnt), int(color)))

    if not items:
        return np.zeros((1, 1), dtype=int)

    items.sort(key=lambda x: (x[0], x[1]))

    max_count = max(cnt for cnt, _ in items)
    out = np.full((len(items), max_count), int(noise_color), dtype=int)

    for r, (cnt, color) in enumerate(items):
        out[r, :cnt] = color

    return out