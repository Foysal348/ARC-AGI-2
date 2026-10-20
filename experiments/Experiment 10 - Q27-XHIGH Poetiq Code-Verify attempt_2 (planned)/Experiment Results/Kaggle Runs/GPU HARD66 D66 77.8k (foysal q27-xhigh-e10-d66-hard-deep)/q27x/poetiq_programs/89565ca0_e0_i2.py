import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid into a small histogram of room counts.

    For each non-noise container color:
      - reconstruct its rectangular wall network,
      - repair small gaps caused by overwritten/noise pixels,
      - count connected empty regions inside it.

    The output rows are sorted by (room_count, color). Each row is filled
    first with the container color, then with the filler/noise color.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.size == 0:
        return np.empty((0, 0), dtype=int)

    h, w = grid.shape
    unique = [int(c) for c in np.unique(grid) if c != 0]
    if not unique:
        return np.zeros((1, 1), dtype=int)

    # ------------------------------------------------------------------
    # Identify the filler/noise color.
    # Noise is the most isolated color: small components, few long runs.
    # ------------------------------------------------------------------
    stats = {}
    for c in unique:
        mask = grid == c
        total = int(mask.sum())
        if total == 0:
            continue

        # Largest 4-connected component.
        visited = np.zeros((h, w), dtype=bool)
        max_comp = 0
        for i, j in np.argwhere(mask):
            if visited[i, j]:
                continue
            stack = [(int(i), int(j))]
            visited[i, j] = True
            size = 0
            while stack:
                y, x = stack.pop()
                size += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            if size > max_comp:
                max_comp = size

        # Maximum straight run length, and fraction of pixels in runs >= 3.
        max_run = 0
        long_mark = np.zeros((h, w), dtype=bool)

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
                        long_mark[i, j:k + 1] = True
                    j = k + 1
                else:
                    j += 1

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
                        long_mark[i:k + 1, j] = True
                    i = k + 1
                else:
                    i += 1

        long_frac = float(long_mark.sum()) / total

        # Prefer small components, then less linear structure.
        stats[c] = (max_comp, long_frac, max_run, -total, c)

    if len(unique) == 1:
        noise = 0
        container_colors = unique
    else:
        noise = min(stats.values())[-1]
        container_colors = [c for c in unique if c != noise]
        if not container_colors:
            noise = 0
            container_colors = unique

    # ------------------------------------------------------------------
    # Count rooms for one container color.
    # ------------------------------------------------------------------
    def count_rooms(color: int) -> int:
        mask = grid == color
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            return 0

        r0, r1 = int(rows[0]), int(rows[-1])
        c0, c1 = int(cols[0]), int(cols[-1])
        H, W = r1 - r0 + 1, c1 - c0 + 1

        if H < 2 or W < 2:
            return 0

        wall = np.zeros((H, W), dtype=bool)

        # Assume the outer bounding box is the container border.
        wall[0, :] = True
        wall[-1, :] = True
        wall[:, 0] = True
        wall[:, -1] = True

        # Visible pixels of this color are wall pixels.
        wall |= mask[r0:r1 + 1, c0:c1 + 1]

        def vertical_support(x: int, y: int) -> bool:
            """Is (y, x) part of a vertical wall segment of length at least 2?"""
            if x < 0 or x >= W or y < 0 or y >= H or not wall[y, x]:
                return False
            if y > 0 and wall[y - 1, x]:
                return True
            if y + 1 < H and wall[y + 1, x]:
                return True
            return False

        def horizontal_support(y: int, x: int) -> bool:
            """Is (y, x) part of a horizontal wall segment of length at least 2?"""
            if x < 0 or x >= W or y < 0 or y >= H or not wall[y, x]:
                return False
            if x > 0 and wall[y, x - 1]:
                return True
            if x + 1 < W and wall[y, x + 1]:
                return True
            return False

        def fill_rows() -> bool:
            """Fill small horizontal gaps between wall runs."""
            changed = False
            for y in range(H):
                x = 0
                while x < W:
                    if not wall[y, x]:
                        x += 1
                        continue

                    start = x
                    while x < W and wall[y, x]:
                        x += 1
                    end = x - 1
                    L = end - start + 1

                    if x < W and not wall[y, x]:
                        gap_start = x
                        while x < W and not wall[y, x]:
                            x += 1
                        gap_len = x - gap_start

                        if x < W:
                            rstart = x
                            while x < W and wall[y, x]:
                                x += 1
                            rend = x - 1
                            R = rend - rstart + 1

                            if gap_len <= 2:
                                left_ep = (
                                    L >= 2
                                    or (start == 0 and L == 1)
                                    or vertical_support(end, y)
                                )
                                right_ep = (
                                    R >= 2
                                    or (rend == W - 1 and R == 1)
                                    or vertical_support(rstart, y)
                                )

                                # Single missing pixel: need some horizontal evidence.
                                if gap_len == 1:
                                    if left_ep and right_ep and (L >= 2 or R >= 2):
                                        wall[y, gap_start:rstart] = True
                                        changed = True

                                # Two missing pixels: be more conservative.
                                elif gap_len == 2:
                                    if L >= 2 and R >= 2:
                                        wall[y, gap_start:rstart] = True
                                        changed = True
            return changed

        def fill_cols() -> bool:
            """Fill small vertical gaps between wall runs."""
            changed = False
            for x in range(W):
                y = 0
                while y < H:
                    if not wall[y, x]:
                        y += 1
                        continue

                    start = y
                    while y < H and wall[y, x]:
                        y += 1
                    end = y - 1
                    L = end - start + 1

                    if y < H and not wall[y, x]:
                        gap_start = y
                        while y < H and not wall[y, x]:
                            y += 1
                        gap_len = y - gap_start

                        if y < H:
                            rstart = y
                            while y < H and wall[y, x]:
                                y += 1
                            rend = y - 1
                            R = rend - rstart + 1

                            if gap_len <= 2:
                                top_ep = (
                                    L >= 2
                                    or (start == 0 and L == 1)
                                    or horizontal_support(end, x)
                                )
                                bottom_ep = (
                                    R >= 2
                                    or (rend == H - 1 and R == 1)
                                    or horizontal_support(rstart, x)
                                )

                                # Single missing pixel: need some vertical evidence.
                                if gap_len == 1:
                                    if top_ep and bottom_ep and (L >= 2 or R >= 2):
                                        wall[gap_start:rstart, x] = True
                                        changed = True

                                # Two missing pixels: be more conservative.
                                elif gap_len == 2:
                                    if L >= 2 and R >= 2:
                                        wall[gap_start:rstart, x] = True
                                        changed = True
            return changed

        # Iterate until no more repairs happen.
        for _ in range(10):
            if not (fill_rows() or fill_cols()):
                break

        # Count connected non-wall regions inside the box.
        interior = ~wall
        visited = np.zeros((H, W), dtype=bool)
        rooms = 0

        for i in range(H):
            for j in range(W):
                if interior[i, j] and not visited[i, j]:
                    rooms += 1
                    stack = [(i, j)]
                    visited[i, j] = True
                    while stack:
                        y, x = stack.pop()
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W:
                                if interior[ny, nx] and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    stack.append((ny, nx))

        return rooms

    # ------------------------------------------------------------------
    # Build output histogram.
    # ------------------------------------------------------------------
    results = []
    for c in container_colors:
        cnt = count_rooms(c)
        if cnt > 0:
            results.append((cnt, int(c)))

    if not results:
        return np.zeros((1, 1), dtype=int)

    results.sort(key=lambda p: (p[0], p[1]))
    max_count = max(cnt for cnt, _ in results)

    out = np.full((len(results), max_count), int(noise), dtype=int)
    for r, (cnt, color) in enumerate(results):
        out[r, :cnt] = color

    return out