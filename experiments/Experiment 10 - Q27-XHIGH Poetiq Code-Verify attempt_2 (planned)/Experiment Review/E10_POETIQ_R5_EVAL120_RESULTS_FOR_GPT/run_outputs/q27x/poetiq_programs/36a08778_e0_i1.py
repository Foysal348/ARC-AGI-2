import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by simulating downward-falling magenta rays.

    Rays start from the bottom cells of existing magenta (6) pixels.
    They fall through background (7) and magenta (6). When a ray is
    directly above a red (2) pixel, it activates the maximal horizontal
    red run in that row. It then draws a magenta border one row above
    that run, extended by one pixel on each side, clipped to the grid.
    New rays start from the two ends of that border. This repeats until
    no new rays are produced.
    """
    g = np.asarray(grid, dtype=int).copy()
    if g.ndim != 2:
        return g

    H, W = g.shape
    if H == 0 or W == 0:
        return g

    BG = 7

    # Identify every maximal horizontal run of red (2) pixels.
    run_id = -np.ones((H, W), dtype=int)
    runs = []  # each entry: (row, c_min, c_max)

    for r in range(H):
        c = 0
        while c < W:
            if g[r, c] == 2:
                c0 = c
                c += 1
                while c < W and g[r, c] == 2:
                    c += 1
                c1 = c - 1
                rid = len(runs)
                runs.append((r, c0, c1))
                run_id[r, c0:c1 + 1] = rid
            else:
                c += 1

    activated = [False] * len(runs)

    # Queue of starting positions for rays.
    q = deque()

    # Initial rays start from magenta pixels with no magenta pixel directly below.
    for r in range(H):
        for c in range(W):
            if g[r, c] == 6 and (r + 1 >= H or g[r + 1, c] != 6):
                q.append((r, c))

    processed_starts = set()

    while q:
        r, c = q.popleft()

        # Avoid re-processing the same ray start.
        if (r, c) in processed_starts:
            continue
        processed_starts.add((r, c))

        if r < 0 or r >= H or c < 0 or c >= W or g[r, c] != 6:
            continue

        cur = r

        # Move this ray downward.
        while cur + 1 < H:
            below = g[cur + 1, c]

            if below == 2:
                # The ray hits a red run.
                rid = int(run_id[cur + 1, c])

                if 0 <= rid < len(runs) and not activated[rid]:
                    _, cmin, cmax = runs[rid]
                    activated[rid] = True

                    # Draw the border one row above the red run.
                    left = max(0, cmin - 1)
                    right = min(W - 1, cmax + 1)

                    for cc in range(left, right + 1):
                        if g[cur, cc] == BG:
                            g[cur, cc] = 6

                    # New rays start from the expanded left and right ends.
                    if cmin - 1 >= 0 and g[cur, cmin - 1] == 6:
                        q.append((cur, cmin - 1))
                    if cmax + 1 < W and g[cur, cmax + 1] == 6:
                        q.append((cur, cmax + 1))

                # Red blocks the ray.
                break

            elif below == BG:
                # Paint background as magenta and continue falling.
                g[cur + 1, c] = 6
                cur += 1

            elif below == 6:
                # Continue through existing magenta.
                cur += 1

            else:
                # Unknown color: stop safely.
                break

    return g