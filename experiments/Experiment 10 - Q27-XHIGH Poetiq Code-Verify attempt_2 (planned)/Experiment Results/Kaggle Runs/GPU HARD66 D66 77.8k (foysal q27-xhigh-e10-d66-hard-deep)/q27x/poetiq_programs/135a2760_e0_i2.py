import numpy as np
from collections import deque
import math

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Find rectangular frames and restore their interiors to the smallest
    well-supported 2D periodic pattern.

    For each detected frame interior:
      * try period sizes in increasing tile-area order;
      * for a period (h, w), choose the most common color for each phase
        (i mod h, j mod w);
      * score the candidate by:
            Hamming errors + 3 * number_of_tied_top_colors
        to avoid accepting misleading wraparound periods;
      * return the first accepted candidate, or fall back to unchanged data.
    """
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)
    if grid.ndim != 2:
        return grid.copy()

    H, W = grid.shape
    if H == 0 or W == 0:
        return grid.copy()

    grid = grid.astype(int, copy=False)
    out = grid.copy()

    # Background is the most common color on the outer border.
    border = np.concatenate((grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]))
    vals, counts = np.unique(border, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # Find connected components of non-background pixels.
    visited = np.zeros((H, W), dtype=bool)
    interiors = []
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for r in range(H):
        for c in range(W):
            if visited[r, c] or grid[r, c] == bg:
                continue

            color = grid[r, c]
            q = deque([(r, c)])
            visited[r, c] = True
            coords = []

            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and grid[nr, nc] == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            # Bounding box of this component.
            rmin = min(x[0] for x in coords)
            rmax = max(x[0] for x in coords)
            cmin = min(x[1] for x in coords)
            cmax = max(x[1] for x in coords)

            bh = rmax - rmin + 1
            bw = cmax - cmin + 1

            # A one-cell-thick rectangular frame has exactly its bounding-box boundary.
            if bh >= 3 and bw >= 3:
                boundary = set()
                for cc in range(cmin, cmax + 1):
                    boundary.add((rmin, cc))
                    boundary.add((rmax, cc))
                for rr in range(rmin, rmax + 1):
                    boundary.add((rr, cmin))
                    boundary.add((rr, cmax))

                if set(coords) == boundary:
                    # Interior with exclusive end indices.
                    interiors.append((rmin + 1, rmax, cmin + 1, cmax))

    def _counts_for(data, h, w, num_colors):
        H2, W2 = data.shape
        cnt = np.zeros((h, w, num_colors), dtype=np.int32)
        for i in range(H2):
            ri = i % h
            row = data[i]
            for j in range(W2):
                cnt[ri, j % w, int(row[j])] += 1
        return cnt

    def _score_and_build(data, h, w, cnt, num_colors):
        H2, W2 = data.shape
        score = 0
        ties = 0

        for a in range(h):
            for b in range(w):
                c = cnt[a, b]
                total = int(c.sum())
                if total == 0:
                    continue
                best = int(c.max())
                score += total - best
                if int(np.count_nonzero(c == best)) > 1:
                    ties += 1

        score += 3 * ties

        tile = np.zeros((h, w), dtype=data.dtype)
        flat_vals, flat_counts = np.unique(data, return_counts=True)
        global_freq = dict(zip(flat_vals.tolist(), flat_counts.tolist()))

        for a in range(h):
            for b in range(w):
                c = cnt[a, b]
                best = int(c.max())
                tied = np.where(c == best)[0]

                if len(tied) == 1:
                    tile[a, b] = int(tied[0])
                else:
                    origin = int(data[a, b])
                    if origin in [int(x) for x in tied]:
                        tile[a, b] = origin
                    else:
                        tile[a, b] = min(
                            [int(x) for x in tied],
                            key=lambda x: (-global_freq.get(x, 0), x)
                        )

        restored = np.empty((H2, W2), dtype=data.dtype)
        for i in range(H2):
            ri = i % h
            for j in range(W2):
                restored[i, j] = tile[ri, j % w]

        return score, restored

    def _reconstruct(data):
        H2, W2 = data.shape
        n = H2 * W2
        if n == 0:
            return data.copy()

        num_colors = max(10, int(data.max()) + 1)

        # Smallest tile area first.
        candidates = []
        for h in range(1, H2 + 1):
            for w in range(1, W2 + 1):
                candidates.append((h * w, h, w))
        candidates.sort()

        threshold = max(2, int(math.ceil(0.12 * n)))
        best = None

        for area, h, w in candidates:
            cnt = _counts_for(data, h, w, num_colors)
            score, restored = _score_and_build(data, h, w, cnt, num_colors)

            if best is None or score < best[0] or (score == best[0] and area < best[1]):
                best = (score, area, restored)

            if score <= threshold:
                return restored

        if best is None:
            return data.copy()

        score, area, restored = best
        # If only the full size (or a very poor fit) remains, keep the original.
        if area == n or score > max(2, int(math.ceil(0.25 * n))):
            return data.copy()
        return restored

    for r1, r2, c1, c2 in interiors:
        if r2 <= r1 or c2 <= c1:
            continue
        region = out[r1:r2, c1:c2]
        out[r1:r2, c1:c2] = _reconstruct(region)

    return out