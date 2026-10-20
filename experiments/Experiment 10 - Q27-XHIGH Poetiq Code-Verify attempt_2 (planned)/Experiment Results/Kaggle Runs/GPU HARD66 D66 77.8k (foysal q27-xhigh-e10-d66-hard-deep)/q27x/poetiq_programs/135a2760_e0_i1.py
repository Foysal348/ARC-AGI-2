import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    For each enclosed rectangular region, infer the smallest 2D periodic pattern
    that best explains its pixels, with a small amount of noise allowed, and fill
    the region with that restored periodic pattern.
    """
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)
    if grid.ndim != 2:
        return grid.copy()

    H, W = grid.shape
    if H == 0 or W == 0:
        return grid.copy()

    out = grid.copy()
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    # Use the most common border color as the global background.
    border = np.concatenate((grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]))
    border_vals, border_counts = np.unique(border, return_counts=True)
    bg_color = int(border_vals[np.argmax(border_counts)])

    visited = np.zeros((H, W), dtype=bool)
    regions = []
    region_keys = set()

    def add_rectangular_holes(holes: np.ndarray) -> None:
        """Extract connected components of holes that are full rectangles."""
        unvisited = holes.copy()
        while True:
            nz = np.nonzero(unvisited)
            if len(nz[0]) == 0:
                break

            sr, sc = int(nz[0][0]), int(nz[1][0])
            hole = np.zeros((H, W), dtype=bool)

            q = deque([(sr, sc)])
            unvisited[sr, sc] = False
            hole[sr, sc] = True

            while q:
                r, c = q.popleft()
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and holes[nr, nc] and not unvisited[nr, nc]:
                        unvisited[nr, nc] = False
                        hole[nr, nc] = True
                        q.append((nr, nc))

            rows = np.any(hole, axis=1)
            cols = np.any(hole, axis=0)
            if not rows.any() or not cols.any():
                continue

            r1 = int(np.argmax(rows))
            r_last = H - 1 - int(np.argmax(rows[::-1]))
            c1 = int(np.argmax(cols))
            c_last = W - 1 - int(np.argmax(cols[::-1]))

            r2 = r_last + 1
            c2 = c_last + 1

            submask = hole[r1:r2, c1:c2]
            if not submask.all():
                continue

            key = (r1, r2, c1, c2)
            if key not in region_keys:
                region_keys.add(key)
                regions.append((r1, r2, c1, c2, submask))

    # Find components that enclose rectangular holes.
    for r in range(H):
        for c in range(W):
            if visited[r, c] or grid[r, c] == bg_color:
                continue

            color = grid[r, c]
            comp = np.zeros((H, W), dtype=bool)

            q = deque([(r, c)])
            comp[r, c] = True
            visited[r, c] = True

            while q:
                cr, cc = q.popleft()
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not comp[nr, nc] and grid[nr, nc] == color:
                        comp[nr, nc] = True
                        visited[nr, nc] = True
                        q.append((nr, nc))

            # Small components cannot be meaningful rectangular frames.
            if comp.sum() < 4:
                continue

            # Flood fill from the outside, treating this component as blocked.
            reach = np.zeros((H, W), dtype=bool)
            q = deque()

            for i in range(H):
                for j in (0, W - 1):
                    if not comp[i, j] and not reach[i, j]:
                        reach[i, j] = True
                        q.append((i, j))
            for j in range(W):
                for i in (0, H - 1):
                    if not comp[i, j] and not reach[i, j]:
                        reach[i, j] = True
                        q.append((i, j))

            while q:
                cr, cc = q.popleft()
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not comp[nr, nc] and not reach[nr, nc]:
                        reach[nr, nc] = True
                        q.append((nr, nc))

            holes = (~comp) & (~reach)
            if holes.any():
                add_rectangular_holes(holes)

    def reconstruct_periodic(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Reconstruct a masked subgrid as the smallest periodic tiling with
        acceptable noise.
        """
        dh, dw = data.shape
        coords = [(i, j) for i in range(dh) for j in range(dw) if mask[i, j]]
        n = len(coords)
        if n <= 1:
            return data.copy()

        # Allowed Hamming distance from the periodic reconstruction.
        threshold = max(2, int(np.ceil(0.10 * n)))

        candidates = []
        for h in range(1, dh + 1):
            for w in range(1, dw + 1):
                # Avoid the trivial “copy the whole box” period.
                if h == dh and w == dw:
                    continue
                # A tile area larger than the number of observed pixels is useless.
                if h * w > n:
                    continue
                candidates.append((h * w, h, w))
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))

        unique_vals, unique_counts = np.unique(data[mask], return_counts=True)
        global_freq = dict(zip(unique_vals.tolist(), unique_counts.tolist()))
        region_has_multiple_colors = len(unique_vals) > 1

        for _, h, w in candidates:
            phase_counts = {}

            for i, j in coords:
                phase = (i % h, j % w)
                val = int(data[i, j])
                if phase not in phase_counts:
                    phase_counts[phase] = {val: 1}
                else:
                    phase_counts[phase][val] = phase_counts[phase].get(val, 0) + 1

            error = 0
            for counts in phase_counts.values():
                total = sum(counts.values())
                best = max(counts.values())
                error += total - best
                if error > threshold:
                    break

            if error > threshold:
                continue

            tile = {}
            for phase, counts in phase_counts.items():
                best = max(counts.values())
                tied = [val for val, cnt in counts.items() if cnt == best]

                if len(tied) == 1:
                    chosen = tied[0]
                else:
                    r, c = phase
                    # Prefer the original prototype if it is one of the tied values.
                    if 0 <= r < dh and 0 <= c < dw and mask[r, c] and int(data[r, c]) in tied:
                        chosen = int(data[r, c])
                    else:
                        chosen = min(tied, key=lambda x: (-global_freq.get(x, 0), x))

                tile[phase] = chosen

            # Avoid erasing a visibly non-uniform region into one color.
            if region_has_multiple_colors and len(set(tile.values())) == 1:
                continue

            new_data = data.copy()
            for i, j in coords:
                new_data[i, j] = tile[(i % h, j % w)]
            return new_data

        return data.copy()

    # Apply reconstruction to each detected rectangular interior.
    for r1, r2, c1, c2, mask in regions:
        box = out[r1:r2, c1:c2]
        new_box = reconstruct_periodic(box, mask)
        box_view = out[r1:r2, c1:c2]
        box_view[mask] = new_box[mask]

    return out