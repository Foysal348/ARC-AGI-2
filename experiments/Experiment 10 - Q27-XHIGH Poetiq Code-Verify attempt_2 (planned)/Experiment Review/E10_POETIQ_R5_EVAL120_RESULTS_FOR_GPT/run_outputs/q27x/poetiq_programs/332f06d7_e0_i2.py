import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Move the 0-colored square toward the 2-colored square as far as the
    square can physically "slide" through the 1/2-colored material.

    The rule used:
    1. The 0 square is the moving square, and its size is N.
    2. Treat colors 0, 1, and 2 as passable material for the path.
    3. Find a shortest cell path from the 0 square to the 2 square.
    4. A square position is valid if its whole N x N footprint lies in passable cells.
    5. From the start position, slide the square by 1-cell steps through valid positions.
    6. Stop at the farthest reachable square position along the shortest path.
       - If the 2 square is reachable, the square ends there.
       - If there is no cell path at all, place the 0 square on the 2 square.
    7. Fill the original 0 square with 1, then put the moved 0 square.
    """
    grid = np.array(grid, copy=True)
    if grid.size == 0:
        return grid

    h, w = grid.shape

    zero_cells = np.argwhere(grid == 0)
    two_cells = np.argwhere(grid == 2)

    if zero_cells.size == 0 or two_cells.size == 0:
        return grid

    zr = zero_cells[:, 0]
    zc = zero_cells[:, 1]
    r0, c0 = int(zr.min()), int(zc.min())
    r1, c1 = int(zr.max()), int(zc.max())

    tr = two_cells[:, 0]
    tc = two_cells[:, 1]
    r2, c2 = int(tr.min()), int(tc.min())
    r3, c3 = int(tr.max()), int(tc.max())

    # The moving square size. In all provided examples the 0 and 2 squares
    # have the same size; use the 0-square size as the authoritative size.
    N = max(r1 - r0 + 1, c1 - c0 + 1)
    if N <= 0 or N > h or N > w:
        return grid

    # Passable cells for the cell-level shortest path and for square placement.
    passable = (grid == 0) | (grid == 1) | (grid == 2)
    passable = passable.copy()

    # Ensure the original squares themselves are considered passable.
    passable[r0:r1 + 1, c0:c1 + 1] = True
    passable[r2:r3 + 1, c2:c3 + 1] = True

    INF = 10**9
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def bfs_distances(start_mask: np.ndarray) -> np.ndarray:
        dist = np.full((h, w), INF, dtype=int)
        q = deque()

        ys, xs = np.where(start_mask & passable)
        for r, c in zip(ys, xs):
            r, c = int(r), int(c)
            dist[r, c] = 0
            q.append((r, c))

        while q:
            r, c = q.popleft()
            nd = dist[r, c] + 1
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if passable[nr, nc] and dist[nr, nc] == INF:
                        dist[nr, nc] = nd
                        q.append((nr, nc))

        return dist

    start_mask = np.zeros((h, w), dtype=bool)
    start_mask[r0:r1 + 1, c0:c1 + 1] = True

    goal_mask = np.zeros((h, w), dtype=bool)
    goal_mask[r2:r3 + 1, c2:c3 + 1] = True

    dist_start = bfs_distances(start_mask)
    dist_goal = bfs_distances(goal_mask)

    goal_dists = dist_start[goal_mask]
    shortest = int(goal_dists.min()) if goal_dists.size > 0 else INF

    # Fallback: use the 2-square position.
    target = (r2, c2)

    if shortest < INF:
        # Cells that lie on at least one shortest cell path.
        on_shortest = (dist_start + dist_goal == shortest)

        H = h - N + 1
        W = w - N + 1

        if H > 0 and W > 0:
            # Prefix sum for valid N x N square positions.
            pref = np.zeros((h + 1, w + 1), dtype=int)
            pref[1:, 1:] = passable.astype(int).cumsum(axis=0).cumsum(axis=1)

            block_sum = (
                pref[N:, N:]
                - pref[:-N, N:]
                - pref[N:, :-N]
                + pref[:-N, :-N]
            )
            valid_anchor = (block_sum == N * N)

            # The start anchor must be usable.
            if 0 <= r0 < H and 0 <= c0 < W:
                valid_anchor[r0, c0] = True

            # The goal anchor may be usable if it really fits in passable cells.
            if 0 <= r2 < H and 0 <= c2 < W:
                if np.all(passable[r2:r2 + N, c2:c2 + N]):
                    valid_anchor[r2, c2] = True

            # BFS in anchor space: sliding the N x N square by 1 cell.
            anchor_dist = np.full((H, W), INF, dtype=int)
            q = deque()

            if valid_anchor[r0, c0]:
                anchor_dist[r0, c0] = 0
                q.append((r0, c0))

            while q:
                r, c = q.popleft()
                nd = anchor_dist[r, c] + 1
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if valid_anchor[nr, nc] and anchor_dist[nr, nc] == INF:
                            anchor_dist[nr, nc] = nd
                            q.append((nr, nc))

            best = None
            best_score = None

            for r in range(H):
                for c in range(W):
                    if not valid_anchor[r, c] or anchor_dist[r, c] == INF:
                        continue

                    ds = dist_start[r:r + N, c:c + N]
                    dg = dist_goal[r:r + N, c:c + N]
                    mask = on_shortest[r:r + N, c:c + N]

                    if not np.any(mask):
                        continue

                    # Progress along the shortest cell path:
                    # the furthest path cell covered by this square position.
                    progress = int(ds[mask].max())

                    # Tie-breakers:
                    # - closer to the 2-square anchor
                    # - farther in anchor-space from the start
                    # - deterministic top-left preference
                    goal_anchor_dist = (r - r2) ** 2 + (c - c2) ** 2
                    score = (
                        -progress,
                        goal_anchor_dist,
                        -anchor_dist[r, c],
                        r,
                        c,
                    )

                    if best_score is None or score < best_score:
                        best_score = score
                        best = (r, c)

            if best is not None:
                # If the square cannot move but a goal exists, move to the goal
                # instead of producing no change.
                if best == (r0, c0) and (r2, c2) != (r0, c0):
                    if 0 <= r2 < H and 0 <= c2 < W and valid_anchor[r2, c2]:
                        best = (r2, c2)
                target = best

    # Apply the transformation.
    out = grid.copy()
    out[r0:r1 + 1, c0:c1 + 1] = 1

    tr, tc = target
    if 0 <= tr < h and 0 <= tc < w and tr + N <= h and tc + N <= w:
        out[tr:tr + N, tc:tc + N] = 0
    else:
        # Defensive fallback: overwrite the original 2-square area.
        out[r2:r3 + 1, c2:c3 + 1] = 0

    return out