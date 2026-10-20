import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    ARC transform:
    - The background is the most frequent color.
    - The wall/frame color is the second most frequent color.
    - Non-wall cells form 4-connected components.
    - For each component, find diagonal "seed" sequences that start at
      a corner of an empty rectangular pocket (two perpendicular walls/outside).
      The diagonal is read inward, including leading background pixels, and
      stops at the first background pixel after a non-background pixel.
    - Each non-wall cell's ring layer is its Chebyshev distance to the nearest
      wall or grid border, minus 1.
    - Each component is filled by cycling its seed sequence over these layers.
      If multiple seed sequences exist in one component, the cell uses the
      nearest seed-corner sequence.
    - Wall cells are unchanged.
    """
    grid = np.array(grid, dtype=int, copy=True)
    h, w = grid.shape

    # Trivial edge cases
    if h == 0 or w == 0:
        return grid

    # Color counts
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[np.argmax(counts)])

    # Wall color: most frequent non-background color.
    non_bg_counts = {
        color: count
        for color, count in zip(colors, counts)
        if int(color) != background
    }

    if not non_bg_counts:
        return grid

    wall = int(max(non_bg_counts, key=lambda c: non_bg_counts[c]))

    # Non-wall mask
    nonwall = grid != wall

    # ------------------------------------------------------------------
    # Chebyshev distance to nearest wall or outside border.
    # Use an expanded grid with a source border.
    # ------------------------------------------------------------------
    H, W = h + 2, w + 2
    INF = 10**9
    dist = np.full((H, W), INF, dtype=int)
    q = deque()

    # Outside border sources
    for i in range(H):
        if dist[i, 0] == INF:
            dist[i, 0] = 0
            q.append((i, 0))
        if dist[i, W - 1] == INF:
            dist[i, W - 1] = 0
            q.append((i, W - 1))

    for j in range(W):
        if dist[0, j] == INF:
            dist[0, j] = 0
            q.append((0, j))
        if dist[H - 1, j] == INF:
            dist[H - 1, j] = 0
            q.append((H - 1, j))

    # Wall cells are sources
    wall_coords = np.argwhere(grid == wall)
    for r, c in wall_coords:
        er, ec = r + 1, c + 1
        if dist[er, ec] == INF:
            dist[er, ec] = 0
            q.append((er, ec))

    # Multi-source BFS with 8-neighbors gives Chebyshev distance
    while q:
        r, c = q.popleft()
        nd = dist[r, c] + 1
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and dist[nr, nc] == INF:
                    dist[nr, nc] = nd
                    q.append((nr, nc))

    # Layer for each original non-wall cell
    layer = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            if nonwall[r, c]:
                d = dist[r + 1, c + 1]
                # d should be at least 1 for non-wall cells
                layer[r, c] = max(0, d - 1)

    # ------------------------------------------------------------------
    # 4-connected components of non-wall cells
    # ------------------------------------------------------------------
    comp_id = np.full((h, w), -1, dtype=int)
    comp_cells = []

    for r in range(h):
        for c in range(w):
            if nonwall[r, c] and comp_id[r, c] == -1:
                cid = len(comp_cells)
                cells = []
                q = deque([(r, c)])
                comp_id[r, c] = cid
                cells.append((r, c))

                while q:
                    cr, cc = q.popleft()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < h and 0 <= nc < w and
                            nonwall[nr, nc] and comp_id[nr, nc] == -1
                        ):
                            comp_id[nr, nc] = cid
                            cells.append((nr, nc))
                            q.append((nr, nc))

                comp_cells.append(cells)

    ncomp = len(comp_cells)

    # ------------------------------------------------------------------
    # Extract diagonal seed sequences from corner pockets.
    # ------------------------------------------------------------------
    def is_component_cell(r, c, cid):
        return (
            0 <= r < h and 0 <= c < w and
            nonwall[r, c] and comp_id[r, c] == cid
        )

    def read_diagonal(r, c, dr, dc, cid):
        """Read diagonal inward from a corner pocket start."""
        seq = []
        seen_non_bg = False
        rr, cc = r, c

        while is_component_cell(rr, cc, cid):
            val = int(grid[rr, cc])
            seq.append(val)

            if val != background:
                seen_non_bg = True
            else:
                if seen_non_bg:
                    break

            rr += dr
            cc += dc

        # Keep only sequences that contain at least one non-background color
        if any(v != background for v in seq):
            return seq
        return None

    sources_by_comp = [[] for _ in range(ncomp)]

    for r in range(h):
        for c in range(w):
            if not nonwall[r, c]:
                continue

            cid = comp_id[r, c]

            up_blocked = (r == 0) or (not nonwall[r - 1, c])
            down_blocked = (r == h - 1) or (not nonwall[r + 1, c])
            left_blocked = (c == 0) or (not nonwall[r, c - 1])
            right_blocked = (c == w - 1) or (not nonwall[r, c + 1])

            # Top-left corner: down-right diagonal
            if up_blocked and left_blocked:
                seq = read_diagonal(r, c, 1, 1, cid)
                if seq is not None:
                    sources_by_comp[cid].append((r, c, seq))

            # Top-right corner: down-left diagonal
            if up_blocked and right_blocked:
                seq = read_diagonal(r, c, 1, -1, cid)
                if seq is not None:
                    sources_by_comp[cid].append((r, c, seq))

            # Bottom-left corner: up-right diagonal
            if down_blocked and left_blocked:
                seq = read_diagonal(r, c, -1, 1, cid)
                if seq is not None:
                    sources_by_comp[cid].append((r, c, seq))

            # Bottom-right corner: up-left diagonal
            if down_blocked and right_blocked:
                seq = read_diagonal(r, c, -1, -1, cid)
                if seq is not None:
                    sources_by_comp[cid].append((r, c, seq))

    # ------------------------------------------------------------------
    # Build output
    # ------------------------------------------------------------------
    out = grid.copy()

    for cid, cells in enumerate(comp_cells):
        sources = sources_by_comp[cid]

        if not sources:
            # No visible seed diagonal: leave as background.
            for r, c in cells:
                out[r, c] = background
            continue

        if len(sources) == 1:
            _, _, seq = sources[0]
            L = len(seq)
            for r, c in cells:
                out[r, c] = seq[layer[r, c] % L]
        else:
            # If several seed diagonals exist in one component, use the
            # nearest seed corner for each cell.
            for r, c in cells:
                best_seq = None
                best_key = None

                for sr, sc, seq in sources:
                    d = max(abs(r - sr), abs(c - sc))
                    # Prefer closer source; tie-break by longer sequence.
                    key = (d, -len(seq))
                    if best_key is None or key < best_key:
                        best_key = key
                        best_seq = seq

                L = len(best_seq)
                out[r, c] = best_seq[layer[r, c] % L]

    # Preserve wall cells exactly as input
    out[grid == wall] = grid[grid == wall]

    return out