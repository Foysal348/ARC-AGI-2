import numpy as np
from collections import deque, Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill each non-wall region with repeating concentric layer colors.

    - Background: most frequent color.
    - Wall: most frequent non-background color.
    - Non-wall cells are grouped into 4-connected regions.
    - Layer of a cell is its Chebyshev distance (8-neighbor distance)
      to the nearest wall cell or grid outside, minus 1.
    - In each region, marker colors define layer colors. Missing layers
      between the border and the farthest marker are background.
      The observed sequence from layer 0 to the farthest marker repeats.
    """
    grid = np.asarray(grid, dtype=int)

    if grid.ndim != 2 or grid.size == 0:
        return grid.copy()

    h, w = grid.shape

    vals, counts = np.unique(grid, return_counts=True)
    if vals.size == 1:
        return grid.copy()

    counts_by_color = {int(vals[i]): int(counts[i]) for i in range(vals.size)}

    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    def border_count(color: int) -> int:
        return int(np.sum(border_mask & (grid == color)))

    def largest_component_size(mask: np.ndarray) -> int:
        """Largest 4-connected component size in mask."""
        visited = np.zeros((h, w), dtype=bool)
        best = 0

        for r in range(h):
            for c in range(w):
                if mask[r, c] and not visited[r, c]:
                    q = deque([(r, c)])
                    visited[r, c] = True
                    size = 0

                    while q:
                        x, y = q.popleft()
                        size += 1

                        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nx, ny = x + dx, y + dy
                            if (
                                0 <= nx < h
                                and 0 <= ny < w
                                and mask[nx, ny]
                                and not visited[nx, ny]
                            ):
                                visited[nx, ny] = True
                                q.append((nx, ny))

                    if size > best:
                        best = size

        return best

    # Choose background.
    max_count = max(counts_by_color.values())
    bg_candidates = [c for c, cnt in counts_by_color.items() if cnt == max_count]

    if len(bg_candidates) == 1:
        background = bg_candidates[0]
    else:
        background = max(
            bg_candidates,
            key=lambda c: (
                border_count(c),
                largest_component_size(grid == c),
                -c,
            ),
        )

    # Choose wall color.
    non_bg = [(c, counts_by_color[c]) for c in counts_by_color if c != background]
    if not non_bg:
        return grid.copy()

    max_non_bg_count = max(cnt for _, cnt in non_bg)
    wall_candidates = [c for c, cnt in non_bg if cnt == max_non_bg_count]

    if len(wall_candidates) == 1:
        wall = wall_candidates[0]
    else:
        wall = max(
            wall_candidates,
            key=lambda c: (largest_component_size(grid == c), -c),
        )

    wall_mask = grid == wall

    # Start with background and keep wall cells unchanged.
    out = np.full((h, w), background, dtype=int)
    out[wall_mask] = wall

    if not np.any(~wall_mask):
        return out

    # ------------------------------------------------------------------
    # Chebyshev distance to nearest wall cell or outside of the grid.
    # Use an expanded grid where the outside is a ring of zero-distance cells.
    # ------------------------------------------------------------------
    H, W = h + 2, w + 2
    INF = 10**9
    dist = np.full((H, W), INF, dtype=int)
    q = deque()

    # Outside ring sources.
    for r in range(H):
        for c in range(W):
            if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                if dist[r, c] != 0:
                    dist[r, c] = 0
                    q.append((r, c))

    # Wall sources.
    for r, c in np.argwhere(wall_mask):
        rr, cc = int(r) + 1, int(c) + 1
        if dist[rr, cc] != 0:
            dist[rr, cc] = 0
            q.append((rr, cc))

    # 8-neighbor BFS gives Chebyshev distance.
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    while q:
        r, c = q.popleft()
        nd = dist[r, c] + 1

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and dist[nr, nc] > nd:
                dist[nr, nc] = nd
                q.append((nr, nc))

    # ------------------------------------------------------------------
    # 4-connected non-wall regions.
    # ------------------------------------------------------------------
    non_wall = ~wall_mask
    labels = -np.ones((h, w), dtype=int)
    components = []

    for r in range(h):
        for c in range(w):
            if non_wall[r, c] and labels[r, c] == -1:
                comp_id = len(components)
                q = deque([(r, c)])
                labels[r, c] = comp_id
                cells = []

                while q:
                    x, y = q.popleft()
                    cells.append((x, y))

                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if (
                            0 <= nx < h
                            and 0 <= ny < w
                            and non_wall[nx, ny]
                            and labels[nx, ny] == -1
                        ):
                            labels[nx, ny] = comp_id
                            q.append((nx, ny))

                components.append(cells)

    # ------------------------------------------------------------------
    # For each region, build and apply its repeating layer-color sequence.
    # ------------------------------------------------------------------
    for cells in components:
        layer_markers = {}

        for r, c in cells:
            val = int(grid[r, c])

            # Markers are non-background, non-wall colors.
            if val != background and val != wall:
                layer = int(dist[r + 1, c + 1]) - 1
                if layer < 0:
                    layer = 0

                if layer not in layer_markers:
                    layer_markers[layer] = Counter()
                layer_markers[layer][val] += 1

        # No markers: leave this region as background.
        if not layer_markers:
            continue

        max_layer = max(layer_markers.keys())

        # Observed pattern from layer 0 through the farthest marker layer.
        # Empty layers in this range are background.
        seq = []
        for layer in range(max_layer + 1):
            if layer in layer_markers:
                counter = layer_markers[layer]
                best_count = max(counter.values())
                chosen = min(
                    color
                    for color, cnt in counter.items()
                    if cnt == best_count
                )
            else:
                chosen = background

            seq.append(chosen)

        seq_len = len(seq)

        for r, c in cells:
            layer = int(dist[r + 1, c + 1]) - 1
            if layer < 0:
                layer = 0

            out[r, c] = seq[layer % seq_len]

    return out