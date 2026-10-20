import numpy as np
from collections import deque, Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by:
      1. Detecting the background color and the wall/barrier color.
      2. Splitting non-wall cells into connected regions.
      3. Computing concentric layers inside each region by BFS from region boundary cells.
      4. In each region, using the non-background marker colors to build a cyclic layer-color sequence.
      5. Filling each region with that cyclic sequence.
    """
    grid = np.asarray(grid, dtype=int)

    if grid.ndim != 2 or grid.size == 0:
        return grid.copy()

    h, w = grid.shape

    # Helper: largest 4-connected component size for a binary mask.
    def largest_component_size(mask: np.ndarray) -> int:
        visited = np.zeros(mask.shape, dtype=bool)
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

    vals, counts = np.unique(grid, return_counts=True)

    # If the grid is one solid color, nothing can be transformed.
    if vals.size == 1:
        return grid.copy()

    # Background: most frequent color. If tied, prefer border presence.
    max_count = int(counts.max())
    bg_candidates = [int(vals[i]) for i in range(len(vals)) if int(counts[i]) == max_count]

    def border_count(color: int) -> int:
        if h == 1 and w == 1:
            return int(grid[0, 0] == color)
        return int(
            np.sum(
                (grid[0, :] == color)
                | (grid[-1, :] == color)
                | (grid[:, 0] == color)
                | (grid[:, -1] == color)
            )
        )

    if len(bg_candidates) == 1:
        background = bg_candidates[0]
    else:
        background = max(bg_candidates, key=lambda c: (border_count(c), -c))

    # Wall color: most frequent non-background color.
    non_bg = [
        (int(vals[i]), int(counts[i]))
        for i in range(len(vals))
        if int(vals[i]) != background
    ]

    if not non_bg:
        return grid.copy()

    max_non_bg_count = max(cnt for _, cnt in non_bg)
    wall_candidates = [color for color, cnt in non_bg if cnt == max_non_bg_count]

    if len(wall_candidates) == 1:
        wall_color = wall_candidates[0]
    else:
        wall_color = max(
            wall_candidates,
            key=lambda c: (largest_component_size(grid == c), -c),
        )

    non_wall = grid != wall_color

    if not np.any(non_wall):
        return grid.copy()

    # Label connected components of non-wall cells.
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

    # Multi-source BFS to compute layer/ring distance from region boundary cells.
    wall_mask = grid == wall_color
    dist = -np.ones((h, w), dtype=int)
    q = deque()

    for r in range(h):
        for c in range(w):
            if not non_wall[r, c]:
                continue

            is_boundary = (r == 0 or r == h - 1 or c == 0 or c == w - 1)

            if not is_boundary:
                if (
                    (r > 0 and wall_mask[r - 1, c])
                    or (r + 1 < h and wall_mask[r + 1, c])
                    or (c > 0 and wall_mask[r, c - 1])
                    or (c + 1 < w and wall_mask[r, c + 1])
                ):
                    is_boundary = True

            if is_boundary:
                dist[r, c] = 0
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = r + dx, c + dy
            if (
                0 <= nx < h
                and 0 <= ny < w
                and non_wall[nx, ny]
                and dist[nx, ny] == -1
            ):
                dist[nx, ny] = dist[r, c] + 1
                q.append((nx, ny))

    # Any unreachable non-wall cell (should not happen) is treated as layer 0.
    dist = np.where(dist == -1, 0, dist)

    def build_layer_sequence(cells):
        """
        Build the cyclic color sequence for one component.

        Returns:
            (sequence, ambiguous)
            sequence is None if there are no markers in the component.
            ambiguous is True if a layer contained multiple marker colors.
        """
        layer_markers = {}

        for r, c in cells:
            value = int(grid[r, c])
            if value != background:
                d = int(dist[r, c])
                layer_markers.setdefault(d, Counter())[value] += 1

        if not layer_markers:
            return None, False

        chosen = {}
        ambiguous = False

        for d, counter in layer_markers.items():
            if len(counter) == 1:
                chosen[d] = next(iter(counter))
            else:
                max_count = max(counter.values())
                candidates = [
                    int(color)
                    for color, cnt in counter.items()
                    if cnt == max_count
                ]
                chosen[d] = min(candidates)
                ambiguous = True

        seed_layers = sorted(chosen)
        first_seed_layer = seed_layers[0]

        # Layers before the first marker are background.
        seq = [background] * first_seed_layer

        started = False
        for d in range(first_seed_layer, max(seed_layers) + 1):
            if d in chosen:
                seq.append(int(chosen[d]))
                started = True
            else:
                # Stop at the first empty layer after markers have started.
                if started:
                    break

        return seq, ambiguous

    # Build output.
    out = np.full(grid.shape, background, dtype=int)
    out[grid == wall_color] = wall_color

    for comp_id, cells in enumerate(components):
        seq, _ambiguous = build_layer_sequence(cells)

        if seq is None or len(seq) == 0:
            # No markers: leave as background.
            continue

        seq_len = len(seq)
        for r, c in cells:
            d = int(dist[r, c])
            if d < 0:
                d = 0
            out[r, c] = seq[d % seq_len]

    return out