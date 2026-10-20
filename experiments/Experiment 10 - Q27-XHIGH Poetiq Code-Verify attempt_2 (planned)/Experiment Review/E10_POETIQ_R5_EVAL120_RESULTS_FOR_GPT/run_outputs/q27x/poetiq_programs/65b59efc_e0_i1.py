import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid according to the following rule:

    The input is organized into columns of M x M blocks:
      - Top band: source patterns.
      - Middle band: clue masks. Each clue mask contains one color, which
        identifies one of the source patterns by color.
      - Bottom band: marker colors. Each bottom column provides the output
        color for the source pattern in that column.

    For each clue mask:
      - Find the source pattern with the same color.
      - Use the bottom marker color from that source pattern's column.
      - Place that recolored source pattern in every output cell indicated
        by a non-zero pixel of the clue mask.

    The output is an (M*M) x (M*M) grid.
    """
    def _as_2d_int(g):
        """Convert input to a 2D int NumPy array, padding ragged rows if needed."""
        if isinstance(g, np.ndarray):
            return g.astype(int, copy=False)

        try:
            arr = np.asarray(g, dtype=int)
            if arr.ndim == 2:
                return arr
        except Exception:
            pass

        # Handle ragged list input by padding with zeros.
        if isinstance(g, (list, tuple)) and len(g) > 0:
            rows = []
            max_w = 0
            for row in g:
                row = list(row)
                max_w = max(max_w, len(row))

            for row in g:
                row = list(row)
                if len(row) < max_w:
                    row.extend([0] * (max_w - len(row)))
                rows.append(row)

            return np.array(rows, dtype=int)

        return np.asarray(g, dtype=int)

    grid = _as_2d_int(grid)

    if grid.ndim != 2 or grid.size == 0:
        return np.zeros_like(grid, dtype=int)

    H, W = grid.shape
    SEP = 5

    # ---------------------------------------------------------------
    # Determine M:
    # Structure is:
    #   rows 0..M-1       top patterns
    #   row M             horizontal separator
    #   rows M+1..2M      middle clue masks
    #   row 2M+1          horizontal separator
    #   rows 2M+2..       bottom markers
    # ---------------------------------------------------------------
    sep_counts = (grid == SEP).sum(axis=1)

    best_M = None
    best_score = -1
    max_M = (H - 1) // 2

    for M in range(1, max_M + 1):
        r1 = M
        r2 = 2 * M + 1
        if r1 < H and r2 < H:
            score = int(sep_counts[r1] + sep_counts[r2])
            if score > best_score:
                best_score = score
                best_M = M

    # Fallback if the separator rows were not obvious.
    if best_M is None or best_score <= 0:
        positive_rows = [r for r in range(1, H) if sep_counts[r] > 0]

        # Prefer exact rows M and 2M+1.
        for M in range(1, max_M + 1):
            r1 = M
            r2 = 2 * M + 1
            if r1 < H and r2 < H and sep_counts[r1] > 0 and sep_counts[r2] > 0:
                best_M = M
                break

        if best_M is None and positive_rows:
            best_M = max(1, positive_rows[0])

        if best_M is None:
            best_M = max(1, min(H, W) // 3)

    M = int(best_M)

    if M <= 0 or M >= H or M > W:
        return np.zeros_like(grid, dtype=int)

    sep1 = M
    sep2 = 2 * M + 1

    if sep2 >= H:
        return np.zeros_like(grid, dtype=int)

    # Column starts for M x M blocks. Blocks are separated by one column.
    starts = list(range(0, W - M + 1, M + 1))
    if not starts:
        starts = [0]

    def _dominant_color(block: np.ndarray):
        """Return the most frequent non-zero, non-separator color in a block."""
        mask = (block != 0) & (block != SEP)
        if not np.any(mask):
            return None
        vals, counts = np.unique(block[mask], return_counts=True)
        return int(vals[np.argmax(counts)])

    # ---------------------------------------------------------------
    # Extract top source patterns.
    # ---------------------------------------------------------------
    top_patterns = []
    color_to_source = {}

    for c0 in starts:
        if c0 + M > W:
            top_patterns.append(None)
            continue

        block = grid[0:M, c0:c0 + M]
        color = _dominant_color(block)

        if color is None:
            top_patterns.append(None)
        else:
            # Shape of the source pattern. Ignore 0; include all other pixels.
            pattern = (block != 0)
            if np.any(pattern):
                idx = len(top_patterns)
                top_patterns.append(pattern)
                color_to_source.setdefault(color, idx)
            else:
                top_patterns.append(None)

    # ---------------------------------------------------------------
    # Extract bottom marker colors.
    # ---------------------------------------------------------------
    bottom_markers = []

    for c0 in starts:
        if c0 + M > W:
            bottom_markers.append(None)
            continue

        block = grid[sep2 + 1:H, c0:c0 + M]
        if block.size == 0:
            bottom_markers.append(None)
        else:
            bottom_markers.append(_dominant_color(block))

    # ---------------------------------------------------------------
    # Build output.
    # ---------------------------------------------------------------
    out = np.zeros((M * M, M * M), dtype=grid.dtype)
    occupied = np.zeros((M, M), dtype=bool)

    mid_start = sep1 + 1
    mid_height = min(M, sep2 - mid_start)

    if mid_height <= 0:
        return out

    for c0 in starts:
        if c0 + M > W:
            continue

        block = np.zeros((M, M), dtype=grid.dtype)
        h = min(mid_height, M)
        w = min(M, W - c0)
        block[:h, :w] = grid[mid_start:mid_start + h, c0:c0 + w]

        foreground = (block != 0) & (block != SEP)
        if not np.any(foreground):
            continue

        vals, counts = np.unique(block[foreground], return_counts=True)
        order = np.argsort(-counts)

        for pos in order:
            color = int(vals[pos])
            source_idx = color_to_source.get(color)

            if source_idx is None:
                continue

            if source_idx >= len(top_patterns) or source_idx >= len(bottom_markers):
                continue

            pattern = top_patterns[source_idx]
            marker = bottom_markers[source_idx]

            if pattern is None or marker is None:
                continue

            mask = (block == color)
            if not np.any(mask):
                continue

            for r in range(M):
                for c in range(M):
                    if mask[r, c] and not occupied[r, c]:
                        occupied[r, c] = True
                        out[r * M:(r + 1) * M, c * M:(c + 1) * M] = (
                            np.where(pattern, marker, 0)
                        )

    return out