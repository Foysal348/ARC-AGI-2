import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid into the output grid.

    Observed rule:
    - There is one background color (the most frequent color).
    - There is one wall color (the non-background color that forms long
      horizontal/vertical line segments).
    - Non-wall cells are split into 4-connected components.
    - For each component, use its bounding box.  Each cell gets a "layer"
      equal to its Chebyshev distance from the bounding-box border:
          layer = min(r - r0, r1 - r, c - c0, c1 - c)
    - The input contains a small sequence of marker colors in each component.
      The color for layer 0, layer 1, layer 2, ... is read from the component
      by looking at non-background colors on those layers.  If the first
      non-background color appears on layer k, earlier layers are filled with
      background.  The resulting sequence is then cycled over all layers.
    - Wall cells are preserved.
    """
    if grid.size == 0:
        return grid.copy()

    g = np.array(grid, dtype=int)
    H, W = g.shape

    # ------------------------------------------------------------------
    # 1. Identify background color: most frequent color.
    # ------------------------------------------------------------------
    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # ------------------------------------------------------------------
    # 2. Identify wall color:
    #    a non-background color with the strongest straight-line structure.
    # ------------------------------------------------------------------
    def line_score(c):
        mask = (g == c)
        hsum = 0
        vmax = 0

        # horizontal runs
        for r in range(H):
            run = 0
            for c2 in range(W + 1):
                if c2 < W and mask[r, c2]:
                    run += 1
                else:
                    if run >= 3:
                        hsum += run
                    if run > vmax:
                        vmax = run
                    run = 0

        vsum = 0
        # vertical runs
        for c in range(W):
            run = 0
            for r2 in range(H + 1):
                if r2 < H and mask[r2, c]:
                    run += 1
                else:
                    if run >= 3:
                        vsum += run
                    if run > vmax:
                        vmax = run
                    run = 0

        return hsum + vsum, vmax, int(np.count_nonzero(mask))

    wall = -1
    best = (0, 0, 0)
    for c in vals:
        c = int(c)
        if c == bg:
            continue
        score = line_score(c)
        if score > best:
            best = score
            wall = c

    # Fallback: if no obvious line color is found, use the second most common.
    if wall == -1:
        order = np.argsort(-counts)
        for idx in order:
            c = int(vals[idx])
            if c != bg:
                wall = c
                break
        if wall == -1:
            wall = bg

    # ------------------------------------------------------------------
    # 3. Label 4-connected components of non-wall cells.
    # ------------------------------------------------------------------
    non_wall = (g != wall)
    labels = -np.ones((H, W), dtype=int)
    comp_cells = []
    comp_id = 0

    for r in range(H):
        for c in range(W):
            if non_wall[r, c] and labels[r, c] == -1:
                q = deque([(r, c)])
                labels[r, c] = comp_id
                cells = []
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if non_wall[nr, nc] and labels[nr, nc] == -1:
                                labels[nr, nc] = comp_id
                                q.append((nr, nc))
                comp_cells.append(cells)
                comp_id += 1

    # ------------------------------------------------------------------
    # 4. Build output: walls stay, other cells start as background.
    # ------------------------------------------------------------------
    out = np.full((H, W), bg, dtype=int)
    out[g == wall] = wall

    # ------------------------------------------------------------------
    # 5. Fill each component.
    # ------------------------------------------------------------------
    for cells in comp_cells:
        if not cells:
            continue

        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)

        max_layer = min(r1 - r0, c1 - c0) // 2

        # Determine the color sequence for layers 0, 1, 2, ...
        seq = []
        started = False

        # For efficiency, bucket non-background cells by layer.
        layer_counts = {}
        for r, c in cells:
            L = min(r - r0, r1 - r, c - c0, c1 - c)
            col = int(g[r, c])
            if col != bg:
                layer_counts.setdefault(L, {})
                layer_counts[L][col] = layer_counts[L].get(col, 0) + 1

        for L in range(max_layer + 1):
            counts_in_layer = layer_counts.get(L, {})
            if counts_in_layer:
                if not started:
                    # If the first marker is on layer L, pad earlier layers
                    # with background color.
                    seq.extend([bg] * L)
                    started = True

                # Choose the dominant non-background color on this layer.
                # Ties are broken by smaller color value.
                chosen = min(counts_in_layer.items(),
                             key=lambda kv: (-kv[1], kv[0]))[0]
                seq.append(chosen)
            else:
                if started:
                    # No more explicit marker layers; cycle the sequence.
                    break

        if not seq:
            # No markers: leave as background.
            continue

        # Paint all cells in this component using cycled layer colors.
        for r, c in cells:
            L = min(r - r0, r1 - r, c - c0, c1 - c)
            out[r, c] = seq[L % len(seq)]

    return out