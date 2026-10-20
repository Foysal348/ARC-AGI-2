import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill keyed enclosed background components with the marker/fill color.

    Inferred rule:
    1. The background is the most common color.
    2. The fill color is a rare/singleton non-background color, preferring
       a pixel on the border and, among those, bottom/right positions.
    3. The key color is a non-background color touching the fill color
       orthogonally (fallback: diagonally, then by rare color).
    4. Consider 4-connected background components that are not the main
       outside background component.
    5. Fill a background component if it is "keyed": a key-color pixel is
       orthogonally adjacent to a non-background boundary pixel of that
       component, or diagonally adjacent when no orthogonal keyed component
       is found.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2:
        return grid.copy()

    H, W = grid.shape
    if H == 0 or W == 0:
        return grid.copy()

    # ---------------------------------------------------------------
    # Background color
    # ---------------------------------------------------------------
    vals, counts = np.unique(grid, return_counts=True)
    color_count = {int(c): int(n) for c, n in zip(vals, counts)}
    max_count = max(color_count.values())

    if len([c for c, n in color_count.items() if n == max_count]) > 1:
        bg = 0 if 0 in color_count else min(color_count)
    else:
        bg = max(color_count, key=lambda c: (color_count[c], -c))

    nonbg_colors = [c for c in color_count if c != bg]
    if not nonbg_colors:
        return grid.copy()

    # ---------------------------------------------------------------
    # Fill color: singleton/rare non-background color, border preferred.
    # ---------------------------------------------------------------
    def positions_of(color):
        return np.argwhere(grid == color)

    singleton_positions = {}
    for c in nonbg_colors:
        if color_count[c] == 1:
            idx = positions_of(c)
            if idx.size:
                singleton_positions[c] = (int(idx[0, 0]), int(idx[0, 1]))

    if singleton_positions:
        def singleton_score(c):
            r, col = singleton_positions[c]
            return (
                r == 0 or r == H - 1 or col == 0 or col == W - 1,
                r == H - 1,
                col == W - 1,
                r,
                col,
                c
            )
        fill = max(singleton_positions, key=singleton_score)
    else:
        # Fallback: least frequent non-background color.
        fill = min(nonbg_colors, key=lambda c: (color_count[c], c))

    # ---------------------------------------------------------------
    # Key color: color adjacent to the fill marker.
    # ---------------------------------------------------------------
    key = None
    fill_pos = positions_of(fill)
    dirs4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    dirs8 = dirs4 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    neigh4 = {}
    for r, c in fill_pos:
        for dr, dc in dirs4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                v = int(grid[nr, nc])
                if v != bg and v != fill:
                    neigh4[v] = neigh4.get(v, 0) + 1

    if neigh4:
        key = max(neigh4, key=lambda c: (neigh4[c], c))
    else:
        neigh8 = {}
        for r, c in fill_pos:
            for dr, dc in dirs8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    v = int(grid[nr, nc])
                    if v != bg and v != fill:
                        neigh8[v] = neigh8.get(v, 0) + 1
        if neigh8:
            key = max(neigh8, key=lambda c: (neigh8[c], c))
        else:
            candidates = [c for c in nonbg_colors if c != fill]
            if not candidates:
                return grid.copy()
            key = max(candidates, key=lambda c: (color_count[c], c))

    # ---------------------------------------------------------------
    # Label 4-connected background components.
    # ---------------------------------------------------------------
    bg_mask = (grid == bg)
    labels = np.zeros((H, W), dtype=int)
    comp_info = []  # (area, cells, touches_border)
    nxt = 0

    for i in range(H):
        for j in range(W):
            if not bg_mask[i, j] or labels[i, j] != 0:
                continue

            nxt += 1
            stack = [(i, j)]
            labels[i, j] = nxt
            cells = []
            touches_border = False

            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                    touches_border = True

                for dr, dc in dirs4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if bg_mask[nr, nc] and labels[nr, nc] == 0:
                            labels[nr, nc] = nxt
                            stack.append((nr, nc))

            comp_info.append((len(cells), cells, touches_border))

    if not comp_info:
        return grid.copy()

    # Main outside component: largest component; if tied, prefer border.
    outside_label = 1 + max(
        range(len(comp_info)),
        key=lambda idx: (comp_info[idx][0], comp_info[idx][2])
    )

    # Key pixels for quick lookup.
    key_mask = (grid == key)
    key_cells = np.argwhere(key_mask)

    def component_keyed_orthogonal(cells):
        """True if key is orthogonally adjacent to a non-bg neighbor of cells."""
        for r, c in cells:
            for dr, dc in dirs4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not bg_mask[nr, nc]:
                    # Is key orthogonally adjacent to this boundary pixel?
                    for dr2, dc2 in dirs4:
                        kr, kc = nr + dr2, nc + dc2
                        if 0 <= kr < H and 0 <= kc < W and key_mask[kr, kc]:
                            return True
        return False

    def component_keyed_diagonal(cells):
        """
        True if key is diagonally adjacent to a non-bg neighbor of cells,
        or directly diagonal/orthogonal adjacent to a component cell.
        """
        for r, c in cells:
            # Direct adjacency to component is a strong fallback.
            for dr, dc in dirs4 + dirs8:
                kr, kc = r + dr, c + dc
                if 0 <= kr < H and 0 <= kc < W and key_mask[kr, kc]:
                    return True

            for dr, dc in dirs4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not bg_mask[nr, nc]:
                    for dr2, dc2 in dirs8:
                        kr, kc = nr + dr2, nc + dc2
                        if 0 <= kr < H and 0 <= kc < W and key_mask[kr, kc]:
                            return True
        return False

    # ---------------------------------------------------------------
    # Select components to fill.
    # ---------------------------------------------------------------
    fill_mask = np.zeros((H, W), dtype=bool)

    # First pass: orthogonal key-contact with boundary.
    for idx, (area, cells, _) in enumerate(comp_info):
        if idx + 1 == outside_label:
            continue
        if area == 0:
            continue
        if component_keyed_orthogonal(cells):
            for r, c in cells:
                fill_mask[r, c] = True

    # Fallback pass: diagonal key-contact if nothing was found.
    if not fill_mask.any():
        for idx, (area, cells, _) in enumerate(comp_info):
            if idx + 1 == outside_label:
                continue
            if area == 0:
                continue
            if component_keyed_diagonal(cells):
                for r, c in cells:
                    fill_mask[r, c] = True

    out = grid.copy()
    out[fill_mask] = fill
    return out