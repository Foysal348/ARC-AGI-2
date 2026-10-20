import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the rightmost 5x5 panel.

    The first 5x5 panel contains a separator wall and a marker cell.
    The marker determines the side on which the second panel's pattern is kept.
    The third panel's pattern is kept on the opposite side.
    The fourth panel is filled with this side-based combination.
    If both patterns are present on the separator, the second pattern wins.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        return g.copy()

    h, w = g.shape
    out = g.copy()
    if h < 2 or w < 2:
        return out

    border = g[0, 0]

    def components(mask, diagonal=False):
        """Connected components using 4- or 8-neighbor adjacency."""
        H, W = mask.shape
        labels = np.zeros((H, W), dtype=int)
        comps = []

        if diagonal:
            neigh = [(dr, dc)
                     for dr in (-1, 0, 1)
                     for dc in (-1, 0, 1)
                     if (dr, dc) != (0, 0)]
        else:
            neigh = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for i in range(H):
            for j in range(W):
                if mask[i, j] and labels[i, j] == 0:
                    lab = len(comps) + 1
                    stack = [(i, j)]
                    labels[i, j] = lab
                    comp = []

                    while stack:
                        r, c = stack.pop()
                        comp.append((r, c))

                        for dr, dc in neigh:
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < H and 0 <= nc < W and
                                    mask[nr, nc] and labels[nr, nc] == 0):
                                labels[nr, nc] = lab
                                stack.append((nr, nc))

                    comps.append(comp)

        return labels, comps

    def separator_is_valid(S, H, W):
        """
        Check whether a set of cells can be the separator wall.

        It should be connected, touch the border, and split the tile into
        at least two 4-neighbor regions when blocked.
        """
        if not S or len(S) < 2:
            return False

        S_mask = np.zeros((H, W), dtype=bool)
        for r, c in S:
            if not (0 <= r < H and 0 <= c < W):
                return False
            S_mask[r, c] = True

        if not S_mask.any():
            return False

        # The wall should be one connected object. Diagonal touching counts.
        _, comps_S = components(S_mask, diagonal=True)
        if len(comps_S) != 1:
            return False

        # It should touch the outer frame.
        sides = set()
        for r, c in S:
            if r == 0:
                sides.add(0)
            if r == H - 1:
                sides.add(1)
            if c == 0:
                sides.add(2)
            if c == W - 1:
                sides.add(3)
        if not sides:
            return False

        # Blocking it must create at least two free regions.
        _, comps_free = components(~S_mask, diagonal=False)
        if len(comps_free) < 2:
            return False

        return True

    # Locate the four inner panels separated by full border-color lines.
    sep_cols = [c for c in range(w) if np.all(g[:, c] == border)]
    sep_rows = [r for r in range(h) if np.all(g[r, :] == border)]

    if len(sep_cols) >= 5 and len(sep_rows) >= 2:
        r0 = sep_rows[0] + 1
        r1 = sep_rows[-1]
        col_bounds = sep_cols
    elif h == 7 and w == 25:
        # Fallback for the standard layout of this task.
        r0, r1 = 1, 6
        col_bounds = [0, 6, 12, 18, 24]
    else:
        return out

    if len(col_bounds) < 5 or r1 <= r0:
        return out

    col_bounds = col_bounds[:5]

    panels = []
    for i in range(4):
        c0 = col_bounds[i] + 1
        c1 = col_bounds[i + 1]
        if c1 <= c0:
            return out
        panels.append(g[r0:r1, c0:c1].copy())

    if len(panels) != 4:
        return out

    H, W = panels[0].shape
    if H == 0 or W == 0:
        return out

    # Background color: the dominant color across all four panels.
    all_interior = np.concatenate([p.flatten() for p in panels])
    vals, counts = np.unique(all_interior, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    def foreground_color(panel):
        """Most common non-background color in a panel."""
        vals = panel[panel != bg]
        if vals.size == 0:
            return bg
        u, cnt = np.unique(vals, return_counts=True)
        return int(u[np.argmax(cnt)])

    b_color = foreground_color(panels[1])
    c_color = foreground_color(panels[2])

    a_mask = panels[0] != bg
    b_mask = panels[1] != bg
    c_mask = panels[2] != bg

    a_coords = [(int(r), int(c))
                for r in range(H)
                for c in range(W)
                if a_mask[r, c]]

    # Find the separator wall and marker cell(s) in panel 1.
    S = None
    markers = None

    if a_coords:
        from itertools import combinations
        n = len(a_coords)

        # Normally there is exactly one marker cell.
        # Try a few small marker sizes just in case.
        max_marker_cells = min(3, max(1, n - 2))

        for k in range(1, max_marker_cells + 1):
            found = False
            for marker_tuple in combinations(a_coords, k):
                S_cand = set(a_coords) - set(marker_tuple)
                if separator_is_valid(S_cand, H, W):
                    S = S_cand
                    markers = list(marker_tuple)
                    found = True
                    break
            if found:
                break

    if S is None:
        # Fallback: use the largest connected foreground part as the wall.
        _, comps_A = components(a_mask, diagonal=True)
        if not comps_A:
            return out
        S = set(max(comps_A, key=len))
        markers = [p for p in a_coords if p not in S]

    S_mask = np.zeros((H, W), dtype=bool)
    for r, c in S:
        if 0 <= r < H and 0 <= c < W:
            S_mask[r, c] = True

    # Determine the two sides of the wall using 4-neighbor flood fill.
    unblocked = ~S_mask
    labels, comps = components(unblocked, diagonal=False)

    side_b = S_mask.copy()
    side_c = S_mask.copy()

    marker_labels = set()
    for r, c in markers:
        if 0 <= r < H and 0 <= c < W and unblocked[r, c]:
            marker_labels.add(int(labels[r, c]))

    if not marker_labels:
        # Graceful fallback: let both sides be all free cells, with B priority.
        side_b[unblocked] = True
        side_c[unblocked] = True
    else:
        for comp_index in range(1, len(comps) + 1):
            comp_mask = labels == comp_index
            if comp_index in marker_labels:
                side_b[comp_mask] = True
            else:
                side_c[comp_mask] = True

    # Build the fourth panel: first place pattern C, then pattern B,
    # so B wins where both patterns overlap on the separator.
    new_d = np.full((H, W), bg, dtype=int)
    new_d[c_mask & side_c] = c_color
    new_d[b_mask & side_b] = b_color

    out[r0:r1, col_bounds[3] + 1:col_bounds[4]] = new_d
    return out