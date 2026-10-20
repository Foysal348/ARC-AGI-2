import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule
    --------------
    1.  The background color is the most frequent color.
    2.  Marker colors are singleton non-background pixels that are not
        directly touching the background.  Each marker determines a
        "trigger color": the non-background color orthogonally adjacent
        to the marker.
    3.  A trigger is a connected component (8-connected) of that trigger
        color having size at least 2.
    4.  Background regions that are enclosed by a single 8-connected
        non-background component are filled with the marker color if the
        boundary of that background region is close to a trigger.

    In practice, this fills the interiors of closed frame-like objects
    that are marked/touched by the appropriate trigger line/region.
    If several markers apply, the marker nearest to the region is used.
    """
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape
    original = grid.copy()

    # ------------------------------------------------------------
    # Background: most frequent color
    # ------------------------------------------------------------
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # ------------------------------------------------------------
    # Label 8-connected non-background components
    # ------------------------------------------------------------
    nonbg = grid != bg
    comp_id = -np.ones((h, w), dtype=int)
    comp_cells = []

    for r in range(h):
        for c in range(w):
            if nonbg[r, c] and comp_id[r, c] == -1:
                cid = len(comp_cells)
                q = deque([(r, c)])
                comp_id[r, c] = cid
                cells = []
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and nonbg[nr, nc] and comp_id[nr, nc] == -1:
                                comp_id[nr, nc] = cid
                                q.append((nr, nc))
                comp_cells.append(cells)

    # ------------------------------------------------------------
    # Find marker colors: singleton, non-background, not touching background
    # ------------------------------------------------------------
    marker_colors = []
    color_masks = {}
    for v in vals:
        v = int(v)
        if v == bg:
            continue
        mask = grid == v
        cnt = int(mask.sum())
        if cnt != 1:
            continue
        r, c = np.where(mask)
        r, c = int(r[0]), int(c[0])
        touches_bg = False
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == bg:
                touches_bg = True
                break
        if not touches_bg:
            marker_colors.append(v)
            color_masks[v] = mask

    if not marker_colors:
        return original

    # ------------------------------------------------------------
    # For each marker, determine trigger colors (non-bg adjacent colors)
    # ------------------------------------------------------------
    marker_info = []  # (marker_color, set of trigger colors, list of marker positions)
    for m in marker_colors:
        trigger_colors = set()
        r, c = np.where(color_masks[m])
        r, c = int(r[0]), int(c[0])
        marker_info.append((m, trigger_colors, [(r, c)]))

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                cc = int(grid[nr, nc])
                if cc != bg and cc != m:
                    trigger_colors.add(cc)

    # Keep only markers with at least one trigger color
    marker_info = [(m, tr, pos) for m, tr, pos in marker_info if tr]
    if not marker_info:
        return original

    # ------------------------------------------------------------
    # Build trigger masks:
    # For each trigger color C, mark cells belonging to 8-connected
    # components of C with size >= 2.
    # ------------------------------------------------------------
    trigger_masks = {}
    for _, tr_colors, _ in marker_info:
        for C in tr_colors:
            if C in trigger_masks:
                continue

            mask = grid == C
            seen = np.zeros((h, w), dtype=bool)
            out_mask = np.zeros((h, w), dtype=bool)

            for r in range(h):
                for c in range(w):
                    if mask[r, c] and not seen[r, c]:
                        q = deque([(r, c)])
                        seen[r, c] = True
                        cells = []
                        while q:
                            cr, cc = q.popleft()
                            cells.append((cr, cc))
                            for dr in (-1, 0, 1):
                                for dc in (-1, 0, 1):
                                    if dr == 0 and dc == 0:
                                        continue
                                    nr, nc = cr + dr, cc + dc
                                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                                        seen[nr, nc] = True
                                        q.append((nr, nc))
                        if len(cells) >= 2:
                            for rr, cc in cells:
                                out_mask[rr, cc] = True
            trigger_masks[C] = out_mask

    # ------------------------------------------------------------
    # Helper: check if a boundary cell set is triggered by color C
    # ------------------------------------------------------------
    def boundary_has_trigger(boundary, C):
        if C not in trigger_masks:
            return False
        trig = trigger_masks[C]

        # For each boundary cell, inspect its 8-neighborhood for a trigger cell.
        for br, bc in boundary:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = br + dr, bc + dc
                    if 0 <= nr < h and 0 <= nc < w and trig[nr, nc]:
                        return True
        return False

    # ------------------------------------------------------------
    # Label 4-connected background components and test each
    # ------------------------------------------------------------
    out = original.copy()
    seen_bg = np.zeros((h, w), dtype=bool)

    for r in range(h):
        for c in range(w):
            if grid[r, c] == bg and not seen_bg[r, c]:
                q = deque([(r, c)])
                seen_bg[r, c] = True
                cells = []
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == bg and not seen_bg[nr, nc]:
                            seen_bg[nr, nc] = True
                            q.append((nr, nc))

                # Determine border sides touched
                sides = set()
                touches_border = False
                for cr, cc in cells:
                    if cr == 0:
                        sides.add('top')
                        touches_border = True
                    if cr == h - 1:
                        sides.add('bottom')
                        touches_border = True
                    if cc == 0:
                        sides.add('left')
                        touches_border = True
                    if cc == w - 1:
                        sides.add('right')
                        touches_border = True

                # Allow interior pockets, or edge pockets touching exactly one side.
                if touches_border and len(sides) != 1:
                    continue

                # Build boundary set: non-background cells 4-adjacent to this bg component
                boundary = set()
                for cr, cc in cells:
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != bg:
                            boundary.add((nr, nc))

                if not boundary:
                    continue

                # The pocket must be surrounded by a single 8-connected non-background component
                enclosing_ids = {comp_id[br, bc] for br, bc in boundary}
                if len(enclosing_ids) != 1:
                    continue

                # Find which markers trigger this pocket
                triggered_markers = []
                for m, tr_colors, mpos in marker_info:
                    ok = False
                    for C in tr_colors:
                        if boundary_has_trigger(boundary, C):
                            ok = True
                            break
                    if ok:
                        # Compute nearest marker distance to this pocket
                        best = min(
                            min(abs(cr - mr) + abs(cc - mc) for cr, cc in cells)
                            for mr, mc in mpos
                        )
                        triggered_markers.append((best, m))

                if not triggered_markers:
                    continue

                # If multiple markers trigger, choose the nearest one.
                _, best_marker = min(triggered_markers, key=lambda x: (x[0], x[1]))
                for cr, cc in cells:
                    out[cr, cc] = best_marker

    return out