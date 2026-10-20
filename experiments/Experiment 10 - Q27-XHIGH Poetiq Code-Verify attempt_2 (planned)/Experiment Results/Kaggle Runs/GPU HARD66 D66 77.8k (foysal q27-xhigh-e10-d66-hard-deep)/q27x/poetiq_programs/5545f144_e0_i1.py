import numpy as np
from itertools import product

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a grid containing one or more vertically separated panels.

    Rule implemented:
    1. Split the input into panels using full vertical separator columns.
       If no such separator columns exist, treat the whole grid as one panel.
    2. For each panel, keep only the main object color (the non-background,
       non-separator color).
    3. Find the largest connected object shape that appears in every panel,
       allowing translation and the 8 dihedral symmetries.
       If there is only one panel, use its largest 8-connected object component.
    4. Output the canonical upright version of that shape on a background grid
       of the panel size.
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape

    # Identify background as the color on the corners.
    bg = int(grid[0, 0])

    # Find full separator columns: a column whose values are all the same,
    # non-background color.
    sep_cols = []
    for c in range(W):
        col = grid[:, c]
        if col.size > 0 and np.all(col == col[0]) and int(col[0]) != bg:
            sep_cols.append(c)

    # Build panel column ranges.
    panel_slices = []
    start = 0
    for c in sep_cols:
        if c > start:
            panel_slices.append((start, c))
        start = c + 1
    if start < W:
        panel_slices.append((start, W))

    if not panel_slices:
        panel_slices = [(0, W)]

    # Object mask per panel.
    panels = []
    object_colors = set()

    for c0, c1 in panel_slices:
        sub = grid[:, c0:c1]
        # The object color is the most common color other than background
        # and the separator color in this strip.
        vals, counts = np.unique(sub, return_counts=True)
        best_obj = None
        best_count = -1
        for val, cnt in zip(vals, counts):
            v = int(val)
            if v == bg:
                continue
            if v in sep_cols:  # separator color may equal a column value
                continue
            if cnt > best_count:
                best_count = cnt
                best_obj = v
        if best_obj is None:
            mask = np.zeros(sub.shape, dtype=bool)
        else:
            object_colors.add(best_obj)
            mask = (sub == best_obj)
        panels.append(mask)

    # If no separator was found, this is a single-panel case.
    def normalize(cells):
        """Normalize a set of coordinates to top-left bbox, return tuple of cells."""
        if not cells:
            return ()
        cells = list(cells)
        min_r = min(r for r, c in cells)
        min_c = min(c for r, c in cells)
        cells = [(r - min_r, c - min_c) for r, c in cells]
        max_r = max(r for r, c in cells)
        max_c = max(c for r, c in cells)
        return tuple(sorted(cells)), max_r + 1, max_c + 1

    def shape_key(cells):
        """Canonical key for a shape under the 8 dihedral transformations."""
        cells, h, w = normalize(cells)
        best = None
        for flip in (False, True):
            cur = [(r, (w - 1 - c) if flip else c) for r, c in cells]
            for rot in range(4):
                # Rotate cur: (r,c) -> (c, h-1-r)
                cur = [(c, h - 1 - r) for r, c in cur]
                h2, w2 = w, h
                # After rotation, bbox dimensions swap every step.
                # We recompute normalize anyway.
                norm, _, _ = normalize(cur)
                if best is None or norm < best:
                    best = norm
        return best

    def connected_8(cells):
        """Check whether a set of cells is 8-connected."""
        cells = set(cells)
        if not cells:
            return False
        start = next(iter(cells))
        seen = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nxt = (r + dr, c + dc)
                    if nxt in cells and nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
        return len(seen) == len(cells)

    def components(mask):
        """Return list of 8-connected components as sets of coordinates."""
        h, w = mask.shape
        visited = np.zeros((h, w), dtype=bool)
        comps = []
        for r in range(h):
            for c in range(w):
                if mask[r, c] and not visited[r, c]:
                    stack = [(r, c)]
                    visited[r, c] = True
                    comp = set()
                    while stack:
                        rr, cc = stack.pop()
                        comp.add((rr, cc))
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = rr + dr, cc + dc
                                if 0 <= nr < h and 0 <= nc < w:
                                    if mask[nr, nc] and not visited[nr, nc]:
                                        visited[nr, nc] = True
                                        stack.append((nr, nc))
                    comps.append(comp)
        return comps

    def contains_shape(panel_mask, shape_cells):
        """
        Return True if panel_mask contains shape_cells under any translation
        and any of the 8 dihedral transformations.
        """
        h, w = panel_mask.shape
        # Precompute transformed shape variants.
        variants = set()
        cells, sh, sw = normalize(shape_cells)
        cur = cells
        for flip in (False, True):
            cur = [(r, (sw - 1 - c) if flip else c) for r, c in cells]
            hh, ww = sh, sw
            for rot in range(4):
                cur = [(c, hh - 1 - r) for r, c in cur]
                hh, ww = ww, hh
                norm, _, _ = normalize(cur)
                variants.add(norm)

        # For each variant, try all top-left translations.
        for var in variants:
            vh = max(r for r, c in var) + 1
            vw = max(c for r, c in var) + 1
            if vh > h or vw > w:
                continue
            for tr in range(h - vh + 1):
                for tc in range(w - vw + 1):
                    ok = True
                    for r, c in var:
                        if not panel_mask[tr + r, tc + c]:
                            ok = False
                            break
                    if ok:
                        return True
        return False

    def largest_common_shape(panel_masks):
        """
        Find the largest connected shape appearing in all panels.
        Returns a normalized set of cells.
        """
        # Use the panel with the fewest object pixels as candidate source.
        counts = [int(m.sum()) for m in panel_masks]
        if min(counts) == 0:
            return None

        idx = counts.index(min(counts))
        source = panel_masks[idx]
        comps = components(source)

        # Sort candidate components by decreasing size; we want the largest
        # common shape, but it may be a subset of a larger component.
        comps.sort(key=lambda comp: -len(comp))

        best_shape = None
        best_size = 0

        # Search subsets up to a practical limit.  The ARC shapes here are small.
        max_subset_size = 12

        for comp in comps:
            if len(comp) < best_size:
                continue
            cells_list = list(comp)
            n = len(cells_list)
            if n == 0:
                continue

            # Enumerate subsets of this component, but only up to max_subset_size.
            # For small n this is fast.  For larger n, restrict to connected
            # DFS subsets of bounded size.
            if n <= max_subset_size:
                from itertools import combinations
                for size in range(min(n, max_subset_size), best_size, -1):
                    found_at_this_size = False
                    for comb in combinations(range(n), size):
                        sub = [cells_list[i] for i in comb]
                        if not connected_8(sub):
                            continue
                        key = shape_key(sub)
                        if key is None:
                            continue
                        # Quick size check.
                        if size <= best_size:
                            continue
                        # Test all other panels.
                        ok = True
                        for j, pm in enumerate(panel_masks):
                            if j == idx:
                                continue
                            if not contains_shape(pm, sub):
                                ok = False
                                break
                        if ok:
                            # Keep lexicographically smallest canonical shape
                            # for deterministic tie-breaking.
                            if best_shape is None or key < shape_key(best_shape):
                                best_shape = set(sub)
                                best_size = size
                            found_at_this_size = True
                    if found_at_this_size:
                        break
            else:
                # Fallback for larger components: only test the component itself
                # and its connected prefixes is too arbitrary; in this task
                # shapes are small, so this path is rarely needed.
                if len(comp) > best_size and connected_8(comp):
                    ok = True
                    for j, pm in enumerate(panel_masks):
                        if j == idx:
                            continue
                        if not contains_shape(pm, comp):
                            ok = False
                            break
                    if ok:
                        best_shape = set(comp)
                        best_size = len(comp)

        return best_shape

    def canonical_upright(cells):
        """
        Choose the 'upright' orientation:
        - prefer smallest height;
        - prefer top row with the fewest cells;
        - then lexicographically smallest normalized shape.
        This matches the examples: V points up, arrow points up, T is upright.
        """
        cells, h, w = normalize(cells)
        best = None
        best_score = None
        cur = cells
        for flip in (False, True):
            cur = [(r, (w - 1 - c) if flip else c) for r, c in cells]
            hh, ww = h, w
            for rot in range(4):
                cur = [(c, hh - 1 - r) for r, c in cur]
                hh, ww = ww, hh
                norm, nh, nw = normalize(cur)
                top_count = sum(1 for r, c in norm if r == 0)
                score = (nh, top_count, norm)
                if best_score is None or score < best_score:
                    best_score = score
                    best = norm
        return best

    def place_shape(output_mask, shape_cells, panel_h, panel_w):
        """
        Place the canonical shape into output_mask.
        If possible, anchor it to the topmost common intersection cell.
        Otherwise place it top-center.
        """
        shape = canonical_upright(shape_cells)
        sh = max(r for r, c in shape) + 1
        sw = max(c for r, c in shape) + 1

        # Find common cells across panels (same relative coordinates).
        common = None
        for i, m in enumerate(panels):
            pts = set(zip(*np.where(m)))
            if i == 0:
                common = pts
            else:
                common &= pts

        # Choose anchor:
        # - if common exists and has horizontal spread, use topmost common cell;
        # - if common is a vertical line, use bottommost common cell;
        # - otherwise use top-center.
        anchor = None
        if common:
            rows = [r for r, c in common]
            cols = [c for r, c in common]
            if max(cols) - min(cols) > 0:
                anchor = min(common, key=lambda x: (x[0], x[1]))
            else:
                anchor = max(common, key=lambda x: (x[0], x[1]))

        # Anchor cell in shape:
        # - if top row has exactly one cell, use it;
        # - else if bottom row has exactly one cell, use it;
        # - else use the top-center cell.
        top_cells = sorted([c for r, c in shape if r == 0])
        bot_cells = sorted([c for r, c in shape if r == sh - 1])
        if len(top_cells) == 1:
            ar, ac = 0, top_cells[0]
        elif len(bot_cells) == 1:
            ar, ac = sh - 1, bot_cells[0]
        else:
            ar, ac = 0, sw // 2

        if anchor is not None:
            ar_row, ar_col = anchor
        else:
            ar_row, ar_col = 0, (panel_w - sw) // 2

        tr = ar_row - ar
        tc = ar_col - ac

        # Keep placement inside the panel.
        tr = max(0, min(tr, panel_h - sh))
        tc = max(0, min(tc, panel_w - sw))

        for r, c in shape:
            output_mask[tr + r, tc + c] = True

    # Special case: all panels are empty.
    if all(int(m.sum()) == 0 for m in panels):
        panel_h = H
        panel_w = panel_slices[0][1] - panel_slices[0][0]
        return np.full((panel_h, panel_w), bg, dtype=grid.dtype)

    if len(panels) == 1:
        # Single panel: output largest connected component, canonical upright.
        comps = components(panels[0])
        if not comps:
            return np.zeros((H, W), dtype=grid.dtype)
        comp = max(comps, key=len)
        target = set(comp)
    else:
        target = largest_common_shape(panels)

    # Output dimensions: same height, panel width.
    panel_h = H
    panel_w = panel_slices[0][1] - panel_slices[0][0]
    out = np.full((panel_h, panel_w), bg, dtype=grid.dtype)

    if target is not None and len(target) > 0:
        out_mask = np.zeros((panel_h, panel_w), dtype=bool)
        place_shape(out_mask, target, panel_h, panel_w)

        # Determine output object color: most common non-bg color in input.
        vals, counts = np.unique(grid, return_counts=True)
        obj = int(grid[0, 0])
        best_cnt = -1
        for v, cnt in zip(vals, counts):
            v = int(v)
            if v == bg:
                continue
            if v in [int(x) for x in grid[:, c] for c in sep_cols] if sep_cols else False:
                continue
            if cnt > best_cnt:
                best_cnt = cnt
                obj = v
        out[out_mask] = obj

    return out