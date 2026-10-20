import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid according to the observed ARC rule.

    The rule implemented here is the one that is consistent with the examples:

    1.  Color 8 is a stencil/template color and is always removed from the output.
    2.  Other non-zero colors are treated as source pixels/components.
    3.  For each source pixel, a local stencil is stamped around it.
    4.  The stencil is determined by the nearest 8-pattern when present; when
        no useful 8-stencil is available, a simple one-cell stencil is used.
    5.  The output color for a source component is chosen as:
         - the color of the nearest single-pixel marker if one exists,
         - otherwise the color of the nearest different source component,
         - otherwise the source color itself.
    6.  Original source pixels and all 8s are erased; only stamped output cells remain.

    This implementation is intentionally heuristic but modular, and handles the
    provided examples and common variations by combining component analysis,
    stencil extraction from 8-components, and nearest-color assignment.
    """
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape
    out = np.zeros((h, w), dtype=int)

    # ---------- helpers ----------

    def in_bounds(r, c):
        return 0 <= r < h and 0 <= c < w

    def neighbors4(r, c):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc):
                yield nr, nc

    # connected components (4-neighborhood)
    comp_id = -np.ones((h, w), dtype=int)
    comps = []  # list of dict: color, cells, bbox

    for r in range(h):
        for c in range(w):
            if comp_id[r, c] != -1 or grid[r, c] == 0:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            comp_id[r, c] = len(comps)
            cells = []
            min_r = max_r = r
            min_c = max_c = c
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                min_r = min(min_r, cr)
                max_r = max(max_r, cr)
                min_c = min(min_c, cc)
                max_c = max(max_c, cc)
                for nr, nc in neighbors4(cr, cc):
                    if comp_id[nr, nc] == -1 and grid[nr, nc] == color:
                        comp_id[nr, nc] = len(comps)
                        stack.append((nr, nc))
            comps.append({
                'color': color,
                'cells': cells,
                'bbox': (min_r, min_c, max_r, max_c),
                'center': (
                    sum(p[0] for p in cells) / max(1, len(cells)),
                    sum(p[1] for p in cells) / max(1, len(cells))
                ),
                'size': len(cells)
            })

    # Split components into 8-components and source components.
    stencil_comps = [c for c in comps if c['color'] == 8]
    source_comps = [c for c in comps if c['color'] != 8]

    if not source_comps:
        return out

    # ---------- extract a local stencil from an 8-component when possible ----------

    def stencil_from_eight_component(comp):
        """
        Try to infer a small stencil from an 8-component.

        For the common 3-cell T shape (three 8s around an empty center),
        return the three offsets relative to the empty center.
        For a straight line of 2/3 8s, return a simple line offset set.
        Otherwise return a one-cell stencil.
        """
        cells = comp['cells']
        n = len(cells)

        if n == 1:
            r, c = cells[0]
            return [(0, 0)]

        # T-shape: three cells with one missing center cell that is adjacent to all three.
        if n == 3:
            pts = set(cells)
            # Try all empty cells adjacent to the cells.
            candidates = set()
            for r, c in pts:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    candidates.add((r + dr, c + dc))
            for center in candidates:
                if center in pts:
                    continue
                arms = []
                ok = True
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if (center[0] + dr, center[1] + dc) in pts:
                        arms.append((dr, dc))
                if len(arms) == 3:
                    # Return T offsets.  Sort for determinism.
                    return sorted(arms)

        # Straight horizontal line of 2 or 3 cells: use rightward offsets.
        rs = sorted({r for r, c in cells})
        cs = sorted({c for r, c in cells})
        if len(rs) == 1 and len(cs) <= 3:
            return [(0, 0), (0, 1)]

        # Straight vertical line of 2 or 3 cells: use downward offsets.
        if len(cs) == 1 and len(rs) <= 3:
            return [(0, 0), (1, 0)]

        # Fallback: use the component's cells relative to its top-left as offsets,
        # clipped to a reasonable radius.
        min_r, min_c = min(cells)
        offsets = []
        for r, c in cells:
            dr, dc = r - min_r, c - min_c
            if abs(dr) <= 3 and abs(dc) <= 3:
                offsets.append((dr, dc))
        if not offsets:
            offsets = [(0, 0)]
        return sorted(set(offsets))

    # Prefer small 8-components as stencils; if none, use a single-cell stencil.
    small_stencils = []
    for comp in stencil_comps:
        if comp['size'] <= 5:
            st = stencil_from_eight_component(comp)
            if st:
                small_stencils.append((comp, st))

    # ---------- choose output color for each source component ----------

    single_markers = [c for c in source_comps if c['size'] == 1 and c['color'] != 0]
    multi_sources = [c for c in source_comps if c['size'] >= 2]

    def nearest_component(comp, others, max_dist=1000000):
        best = None
        best_d = max_dist
        for other in others:
            if other is comp:
                continue
            if other['color'] == comp['color']:
                continue
            d = abs(comp['center'][0] - other['center'][0]) + abs(comp['center'][1] - other['center'][1])
            if d < best_d:
                best_d = d
                best = other
        return best, best_d

    def color_for_source(comp):
        # If there is a single marker nearby, use its color.
        if single_markers:
            marker, d = nearest_component(comp, single_markers)
            if marker is not None and d <= 20:
                return int(marker['color'])
        # Otherwise, use the nearest different multi-source color.
        if multi_sources:
            other, d = nearest_component(comp, multi_sources)
            if other is not None:
                return int(other['color'])
        # Fallback: use own color.
        return int(comp['color'])

    # ---------- choose a stencil for each source component ----------

    def stencil_for_source(comp):
        # If there are small 8-stencils, choose the nearest one.
        if small_stencils:
            best = None
            best_d = None
            for sc, st in small_stencils:
                d = abs(comp['center'][0] - sc['center'][0]) + abs(comp['center'][1] - sc['center'][1])
                if best is None or d < best_d:
                    best_d = d
                    best = st
            if best:
                return best

        # Fallback orientation-aware local stencil.
        # Estimate if component is mostly horizontal, vertical, or arbitrary.
        rs = [r for r, c in comp['cells']]
        cs = [c for r, c in comp['cells']]
        height = max(rs) - min(rs)
        width = max(cs) - min(cs)

        if height == 0:
            # horizontal line: stamp above and around to make a checkerboard edge
            return [(-1, 0), (0, -1), (0, 1), (0, 0)]
        if width == 0:
            # vertical line: stamp beside and around
            return [(-1, 0), (1, 0), (0, -1), (0, 0)]

        # General fallback: 4-neighbor expansion plus center.
        return [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

    # ---------- stamp all source components ----------

    # Process larger components first so prominent shapes are less likely to be
    # overwritten by tiny stray components.
    order = sorted(source_comps, key=lambda x: -x['size'])

    for comp in order:
        color = color_for_source(comp)
        stencil = stencil_for_source(comp)
        for r, c in comp['cells']:
            for dr, dc in stencil:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = color

    # If nothing was produced (e.g., only singleton sources), simply copy the
    # non-8 colors while removing 8.  This is a safe fallback.
    if out.max() == 0:
        out = grid.copy()
        out[grid == 8] = 0

    return out