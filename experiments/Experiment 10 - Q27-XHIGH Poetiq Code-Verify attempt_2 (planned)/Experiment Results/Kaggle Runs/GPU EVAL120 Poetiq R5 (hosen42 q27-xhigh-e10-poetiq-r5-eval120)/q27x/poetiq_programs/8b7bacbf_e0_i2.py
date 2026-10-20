import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill enclosed background pockets that are connected to a singleton marker
    through a chain of trigger-color components and other enclosed pockets.

    Rule:
    - Background is the most frequent color.
    - A marker is a singleton non-background pixel that does not orthogonally
      touch the background.
    - The trigger color is the non-background color orthogonally adjacent to
      the marker.
    - A pocket is a 4-connected background component whose 4-neighbor boundary
      belongs to exactly one 8-connected non-background component.  Pockets
      touching the grid border are allowed only if they touch exactly one side.
    - Starting from trigger components adjacent to the marker, alternate between
      trigger components and pockets.  A trigger component is adjacent to a
      pocket if any of its cells is 8-adjacent to a boundary cell of the pocket.
    - Paint every reached pocket with the marker color.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.size == 0 or grid.ndim != 2:
        return grid.copy()

    h, w = grid.shape
    out = grid.copy()

    dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    dirs8 = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    vals, counts = np.unique(grid, return_counts=True)
    max_count = int(counts.max())
    candidates = vals[counts == max_count]
    if 0 in candidates:
        bg = int(candidates[0])
    else:
        bg = int(candidates[0])

    color_count = {int(v): int(c) for v, c in zip(vals, counts)}

    # ---------------------------------------------------------------
    # Label 8-connected non-background components.
    # These are used to require that a pocket boundary is one object.
    # ---------------------------------------------------------------
    nonbg = grid != bg
    comp_id = -np.ones((h, w), dtype=int)
    next_comp = 0

    for r in range(h):
        for c in range(w):
            if nonbg[r, c] and comp_id[r, c] == -1:
                q = deque([(r, c)])
                comp_id[r, c] = next_comp
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in dirs8:
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < h
                            and 0 <= nc < w
                            and nonbg[nr, nc]
                            and comp_id[nr, nc] == -1
                        ):
                            comp_id[nr, nc] = next_comp
                            q.append((nr, nc))
                next_comp += 1

    # ---------------------------------------------------------------
    # Find candidate pockets: enclosed background components.
    # ---------------------------------------------------------------
    holes = []
    seen_bg = np.zeros((h, w), dtype=bool)

    for r in range(h):
        for c in range(w):
            if grid[r, c] == bg and not seen_bg[r, c]:
                q = deque([(r, c)])
                seen_bg[r, c] = True
                cells = []
                sides = set()

                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))

                    if cr == 0:
                        sides.add("top")
                    if cr == h - 1:
                        sides.add("bottom")
                    if cc == 0:
                        sides.add("left")
                    if cc == w - 1:
                        sides.add("right")

                    for dr, dc in dirs4:
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < h
                            and 0 <= nc < w
                            and grid[nr, nc] == bg
                            and not seen_bg[nr, nc]
                        ):
                            seen_bg[nr, nc] = True
                            q.append((nr, nc))

                # Internal pockets are allowed.  Border pockets are allowed
                # only when they touch exactly one border side.
                if sides and len(sides) > 1:
                    continue

                boundary = []
                boundary_set = set()
                for cr, cc in cells:
                    for dr, dc in dirs4:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != bg:
                            pos = (nr, nc)
                            if pos not in boundary_set:
                                boundary_set.add(pos)
                                boundary.append(pos)

                if not boundary:
                    continue

                boundary_comps = {int(comp_id[nr, nc]) for nr, nc in boundary}
                if len(boundary_comps) != 1:
                    continue

                holes.append(
                    {
                        "cells": cells,
                        "boundary": boundary,
                        "boundary_set": boundary_set,
                    }
                )

    if not holes:
        return out

    # ---------------------------------------------------------------
    # Label 8-connected components of every non-background color.
    # These are the possible trigger components.
    # ---------------------------------------------------------------
    color_comps = []
    comps_by_color = {}

    for v in vals:
        v = int(v)
        if v == bg:
            continue

        mask = (grid == v)
        seen = np.zeros((h, w), dtype=bool)

        for r in range(h):
            for c in range(w):
                if mask[r, c] and not seen[r, c]:
                    cid = len(color_comps)
                    q = deque([(r, c)])
                    seen[r, c] = True
                    cells = []

                    while q:
                        cr, cc = q.popleft()
                        cells.append((cr, cc))
                        for dr, dc in dirs8:
                            nr, nc = cr + dr, cc + dc
                            if (
                                0 <= nr < h
                                and 0 <= nc < w
                                and mask[nr, nc]
                                and not seen[nr, nc]
                            ):
                                seen[nr, nc] = True
                                q.append((nr, nc))

                    cell_set = set(cells)
                    neighbor_set = set()
                    for cr, cc in cells:
                        for dr, dc in dirs8:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                neighbor_set.add((nr, nc))

                    color_comps.append(
                        {
                            "color": v,
                            "cells": cells,
                            "cell_set": cell_set,
                            "neighbor_set": neighbor_set,
                        }
                    )
                    comps_by_color.setdefault(v, []).append(cid)

    # ---------------------------------------------------------------
    # Find markers: singleton non-background pixels not orthogonally
    # adjacent to background.
    # ---------------------------------------------------------------
    markers = []

    for v in vals:
        v = int(v)
        if v == bg or color_count.get(v, 0) != 1:
            continue

        rr, cc = np.where(grid == v)
        r = int(rr[0])
        c = int(cc[0])

        touches_bg = False
        trigger_colors = set()

        for dr, dc in dirs4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            val = int(grid[nr, nc])
            if val == bg:
                touches_bg = True
            elif val != v and color_count.get(val, 0) > 1:
                trigger_colors.add(val)

        if touches_bg:
            continue
        if not trigger_colors:
            continue

        markers.append(
            {
                "color": v,
                "pos": (r, c),
                "trig_colors": trigger_colors,
            }
        )

    if not markers:
        return out

    # ---------------------------------------------------------------
    # Build global coordinate -> holes mapping for pocket boundaries.
    # ---------------------------------------------------------------
    coord_to_holes = {}
    for hid, hole in enumerate(holes):
        for pos in hole["boundary"]:
            coord_to_holes.setdefault(pos, []).append(hid)

    best_fill = {}

    # ---------------------------------------------------------------
    # Process each marker independently.
    # ---------------------------------------------------------------
    for marker in markers:
        marker_color = marker["color"]
        mr, mc = marker["pos"]
        marker_neighbors = []
        for dr, dc in dirs4:
            nr, nc = mr + dr, mc + dc
            if 0 <= nr < h and 0 <= nc < w:
                marker_neighbors.append((nr, nc))

        # Relevant trigger components for this marker.
        trig_ids = []
        for tc in marker["trig_colors"]:
            trig_ids.extend(comps_by_color.get(tc, []))

        if not trig_ids:
            continue

        # Coordinate -> trigger component IDs, based on 8-neighborhood.
        coord_to_trig = {}
        for idx in trig_ids:
            for pos in color_comps[idx]["neighbor_set"]:
                coord_to_trig.setdefault(pos, []).append(idx)

        # Seed trigger components that touch the marker orthogonally.
        seed_ids = set()
        for idx in trig_ids:
            for pos in marker_neighbors:
                if pos in color_comps[idx]["cell_set"]:
                    seed_ids.add(idx)
                    break

        if not seed_ids:
            continue

        visited_trig = set(seed_ids)
        visited_holes = set()
        queue = deque([("t", idx) for idx in seed_ids])

        comp_adj_cache = {}
        hole_adj_cache = {}

        while queue:
            typ, ident = queue.popleft()

            if typ == "t":
                if ident not in comp_adj_cache:
                    adjacent_holes = set()
                    for pos in color_comps[ident]["neighbor_set"]:
                        adjacent_holes.update(coord_to_holes.get(pos, ()))
                    comp_adj_cache[ident] = list(adjacent_holes)

                for hid in comp_adj_cache[ident]:
                    if hid not in visited_holes:
                        visited_holes.add(hid)
                        queue.append(("h", hid))

            else:
                if ident not in hole_adj_cache:
                    adjacent_trig = set()
                    for pos in holes[ident]["boundary"]:
                        adjacent_trig.update(coord_to_trig.get(pos, ()))
                    hole_adj_cache[ident] = list(adjacent_trig)

                for tid in hole_adj_cache[ident]:
                    if tid not in visited_trig:
                        visited_trig.add(tid)
                        queue.append(("t", tid))

        # Record the best marker for each reachable pocket.
        for hid in visited_holes:
            min_dist = min(
                abs(r - mr) + abs(c - mc) for r, c in holes[hid]["cells"]
            )
            cur = best_fill.get(hid)
            if cur is None or min_dist < cur[0] or (
                min_dist == cur[0] and marker_color < cur[1]
            ):
                best_fill[hid] = (min_dist, marker_color)

    # ---------------------------------------------------------------
    # Apply fills.
    # ---------------------------------------------------------------
    for hid, (_, color) in best_fill.items():
        for r, c in holes[hid]["cells"]:
            out[r, c] = color

    return out