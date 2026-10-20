import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill selected background pockets with the lone marker color.

    Rule implemented:
    1. The background color is the most frequent color.
    2. The marker/fill color is the highest-valued non-background color
       that occurs exactly once.
    3. The selector color is a non-background color adjacent to the marker.
       If several such colors exist, choose the one that explains the most
       fillable pockets; ties are broken by frequency, then value.
    4. A background pocket is filled when:
       - it is not the main outside background component,
       - all of its in-grid non-background neighbors have one uniform color B,
       - B is not the marker or selector color,
       - the connected component of B forming the pocket boundary touches
         the selector color (8-neighborhood).
    """
    arr = np.array(grid, dtype=int)
    if arr.size == 0:
        return arr

    h, w = arr.shape

    # 1) Background = most frequent color.
    colors, counts = np.unique(arr, return_counts=True)
    bg = int(colors[np.argmax(counts)])

    non_bg = [int(c) for c in colors if int(c) != bg]
    if not non_bg:
        return arr

    count = {int(c): int(n) for c, n in zip(colors, counts)}

    # 2) Marker/fill color = highest-valued singleton non-background color.
    singletons = [c for c in non_bg if count[c] == 1]
    if not singletons:
        return arr
    fill = max(singletons)

    marker_pos = np.argwhere(arr == fill)
    if marker_pos.size == 0:
        return arr
    mr, mc = marker_pos[0]

    # 3) Candidate selector colors around the marker.
    candidate_selectors = set()

    # Prefer 4-neighbor, then 8-neighbor if necessary.
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = mr + dr, mc + dc
        if 0 <= nr < h and 0 <= nc < w:
            v = int(arr[nr, nc])
            if v != bg and v != fill:
                candidate_selectors.add(v)

    if not candidate_selectors:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = mr + dr, mc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    v = int(arr[nr, nc])
                    if v != bg and v != fill:
                        candidate_selectors.add(v)

    if not candidate_selectors:
        return arr

    # Label all non-background, non-marker components (4-connectivity).
    comp_id = -np.ones((h, w), dtype=int)
    comp_color = []
    comp_touches_selector = {}

    dirs4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
    dirs8 = ((-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0, 1),
             (1, -1),  (1, 0),  (1, 1))

    for i in range(h):
        for j in range(w):
            v = int(arr[i, j])
            if v == bg or v == fill or comp_id[i, j] != -1:
                continue

            cid = len(comp_color)
            comp_id[i, j] = cid
            comp_color.append(v)
            q = deque([(i, j)])

            while q:
                r, c = q.popleft()
                for dr, dc in dirs4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        nv = int(arr[nr, nc])
                        if nv != bg and nv != fill and nv == v and comp_id[nr, nc] == -1:
                            comp_id[nr, nc] = cid
                            q.append((nr, nc))

    # For each selector candidate, mark components that touch it by 8-neighborhood.
    selector_component_touch = {}
    for sel in candidate_selectors:
        touch = np.zeros(len(comp_color), dtype=bool)
        sel_pixels = np.argwhere(arr == sel)
        for r, c in sel_pixels:
            for dr, dc in dirs8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    cid = comp_id[nr, nc]
                    if cid != -1:
                        touch[cid] = True
        selector_component_touch[sel] = touch

    # Label background components.
    bg_id = -np.ones((h, w), dtype=int)
    bg_comps = []

    for i in range(h):
        for j in range(w):
            if arr[i, j] == bg and bg_id[i, j] == -1:
                bid = len(bg_comps)
                bg_id[i, j] = bid
                q = deque([(i, j)])
                pixels = []
                touches_border = False
                border_count = 0
                boundary_colors = set()
                boundary_comp_ids = set()

                while q:
                    r, c = q.popleft()
                    pixels.append((r, c))

                    if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                        touches_border = True
                        border_count += 1

                    for dr, dc in dirs4:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            v = int(arr[nr, nc])
                            if v == bg:
                                if bg_id[nr, nc] == -1:
                                    bg_id[nr, nc] = bid
                                    q.append((nr, nc))
                            else:
                                if v != fill:
                                    boundary_colors.add(v)
                                    boundary_comp_ids.add(int(bg_id[nr, nc]) if False else int(comp_id[nr, nc]))
                    # The non-background neighbor id comes from comp_id, not bg_id.
                    for dr, dc in dirs4:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            v = int(arr[nr, nc])
                            if v != bg and v != fill:
                                cid = comp_id[nr, nc]
                                if cid != -1:
                                    boundary_comp_ids.add(int(cid))

                bg_comps.append({
                    "pixels": pixels,
                    "touches_border": touches_border,
                    "border_count": border_count,
                    "area": len(pixels),
                    "boundary_colors": boundary_colors,
                    "boundary_comp_ids": boundary_comp_ids,
                })

    if not bg_comps:
        return arr

    # Main outside background = largest border-touching background component.
    main_bg_id = -1
    if any(c["touches_border"] for c in bg_comps):
        main_bg_id = max(
            range(len(bg_comps)),
            key=lambda i: (bg_comps[i]["border_count"], bg_comps[i]["area"])
        )

    def filled_cells_for_selector(sel):
        """Return the number of background cells that would be filled for sel."""
        touched = selector_component_touch.get(sel, np.zeros(len(comp_color), dtype=bool))
        total = 0

        for bid, comp in enumerate(bg_comps):
            if bid == main_bg_id:
                continue
            if len(comp["boundary_colors"]) != 1:
                continue

            border_color = next(iter(comp["boundary_colors"]))
            if border_color == fill or border_color == sel:
                continue

            ok = False
            for cid in comp["boundary_comp_ids"]:
                if comp_color[cid] == border_color and touched[cid]:
                    ok = True
                    break

            if ok:
                total += comp["area"]

        return total

    # Choose selector by explanatory power, then frequency, then value.
    selector = max(
        candidate_selectors,
        key=lambda s: (filled_cells_for_selector(s), count.get(s, 0), s)
    )

    touched = selector_component_touch[selector]
    result = arr.copy()

    for bid, comp in enumerate(bg_comps):
        if bid == main_bg_id:
            continue
        if len(comp["boundary_colors"]) != 1:
            continue

        border_color = next(iter(comp["boundary_colors"]))
        if border_color == fill or border_color == selector:
            continue

        selected = False
        for cid in comp["boundary_comp_ids"]:
            if comp_color[cid] == border_color and touched[cid]:
                selected = True
                break

        if selected:
            for r, c in comp["pixels"]:
                result[r, c] = fill

    return result