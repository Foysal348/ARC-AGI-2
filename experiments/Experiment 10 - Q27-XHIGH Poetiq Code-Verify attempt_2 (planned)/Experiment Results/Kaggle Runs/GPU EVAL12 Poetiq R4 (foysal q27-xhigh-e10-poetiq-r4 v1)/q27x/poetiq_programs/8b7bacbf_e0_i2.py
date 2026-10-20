import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill selected background pockets with the singleton fill color.

    Rule used:
    1. The background is the most frequent color.
    2. The fill color is the non-background color that appears once.
       If several singleton colors exist, use the largest value.
    3. The selector color is the non-background color adjacent to the fill pixel.
    4. A background pocket is filled if:
       - it is not connected to the grid border by 4-neighbor background paths,
         or it lies on the border but forms a compact border pocket,
       - its boundary contains a wall cell of a non-background color,
       - that wall cell is adjacent to the selector color,
       - the pocket is sufficiently compact for border-touching cases.
    """
    arr = np.array(grid, dtype=int)
    if arr.size == 0:
        return arr

    h, w = arr.shape

    # Background color: most frequent color.
    colors, counts = np.unique(arr, return_counts=True)
    bg = int(colors[np.argmax(counts)])

    non_bg = [int(c) for c in colors if int(c) != bg]
    if not non_bg:
        return arr

    # Fill color: singleton non-background color; highest value if multiple.
    singletons = [c for c in non_bg if counts[list(colors).index(c)] == 1]
    if singletons:
        fill_color = max(singletons)
    else:
        # Fallback: least frequent non-background color.
        fill_color = min(non_bg, key=lambda c: (counts[list(colors).index(c)], -c))

    # Find the fill pixel.
    fill_pixels = np.argwhere(arr == fill_color)
    if fill_pixels.size == 0:
        return arr

    r0, c0 = fill_pixels[0]

    # Selector color: adjacent non-background, non-fill color.
    selector = None
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nr, nc = r0 + dr, c0 + dc
        if 0 <= nr < h and 0 <= nc < w:
            v = int(arr[nr, nc])
            if v != bg and v != fill_color:
                selector = v
                break

    if selector is None:
        # Fallback: most frequent non-background, non-fill color.
        selector = max((c for c in non_bg if c != fill_color),
                       key=lambda c: counts[list(colors).index(c)])

    # Label background components using 4-connectivity.
    comp_id = -np.ones((h, w), dtype=int)
    comps = []

    for i in range(h):
        for j in range(w):
            if arr[i, j] == bg and comp_id[i, j] == -1:
                cid = len(comps)
                q = deque([(i, j)])
                comp_id[i, j] = cid
                pixels = []
                touches_border = False
                min_r = max_r = i
                min_c = max_c = j

                while q:
                    x, y = q.popleft()
                    pixels.append((x, y))
                    if x == 0 or x == h - 1 or y == 0 or y == w - 1:
                        touches_border = True
                    min_r = min(min_r, x)
                    max_r = max(max_r, x)
                    min_c = min(min_c, y)
                    max_c = max(max_c, y)

                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w:
                            if comp_id[nx, ny] == -1 and arr[nx, ny] == bg:
                                comp_id[nx, ny] = cid
                                q.append((nx, ny))

                comps.append({
                    "pixels": pixels,
                    "touch": touches_border,
                    "area": len(pixels),
                    "bbox": (min_r, max_r, min_c, max_c)
                })

    if not comps:
        return arr

    # Selector-adjacent wall mask: a non-background cell adjacent to selector.
    selector_wall = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(arr == selector):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if arr[nr, nc] != bg and arr[nr, nc] != fill_color:
                        selector_wall[nr, nc] = True

    def is_compact_border_pocket(comp):
        """
        Border-touching pockets are only filled when they look like a small
        completed pocket rather than an open side region.
        """
        area = comp["area"]
        min_r, max_r, min_c, max_c = comp["bbox"]
        bbox_area = (max_r - min_r + 1) * (max_c - min_c + 1)
        if area > 8:
            return False
        if bbox_area > 12:
            return False
        # Very elongated border regions are rejected.
        if (max_r - min_r + 1) > 3 and (max_c - min_c + 1) > 3:
            return False
        return True

    # Fill selected components.
    for comp in comps:
        # Interior pockets are primary candidates.
        if not comp["touch"]:
            candidate = True
        else:
            candidate = is_compact_border_pocket(comp)

        if not candidate:
            continue

        # Check whether this background component touches a selector-adjacent wall.
        selected = False
        for x, y in comp["pixels"]:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w:
                    if selector_wall[nx, ny]:
                        selected = True
                        break
            if selected:
                break

        if selected:
            for x, y in comp["pixels"]:
                arr[x, y] = fill_color

    return arr