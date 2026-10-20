import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill selected enclosed background regions with the singleton fill color.

    Rule:
    1. The background color is the most frequent color.
    2. The fill color is the non-background color that appears exactly once.
       If there are several, use the highest-valued one.
    3. The selector color is the non-background color adjacent to the fill pixel.
    4. For every background component that is not the main outside component,
       fill it with the fill color if any selector-colored pixel touches
       (4- or 8-neighbour) a boundary cell surrounding that background component.
    """
    arr = np.array(grid, dtype=int)
    if arr.size == 0:
        return arr

    h, w = arr.shape
    colors, counts = np.unique(arr, return_counts=True)
    count = {int(c): int(n) for c, n in zip(colors, counts)}

    # Background = most frequent color.
    bg = int(colors[np.argmax(counts)])

    non_bg = [int(c) for c in colors if int(c) != bg]
    if not non_bg:
        return arr

    # Fill color: singleton non-background color; highest value if multiple.
    singletons = [c for c in non_bg if count[c] == 1]
    if singletons:
        fill_color = max(singletons)
    else:
        # Fallback, should not be needed for valid cases.
        fill_color = max(non_bg, key=lambda c: (count[c], c))

    # Find the fill pixel.
    fill_pos = np.argwhere(arr == fill_color)
    if fill_pos.size == 0:
        return arr

    # Selector color: adjacent non-background color next to the fill pixel.
    r0, c0 = fill_pos[0]
    neighbor_colors = set()

    # 4-neighbour first.
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r0 + dr, c0 + dc
        if 0 <= nr < h and 0 <= nc < w:
            v = int(arr[nr, nc])
            if v != bg and v != fill_color:
                neighbor_colors.add(v)

    # If none, try 8-neighbour.
    if not neighbor_colors:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r0 + dr, c0 + dc
                if 0 <= nr < h and 0 <= nc < w:
                    v = int(arr[nr, nc])
                    if v != bg and v != fill_color:
                        neighbor_colors.add(v)

    if neighbor_colors:
        selector = max(neighbor_colors, key=lambda c: (count.get(c, 0), c))
    else:
        # Fallback: most frequent non-background color.
        selector = max(non_bg, key=lambda c: (count[c], c))

    # Label background components.
    comp_id = -np.ones((h, w), dtype=int)
    comp_pixels = []
    comp_border_count = []
    comp_size = []

    dirs4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for i in range(h):
        for j in range(w):
            if arr[i, j] == bg and comp_id[i, j] == -1:
                cid = len(comp_pixels)
                q = deque([(i, j)])
                comp_id[i, j] = cid
                pixels = []
                border_count = 0

                while q:
                    x, y = q.popleft()
                    pixels.append((x, y))
                    if x == 0 or x == h - 1 or y == 0 or y == w - 1:
                        border_count += 1

                    for dx, dy in dirs4:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w:
                            if comp_id[nx, ny] == -1 and arr[nx, ny] == bg:
                                comp_id[nx, ny] = cid
                                q.append((nx, ny))

                comp_pixels.append(pixels)
                comp_border_count.append(border_count)
                comp_size.append(len(pixels))

    if not comp_pixels:
        return arr

    # Main outside background component: the one touching the border most, then largest.
    main_comp = max(
        range(len(comp_pixels)),
        key=lambda idx: (comp_border_count[idx], comp_size[idx])
    )

    # Mark cells near the selector color (3x3 neighbourhood).
    selector_mask = (arr == selector)
    selector_near = np.zeros((h, w), dtype=bool)
    for sr, sc in np.argwhere(selector_mask):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = sr + dr, sc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    selector_near[nr, nc] = True

    selected_components = []

    # A non-main background component is filled if selector touches its boundary.
    for cid, pixels in enumerate(comp_pixels):
        if cid == main_comp:
            continue

        selected = False
        for x, y in pixels:
            for dx, dy in dirs4:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w:
                    # Boundary cell of this background component.
                    if arr[nx, ny] != bg and selector_near[nx, ny]:
                        selected = True
                        break
            if selected:
                break

        if selected:
            selected_components.append(cid)

    # Apply fill.
    for cid in selected_components:
        for x, y in comp_pixels[cid]:
            arr[x, y] = fill_color

    return arr