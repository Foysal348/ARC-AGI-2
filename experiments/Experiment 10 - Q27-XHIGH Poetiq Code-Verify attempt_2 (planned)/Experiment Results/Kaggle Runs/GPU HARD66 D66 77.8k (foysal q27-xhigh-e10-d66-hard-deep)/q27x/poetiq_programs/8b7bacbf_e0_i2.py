import numpy as np


def _label_components(mask: np.ndarray, directions, store_cells: bool = False):
    """
    Label connected components in a boolean mask.

    Returns:
        labels: integer component labels, 0 means no component
        cells: list of component cells if store_cells is True, else []
        next_label: number of components
    """
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    cells = []
    next_label = 0

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue

            next_label += 1
            if store_cells:
                cells.append([])

            labels[r, c] = next_label
            stack = [(r, c)]

            while stack:
                y, x = stack.pop()
                if store_cells:
                    cells[-1].append((y, x))

                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = next_label
                            stack.append((ny, nx))

    return labels, cells, next_label


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill active enclosed background cavities with the rare marker color(s).

    Active foreground is the 8-connected non-background component(s) containing
    the rare marker pixel(s). A background cavity is filled only if it does not
    contain a grid corner and all its orthogonally adjacent non-background pixels
    are in active components.
    """
    if grid is None:
        return np.zeros((0, 0), dtype=int)

    arr = np.array(grid, dtype=int, copy=True)
    if arr.ndim != 2:
        return arr

    h, w = arr.shape
    if h == 0 or w == 0:
        return arr

    # Background = most frequent color; prefer 0 on ties.
    vals, counts = np.unique(arr, return_counts=True)
    max_count = counts.max()
    bg_choices = vals[counts == max_count]
    background = int(0 if 0 in bg_choices else bg_choices[0])

    non_bg_counts = {}
    for color, count in zip(vals, counts):
        color = int(color)
        if color != background:
            non_bg_counts[color] = int(count)

    if not non_bg_counts:
        return arr

    # Marker colors: singleton non-background colors if present.
    singleton_colors = [c for c, n in non_bg_counts.items() if n == 1]
    if singleton_colors:
        marker_colors = sorted(singleton_colors)
    else:
        min_count = min(non_bg_counts.values())
        marker_colors = sorted([c for c, n in non_bg_counts.items() if n == min_count])

    if not marker_colors:
        return arr

    marker_positions = {
        int(c): np.argwhere(arr == int(c)).astype(float)
        for c in marker_colors
    }

    # 8-connected foreground components.
    directions8 = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    non_bg_mask = arr != background
    non_bg_labels, _, _ = _label_components(non_bg_mask, directions8, store_cells=False)

    # Active foreground components are those containing at least one marker pixel.
    comp_markers = {}
    for color in marker_colors:
        for r, c in np.argwhere(arr == color):
            lbl = int(non_bg_labels[r, c])
            if lbl != 0:
                comp_markers.setdefault(lbl, set()).add(int(color))

    active_labels = set(comp_markers.keys())
    if not active_labels:
        return arr

    # 4-connected background components.
    directions4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    bg_mask = arr == background
    bg_labels, bg_cells, n_bg_components = _label_components(
        bg_mask, directions4, store_cells=True
    )

    if n_bg_components == 0:
        return arr

    # Exterior background components: those containing any grid corner.
    exterior = set()
    for r, c in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        if bg_mask[r, c]:
            exterior.add(int(bg_labels[r, c]))

    out = arr.copy()

    for lbl, cells in enumerate(bg_cells, start=1):
        if lbl in exterior or not cells:
            continue

        candidate_colors = set()
        has_active = False
        has_inactive = False

        # Inspect orthogonal foreground boundary of this background component.
        for r, c in cells:
            for dy, dx in directions4:
                nr, nc = r + dy, c + dx
                if 0 <= nr < h and 0 <= nc < w and non_bg_mask[nr, nc]:
                    fg_lbl = int(non_bg_labels[nr, nc])
                    if fg_lbl in active_labels:
                        has_active = True
                        candidate_colors.update(comp_markers.get(fg_lbl, set()))
                    else:
                        has_inactive = True

        # Fill only cavities bounded exclusively by active foreground.
        if not has_active or has_inactive or not candidate_colors:
            continue

        if len(candidate_colors) == 1:
            fill_color = next(iter(candidate_colors))
        else:
            # Choose nearest marker color by cavity centroid distance.
            avg_r = sum(r for r, c in cells) / len(cells)
            avg_c = sum(c for r, c in cells) / len(cells)
            center = np.array([avg_r, avg_c], dtype=float)

            best_color = None
            best_dist = float("inf")
            for color in candidate_colors:
                pts = marker_positions[color]
                dist = float(np.min(np.sum((pts - center) ** 2, axis=1)))
                if dist < best_dist - 1e-12:
                    best_dist = dist
                    best_color = color
                elif abs(dist - best_dist) <= 1e-12 and (
                    best_color is None or color < best_color
                ):
                    best_color = color

            fill_color = best_color

        for r, c in cells:
            out[r, c] = fill_color

    return out