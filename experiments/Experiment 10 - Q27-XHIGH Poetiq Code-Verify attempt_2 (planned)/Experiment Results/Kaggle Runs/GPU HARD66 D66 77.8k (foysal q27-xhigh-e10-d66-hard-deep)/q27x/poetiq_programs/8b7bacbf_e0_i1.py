import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill enclosed background regions with the rare marker color(s).

    Rule inferred from the examples:
      * The background is the most frequent color.
      * The fill color is the rare non-background color(s) -- in the given
        examples this is the unique color.
      * Enclosed background components are filled.  The exterior background is
        identified as the background component(s) containing a grid corner
        (with a largest-border-component fallback if no corner is background).
      * If there is more than one marker color, each enclosed component is
        filled with the nearest marker color (Euclidean distance from the
        component centroid to the marker pixel).
    """
    if grid is None:
        return np.zeros((0, 0), dtype=int)

    arr = np.array(grid, dtype=int, copy=True)
    if arr.ndim != 2:
        return arr

    h, w = arr.shape
    if h == 0 or w == 0:
        return arr

    # Background = most frequent color.  Prefer 0 on ties if present.
    vals, counts = np.unique(arr, return_counts=True)
    max_count = counts.max()
    bg_choices = vals[counts == max_count]
    background = int(0 if 0 in bg_choices else bg_choices[0])

    # Marker colors: first singleton non-background colors.
    marker_positions = {}
    if len(vals) > 1:
        non_bg = [(int(c), int(n)) for c, n in zip(vals, counts) if c != background]
        singletons = [(c, n) for c, n in non_bg if n == 1]
        if singletons:
            for c, _ in singletons:
                marker_positions[c] = np.argwhere(arr == c).astype(float)
        else:
            # Fallback: rarest non-background color(s).
            min_count = min(n for _, n in non_bg)
            for c, n in non_bg:
                if n == min_count:
                    marker_positions[c] = np.argwhere(arr == c).astype(float)

    if not marker_positions:
        return arr

    bg_mask = (arr == background)

    # Label 4-connected background components.
    labels = np.zeros((h, w), dtype=np.int32)
    comp_size = []
    comp_sum_r = []
    comp_sum_c = []
    next_label = 0

    for r in range(h):
        for c in range(w):
            if bg_mask[r, c] and labels[r, c] == 0:
                next_label += 1
                labels[r, c] = next_label
                stack = [(r, c)]
                size = 0
                sum_r = 0
                sum_c = 0

                while stack:
                    y, x = stack.pop()
                    size += 1
                    sum_r += y
                    sum_c += x

                    if y > 0 and bg_mask[y - 1, x] and labels[y - 1, x] == 0:
                        labels[y - 1, x] = next_label
                        stack.append((y - 1, x))
                    if y + 1 < h and bg_mask[y + 1, x] and labels[y + 1, x] == 0:
                        labels[y + 1, x] = next_label
                        stack.append((y + 1, x))
                    if x > 0 and bg_mask[y, x - 1] and labels[y, x - 1] == 0:
                        labels[y, x - 1] = next_label
                        stack.append((y, x - 1))
                    if x + 1 < w and bg_mask[y, x + 1] and labels[y, x + 1] == 0:
                        labels[y, x + 1] = next_label
                        stack.append((y, x + 1))

                comp_size.append(size)
                comp_sum_r.append(sum_r)
                comp_sum_c.append(sum_c)

    if next_label == 0:
        return arr

    # Exterior components: those containing a corner.
    exterior = set()
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    for r, c in corners:
        lbl = labels[r, c]
        if lbl != 0:
            exterior.add(int(lbl))

    # Fallback if no background corner exists: use largest border-touching
    # background component as exterior.
    if not exterior:
        border_labels = set()
        for c in range(w):
            if labels[0, c] != 0:
                border_labels.add(int(labels[0, c]))
            if labels[h - 1, c] != 0:
                border_labels.add(int(labels[h - 1, c]))
        for r in range(h):
            if labels[r, 0] != 0:
                border_labels.add(int(labels[r, 0]))
            if labels[r, w - 1] != 0:
                border_labels.add(int(labels[r, w - 1]))

        if border_labels:
            exterior.add(max(border_labels, key=lambda lbl: comp_size[lbl - 1]))

    markers = list(marker_positions.items())

    # Assign a fill color to every enclosed component.
    fill_label_color = {}
    if len(markers) == 1:
        only_color = markers[0][0]
        for lbl in range(1, next_label + 1):
            if lbl not in exterior:
                fill_label_color[lbl] = only_color
    else:
        for lbl in range(1, next_label + 1):
            if lbl in exterior:
                continue

            size = comp_size[lbl - 1]
            if size == 0:
                continue

            cy = comp_sum_r[lbl - 1] / size
            cx = comp_sum_c[lbl - 1] / size
            point = np.array([cy, cx], dtype=float)

            best_color = None
            best_dist = float("inf")

            for color, pts in markers:
                d = np.min(np.sum((pts - point) ** 2, axis=1))
                if d < best_dist - 1e-12:
                    best_dist = d
                    best_color = color
                elif abs(d - best_dist) <= 1e-12 and (best_color is None or color < best_color):
                    best_dist = d
                    best_color = color

            if best_color is not None:
                fill_label_color[lbl] = best_color

    # Apply fills.
    for lbl, color in fill_label_color.items():
        arr[labels == lbl] = color

    return arr