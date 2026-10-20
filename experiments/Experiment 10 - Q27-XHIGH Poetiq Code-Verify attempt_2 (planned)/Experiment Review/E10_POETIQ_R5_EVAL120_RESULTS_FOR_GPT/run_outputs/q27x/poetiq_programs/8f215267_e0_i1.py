import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by keeping hollow rectangular frames, removing all other
    colored objects, and placing a row of dots inside each frame.

    For each frame color:
      - count the number of other connected components of that color
      - place that many pixels of the frame color in the middle interior row
      - align them to the right, starting two columns left of the right border
        and skipping one column between markers
    """
    grid = np.asarray(grid, dtype=int)

    if grid.ndim != 2:
        return grid.copy()

    H, W = grid.shape
    if H == 0 or W == 0:
        return grid.copy()

    # Background is the most frequent color.
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[np.argmax(counts)])

    # Output will contain only background plus reconstructed frames.
    out = np.full((H, W), bg, dtype=grid.dtype)

    visited = np.zeros((H, W), dtype=bool)
    comps_by_color = {}

    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    # Find all 4-connected non-background components.
    for r in range(H):
        for c in range(W):
            if grid[r, c] == bg or visited[r, c]:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True

            minr = maxr = r
            minc = maxc = c
            size = 0

            while stack:
                cr, cc = stack.pop()
                size += 1

                if cr < minr:
                    minr = cr
                if cr > maxr:
                    maxr = cr
                if cc < minc:
                    minc = cc
                if cc > maxc:
                    maxc = cc

                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < H
                        and 0 <= nc < W
                        and not visited[nr, nc]
                        and grid[nr, nc] == color
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            h = maxr - minr + 1
            w = maxc - minc + 1

            # A frame is a hollow rectangle.
            is_rect_frame = False
            if h >= 3 and w >= 3:
                expected_perimeter_pixels = 2 * h + 2 * w - 4
                if size == expected_perimeter_pixels:
                    ok = True

                    # Check top and bottom edges.
                    for cc in range(minc, maxc + 1):
                        if grid[minr, cc] != color or grid[maxr, cc] != color:
                            ok = False
                            break

                    # Check left and right edges.
                    if ok:
                        for rr in range(minr, maxr + 1):
                            if grid[rr, minc] != color or grid[rr, maxc] != color:
                                ok = False
                                break

                    is_rect_frame = ok

            comp = {
                "color": color,
                "rmin": minr,
                "rmax": maxr,
                "cmin": minc,
                "cmax": maxc,
                "size": size,
                "rect_frame": is_rect_frame,
            }
            comps_by_color.setdefault(color, []).append(comp)

    # Choose one frame per color: the largest rectangular frame if present.
    frames = {}
    for color, comps in comps_by_color.items():
        rect_comps = [comp for comp in comps if comp["rect_frame"]]
        if rect_comps:
            frames[color] = max(rect_comps, key=lambda comp: comp["size"])

    # Reconstruct selected frames in the output.
    for color, f in frames.items():
        r1, r2, c1, c2 = f["rmin"], f["rmax"], f["cmin"], f["cmax"]

        out[r1, c1:c2 + 1] = color
        out[r2, c1:c2 + 1] = color
        out[r1:r2 + 1, c1] = color
        out[r1:r2 + 1, c2] = color

    # For each frame, add markers based on the number of removed objects.
    for color, f in frames.items():
        comps = comps_by_color[color]
        n_objects = len(comps) - 1  # exclude the frame component

        if n_objects <= 0:
            continue

        r1, r2, c1, c2 = f["rmin"], f["rmax"], f["cmin"], f["cmax"]

        mid_r = (r1 + r2) // 2
        if mid_r <= r1 or mid_r >= r2:
            mid_r = r1 + (r2 - r1) // 2

        # Valid marker columns: start two columns left of the right border,
        # then move left every second column.
        slots = list(range(c2 - 2, c1, -2))

        # Fallback for unusually small frames.
        if not slots:
            slots = list(range(c2 - 1, c1, -1))

        if n_objects <= len(slots):
            marker_cols = slots[:n_objects]
        else:
            # Graceful fallback if there are more objects than normal slots.
            marker_cols = slots[:]
            chosen = set(marker_cols)

            for c in range(c2 - 1, c1, -1):
                if c not in chosen:
                    marker_cols.append(c)
                    chosen.add(c)
                    if len(marker_cols) == n_objects:
                        break

        for c in marker_cols:
            if c1 < c < c2 and r1 < mid_r < r2:
                out[mid_r, c] = color

    return out