import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform input grid according to the inferred ARC rule:

    - Find 8-connected clusters of non-zero pixels.
    - In clusters with multiple colors, choose one color as the 'container'.
      The container is the color that best contains the other colors.
    - Remove the non-container colors in that cluster.
    - For each removed color, draw a line of that color outside the container:
        * Side: bounding-box side of the container with the fewest container pixels.
          Ties are broken in the order: up, left, right, down.
        * Anchor: the container pixel on that side closest to the removed color's center.
        * Length: removed color's bbox height for vertical directions,
                  removed color's bbox width for horizontal directions.
        * Clip the line to the grid boundary and stop if it hits an occupied cell.
    """
    arr = np.array(grid, dtype=int)

    if arr.ndim != 2:
        return arr.copy()

    H, W = arr.shape
    if H == 0 or W == 0:
        return arr.copy()

    out = arr.copy()
    nonzero = arr != 0
    visited = np.zeros((H, W), dtype=bool)

    offsets8 = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    def bbox(pixels):
        rmin = min(r for r, _ in pixels)
        rmax = max(r for r, _ in pixels)
        cmin = min(c for _, c in pixels)
        cmax = max(c for _, c in pixels)
        return rmin, rmax, cmin, cmax

    for r0 in range(H):
        for c0 in range(W):
            if not nonzero[r0, c0] or visited[r0, c0]:
                continue

            # Collect one 8-connected non-zero cluster.
            stack = [(r0, c0)]
            visited[r0, c0] = True
            comp = []

            while stack:
                r, c = stack.pop()
                comp.append((r, c))

                for dr, dc in offsets8:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if nonzero[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))

            # Group pixels in this cluster by color.
            pix_by_color = {}
            for r, c in comp:
                color = int(arr[r, c])
                pix_by_color.setdefault(color, []).append((r, c))

            # If only one color is present, leave it unchanged.
            if len(pix_by_color) <= 1:
                continue

            colors = list(pix_by_color.keys())
            bboxes = {color: bbox(pix_by_color[color]) for color in colors}

            # Choose the container color.
            # Prefer colors whose bbox contains more pixels of other colors,
            # then larger bbox area, then larger pixel count.
            def container_score(color):
                rmin, rmax, cmin, cmax = bboxes[color]
                area = (rmax - rmin + 1) * (cmax - cmin + 1)
                count = len(pix_by_color[color])

                contained = 0
                for other in colors:
                    if other == color:
                        continue
                    for rr, cc in pix_by_color[other]:
                        if rmin <= rr <= rmax and cmin <= cc <= cmax:
                            contained += 1

                return (contained, area, count)

            main_color = max(colors, key=container_score)
            main_pixels = pix_by_color[main_color]
            main_rmin, main_rmax, main_cmin, main_cmax = bboxes[main_color]

            # Count container pixels on each side of its bounding box.
            top_count = sum(1 for r, c in main_pixels if r == main_rmin)
            bottom_count = sum(1 for r, c in main_pixels if r == main_rmax)
            left_count = sum(1 for r, c in main_pixels if c == main_cmin)
            right_count = sum(1 for r, c in main_pixels if c == main_cmax)

            # Pick side with the fewest container pixels.
            # Tie order: up, left, right, down.
            side_options = [
                ("up", top_count),
                ("left", left_count),
                ("right", right_count),
                ("down", bottom_count),
            ]
            min_count = min(cnt for _, cnt in side_options)
            direction = "up"
            for side, cnt in side_options:
                if cnt == min_count:
                    direction = side
                    break

            # Process every non-container color in this cluster.
            for color in sorted(colors):
                if color == main_color:
                    continue

                pixels = pix_by_color[color]

                # Remove the original pixels of this color from this cluster.
                for r, c in pixels:
                    out[r, c] = 0

                prmin, prmax, pcmin, pcmax = bboxes[color]

                # Length and centering info.
                if direction in ("up", "down"):
                    length = prmax - prmin + 1
                    center_col = (pcmin + pcmax) / 2.0
                else:
                    length = pcmax - pcmin + 1
                    center_row = (prmin + prmax) / 2.0

                if direction == "up":
                    candidates = [(r, c) for r, c in main_pixels if r == main_rmin]
                    anchor = min(
                        candidates,
                        key=lambda rc: (abs(rc[1] - center_col), rc[1])
                    )
                    ar, ac = anchor
                    max_len = min(length, ar)

                    for step in range(1, max_len + 1):
                        nr, nc = ar - step, ac
                        if 0 <= nr < H and 0 <= nc < W and out[nr, nc] == 0:
                            out[nr, nc] = color
                        else:
                            break

                elif direction == "down":
                    candidates = [(r, c) for r, c in main_pixels if r == main_rmax]
                    anchor = min(
                        candidates,
                        key=lambda rc: (abs(rc[1] - center_col), rc[1])
                    )
                    ar, ac = anchor
                    max_len = min(length, H - 1 - ar)

                    for step in range(1, max_len + 1):
                        nr, nc = ar + step, ac
                        if 0 <= nr < H and 0 <= nc < W and out[nr, nc] == 0:
                            out[nr, nc] = color
                        else:
                            break

                elif direction == "left":
                    candidates = [(r, c) for r, c in main_pixels if c == main_cmin]
                    anchor = min(
                        candidates,
                        key=lambda rc: (abs(rc[0] - center_row), rc[0])
                    )
                    ar, ac = anchor
                    max_len = min(length, ac)

                    for step in range(1, max_len + 1):
                        nr, nc = ar, ac - step
                        if 0 <= nr < H and 0 <= nc < W and out[nr, nc] == 0:
                            out[nr, nc] = color
                        else:
                            break

                else:  # direction == "right"
                    candidates = [(r, c) for r, c in main_pixels if c == main_cmax]
                    anchor = min(
                        candidates,
                        key=lambda rc: (abs(rc[0] - center_row), rc[0])
                    )
                    ar, ac = anchor
                    max_len = min(length, W - 1 - ac)

                    for step in range(1, max_len + 1):
                        nr, nc = ar, ac + step
                        if 0 <= nr < H and 0 <= nc < W and out[nr, nc] == 0:
                            out[nr, nc] = color
                        else:
                            break

    return out