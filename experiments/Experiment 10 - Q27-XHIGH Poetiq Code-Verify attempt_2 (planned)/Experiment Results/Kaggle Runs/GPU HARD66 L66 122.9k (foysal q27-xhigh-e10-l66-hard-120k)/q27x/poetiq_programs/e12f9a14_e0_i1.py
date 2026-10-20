import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    For each frame-enclosed colored region, extend the region's color outward
    through any missing frame cells.

    The background is the most frequent color. The frame color is the next most
    frequent color. Other colors are treated as interior colors.

    For each interior component:
      - expand its bounding box by 1 to form the expected frame border;
      - background cells on that border are openings;
      - from each opening, draw a straight ray outward in the direction from
        the nearest interior pixel through that opening;
      - stop when the grid edge or a blocking non-background color is reached.
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape
    out = grid.copy()

    # Identify background and frame color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    nonbg = [(int(v), int(c)) for v, c in zip(vals, counts) if int(v) != bg]
    if not nonbg:
        return out

    frame_color = int(nonbg[0][0])
    max_frame_count = int(nonbg[0][1])
    for v, c in nonbg[1:]:
        if c > max_frame_count:
            frame_color = int(v)
            max_frame_count = int(c)

    interior_colors = [int(v) for v, c in nonbg if int(v) != frame_color]
    if not interior_colors:
        return out

    # Helper: shoot a straight ray.
    def shoot(r: int, c: int, dr: int, dc: int, color: int) -> None:
        while 0 <= r < H and 0 <= c < W:
            cur = out[r, c]

            # Stop before overwriting another non-background color.
            if cur == frame_color:
                break
            if cur != bg and cur != color:
                break

            out[r, c] = color
            r += dr
            c += dc

    # Process interior colors in a deterministic order.
    for color in sorted(interior_colors):
        mask = (grid == color)
        visited = np.zeros((H, W), dtype=bool)

        for start_r in range(H):
            for start_c in range(W):
                if not mask[start_r, start_c] or visited[start_r, start_c]:
                    continue

                # 4-connected component of this interior color.
                stack = [(start_r, start_c)]
                visited[start_r, start_c] = True

                pixels = []
                rmin = rmax = start_r
                cmin = cmax = start_c

                while stack:
                    r, c = stack.pop()
                    pixels.append((r, c))

                    rmin = min(rmin, r)
                    rmax = max(rmax, r)
                    cmin = min(cmin, c)
                    cmax = max(cmax, c)

                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if mask[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                stack.append((nr, nc))

                # Expected frame boundary is the interior bbox expanded by 1.
                r0 = max(0, rmin - 1)
                r1 = min(H - 1, rmax + 1)
                c0 = max(0, cmin - 1)
                c1 = min(W - 1, cmax + 1)

                # Collect unique perimeter cells of the expected frame.
                perimeter = set()
                for rr in range(r0, r1 + 1):
                    perimeter.add((rr, c0))
                    perimeter.add((rr, c1))
                for cc in range(c0, c1 + 1):
                    perimeter.add((r0, cc))
                    perimeter.add((r1, cc))

                # There should be at least some frame color on this border.
                has_frame = any(grid[rr, cc] == frame_color for rr, cc in perimeter)
                if not has_frame:
                    continue

                center_r = (rmin + rmax) / 2.0
                center_c = (cmin + cmax) / 2.0

                for rr, cc in perimeter:
                    # Openings are background cells on the expected border.
                    if grid[rr, cc] != bg:
                        continue

                    # Find the nearest interior pixel to this opening.
                    best_pixel = None
                    best_dist = 10**18
                    for pr, pc in pixels:
                        d = (rr - pr) ** 2 + (cc - pc) ** 2
                        if d < best_dist:
                            best_dist = d
                            best_pixel = (pr, pc)

                    if best_pixel is None or best_dist == 0:
                        continue

                    pr, pc = best_pixel

                    # Direction from the nearest interior pixel to the opening.
                    dr = 1 if rr > pr else -1 if rr < pr else 0
                    dc = 1 if cc > pc else -1 if cc < pc else 0
                    if dr == 0 and dc == 0:
                        continue

                    # Only shoot outward.
                    if dr * (rr - center_r) + dc * (cc - center_c) <= 0:
                        continue

                    shoot(rr, cc, dr, dc, color)

    return out