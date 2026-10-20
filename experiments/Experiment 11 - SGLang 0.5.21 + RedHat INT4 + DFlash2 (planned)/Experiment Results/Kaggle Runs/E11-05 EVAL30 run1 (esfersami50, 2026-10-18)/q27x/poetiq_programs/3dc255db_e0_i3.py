import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform each two-color connected "container + payload" object.

    For every 8-connected component containing more than one color:
      * the most frequent color is treated as the container,
      * all other colors in that component are treated as payload,
      * payload cells are removed,
      * a straight line of the payload color is drawn outside the container.

    The line direction is chosen from the sides of the container's bounding box
    with the fewest container cells.  Among those sides, the side most opposite
    to the payload centroid is preferred.  The line length is the payload
    bounding-box extent in the chosen direction (horizontal -> width,
    vertical -> height), clipped naturally by the grid boundary.
    """
    if grid is None:
        return np.array([], dtype=int)

    grid = np.array(grid, dtype=int, copy=True)
    if grid.size == 0:
        return grid

    H, W = grid.shape
    out = grid.copy()
    visited = np.zeros((H, W), dtype=bool)

    def component_cells(start_r, start_c):
        """8-connected component of non-zero cells."""
        q = deque([(start_r, start_c)])
        visited[start_r, start_c] = True
        cells = []

        while q:
            r, c = q.popleft()
            cells.append((r, c))

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if not visited[nr, nc] and grid[nr, nc] != 0:
                            visited[nr, nc] = True
                            q.append((nr, nc))

        return cells

    def bbox(cells):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        return min(rows), max(rows), min(cols), max(cols)

    def side_tip_score(side, cells, rmin, rmax, cmin, cmax):
        """
        Lower is better.  Prefer the side whose cells are closest to the
        perpendicular center of the bounding box.
        """
        if side == "U":
            pts = [c for r, c in cells if r == rmin]
            center = (cmin + cmax) / 2.0
        elif side == "D":
            pts = [c for r, c in cells if r == rmax]
            center = (cmin + cmax) / 2.0
        elif side == "L":
            pts = [r for r, c in cells if c == cmin]
            center = (rmin + rmax) / 2.0
        else:  # "R"
            pts = [r for r, c in cells if c == cmax]
            center = (rmin + rmax) / 2.0

        if not pts:
            return float("inf")

        return abs(sum(pts) / len(pts) - center)

    def draw_payload_line(container_cells, payload_cells, color):
        if not container_cells or not payload_cells:
            return

        rmin, rmax, cmin, cmax = bbox(container_cells)

        top = sum(1 for r, c in container_cells if r == rmin)
        bottom = sum(1 for r, c in container_cells if r == rmax)
        left = sum(1 for r, c in container_cells if c == cmin)
        right = sum(1 for r, c in container_cells if c == cmax)

        side_counts = {"U": top, "D": bottom, "L": left, "R": right}
        min_count = min(side_counts.values())
        candidates = [s for s in ("U", "D", "L", "R") if side_counts[s] == min_count]

        cr = sum(r for r, _ in container_cells) / len(container_cells)
        cc = sum(c for _, c in container_cells) / len(container_cells)

        pr = sum(r for r, _ in payload_cells) / len(payload_cells)
        pc = sum(c for _, c in payload_cells) / len(payload_cells)

        # Vector from payload centroid to container centroid: direction away from payload.
        away_r = cr - pr
        away_c = cc - pc

        dir_vec = {
            "U": (-1, 0),
            "D": (1, 0),
            "L": (0, -1),
            "R": (0, 1),
        }

        def score_side(side):
            dr, dc = dir_vec[side]
            dot = away_r * dr + away_c * dc
            tip = side_tip_score(side, container_cells, rmin, rmax, cmin, cmax)
            # Higher is better:
            # 1) larger dot with the away vector,
            # 2) smaller tip_score (more centered protrusion),
            # 3) deterministic tie-break.
            tie = {"U": 3, "L": 2, "D": 1, "R": 0}[side]
            return (dot, -tip, tie)

        side = max(candidates, key=score_side)

        # Choose the anchor cell on that side, preferring the centered tip.
        if side == "U":
            row = rmin
            pts = [c for r, c in container_cells if r == row]
            center = (cmin + cmax) / 2.0
            anchor_c = min(pts, key=lambda c: (abs(c - center), abs(c - pc)))
            length = max(r for r, _ in payload_cells) - min(r for r, _ in payload_cells) + 1
            r0 = rmin - 1
            for i in range(length):
                r = r0 - i
                c = anchor_c
                if 0 <= r < H and 0 <= c < W and out[r, c] in (0, color):
                    out[r, c] = color
                elif not (0 <= r < H and 0 <= c < W):
                    break
                else:
                    break

        elif side == "D":
            row = rmax
            pts = [c for r, c in container_cells if r == row]
            center = (cmin + cmax) / 2.0
            anchor_c = min(pts, key=lambda c: (abs(c - center), abs(c - pc)))
            length = max(r for r, _ in payload_cells) - min(r for r, _ in payload_cells) + 1
            r0 = rmax + 1
            for i in range(length):
                r = r0 + i
                c = anchor_c
                if 0 <= r < H and 0 <= c < W and out[r, c] in (0, color):
                    out[r, c] = color
                elif not (0 <= r < H and 0 <= c < W):
                    break
                else:
                    break

        elif side == "L":
            col = cmin
            pts = [r for r, c in container_cells if c == col]
            center = (rmin + rmax) / 2.0
            anchor_r = min(pts, key=lambda r: (abs(r - center), abs(r - pr)))
            length = max(c for _, c in payload_cells) - min(c for _, c in payload_cells) + 1
            c0 = cmin - 1
            for i in range(length):
                r = anchor_r
                c = c0 - i
                if 0 <= r < H and 0 <= c < W and out[r, c] in (0, color):
                    out[r, c] = color
                elif not (0 <= r < H and 0 <= c < W):
                    break
                else:
                    break

        else:  # side == "R"
            col = cmax
            pts = [r for r, c in container_cells if c == col]
            center = (rmin + rmax) / 2.0
            anchor_r = min(pts, key=lambda r: (abs(r - center), abs(r - pr)))
            length = max(c for _, c in payload_cells) - min(c for _, c in payload_cells) + 1
            c0 = cmax + 1
            for i in range(length):
                r = anchor_r
                c = c0 + i
                if 0 <= r < H and 0 <= c < W and out[r, c] in (0, color):
                    out[r, c] = color
                elif not (0 <= r < H and 0 <= c < W):
                    break
                else:
                    break

    # Process each 8-connected non-zero component.
    for r in range(H):
        for c in range(W):
            if not visited[r, c] and grid[r, c] != 0:
                cells = component_cells(r, c)

                color_cells = {}
                for rr, cc in cells:
                    col = int(grid[rr, cc])
                    color_cells.setdefault(col, []).append((rr, cc))

                # Leave one-color components unchanged.
                if len(color_cells) < 2:
                    continue

                # Container: most frequent color; use bbox area as tie-breaker.
                def color_score(col):
                    area = 1
                    if color_cells[col]:
                        r1, r2, c1, c2 = bbox(color_cells[col])
                        area = (r2 - r1 + 1) * (c2 - c1 + 1)
                    return (len(color_cells[col]), area, -col)

                container_color = max(color_cells.keys(), key=color_score)
                container_cells = color_cells[container_color]
                payload_colors = [col for col in color_cells if col != container_color]

                # Remove payload cells first.
                for col in payload_colors:
                    for rr, cc in color_cells[col]:
                        out[rr, cc] = 0

                # Draw a line for each payload color.
                for col in sorted(payload_colors, key=lambda x: -len(color_cells[x])):
                    draw_payload_line(container_cells, color_cells[col], col)

    return out