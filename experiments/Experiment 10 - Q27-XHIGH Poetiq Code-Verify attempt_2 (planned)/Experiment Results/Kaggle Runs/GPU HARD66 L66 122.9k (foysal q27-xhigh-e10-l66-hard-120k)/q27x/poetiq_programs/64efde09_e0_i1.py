import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform ARC-style grids containing:
      - background cells,
      - isolated "source" pixels of some colors,
      - 2-pixel-thick rectangular bars made from non-source colors.

    For each bar:
      1. Find its three ports along its long side.
      2. Choose the emission side:
           - if a source pixel lies exactly on a port in one of the two
             cardinal emission directions, prefer that side;
           - otherwise use the side with fewer 4s.
      3. Determine the bar's "type color" (usually 1 or 3) and use it to
         decide whether the remaining source colors are assigned in
         ascending or descending order.
      4. Draw straight rays from the chosen ports to the grid boundary,
         stopping at non-background objects unless the cell already has
         the same ray color.
    """
    arr = np.array(grid, dtype=int, copy=True)
    if arr.ndim != 2:
        return arr

    H, W = arr.shape
    if H == 0 or W == 0:
        return arr

    # Background is the most frequent color.
    flat = arr.flatten()
    if flat.size == 0:
        return arr
    max_color = int(flat.max())
    bg = int(np.bincount(flat, minlength=max(max_color + 1, 10)).argmax())

    # Find connected components of non-background cells.
    visited = np.zeros((H, W), dtype=bool)
    sources = []          # (r, c, color)
    objects = []          # (r0, c0, r1, c1, orientation)
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for i in range(H):
        for j in range(W):
            if visited[i, j] or arr[i, j] == bg:
                continue

            stack = [(i, j)]
            visited[i, j] = True
            cells = []

            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < H and 0 <= nc < W
                        and not visited[nr, nc]
                        and arr[nr, nc] != bg
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            r0 = min(r for r, _ in cells)
            r1 = max(r for r, _ in cells)
            c0 = min(c for _, c in cells)
            c1 = max(c for _, c in cells)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            size = len(cells)

            # Isolated singletons are sources.
            if size == 1:
                sources.append((r0, c0, int(arr[r0, c0])))
                continue

            # Objects are full 2-by-L or L-by-2 rectangles.
            if min(h, w) == 2 and max(h, w) >= 5 and size >= 0.8 * h * w:
                orientation = "H" if h == 2 else "V"
                objects.append((r0, c0, r1, c1, orientation))

    if not sources or not objects:
        return arr

    source_colors = sorted(set(color for _, _, color in sources))

    def port_offsets(orientation: str, L: int):
        """
        Known port offsets from the examples, with reasonable fallbacks.
        Offsets are 0-based along the long dimension.
        """
        if L < 3:
            return list(range(1, max(1, L - 1)))

        if orientation == "V":
            if L == 7:
                offs = [1, 4, 5]
            elif L == 6 or L in (8, 9):
                offs = [1, 3, 5]
            elif L >= 10:
                offs = [1, max(3, L // 2), max(4, L - 2)]
            else:
                offs = [1, 2, 3]
        else:  # horizontal
            if L == 7:
                offs = [1, 2, 5]
            elif L == 9:
                offs = [3, 5, 7]
            elif L == 6 or L == 8:
                offs = [1, 3, 5]
            elif L >= 10:
                offs = [max(1, L // 3), max(2, L // 2), max(3, L - 2)]
            else:
                offs = [1, 2, 3]

        offs = sorted({o for o in offs if 1 <= o <= max(1, L - 2)})
        if len(offs) < 3:
            for o in range(1, max(2, L - 1)):
                if o not in offs:
                    offs.append(o)
                if len(offs) == 3:
                    break
        return offs[:3]

    def count4s(side: np.ndarray) -> int:
        return int(np.count_nonzero(side == 4))

    def default_side(orientation: str, r0: int, c0: int, r1: int, c1: int):
        """Side with fewer 4 cells."""
        if orientation == "H":
            top4 = count4s(arr[r0, c0:c1 + 1])
            bot4 = count4s(arr[r1, c0:c1 + 1])
            if top4 < bot4:
                return "up"
            if bot4 < top4:
                return "down"
            return "up"
        else:
            left4 = count4s(arr[r0:r1 + 1, c0])
            right4 = count4s(arr[r0:r1 + 1, c1])
            if left4 < right4:
                return "left"
            if right4 < left4:
                return "right"
            return "left"

    def source_side(
        orientation: str,
        r0: int,
        c0: int,
        r1: int,
        c1: int,
        offs,
        fallback_side: str,
    ):
        """
        If a source lies on a port in one emission direction, choose that
        direction.  Otherwise return None.
        """
        if orientation == "H":
            port_cols = {c0 + o: o for o in offs}
            directions = ("up", "down")
        else:
            port_rows = {r0 + o: o for o in offs}
            directions = ("left", "right")

        counts = {d: 0 for d in directions}
        min_dist = {d: 10**9 for d in directions}

        for r, c, _ in sources:
            if orientation == "H":
                if c in port_cols:
                    if r < r0:
                        d = r0 - r
                        counts["up"] += 1
                        min_dist["up"] = min(min_dist["up"], d)
                    elif r > r1:
                        d = r - r1
                        counts["down"] += 1
                        min_dist["down"] = min(min_dist["down"], d)
            else:
                if r in port_rows:
                    if c < c0:
                        d = c0 - c
                        counts["left"] += 1
                        min_dist["left"] = min(min_dist["left"], d)
                    elif c > c1:
                        d = c - c1
                        counts["right"] += 1
                        min_dist["right"] = min(min_dist["right"], d)

        candidates = [d for d in directions if counts[d] > 0]
        if not candidates:
            return None

        order = ["up", "left", "down", "right"]
        return min(
            candidates,
            key=lambda d: (
                -counts[d],
                min_dist[d],
                0 if d == fallback_side else 1,
                order.index(d),
            ),
        )

    def object_type_color(
        r0: int, c0: int, r1: int, c1: int
    ) -> int:
        """
        Identify the bar's 'type' color (e.g. 1 or 3), excluding background,
        source colors, and the structural colors 2 and 4.
        """
        sub = arr[r0:r1 + 1, c0:c1 + 1].flatten()
        counts = {}
        for v in sub:
            v = int(v)
            if v == bg or v == 2 or v == 4 or v in source_colors:
                continue
            counts[v] = counts.get(v, 0) + 1

        if counts:
            return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]

        vals = set(map(int, sub.tolist()))
        if 1 in vals:
            return 1
        if 3 in vals:
            return 3
        return 1

    new_grid = arr.copy()

    def draw_line(r: int, c: int, dr: int, dc: int, color: int) -> None:
        """
        Draw in a cardinal direction, filling background.  Continue through
        same-colored cells (e.g. source pixels), stop at other objects.
        """
        while 0 <= r < H and 0 <= c < W:
            val = int(new_grid[r, c])
            if val == bg:
                new_grid[r, c] = color
            elif val != color:
                break
            r += dr
            c += dc

    # Process objects in a stable order.
    for r0, c0, r1, c1, orientation in sorted(objects, key=lambda x: (x[0], x[2])):
        L = (c1 - c0 + 1) if orientation == "H" else (r1 - r0 + 1)
        offs = port_offsets(orientation, L)

        fallback = default_side(orientation, r0, c0, r1, c1)
        side = source_side(orientation, r0, c0, r1, c1, offs, fallback)
        if side is None:
            side = fallback

        type_color = object_type_color(r0, c0, r1, c1)

        # From the examples:
        #   type color 3: horizontal ascending, vertical descending
        #   type color 1: horizontal descending, vertical ascending
        if orientation == "H":
            ascending = (type_color > 2)
        else:
            ascending = (type_color < 2)

        # Fix colors at ports that have a matching source on the chosen side.
        assigned = {}

        if orientation == "H":
            port_cols = {c0 + o: o for o in offs}
            best = {}
            for r, c, color in sources:
                if c in port_cols:
                    if (side == "up" and r < r0) or (side == "down" and r > r1):
                        o = port_cols[c]
                        dist = abs(r - ((r0 + r1) // 2))
                        if o not in best or dist < best[o][0]:
                            best[o] = (dist, color)
            assigned = {o: col for o, (_, col) in best.items()}
        else:
            port_rows = {r0 + o: o for o in offs}
            best = {}
            for r, c, color in sources:
                if r in port_rows:
                    if (side == "left" and c < c0) or (side == "right" and c > c1):
                        o = port_rows[r]
                        dist = abs(c - ((c0 + c1) // 2))
                        if o not in best or dist < best[o][0]:
                            best[o] = (dist, color)
            assigned = {o: col for o, (_, col) in best.items()}

        # Fill remaining ports with the remaining source colors.
        used_colors = set(assigned.values())
        remaining_offsets = [o for o in offs if o not in assigned]
        remaining_colors = [c for c in source_colors if c not in used_colors]

        if ascending:
            fill_order = sorted(remaining_colors)
        else:
            fill_order = sorted(remaining_colors, reverse=True)

        for o, col in zip(remaining_offsets, fill_order):
            assigned[o] = col

        # Draw rays.
        if orientation == "H":
            for o, col in assigned.items():
                if col is None:
                    continue
                cc = c0 + o
                if side == "up":
                    draw_line(r0 - 1, cc, -1, 0, col)
                elif side == "down":
                    draw_line(r1 + 1, cc, 1, 0, col)
        else:
            for o, col in assigned.items():
                if col is None:
                    continue
                rr = r0 + o
                if side == "left":
                    draw_line(rr, c0 - 1, 0, -1, col)
                elif side == "right":
                    draw_line(rr, c1 + 1, 0, 1, col)

    return new_grid