import numpy as np
from collections import Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Solve the arrow/color-transfer ARC task.

    Rules inferred from the examples:
      * 1s form small T-shaped arrow markers.
      * An arrow has three orthogonal 1-neighbors around a center.
        The arrow points in the direction of the single 1-arm.
      * If the arrow center is not 1, the source color is the center color.
        If the arrow center is 1, the source color is the color of the
        region on the tail side (opposite the arrow direction).
      * The target is the first non-1 region reached by moving in the
        arrow direction.
      * The target region is repainted with the source color.
      * If several arrows target the same region, the arrow with the larger
        source region wins.
      * Non-square grids are rotated 90 degrees clockwise in the final output.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    if H == 0 or W == 0:
        return g

    def inb(r, c):
        return 0 <= r < H and 0 <= c < W

    def first_non1(r, c, dr, dc):
        while inb(r, c):
            if g[r, c] != 1:
                return r, c
            r += dr
            c += dc
        return None

    def region_seed(r, c, dr, dc):
        p = first_non1(r, c, dr, dc)
        return int(g[p[0], p[1]]) if p else None

    def component_id_from_seed(r, c):
        if not inb(r, c):
            return None
        color = int(g[r, c])
        if color == 1:
            return None

        # 4-connected component of the same color.
        seen = np.zeros((H, W), dtype=bool)
        stack = [(r, c)]
        seen[r, c] = True
        comp = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = cr + dr, cc + dc
                if inb(nr, nc) and not seen[nr, nc] and g[nr, nc] == color:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        return comp

    arrows = []

    # Detect T-shaped arrows.
    # The three 1s determine the direction: the lone arm is the pointing side.
    for r in range(H):
        for c in range(W):
            # For each possible direction, check whether the three needed
            # neighboring cells are all 1 and the fourth (tail) is not 1.
            for dname, (dr, dc) in [
                ("up", (-1, 0)),
                ("down", (1, 0)),
                ("left", (0, -1)),
                ("right", (0, 1)),
            ]:
                if dname == "up":
                    arm = (r - 1, c)
                    bar = [(r, c - 1), (r, c + 1)]
                    tail = (r + 1, c)
                elif dname == "down":
                    arm = (r + 1, c)
                    bar = [(r, c - 1), (r, c + 1)]
                    tail = (r - 1, c)
                elif dname == "left":
                    arm = (r, c - 1)
                    bar = [(r - 1, c), (r + 1, c)]
                    tail = (r, c + 1)
                else:  # right
                    arm = (r, c + 1)
                    bar = [(r - 1, c), (r + 1, c)]
                    tail = (r, c - 1)

                ok = inb(*arm) and g[arm] == 1
                if ok:
                    for br, bc in bar:
                        if not (inb(br, bc) and g[br, bc] == 1):
                            ok = False
                            break
                if not ok:
                    continue

                # The tail should not also be a 1.
                if inb(*tail) and g[tail] == 1:
                    continue

                center = int(g[r, c])

                # Determine source color.
                if center != 1:
                    source_color = center
                    source_cells = component_id_from_seed(r, c)
                    if source_cells is None:
                        source_cells = [(r, c)]
                else:
                    src = region_seed(*tail, -dr, -dc)
                    if src is None:
                        continue
                    source_color = src
                    source_cells = component_id_from_seed(*tail)
                    if source_cells is None:
                        source_cells = [tail]

                # Determine target region seed by moving in arrow direction.
                tgt = first_non1(*arm, dr, dc)
                if tgt is None:
                    continue

                target_seed = tgt
                target_color = int(g[tgt])

                # Ignore arrows whose source and target are already the same color.
                if source_color == target_color:
                    continue

                arrows.append({
                    "dir": dname,
                    "source_color": int(source_color),
                    "source_size": len(source_cells),
                    "target_seed": target_seed,
                })

    # Build output.
    out = g.copy()

    # Apply color transfers. Larger source regions have priority.
    # We do it in-place on non-1 cells; target regions are identified by
    # connected components of the original grid color at the target seed.
    # For each target component, remember the winning source color.
    target_win = {}

    # Precompute target components for each arrow.
    for a in arrows:
        tr, tc = a["target_seed"]
        tcolor = int(g[tr, tc])
        comp = component_id_from_seed(tr, tc)
        if comp is None:
            continue

        # Use component tuple as key.
        key = frozenset(comp)
        # Keep arrow with maximum source size.
        if key not in target_win or a["source_size"] > target_win[key][1]:
            target_win[key] = (a["source_color"], a["source_size"])

    # Repaint all target components.
    for comp, (color, _) in target_win.items():
        for r, c in comp:
            if g[r, c] != 1:
                out[r, c] = color

    # Replace remaining 1s.
    # Fill each 1 with the dominant neighboring non-1 color in the transformed grid.
    for r in range(H):
        for c in range(W):
            if out[r, c] != 1:
                continue

            neigh = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)):
                nr, nc = r + dr, c + dc
                if inb(nr, nc) and out[nr, nc] != 1:
                    neigh.append(int(out[nr, nc]))

            if neigh:
                out[r, c] = Counter(neigh).most_common(1)[0][0]
            else:
                out[r, c] = 0

    # Non-square inputs are rotated clockwise in the provided examples.
    if H != W:
        out = np.rot90(out, k=-1)

    return out