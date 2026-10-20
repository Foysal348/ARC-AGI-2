import numpy as np
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid into a small triangular summary grid.

    Rule inferred from examples
    ---------------------------
    1. Non-zero colors that form rectangular/outline-like shapes are treated
       as "objects". The remaining non-zero color is used as the filler color
       for the upper triangle of the output.

    2. Objects are grouped into connected components using bounding-box
       overlap.

    3. Inside each component:
       - If the component is a simple path, traverse it from the endpoint
         with the smaller area (ties broken by rightmost column).
       - Otherwise, fall back to sorting by right edge.

    4. Components are sorted by:
       - minimum object area in the component
       - first object's right edge
       - larger component first

    5. The final output is an N x N matrix where row i is filled with the
       i-th object color from column 0 through column i, and the rest is the
       filler color.
    """
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid, dtype=int)

    if grid.size == 0:
        return np.array([], dtype=int)

    h, w = grid.shape
    colors = np.unique(grid)
    colors = colors[colors != 0]

    objects = []
    color_counts = {}

    for c in colors:
        mask = (grid == c)
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue

        top, bottom = ys.min(), ys.max()
        left, right = xs.min(), xs.max()
        bh = bottom - top + 1
        bw = right - left + 1
        area = bh * bw

        # Ignore tiny specks.
        if bh < 2 or bw < 2 or len(ys) < 4:
            continue

        perimeter = 2 * bh + 2 * bw - 4
        on_border = ((ys == top) | (ys == bottom) | (xs == left) | (xs == right)).sum()

        # Outline-like shapes have most pixels on the bbox boundary.
        border_ratio = on_border / max(1, len(ys))
        coverage = on_border / max(1, perimeter)

        if border_ratio >= 0.55 and coverage >= 0.45:
            objects.append({
                "color": int(c),
                "top": int(top),
                "bottom": int(bottom),
                "left": int(left),
                "right": int(right),
                "area": int(area),
            })
        else:
            color_counts[int(c)] = color_counts.get(int(c), 0) + len(ys)

    # If no object is detected, gracefully return a zero grid.
    if not objects:
        return np.zeros_like(grid, dtype=int)

    # Filler color: most frequent non-object color, or 0 if none.
    filler = 0
    if color_counts:
        filler = max(color_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

    n = len(objects)

    # Build overlap graph.
    adj = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            a = objects[i]
            b = objects[j]
            if (
                max(a["top"], b["top"]) <= min(a["bottom"], b["bottom"])
                and max(a["left"], b["left"]) <= min(a["right"], b["right"])
            ):
                adj[i].append(j)
                adj[j].append(i)

    # Connected components.
    seen = [False] * n
    components = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        components.append(comp)

    def component_order(comp):
        """Order objects inside one component."""
        if len(comp) == 1:
            return comp[:]

        # If the component is a simple path, follow it.
        degrees = {u: len(adj[u]) for u in comp}
        if all(deg <= 2 for deg in degrees.values()) and sum(1 for d in degrees.values() if d == 1) == 2:
            endpoints = [u for u in comp if degrees[u] == 1]
            a, b = endpoints
            # Start from the smaller-area endpoint.
            if objects[a]["area"] != objects[b]["area"]:
                start = a if objects[a]["area"] < objects[b]["area"] else b
            elif objects[a]["right"] != objects[b]["right"]:
                start = a if objects[a]["right"] < objects[b]["right"] else b
            elif objects[a]["top"] != objects[b]["top"]:
                start = a if objects[a]["top"] < objects[b]["top"] else b
            else:
                start = a if objects[a]["left"] < objects[b]["left"] else b

            order = []
            prev = -1
            cur = start
            while True:
                order.append(cur)
                nxts = [x for x in adj[cur] if x != prev]
                if not nxts:
                    break
                # For a path, at most one unvisited neighbor remains.
                prev, cur = cur, nxts[0]
            if len(order) == len(comp):
                return order

        # Fallback: left-to-right by right edge.
        return sorted(comp, key=lambda i: (
            objects[i]["right"],
            objects[i]["top"],
            objects[i]["area"],
            objects[i]["color"],
        ))

    ordered_components = []
    for comp in components:
        order = component_order(comp)
        min_area = min(objects[i]["area"] for i in comp)
        first_right = objects[order[0]]["right"]
        total_area = sum(objects[i]["area"] for i in comp)
        ordered_components.append((min_area, first_right, -len(comp), total_area, order))

    ordered_components.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    sequence = [obj["color"] for _, _, _, _, order in ordered_components for obj in
                (objects[i] for i in order)]

    # Build final triangular output.
    m = len(sequence)
    out = np.full((m, m), filler, dtype=int)
    for i, c in enumerate(sequence):
        out[i, : i + 1] = c

    return out