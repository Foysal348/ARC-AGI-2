import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Arrange separated tiles into a straight line by connecting matching border stubs.

    Tiles are separated by full rows/columns of 6. For each tile, we look at the
    non-background pixels and record where they touch the tile border. Then we
    find a sequence of all tiles such that consecutive tiles have matching
    border coordinates:
      - vertical output: bottom -> top
      - horizontal output: right -> left
    """
    g = np.array(grid, dtype=int)

    if g.ndim != 2:
        return g.copy()

    H, W = g.shape
    if H == 0 or W == 0:
        return g.copy()

    # Most frequent non-separator color is treated as the background.
    vals, counts = np.unique(g, return_counts=True)
    non_sep = vals != 6
    if non_sep.any():
        bg = int(vals[non_sep][np.argmax(counts[non_sep])])
    else:
        bg = 0

    def make_intervals(lines, length):
        """Return non-empty intervals between separator lines."""
        bounds = [0] + list(lines) + [length]
        intervals = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            start = a + 1
            end = b - 1
            if start <= end:
                intervals.append((start, end))
        return intervals

    # Separator lines are full rows or full columns of 6.
    sep_rows = [r for r in range(H) if np.all(g[r] == 6)]
    sep_cols = [c for c in range(W) if np.all(g[:, c] == 6)]

    row_intervals = make_intervals(sep_rows, H) or [(0, H - 1)]
    col_intervals = make_intervals(sep_cols, W) or [(0, W - 1)]

    panels = []

    # Extract each tile and its border endpoints.
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            sub = g[r0:r1 + 1, c0:c1 + 1]
            if sub.size == 0:
                continue

            mask = (sub != bg) & (sub != 6)
            if not mask.any():
                continue

            h, w = mask.shape

            top = set(np.where(mask[0, :])[0].tolist())
            bottom = set(np.where(mask[-1, :])[0].tolist())
            left = set(np.where(mask[:, 0])[0].tolist())
            right = set(np.where(mask[:, -1])[0].tolist())

            panels.append({
                "grid": sub,
                "endpoints": {
                    "top": top,
                    "bottom": bottom,
                    "left": left,
                    "right": right,
                }
            })

    n = len(panels)

    if n == 0:
        return g.copy()

    if n == 1:
        return panels[0]["grid"].copy()

    def build_graph(out_side, in_side):
        """
        Directed graph edge i -> j if tile i has an endpoint on out_side
        matching an endpoint on tile j's in_side.
        """
        adj = [set() for _ in range(n)]
        for i in range(n):
            out_set = panels[i]["endpoints"][out_side]
            if not out_set:
                continue
            for j in range(n):
                if i == j:
                    continue
                if out_set & panels[j]["endpoints"][in_side]:
                    adj[i].add(j)
        return [sorted(a) for a in adj]

    def find_path(adj):
        """Find a Hamiltonian path in a small directed graph."""
        m = len(adj)
        if m == 0:
            return None
        if m == 1:
            return [0]

        indeg = [0] * m
        for i in range(m):
            for j in adj[i]:
                indeg[j] += 1

        outdeg = [len(adj[i]) for i in range(m)]
        total_deg = [indeg[i] + outdeg[i] for i in range(m)]

        # Prefer starts with no incoming edges, then low-degree nodes.
        start_order = sorted(range(m), key=lambda i: (indeg[i] != 0, total_deg[i], i))

        def dfs(start):
            path = [start]
            visited = [False] * m
            visited[start] = True

            def rec(u):
                if len(path) == m:
                    return path[:]

                candidates = [v for v in adj[u] if not visited[v]]
                candidates.sort(key=lambda v: (total_deg[v], indeg[v], v))

                for v in candidates:
                    visited[v] = True
                    path.append(v)

                    res = rec(v)
                    if res is not None:
                        return res

                    path.pop()
                    visited[v] = False

                return None

            return rec(start)

        for s in start_order:
            path = dfs(s)
            if path is not None:
                return path

        return None

    # Vertical path: bottom -> top.
    adj_v = build_graph("bottom", "top")
    path_v = find_path(adj_v)

    # Horizontal path: right -> left.
    adj_h = build_graph("right", "left")
    path_h = find_path(adj_h)

    def endpoint_count(idx):
        return sum(len(s) for s in panels[idx]["endpoints"].values())

    choices = []

    if path_v is not None:
        score = (
            sum(len(a) for a in adj_v),
            endpoint_count(path_v[0]),
            endpoint_count(path_v[-1]),
        )
        choices.append(("v", path_v, score))

    if path_h is not None:
        score = (
            sum(len(a) for a in adj_h),
            endpoint_count(path_h[0]),
            endpoint_count(path_h[-1]),
        )
        choices.append(("h", path_h, score))

    if choices:
        # Prefer the more constrained graph, then prefer vertical on ties.
        choices.sort(key=lambda x: (x[2], 0 if x[0] == "v" else 1))
        orientation, path = choices[0][0], choices[0][1]
    else:
        # Fallback for degenerate cases: use the obvious input layout order.
        if len(row_intervals) == 1 and len(col_intervals) > 1:
            orientation = "v"
        elif len(col_intervals) == 1 and len(row_intervals) > 1:
            orientation = "h"
        elif len(col_intervals) >= len(row_intervals):
            orientation = "v"
        else:
            orientation = "h"
        path = list(range(n))

    selected = [panels[i] for i in path]

    if orientation == "v":
        out_h = sum(p["grid"].shape[0] for p in selected) + len(selected) - 1
        out_w = max(p["grid"].shape[1] for p in selected)
        out = np.full((out_h, out_w), bg, dtype=int)

        r = 0
        for i, p in enumerate(selected):
            h, w = p["grid"].shape
            out[r:r + h, :w] = p["grid"]
            r += h

            if i != len(selected) - 1:
                out[r, :] = 6
                r += 1

        return out

    else:
        out_h = max(p["grid"].shape[0] for p in selected)
        out_w = sum(p["grid"].shape[1] for p in selected) + len(selected) - 1
        out = np.full((out_h, out_w), bg, dtype=int)

        c = 0
        for i, p in enumerate(selected):
            h, w = p["grid"].shape
            out[:h, c:c + w] = p["grid"]
            c += w

            if i != len(selected) - 1:
                out[:, c] = 6
                c += 1

        return out