import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    ARC-style restoration of thin orthogonal lines.

    Interpretation used by the examples:
    - Real line colours are all non-zero colours except the placeholder colours 2 and 4.
    - Each real colour forms one continuous 1-pixel-wide orthogonal path.
    - The two ends of the path touch the grid border.
    - Missing path cells may be 0, placeholder 2/4, or a higher-priority colour.
    - When paths cross, the smaller numeric colour is shown.

    The solver:
    1. For each real colour, builds a weighted graph of likely path cells.
    2. Finds the best border-to-border orthogonal path for that colour.
    3. Expands that path to full cells.
    4. Paints all colour paths, letting the lowest colour win at overlaps.
    """
    grid = np.asarray(grid, dtype=int)
    H, W = grid.shape
    if H == 0 or W == 0:
        return grid.copy()

    placeholder_colors = {2, 4}
    real_colors = [c for c in np.unique(grid) if c != 0 and c not in placeholder_colors]

    path_masks = {c: set() for c in real_colors}

    # ---------- helpers ----------
    def is_border(r, c):
        return r == 0 or r == H - 1 or c == 0 or c == W - 1

    def cell_score(r, c, color):
        """Score for visiting this cell by the current colour."""
        val = grid[r, c]
        if val == color:
            s = 1
        elif val in placeholder_colors:
            s = 20
        else:
            s = -1
        if is_border(r, c):
            s += 5
        return s

    def relevant_placeholder_for_color(r, c, color):
        """
        A placeholder is allowed into colour c's graph if it is close to c
        in its row/column, or lies between existing c cells in that row/column.
        """
        # Close in the same row / column.
        for nc in range(max(0, c - 6), min(W, c + 7)):
            if nc != c and grid[r, nc] == color:
                return True
        for nr in range(max(0, r - 6), min(H, r + 7)):
            if nr != r and grid[nr, c] == color:
                return True

        # Between existing cells in the same row / column.
        row_cols = [cc for cc in range(W) if grid[r, cc] == color]
        if row_cols and min(row_cols) <= c <= max(row_cols):
            return True

        col_rows = [rr for rr in range(H) if grid[rr, c] == color]
        if col_rows and min(col_rows) <= r <= max(col_rows):
            return True

        return False

    def direct_edge_weight(a, b, color):
        """Weight for moving from a to b, counting all cells after a up to b."""
        r1, c1 = a
        r2, c2 = b
        dr = 0 if r1 == r2 else (1 if r2 > r1 else -1)
        dc = 0 if c1 == c2 else (1 if c2 > c1 else -1)

        weight = 0
        r, c = r1 + dr, c1 + dc
        while (r, c) != (r1, c1):
            weight += cell_score(r, c, color)
            if (r, c) == (r2, c2):
                break
            r += dr
            c += dc
        return weight

    def add_directed_edge(adj, a, b, w):
        adj[a][b] = w
        adj[b].setdefault(a, None)

    def remove_node(adj, v):
        for u in list(adj[v].keys()):
            if u in adj:
                adj[u].pop(v, None)
        adj.pop(v, None)

    def straight_line_cells(a, b):
        r1, c1 = a
        r2, c2 = b
        if r1 == r2:
            step = 1 if c2 >= c1 else -1
            return [(r1, c) for c in range(c1, c2 + step, step)]
        elif c1 == c2:
            step = 1 if r2 >= r1 else -1
            return [(r, c1) for r in range(r1, r2 + step, step)]
        else:
            # Should not happen for compressed straight edges.
            return [a, b]

    # ---------- per-colour path solving ----------
    for color in real_colors:
        # 1. Base key cells.
        key_cells = set()
        for r in range(H):
            for c in range(W):
                if grid[r, c] == color:
                    key_cells.add((r, c))
                elif grid[r, c] in placeholder_colors:
                    if relevant_placeholder_for_color(r, c, color):
                        key_cells.add((r, c))

        if not key_cells:
            continue

        nodes = set(key_cells)

        # 2. Add plausible border endpoints.
        max_border_ext = max(10, max(H, W) // 2)

        row_cols = {}
        col_rows = {}
        for (r, c) in key_cells:
            row_cols.setdefault(r, []).append(c)
            col_rows.setdefault(c, []).append(r)

        for r, cols in row_cols.items():
            mn, mx = min(cols), max(cols)
            if mn <= max_border_ext:
                nodes.add((r, 0))
            if (W - 1 - mx) <= max_border_ext:
                nodes.add((r, W - 1))

        for c, rows in col_rows.items():
            mn, mx = min(rows), max(rows)
            if mn <= max_border_ext:
                nodes.add((0, c))
            if (H - 1 - mx) <= max_border_ext:
                nodes.add((H - 1, c))

        # 3. Initial graph: connect consecutive nodes in rows/columns.
        max_gap = 8
        adj = {v: {} for v in nodes}

        def connect(a, b):
            if a == b:
                return
            dr = abs(a[0] - b[0])
            dc = abs(a[1] - b[1])
            if dr != 0 and dc != 0:
                return
            dist = dr + dc
            if dist == 0:
                return

            a_border = is_border(*a)
            b_border = is_border(*b)
            limit = max_border_ext if (a_border or b_border) else max_gap
            if dist > limit:
                return

            w_ab = direct_edge_weight(a, b, color)
            w_ba = direct_edge_weight(b, a, color)
            # Keep the better parallel edge if one already exists.
            if b in adj[a]:
                w_ab = max(w_ab, adj[a][b])
            if a in adj[b]:
                w_ba = max(w_ba, adj[b][a])
            adj[a][b] = w_ab
            adj[b][a] = w_ba

        # Rows.
        row_nodes = {}
        for (r, c) in nodes:
            row_nodes.setdefault(r, []).append(c)
        for r, cols in row_nodes.items():
            cols = sorted(cols)
            for c1, c2 in zip(cols[:-1], cols[1:]):
                connect((r, c1), (r, c2))

        # Columns.
        col_nodes = {}
        for (r, c) in nodes:
            col_nodes.setdefault(c, []).append(r)
        for c, rows in col_nodes.items():
            rows = sorted(rows)
            for r1, r2 in zip(rows[:-1], rows[1:]):
                connect((r1, c), (r2, c))

        # Drop isolated nodes.
        nodes = set(v for v in adj if adj[v])
        adj = {v: adj[v] for v in nodes}

        # 4. Prune non-border dead ends.
        while True:
            dead = [v for v in adj if not is_border(*v) and len(adj[v]) == 1]
            if not dead:
                break
            for v in dead:
                remove_node(adj, v)

        # 5. Compress straight degree-2 non-border nodes.
        changed = True
        while changed:
            changed = False
            for v in list(adj.keys()):
                if not adj[v]:
                    continue
                if is_border(*v) or len(adj[v]) != 2:
                    continue

                u, w = list(adj[v].items())
                a = u
                b = w
                # The two neighbours must be in the list values, not the keys.
                neigh = list(adj[v].keys())
                if len(neigh) != 2:
                    continue
                a, b = neigh

                # Only suppress collinear chains, never turns.
                r, c = v
                if not ((a[0] == r == b[0]) or (a[1] == c == b[1])):
                    continue

                wa = adj[v][a]
                wb = adj[v][b]
                w_ab = wa + adj[a][v]
                w_ba = wb + adj[b][v]

                # Remove v.
                remove_node(adj, v)

                # Add combined edge, keeping better parallel edge.
                if b in adj[a]:
                    w_ab = max(w_ab, adj[a][b])
                if a in adj[b]:
                    w_ba = max(w_ba, adj[b][a])

                if w_ab > -10**9:
                    adj[a][b] = w_ab
                if w_ba > -10**9:
                    adj[b][a] = w_ba

                changed = True

        if not adj:
            continue

        # 6. Find best border-to-border simple path.
        border_nodes = [v for v in adj if is_border(*v) and adj[v]]
        if len(border_nodes) < 2:
            # Fallback: keep original visible colour cells if no valid path is found.
            for r in range(H):
                for c in range(W):
                    if grid[r, c] == color:
                        path_masks[color].add((r, c))
            continue

        # Small graphs: exact DFS with light pruning.
        n_nodes = len(adj)
        best_score = -10**18
        best_path = None

        # Positive total edge score is a loose upper bound.
        total_positive = 0
        for u in adj:
            for v, w in adj[u].items():
                if w > 0:
                    total_positive += w

        # Start from each border node.
        for start in border_nodes:
            stack = [(start, {start}, cell_score(*start, color), [start])]
            while stack:
                u, visited, score, path = stack.pop()

                if len(path) > 1 and is_border(*u) and score > best_score:
                    best_score = score
                    best_path = path

                # Loose branch-and-bound.
                if score + total_positive <= best_score:
                    continue

                # Neighbours in descending immediate gain first.
                neighbours = sorted(adj[u].items(), key=lambda x: x[1], reverse=True)
                for v, w in neighbours:
                    if v not in visited:
                        nvis = visited | {v}
                        npath = path + [v]
                        nsc = score + w
                        stack.append((v, nvis, nsc, npath))

        # If DFS failed, fallback to visible cells.
        if best_path is None:
            for r in range(H):
                for c in range(W):
                    if grid[r, c] == color:
                        path_masks[color].add((r, c))
            continue

        # 7. Expand selected path.
        for a, b in zip(best_path[:-1], best_path[1:]):
            if a in adj and b in adj[a]:
                for cell in straight_line_cells(a, b):
                    path_masks[color].add(cell)
            else:
                # Should not happen, but keep endpoints safe.
                path_masks[color].add(a)
                path_masks[color].add(b)

    # ---------- compose final grid with lower colour priority ----------
    out = np.zeros((H, W), dtype=int)

    for color in sorted(path_masks.keys()):
        for (r, c) in path_masks[color]:
            if 0 <= r < H and 0 <= c < W:
                if out[r, c] == 0 or color < out[r, c]:
                    out[r, c] = color

    return out