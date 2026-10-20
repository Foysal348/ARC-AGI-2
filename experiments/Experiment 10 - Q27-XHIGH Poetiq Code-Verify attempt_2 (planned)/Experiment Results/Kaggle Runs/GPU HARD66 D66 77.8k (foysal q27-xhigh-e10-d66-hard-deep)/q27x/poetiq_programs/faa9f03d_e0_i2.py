import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct one orthogonal path per real colour.

    Observed rule from the examples:
    - Colours 2 and 4 are "placeholder / damaged" cells, not final colours.
    - Every other non-zero colour belongs to one true 1-pixel-wide orthogonal
      path whose endpoints lie on the grid border.
    - In the input, parts of the true path may be missing (0), damaged (2/4),
      or hidden at crossings by other colours.
    - Decoy fragments of the same colour may also appear.
    - The intended path is the border-to-border path with the best evidence
      score: visible cells are rewarded, placeholders are strongly rewarded,
      and long unsupported zero stretches are penalised.
    - If reconstructed paths cross, the smaller colour number is shown.

    The solver works independently for each real colour:
    1. Collect anchor cells: visible cells of the colour and relevant
       placeholder cells.
    2. Add plausible border endpoint cells.
    3. Build a graph whose nodes are anchors / border endpoints.  Edges connect
       consecutive nodes in the same row/column; the edge stores the score of
       the cells strictly between them.
    4. If the graph is a forest, exactly find the best border-to-border path
       by DFS from each border node.  If cycles are present, fall back to a
       maximum-score spanning forest and then do the same DFS.
    5. Expand the selected path to all its grid cells.
    6. Overlay colours from highest to lowest so smaller colours win crossings.
    """
    grid = np.asarray(grid, dtype=int)
    H, W = grid.shape
    if H == 0 or W == 0:
        return grid.copy()

    placeholders = {2, 4}

    # Real colours are all non-zero colours except the placeholder colours.
    real_colors = [int(c) for c in np.unique(grid) if c != 0 and c not in placeholders]

    # Scoring constants.
    VISIBLE_SCORE = 10
    PLACEHOLDER_SCORE = 20
    ZERO_SCORE = -1
    OTHER_SCORE = -5

    # A candidate path must contain a reasonable amount of evidence.
    MIN_ANCHOR = 3
    MIN_VISIBLE = 2

    def cell_categories(val):
        """Return (score, anchor, visible, placeholder) for one cell value."""
        if val == c:
            return VISIBLE_SCORE, 1, 1, 0
        if val in placeholders:
            return PLACEHOLDER_SCORE, 1, 0, 1
        if val == 0:
            return ZERO_SCORE, 0, 0, 0
        return OTHER_SCORE, 0, 0, 0

    def reconstruct_color(c):
        """Reconstruct the selected path for colour c as a set of (r, col)."""
        # Visible cells of this colour.
        base_nodes = set()
        row_base_cols = [[] for _ in range(H)]
        col_base_rows = [[] for _ in range(W)]

        for r in range(H):
            for cc in range(W):
                if grid[r, cc] == c:
                    base_nodes.add((r, cc))
                    row_base_cols[r].append(cc)
                    col_base_rows[cc].append(r)

        if not base_nodes:
            return set()

        row_base_count = [len(x) for x in row_base_cols]
        col_base_count = [len(x) for x in col_base_rows]

        # Placeholder relevance: a placeholder is useful if it is close to a
        # row/column that already contains at least two visible cells, or if
        # it lies between such visible cells.
        max_ext = max(5, min(H, W) // 2)
        nodes = set(base_nodes)

        for r in range(H):
            for cc in range(W):
                val = grid[r, cc]
                if val not in placeholders:
                    continue

                relevant = False

                if row_base_count[r] >= 2:
                    cols = row_base_cols[r]
                    mn, mx = min(cols), max(cols)
                    nearest = min(abs(cc - x) for x in cols)
                    if mn <= cc <= mx or nearest <= max_ext:
                        relevant = True

                if not relevant and col_base_count[cc] >= 2:
                    rows = col_base_rows[cc]
                    mn, mx = min(rows), max(rows)
                    nearest = min(abs(r - x) for x in rows)
                    if mn <= r <= mx or nearest <= max_ext:
                        relevant = True

                if relevant:
                    nodes.add((r, cc))

        # Anchor counts used to justify zero/other-colour border endpoints.
        row_anchor_count = [0] * H
        col_anchor_count = [0] * W
        for (r, cc) in nodes:
            if grid[r, cc] == c or grid[r, cc] in placeholders:
                row_anchor_count[r] += 1
                col_anchor_count[cc] += 1

        # Add plausible border endpoints.  Existing anchor border cells are
        # already present.  Zero/other border cells are added only when the
        # row or column has at least one anchor in the same line.
        anchor_nodes = set(nodes)
        for r in range(H):
            for cc in (0, W - 1):
                if (r, cc) not in anchor_nodes and (
                    row_anchor_count[r] >= 1 or col_anchor_count[cc] >= 1
                ):
                    nodes.add((r, cc))

        for cc in range(W):
            for r in (0, H - 1):
                if (r, cc) not in anchor_nodes and (
                    row_anchor_count[r] >= 1 or col_anchor_count[cc] >= 1
                ):
                    nodes.add((r, cc))

        if not nodes:
            return set()

        nodes = sorted(nodes)
        node_id = {node: i for i, node in enumerate(nodes)}
        N = len(nodes)

        node_score = [0] * N
        node_anchor = [0] * N
        node_visible = [0] * N
        node_placeholder = [0] * N
        is_border = [False] * N

        for i, (r, cc) in enumerate(nodes):
            s, a, v, p = cell_categories(grid[r, cc])
            node_score[i] = s
            node_anchor[i] = a
            node_visible[i] = v
            node_placeholder[i] = p
            is_border[i] = (r == 0 or r == H - 1 or cc == 0 or cc == W - 1)

        adj = [[] for _ in range(N)]
        edge_set = set()

        def add_edge(u, v):
            if u == v:
                return
            key = (u, v) if u < v else (v, u)
            if key in edge_set:
                return
            edge_set.add(key)

            r1, c1 = nodes[u]
            r2, c2 = nodes[v]
            dist = abs(r1 - r2) + abs(c1 - c2)

            gap_score = 0
            gap_anchor = 0
            gap_visible = 0
            gap_placeholder = 0

            if r1 == r2:
                lo, hi = sorted((c1, c2))
                for cc in range(lo + 1, hi):
                    s, a, v, p = cell_categories(grid[r1, cc])
                    gap_score += s
                    gap_anchor += a
                    gap_visible += v
                    gap_placeholder += p
            else:
                lo, hi = sorted((r1, r2))
                for rr in range(lo + 1, hi):
                    s, a, v, p = cell_categories(grid[rr, c1])
                    gap_score += s
                    gap_anchor += a
                    gap_visible += v
                    gap_placeholder += p

            # Edge stores gap only; endpoint scores are added when entering it.
            adj[u].append((v, dist, gap_score, gap_anchor, gap_visible, gap_placeholder))
            adj[v].append((u, dist, gap_score, gap_anchor, gap_visible, gap_placeholder))

        # Connect consecutive nodes in every row and column.
        by_row = {}
        by_col = {}
        for i, (r, cc) in enumerate(nodes):
            by_row.setdefault(r, []).append((cc, i))
            by_col.setdefault(cc, []).append((r, i))

        for r, items in by_row.items():
            items.sort()
            for (_, u), (_, v) in zip(items[:-1], items[1:]):
                add_edge(u, v)

        for cc, items in by_col.items():
            items.sort()
            for (_, u), (_, v) in zip(items[:-1], items[1:]):
                add_edge(u, v)

        border_ids = [i for i in range(N) if is_border[i]]
        if len(border_ids) < 2:
            return set()

        # Check whether the graph is a forest.
        visited = [False] * N
        cyclic = False
        for i in range(N):
            if visited[i]:
                continue
            stack = [i]
            visited[i] = True
            comp_nodes = 0
            comp_edges = 0
            while stack:
                u = stack.pop()
                comp_nodes += 1
                for v, *_ in adj[u]:
                    comp_edges += 1
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            comp_edges //= 2
            if comp_edges != comp_nodes - 1:
                cyclic = True
                break

        # If cycles exist, fall back to a maximum-score spanning forest.
        work_adj = adj
        if cyclic:
            parent = list(range(N))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra == rb:
                    return False
                parent[rb] = ra
                return True

            edges = []
            for u in range(N):
                for v, dist, gs, ga, gv, gp in adj[u]:
                    if u < v:
                        # Symmetric edge score used only for choosing the tree.
                        gain = gs + node_score[u] + node_score[v]
                        edges.append((gain, u, v, dist, gs, ga, gv, gp))

            edges.sort(key=lambda x: x[0], reverse=True)

            work_adj = [[] for _ in range(N)]
            for gain, u, v, dist, gs, ga, gv, gp in edges:
                if union(u, v):
                    work_adj[u].append((v, dist, gs, ga, gv, gp))
                    work_adj[v].append((u, dist, gs, ga, gv, gp))

        best_tuple = None
        best_path = None

        # In a forest, unique path from start to every node.
        for start in border_ids:
            par = [-1] * N
            score = [0] * N
            anchor = [0] * N
            visible = [0] * N
            placeholder = [0] * N
            length = [0] * N

            par[start] = -2
            score[start] = node_score[start]
            anchor[start] = node_anchor[start]
            visible[start] = node_visible[start]
            placeholder[start] = node_placeholder[start]

            stack = [start]
            while stack:
                u = stack.pop()

                if is_border[u] and u != start:
                    if (
                        anchor[u] >= MIN_ANCHOR
                        and visible[u] >= MIN_VISIBLE
                        and score[u] > 0
                    ):
                        tup = (
                            score[u],
                            placeholder[u],
                            anchor[u],
                            visible[u],
                            -length[u],
                        )
                        if best_tuple is None or tup > best_tuple:
                            best_tuple = tup
                            path = []
                            cur = u
                            while cur != -1:
                                path.append(cur)
                                if cur == start:
                                    break
                                cur = par[cur]
                            best_path = path

                for v, dist, gs, ga, gv, gp in work_adj[u]:
                    if par[v] != -1:
                        continue
                    par[v] = u
                    score[v] = score[u] + gs + node_score[v]
                    anchor[v] = anchor[u] + ga + node_anchor[v]
                    visible[v] = visible[u] + gv + node_visible[v]
                    placeholder[v] = placeholder[u] + gp + node_placeholder[v]
                    length[v] = length[u] + dist
                    stack.append(v)

        if best_path is None or len(best_path) < 2:
            return set()

        mask = set()
        for a, b in zip(best_path[:-1], best_path[1:]):
            r1, c1 = nodes[a]
            r2, c2 = nodes[b]
            if r1 == r2:
                lo, hi = sorted((c1, c2))
                for cc in range(lo, hi + 1):
                    mask.add((r1, cc))
            else:
                lo, hi = sorted((r1, r2))
                for rr in range(lo, hi + 1):
                    mask.add((rr, c1))

        return mask

    masks = {}
    for c in real_colors:
        mask = reconstruct_color(c)
        if mask:
            masks[c] = mask

    # Overlay higher colours first so smaller colours win at crossings.
    out = np.zeros((H, W), dtype=int)
    for c in sorted(masks.keys(), reverse=True):
        for r, cc in masks[c]:
            if 0 <= r < H and 0 <= cc < W:
                out[r, cc] = c

    return out