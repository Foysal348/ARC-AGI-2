import numpy as np
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct straight orthogonal paths for each main color.

    The examples follow this pattern:
    - Colors 2 and 4 are endpoint / marker / noise cells.
    - Every other color forms one or more orthogonal line paths.
    - The intended path for a color is the best border-to-border path
      supported by its own cells and marker cells.
    - Where reconstructed paths of different colors overlap, the lower
      numeric color wins.
    """
    grid = np.array(grid, dtype=np.uint8, copy=True)
    H, W = grid.shape
    if H == 0 or W == 0:
        return grid

    special_values = {2, 4}

    # ---------------------------------------------------------------
    # Associate marker cells with nearby main colors.
    # A marker is assigned to a color if, in some straight direction,
    # the first non-marker non-zero cell is that color.
    # ---------------------------------------------------------------
    marker_assoc = defaultdict(set)

    def assoc_marker(r, c):
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            rr, cc = r + dr, c + dc
            while 0 <= rr < H and 0 <= cc < W:
                v = int(grid[rr, cc])
                if v in special_values:
                    rr += dr
                    cc += dc
                    continue
                if v != 0:
                    marker_assoc[(r, c)].add(v)
                break

    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) in special_values:
                assoc_marker(r, c)

    main_colors = sorted({int(v) for v in grid.flat if v != 0 and v not in special_values})

    output = np.zeros_like(grid, dtype=np.uint8)

    # Edge weight helpers.
    # Evidence-to-evidence edges: prefer short gaps.
    # Border-extension edges: prefer reaching the border.
    def edge_weight_between(d):
        return 10 - max(0, d - 1) * 3

    def edge_weight_border(d):
        return 5 + d

    NEG_INF = -10**9

    # ---------------------------------------------------------------
    # Reconstruct the best path for one color.
    # ---------------------------------------------------------------
    def reconstruct_color(color):
        # Evidence cells: true color cells + associated marker cells.
        evidence = set()
        for r in range(H):
            for c in range(W):
                v = int(grid[r, c])
                if v == color:
                    evidence.add((r, c))
                elif v in special_values and color in marker_assoc.get((r, c), ()):
                    evidence.add((r, c))

        if not evidence:
            return None

        node_id = {}
        nodes = []          # (r, c, weight, is_border)
        adj = []            # adjacency list for graph

        def get_node(r, c, is_evidence):
            key = (r, c)
            if key in node_id:
                return node_id[key]
            if is_evidence:
                if int(grid[r, c]) == color:
                    w = 10
                else:
                    w = 100   # markers are strong evidence
            else:
                w = 0
            b = (r == 0 or r == H - 1 or c == 0 or c == W - 1)
            idx = len(nodes)
            node_id[key] = idx
            nodes.append((r, c, w, b))
            adj.append([])
            return idx

        # Add evidence nodes.
        for r, c in evidence:
            get_node(r, c, True)

        # Group evidence by rows and columns.
        by_row = defaultdict(list)
        by_col = defaultdict(list)
        for (r, c), idx in node_id.items():
            by_row[r].append((c, idx))
            by_col[c].append((r, idx))

        # Connect consecutive evidence cells in the same row / column.
        for r, items in by_row.items():
            items.sort()
            for i in range(len(items) - 1):
                c1, u = items[i]
                c2, v = items[i + 1]
                d = c2 - c1
                if d > 1:
                    w = edge_weight_between(d)
                    adj[u].append((v, w))
                    adj[v].append((u, w))

        for c, items in by_col.items():
            items.sort()
            for i in range(len(items) - 1):
                r1, u = items[i]
                r2, v = items[i + 1]
                d = r2 - r1
                if d > 1:
                    w = edge_weight_between(d)
                    adj[u].append((v, w))
                    adj[v].append((u, w))

        # Add border extension nodes from outermost evidence in each line.
        for r, items in by_row.items():
            items.sort()
            if not items:
                continue
            c_left, u_left = items[0]
            if c_left > 0:
                b = get_node(r, 0, False)
                w = edge_weight_border(c_left)
                adj[u_left].append((b, w))
                adj[b].append((u_left, w))

            c_right, u_right = items[-1]
            if c_right < W - 1:
                b = get_node(r, W - 1, False)
                w = edge_weight_border(W - 1 - c_right)
                adj[u_right].append((b, w))
                adj[b].append((u_right, w))

        for c, items in by_col.items():
            items.sort()
            if not items:
                continue
            r_top, u_top = items[0]
            if r_top > 0:
                b = get_node(0, c, False)
                w = edge_weight_border(r_top)
                adj[u_top].append((b, w))
                adj[b].append((u_top, w))

            r_bot, u_bot = items[-1]
            if r_bot < H - 1:
                b = get_node(H - 1, c, False)
                w = edge_weight_border(H - 1 - r_bot)
                adj[u_bot].append((b, w))
                adj[b].append((u_bot, w))

        n = len(nodes)
        if n < 2:
            return None

        # Connected components.
        seen = [False] * n
        components = []
        for s in range(n):
            if seen[s]:
                continue
            stack = [s]
            comp = []
            seen[s] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v, _ in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            components.append(comp)

        best_mask = None
        best_mask_score = NEG_INF

        for comp in components:
            # Need at least two border endpoints.
            border_count = sum(1 for u in comp if nodes[u][3])
            if border_count < 2:
                continue

            comp_set = set(comp)

            # Build a maximum spanning tree of this component.
            # This turns cycles into a tree while keeping stronger edges.
            in_tree = {comp[0]: True}
            tree_adj = {u: [] for u in comp}
            heap = []

            for v, w in adj[comp[0]]:
                if v in comp_set:
                    import heapq
                    heap.append((-w, comp[0], v))

            import heapq
            heapq.heapify(heap)

            while heap and len(in_tree) < len(comp):
                nw, u, v = heapq.heappop(heap)
                w = -nw
                if v in in_tree:
                    continue
                in_tree[v] = True
                tree_adj[u].append((v, w))
                tree_adj[v].append((u, w))
                for x, wx in adj[v]:
                    if x in comp_set and x not in in_tree:
                        heapq.heappush(heap, (-wx, v, x))

            # If Prim failed to connect all nodes (should not happen),
            # fall back to any spanning tree.
            if len(in_tree) < len(comp):
                tree_adj = {u: [] for u in comp}
                stack = [comp[0]]
                in_tree = {comp[0]: True}
                while stack:
                    u = stack.pop()
                    for v, w in adj[u]:
                        if v in comp_set and v not in in_tree:
                            in_tree[v] = True
                            tree_adj[u].append((v, w))
                            tree_adj[v].append((u, w))
                            stack.append(v)

            # Root the tree.
            root = comp[0]
            parent = {root: -1}
            order = []
            stack = [root]
            while stack:
                u = stack.pop()
                order.append(u)
                for v, _ in tree_adj[u]:
                    if v != parent.get(u, -1):
                        parent[v] = u
                        stack.append(v)

            # If tree not connected (rare), use the first connected part.
            if len(order) < len(comp):
                usable = set(order)
                if len(usable) < 2:
                    continue
                # Recompute tree adjacency restricted to usable nodes.
                tree_adj = {u: [(v, w) for (v, w) in tree_adj.get(u, []) if v in usable] for u in usable}

            weight = [nodes[u][2] for u in range(n)]
            is_border = [nodes[u][3] for u in range(n)]

            down_score = {u: NEG_INF for u in tree_adj}
            down_choice = {u: -1 for u in tree_adj}

            global_best = NEG_INF
            global_info = None

            # Post-order DP.
            for u in reversed(order):
                contribs = []
                for v, w in tree_adj[u]:
                    if parent.get(v, -1) == u and down_score[v] > NEG_INF // 2:
                        contribs.append((down_score[v] + w, v, w))

                contribs.sort(reverse=True)

                if is_border[u]:
                    if contribs:
                        best_child_val = max(0, contribs[0][0])
                        if contribs[0][0] > 0:
                            down_score[u] = weight[u] + contribs[0][0]
                            down_choice[u] = contribs[0][1]
                        else:
                            down_score[u] = weight[u]
                            down_choice[u] = -1
                    else:
                        down_score[u] = weight[u]
                        down_choice[u] = -1
                else:
                    if contribs:
                        down_score[u] = weight[u] + contribs[0][0]
                        down_choice[u] = contribs[0][1]
                    else:
                        down_score[u] = NEG_INF
                        down_choice[u] = -1

                # Candidate path whose LCA is u.
                if is_border[u]:
                    if contribs:
                        cand = weight[u] + max(0, contribs[0][0])
                        if cand > global_best:
                            global_best = cand
                            global_info = (u, "border_child", contribs[0][1])
                else:
                    if len(contribs) >= 2:
                        cand = weight[u] + contribs[0][0] + contribs[1][0]
                        if cand > global_best:
                            global_best = cand
                            global_info = (u, "two", contribs[0][1], contribs[1][1])

            if global_info is None:
                continue

            # Recover endpoints.
            def endpoint_from_child(child):
                u = child
                while True:
                    if is_border[u] and down_choice[u] == -1:
                        return u
                    nxt = down_choice[u]
                    if nxt == -1:
                        # Should not happen for a valid branch; fallback to nearest border.
                        return u
                    u = nxt

            def path_from_ancestor(anc, desc):
                path = []
                cur = desc
                while cur != anc:
                    path.append(cur)
                    cur = parent[cur]
                path.append(anc)
                path.reverse()
                return path

            info = global_info
            if info[1] == "border_child":
                u = info[2]
                child = info[3]
                end2 = endpoint_from_child(child)
                seq = path_from_ancestor(u, end2)
            else:
                u = info[2]
                c1, c2 = info[3], info[4]
                e1 = endpoint_from_child(c1)
                e2 = endpoint_from_child(c2)
                p1 = path_from_ancestor(u, e1)  # u -> e1
                p2 = path_from_ancestor(u, e2)  # u -> e2
                seq = p1[::-1] + p2[1:]

            if len(seq) < 2:
                continue

            mask = np.zeros((H, W), dtype=bool)
            for a, b in zip(seq[:-1], seq[1:]):
                r1, c1, _, _ = nodes[a]
                r2, c2, _, _ = nodes[b]
                if r1 == r2:
                    cmin, cmax = sorted((c1, c2))
                    mask[r1, cmin:cmax + 1] = True
                elif c1 == c2:
                    rmin, rmax = sorted((r1, r2))
                    mask[rmin:rmax + 1, c1] = True
                else:
                    # Should not happen in the constructed tree.
                    mask[r1, c1] = True
                    mask[r2, c2] = True

            if mask.sum() > best_mask_score:
                best_mask_score = mask.sum()
                best_mask = mask

        return best_mask

    # ---------------------------------------------------------------
    # Compose final image.
    # Lower colors have priority, so process in increasing color order.
    # ---------------------------------------------------------------
    for color in main_colors:
        mask = reconstruct_color(color)
        if mask is None:
            continue
        for r in range(H):
            for c in range(W):
                if mask[r, c]:
                    # Keep already-painted lower colors.
                    if output[r, c] == 0 or output[r, c] == color:
                        output[r, c] = color

    return output