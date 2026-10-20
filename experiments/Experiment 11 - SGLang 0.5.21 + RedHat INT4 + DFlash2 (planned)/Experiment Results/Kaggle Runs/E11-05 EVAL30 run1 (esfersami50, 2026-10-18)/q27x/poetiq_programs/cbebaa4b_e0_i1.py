import numpy as np
from collections import defaultdict, deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Assemble separated colored pieces into one connected structure.

    Interpretation:
      - Colors 1..9 (except 2) are rigid colored pieces.
      - Color 2 cells are "ports" attached to nearby pieces.
      - In the output, pieces are translated (not rotated) so that ports
        from different pieces coincide, forming a connected assembly.
      - The color-4 piece is kept in its original position as the anchor.
      - The best assembly is selected by minimizing a few tie-breakers
        (direct color adjacency, bounding-box area, movement from input).
    """
    grid = np.array(grid, dtype=np.uint8, copy=True)
    H, W = grid.shape

    if H == 0 or W == 0:
        return grid

    # ------------------------------------------------------------------
    # 1. Extract rigid colored components (ignore 0 and 2).
    # ------------------------------------------------------------------
    class Piece:
        __slots__ = ("pid", "color", "r0", "c0", "colors", "ports", "input_trans")

    visited = np.zeros((H, W), dtype=bool)
    pieces = []
    coord_to_piece = {}

    def add_piece(color, cells):
        min_r = min(r for r, c in cells)
        min_c = min(c for r, c in cells)
        color_offsets = {(r - min_r, c - min_c) for r, c in cells}
        p = Piece()
        p.pid = len(pieces)
        p.color = color
        p.r0, p.c0 = min_r, min_c
        p.colors = color_offsets
        p.ports = set()
        p.input_trans = (min_r, min_c)
        pieces.append(p)
        for r, c in cells:
            coord_to_piece[(r, c)] = p.pid
        return p

    for r in range(H):
        for c in range(W):
            v = grid[r, c]
            if v != 0 and v != 2 and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                cells = []
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                            if grid[nr, nc] == v:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                add_piece(v, cells)

    if not pieces:
        return np.zeros_like(grid)

    # ------------------------------------------------------------------
    # 2. Assign 2-cells to nearby pieces as local ports.
    # ------------------------------------------------------------------
    def piece_abs_color_cells(p, t):
        return {(t[0] + dr, t[1] + dc) for dr, dc in p.colors}

    def piece_abs_port_cells(p, t):
        return {(t[0] + dr, t[1] + dc) for dr, dc in p.ports}

    for r in range(H):
        for c in range(W):
            if grid[r, c] != 2:
                continue
            neigh = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                pid = coord_to_piece.get((r + dr, c + dc))
                if pid is not None:
                    neigh.append(pid)
            if not neigh:
                continue
            if len(set(neigh)) == 1:
                pid = neigh[0]
            else:
                # Rare fallback: choose the nearest piece if a 2 touches more than one.
                def dist2(pid):
                    p = pieces[pid]
                    return min((r - (p.r0 + dr)) * (r - (p.r0 + dr)) +
                               (c - (p.c0 + dc)) * (c - (p.c0 + dc))
                               for dr, dc in p.colors)
                pid = min(set(neigh), key=dist2)
            p = pieces[pid]
            p.ports.add((r - p.r0, c - p.c0))

    # Remove pieces with no ports if there are too many? Keep them, but they cannot
    # participate in port overlap unless anchor. This problem's instances all have ports.

    # ------------------------------------------------------------------
    # 3. Choose anchor: preferably color 4, otherwise largest/most-ported piece.
    # ------------------------------------------------------------------
    def anchor_key(p):
        return (
            1 if p.color == 4 else 0,
            len(p.ports),
            len(p.colors)
        )
    anchor = max(pieces, key=anchor_key)

    def in_bounds(p, t):
        for dr, dc in p.colors | p.ports:
            rr, cc = t[0] + dr, t[1] + dc
            if rr < 0 or rr >= H or cc < 0 or cc >= W:
                return False
        return True

    # Precompute piece-specific offset sets
    for p in pieces:
        p.offset_union = p.colors | p.ports

    # ------------------------------------------------------------------
    # 4. Search over translations.
    # ------------------------------------------------------------------
    best_score = None
    best_translations = None

    # For fast conflict checks
    def conflict(p, t, placed_colors, placed_ports):
        # Bound check
        if not in_bounds(p, t):
            return True

        p_colors = {(t[0] + dr, t[1] + dc) for dr, dc in p.colors}
        p_ports = {(t[0] + dr, t[1] + dc) for dr, dc in p.ports}

        # Color-color overlap
        if any(c in placed_colors for c in p_colors):
            return True
        # Port-color overlap
        if any(c in placed_ports for c in p_ports):
            return True
        # Color-port overlap
        if any(c in placed_ports for c in p_colors):
            return True
        return False

    def direct_adj_count(p, t, placed_colors):
        """Count 4-adjacencies between new piece colors and already placed colors."""
        cnt = 0
        p_colors = {(t[0] + dr, t[1] + dc) for dr, dc in p.colors}
        for r, c in p_colors:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (r + dr, c + dc) in placed_colors:
                    cnt += 1
        return cnt // 2  # each adjacency counted twice? Actually only new vs placed, counted once per neighbor; no need to halve, but keep stable

    def solution_score(trans):
        # Build final color and port sets, and port sharing counts.
        all_colors = set()
        all_ports = set()
        port_by_coord = defaultdict(set)
        adj_count = 0

        piece_cells = {}
        for p in pieces:
            t = trans[p.pid]
            cells = {(t[0] + dr, t[1] + dc) for dr, dc in p.colors}
            piece_cells[p.pid] = cells
            all_colors |= cells
            for dr, dc in p.ports:
                pr, pc = t[0] + dr, t[1] + dc
                all_ports.add((pr, pc))
                port_by_coord[(pr, pc)].add(p.pid)

        # Direct color adjacency between different pieces
        for i, ci in piece_cells.items():
            for r, c in ci:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (r + dr, c + dc)
                    if n in all_colors:
                        # find if different piece
                        for j, cj in piece_cells.items():
                            if i != j and n in cj:
                                adj_count += 1
                                break
        adj_count //= 2

        # All ports must be shared by at least two pieces
        valid = True
        triple_penalties = 0
        for coord, ids in port_by_coord.items():
            if len(ids) < 2:
                valid = False
            if len(ids) > 2:
                triple_penalties += len(ids) - 2
        if not valid:
            return None

        # Connectivity via shared ports
        parent = {p.pid: p.pid for p in pieces}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        for coord, ids in port_by_coord.items():
            lst = list(ids)
            for x in lst[1:]:
                union(lst[0], x)
        roots = {find(p.pid) for p in pieces}
        connected = (len(roots) <= 1)
        if not connected:
            return None

        # Bounding box over colors and ports
        if not all_colors:
            bbox_area = 0
        else:
            rmin = min(r for r, c in all_colors | all_ports)
            rmax = max(r for r, c in all_colors | all_ports)
            cmin = min(c for r, c in all_colors | all_ports)
            cmax = max(c for r, c in all_colors | all_ports)
            bbox_area = (rmax - rmin + 1) * (cmax - cmin + 1)

        movement = sum((trans[p.pid][0] - p.r0) ** 2 + (trans[p.pid][1] - p.c0) ** 2 for p in pieces)

        # Deterministic tie-breaker
        tie = tuple(sorted((trans[p.pid], p.pid) for p in pieces))

        return (adj_count, bbox_area, movement, triple_penalties, tie)

    # Initial state
    initial_trans = {p.pid: p.input_trans for p in pieces}
    anchor_pid = anchor.pid
    initial_trans[anchor_pid] = (anchor.r0, anchor.c0)

    initial_colors = piece_abs_color_cells(anchor, initial_trans[anchor_pid])
    initial_ports = piece_abs_port_cells(anchor, initial_trans[anchor_pid])

    placed = {anchor_pid}
    translations = dict(initial_trans)
    placed_colors = set(initial_colors)
    placed_ports = set(initial_ports)

    # Candidate generation: a new piece must share at least one port with placed pieces.
    def candidates_for(p, placed_ids):
        cands = set()
        for qpid in placed_ids:
            q = pieces[qpid]
            qt = translations[qpid]
            for pa in p.ports:
                for qb in q.ports:
                    tr = qt[0] + qb[0] - pa[0]
                    tc = qt[1] + qb[1] - pa[1]
                    if in_bounds(p, (tr, tc)):
                        cands.add((tr, tc))
        valid = []
        for t in cands:
            if not conflict(p, t, placed_colors, placed_ports):
                valid.append(t)
        # Order candidates by a quick heuristic: smaller local movement and adjacency
        def key(t):
            move = (t[0] - p.r0) ** 2 + (t[1] - p.c0) ** 2
            adj = direct_adj_count(p, t, placed_colors)
            return (adj, move, t[0], t[1])
        valid.sort(key=key)
        return valid

    # Optional pruning: ensure every currently unpaired placed port could still
    # potentially meet some unplaced port.
    def potential_ok():
        unplaced = [p for p in pieces if p.pid not in placed]
        if not unplaced:
            return True

        # Collect unpaired port coords of placed pieces
        port_ids = defaultdict(set)
        for pid in placed:
            p = pieces[pid]
            t = translations[pid]
            for dr, dc in p.ports:
                port_ids[(t[0] + dr, t[1] + dc)].add(pid)

        for coord, ids in port_ids.items():
            if len(ids) >= 2:
                continue
            # This port needs a future partner.
            possible = False
            for p in unplaced:
                for dr, dc in p.ports:
                    # If p were placed so that this port coincides with coord
                    tr = coord[0] - dr
                    tc = coord[1] - dc
                    if in_bounds(p, (tr, tc)):
                        # Quick check: would not immediately overlap color/port
                        temp_colors = {(tr + x, tc + y) for x, y in p.colors}
                        temp_ports = {(tr + x, tc + y) for x, y in p.ports}
                        if not (any(c in placed_colors for c in temp_colors) or
                                any(c in placed_ports for c in temp_ports) or
                                any(c in placed_ports for c in temp_colors)):
                            possible = True
                            break
                if possible:
                    break
            if not possible:
                return False
        return True

    # DFS
    def dfs(placed_count):
        nonlocal best_score, best_translations

        if placed_count == len(pieces):
            score = solution_score(translations)
            if score is not None and (best_score is None or score < best_score):
                best_score = score
                best_translations = dict(translations)
            return

        # Choose next piece with fewest candidates (MRV)
        unplaced = [p for p in pieces if p.pid not in placed]
        best_p = None
        best_cands = None
        for p in unplaced:
            cands = candidates_for(p, placed)
            if cands:
                if best_cands is None or len(cands) < len(best_cands):
                    best_cands = cands
                    best_p = p
                if len(best_cands) == 1:
                    break

        if best_p is None:
            # No piece can connect to current placed set; dead end.
            return

        p = best_p
        cands = best_cands
        t_new = None
        # Try candidates; stop after best found? We continue for safety but limit by score.
        for t in cands:
            # Place
            translations[p.pid] = t
            new_colors = {(t[0] + dr, t[1] + dc) for dr, dc in p.colors}
            new_ports = {(t[0] + dr, t[1] + dc) for dr, dc in p.ports}

            placed.add(p.pid)
            old_colors_len = len(placed_colors)
            old_ports_len = len(placed_ports)
            placed_colors |= new_colors
            placed_ports |= new_ports

            if potential_ok():
                dfs(placed_count + 1)

            # Undo
            placed.remove(p.pid)
            # Remove only coordinates added (careful with overlaps; use sets and rebuild small)
            placed_colors = set()
            placed_ports = set()
            for pid in placed:
                pp = pieces[pid]
                tt = translations[pid]
                placed_colors |= {(tt[0] + dr, tt[1] + dc) for dr, dc in pp.colors}
                placed_ports |= {(tt[0] + dr, tt[1] + dc) for dr, dc in pp.ports}
            del translations[p.pid]

    dfs(1)

    # Fallback: if no solution found by search, return input (should not happen for valid tasks)
    if best_translations is None:
        # As a conservative fallback, keep all pieces in place and draw ports.
        out = np.zeros_like(grid)
        for p in pieces:
            for dr, dc in p.ports:
                r, c = p.r0 + dr, p.c0 + dc
                if 0 <= r < H and 0 <= c < W:
                    out[r, c] = 2
            for dr, dc in p.colors:
                r, c = p.r0 + dr, p.c0 + dc
                if 0 <= r < H and 0 <= c < W:
                    out[r, c] = p.color
        return out

    # ------------------------------------------------------------------
    # 5. Build output grid.
    # ------------------------------------------------------------------
    out = np.zeros_like(grid)
    # Place ports first
    for p in pieces:
        tr, tc = best_translations[p.pid]
        for dr, dc in p.ports:
            r, c = tr + dr, tc + dc
            if 0 <= r < H and 0 <= c < W:
                out[r, c] = 2
    # Place colors on top (should not conflict)
    for p in pieces:
        tr, tc = best_translations[p.pid]
        for dr, dc in p.colors:
            r, c = tr + dr, tc + dc
            if 0 <= r < H and 0 <= c < W:
                out[r, c] = p.color

    return out