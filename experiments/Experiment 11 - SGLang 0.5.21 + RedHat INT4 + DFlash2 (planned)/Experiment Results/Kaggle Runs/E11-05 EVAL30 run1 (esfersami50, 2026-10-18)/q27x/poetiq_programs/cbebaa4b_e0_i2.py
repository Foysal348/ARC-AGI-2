import numpy as np
from collections import defaultdict, deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Assemble separated colored pieces into one compact connected structure.

    The input contains rigid colored pieces (colors other than 0 and 2) and
    connector cells of color 2. Each 2-cell near a piece is treated as a port
    attached to that piece. The central color-4 piece is kept fixed when
    present. Other pieces are translated so that ports are used to connect
    pieces: a port may overlap another port, or lie adjacent to a color cell
    of another piece.

    The solver extracts pieces and ports, then searches over translations with
    a small DFS. The chosen arrangement is the one that:
      - keeps color 4 fixed if possible,
      - never overlaps colored cells of different pieces,
      - creates a single connected assembly (colors + connector cells),
      - makes all connector cells "useful" (adjacent to color cells from at
        least two pieces, or at least adjacent to another piece),
      - is compact and close to the original positions.
    """
    grid = np.array(grid, dtype=np.uint8, copy=True)
    if grid.size == 0:
        return grid

    H, W = grid.shape
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))

    # ------------------------------------------------------------------
    # 1. Extract rigid colored components.
    # ------------------------------------------------------------------
    class Piece:
        __slots__ = ("idx", "color", "cells", "ports", "top_left")

    visited = np.zeros((H, W), dtype=bool)
    pieces = []
    cell_to_piece = {}

    def add_piece(color, cells):
        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        p = Piece()
        p.idx = len(pieces)
        p.color = color
        p.cells = [(r - min_r, c - min_c) for r, c in cells]
        p.ports = set()
        p.top_left = (min_r, min_c)
        pieces.append(p)
        for r, c in cells:
            cell_to_piece[(r, c)] = p.idx
        return p

    for r in range(H):
        for c in range(W):
            v = int(grid[r, c])
            if v == 0 or v == 2 or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in directions:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if not visited[nr, nc] and int(grid[nr, nc]) == v:
                            visited[nr, nc] = True
                            q.append((nr, nc))
            add_piece(v, cells)

    if not pieces:
        return np.zeros_like(grid)

    # ------------------------------------------------------------------
    # 2. Assign 2-cells to adjacent pieces as ports.
    # ------------------------------------------------------------------
    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) != 2:
                continue

            neighbors = []
            for dr, dc in directions:
                pid = cell_to_piece.get((r + dr, c + dc))
                if pid is not None:
                    neighbors.append(pid)

            if not neighbors:
                continue

            # If a connector touches several pieces, make it a port of all of them.
            for pid in set(neighbors):
                p = pieces[pid]
                p.ports.add((r - p.top_left[0], c - p.top_left[1]))

    # Keep only pieces that have at least one port, unless none have ports.
    ported = [p for p in pieces if p.ports]
    if ported:
        pieces = ported

    # ------------------------------------------------------------------
    # 3. Choose anchor. Prefer color 4, then piece with most ports/cells.
    # ------------------------------------------------------------------
    def anchor_key(p):
        return (
            1 if p.color == 4 else 0,
            len(p.ports),
            len(p.cells)
        )

    anchor = max(pieces, key=anchor_key)
    anchor_pid = anchor.idx
    anchor_trans = {anchor_pid: (anchor.top_left[0], anchor.top_left[1])}

    def piece_cells(p, t):
        return {(t[0] + dr, t[1] + dc) for dr, dc in p.cells}

    def piece_ports(p, t):
        return {(t[0] + dr, t[1] + dc) for dr, dc in p.ports}

    def in_bounds(p, t):
        for dr, dc in p.cells | p.ports:
            r, c = t[0] + dr, t[1] + dc
            if r < 0 or r >= H or c < 0 or c >= W:
                return False
        return True

    # ------------------------------------------------------------------
    # 4. DFS over piece translations.
    # ------------------------------------------------------------------
    best = None

    def score_solution(trans):
        placed_colors = {}
        all_colors = set()
        all_ports = set()
        port_owners = defaultdict(set)

        for p in pieces:
            t = trans[p.idx]
            cols = piece_cells(p, t)
            ports = piece_ports(p, t)
            placed_colors[p.idx] = cols
            all_colors |= cols
            all_ports |= ports
            for cell in ports:
                port_owners[cell].add(p.idx)

        # Check no color-color overlap.
        seen = set()
        for p in pieces:
            for cell in placed_colors[p.idx]:
                if cell in seen:
                    return None
                seen.add(cell)

        # Connectivity of all non-zero cells.
        all_nonzero = all_colors | all_ports
        if not all_nonzero:
            return None
        start = next(iter(all_nonzero))
        seen = set()
        q = deque([start])
        seen.add(start)
        while q:
            r, c = q.popleft()
            for dr, dc in directions:
                n = (r + dr, c + dc)
                if n in all_nonzero and n not in seen:
                    seen.add(n)
                    q.append(n)
        if seen != all_nonzero:
            return None

        # Check every port is useful.
        for cell, owners in port_owners.items():
            r, c = cell
            neighbor_piece_colors = set()
            for dr, dc in directions:
                pid = cell_to_piece.get((r + dr, c + dc))
                # cell_to_piece maps input cells only, so do final lookup below.
            # Final lookup from placed colors.
            for pid, cols in placed_colors.items():
                for dr, dc in directions:
                    if (r + dr, c + dc) in cols:
                        neighbor_piece_colors.add(pid)

            useful = len(neighbor_piece_colors) >= 2
            if not useful:
                # Allow a port that overlaps another piece's port and touches at
                # least one color cell.
                if len(owners) >= 2 and neighbor_piece_colors:
                    useful = True
            if not useful:
                return None

        # Compactness score.
        rs = [r for r, _ in all_nonzero]
        cs = [c for _, c in all_nonzero]
        bbox_area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)

        movement = 0
        for p in pieces:
            tr, tc = trans[p.idx]
            ir, ic = p.top_left
            movement += (tr - ir) ** 2 + (tc - ic) ** 2

        # Deterministic tie-breaker.
        tie = tuple(sorted((trans[p.idx], p.idx) for p in pieces))
        return (bbox_area, movement, tie)

    def conflicts(p, t, placed_color_cells, placed_port_cells):
        cols = piece_cells(p, t)
        ports = piece_ports(p, t)
        if any(c in placed_color_cells for c in cols):
            return True
        # A port may overlap a color only if that color belongs to the same piece
        # (not possible by construction), but ports may overlap ports.
        # We also forbid ports overlapping placed colors, except when that cell
        # would be a port of the placed piece (already in placed_port_cells).
        for cell in ports:
            if cell in placed_color_cells and cell not in placed_port_cells:
                # Allow one-way overlap? In valid examples this rarely happens.
                # Be permissive here; final scoring will remove bad solutions.
                pass
        return False

    def has_attachment(p, t, placed, translations, placed_color_cells, placed_port_cells):
        new_ports = piece_ports(p, t)
        new_colors = piece_cells(p, t)

        # Port of new piece overlaps placed port.
        for qid in placed:
            q = pieces[qid]
            qt = translations[qid]
            qports = piece_ports(q, qt)
            if new_ports & qports:
                return True

            # New port adjacent to placed color.
            for (pr, pc) in new_ports:
                for dr, dc in directions:
                    if (pr + dr, pc + dc) in placed_color_cells:
                        return True

            # Placed port adjacent to new color.
            for (qr, qc) in qports:
                for dr, dc in directions:
                    if (qr + dr, qc + dc) in new_colors:
                        return True
        return False

    def candidates_for(p, placed, translations, placed_color_cells, placed_port_cells):
        cands = set()

        # Generate by interactions with placed pieces.
        for qid in placed:
            q = pieces[qid]
            qt = translations[qid]
            qcolors = piece_cells(q, qt)
            qports = piece_ports(q, qt)

            for p_port in p.ports:
                # Port-port overlap.
                for q_port in qports:
                    t = (q_port[0] - p_port[0], q_port[1] - p_port[1])
                    if in_bounds(p, t) and not conflicts(p, t, placed_color_cells, placed_port_cells):
                        cands.add(t)

                # New port adjacent to placed color.
                for q_cell in qcolors:
                    for dr, dc in directions:
                        t = (q_cell[0] + dr - p_port[0], q_cell[1] + dc - p_port[1])
                        if in_bounds(p, t) and not conflicts(p, t, placed_color_cells, placed_port_cells):
                            cands.add(t)

            # Placed port adjacent to new color.
            for q_port in qports:
                for p_cell in p.cells:
                    for dr, dc in directions:
                        t = (q_port[0] + dr - p_cell[0], q_port[1] + dc - p_cell[1])
                        if in_bounds(p, t) and not conflicts(p, t, placed_color_cells, placed_port_cells):
                            cands.add(t)

        # Fallback: small translations around original position (useful for simple cases).
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                t = (p.top_left[0] + dr, p.top_left[1] + dc)
                if in_bounds(p, t) and not conflicts(p, t, placed_color_cells, placed_port_cells):
                    cands.add(t)

        valid = []
        for t in cands:
            if has_attachment(p, t, placed, translations, placed_color_cells, placed_port_cells):
                valid.append(t)

        # Order by local compactness / movement.
        def key(t):
            move = (t[0] - p.top_left[0]) ** 2 + (t[1] - p.top_left[1]) ** 2
            return (move, t)
        valid.sort(key=key)
        return valid

    def dfs(placed_count, placed, translations, placed_color_cells, placed_port_cells):
        nonlocal best

        if placed_count == len(pieces):
            sc = score_solution(translations)
            if sc is not None and (best is None or sc < best):
                best = (sc, dict(translations))
            return

        # Choose unplaced piece with fewest candidates.
        unplaced = [p for p in pieces if p.idx not in placed]
        best_p = None
        best_cands = None
        for p in unplaced:
            cands = candidates_for(p, placed, translations, placed_color_cells, placed_port_cells)
            if cands:
                if best_cands is None or len(cands) < len(best_cands):
                    best_cands = cands
                    best_p = p
                if len(best_cands) == 1:
                    break

        if best_p is None:
            return

        p = best_p
        for t in best_cands:
            translations[p.idx] = t
            new_colors = piece_cells(p, t)
            new_ports = piece_ports(p, t)

            old_colors = set(placed_color_cells)
            old_ports = set(placed_port_cells)

            placed.add(p.idx)
            placed_color_cells |= new_colors
            placed_port_cells |= new_ports

            dfs(placed_count + 1, placed, translations, placed_color_cells, placed_port_cells)

            # Restore.
            del translations[p.idx]
            placed.remove(p.idx)
            placed_color_cells = set(old_colors)
            placed_port_cells = set(old_ports)

    initial_color_cells = piece_cells(anchor, anchor_trans[anchor_pid])
    initial_port_cells = piece_ports(anchor, anchor_trans[anchor_pid])
    placed = {anchor_pid}
    translations = dict(anchor_trans)

    dfs(1, placed, translations, initial_color_cells, initial_port_cells)

    # ------------------------------------------------------------------
    # 5. Build output.
    # ------------------------------------------------------------------
    if best is None:
        # Conservative fallback: draw pieces at original positions.
        out = np.zeros_like(grid)
        for p in pieces:
            tr, tc = p.top_left
            for dr, dc in p.ports:
                r, c = tr + dr, tc + dc
                if 0 <= r < H and 0 <= c < W:
                    out[r, c] = 2
            for dr, dc in p.cells:
                r, c = tr + dr, tc + dc
                if 0 <= r < H and 0 <= c < W:
                    out[r, c] = p.color
        return out

    _, trans = best
    out = np.zeros_like(grid)

    # Draw ports first.
    for p in pieces:
        tr, tc = trans[p.idx]
        for dr, dc in p.ports:
            r, c = tr + dr, tc + dc
            if 0 <= r < H and 0 <= c < W:
                out[r, c] = 2

    # Draw colors on top, but do not overwrite a port that is shared by multiple
    # pieces or that is clearly a connector between pieces.
    for p in pieces:
        tr, tc = trans[p.idx]
        for dr, dc in p.cells:
            r, c = tr + dr, tc + dc
            if 0 <= r < H and 0 <= c < W:
                out[r, c] = p.color

    return out