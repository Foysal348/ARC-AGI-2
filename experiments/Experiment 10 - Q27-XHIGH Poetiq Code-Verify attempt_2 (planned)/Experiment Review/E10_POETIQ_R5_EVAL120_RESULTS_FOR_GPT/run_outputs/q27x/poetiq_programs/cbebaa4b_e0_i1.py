import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid by:
      1. Extracting the non-zero, non-2 coloured components as pieces.
      2. Keeping the colour-4 piece fixed as the base when present.
      3. Moving the other pieces rigidly so that:
           - pieces do not overlap,
           - the final assembly is connected,
           - as many 2-cells as possible become internal (i.e. adjacent
             to a different piece).
      4. Among valid arrangements, prefer the one with the largest number
         of internal 2-cells, then the most top-left-justified occupied
         cells.
    The returned grid contains only the moved pieces; 2-cells are kept only
    if they are adjacent to a different piece.
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape

    # ------------------------------------------------------------------
    # Basic connected-component helper
    # ------------------------------------------------------------------
    def component_mask(mask: np.ndarray) -> list:
        """Return one 4-connected component as a list of (r,c) or []."""
        seen = np.zeros(mask.shape, dtype=bool)
        stack = []

        # start from first True cell
        it = np.argwhere(mask)
        if len(it) == 0:
            return []
        r0, c0 = it[0]
        stack.append((int(r0), int(c0)))
        seen[r0, c0] = True
        comp = []

        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        return comp

    def bbox(cells) -> tuple:
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return min(rs), min(cs), max(rs), max(cs)

    # ------------------------------------------------------------------
    # Extract pieces.
    # A piece is a same-colour component of non-zero, non-2 cells.
    # Nearby 2-cells are attached as "holes" to that piece.
    # ------------------------------------------------------------------
    non2 = (grid > 0) & (grid != 2)
    visited = np.zeros((H, W), dtype=bool)
    pieces = []

    for color in sorted(np.unique(grid)):
        if color in (0, 2):
            continue
        mask = (grid == color) & (~visited)
        while True:
            comp = component_mask(mask)
            if not comp:
                break
            for r, c in comp:
                visited[r, c] = True
            # remove this component from mask to find next one of same colour
            mask[np.array([r for r, _ in comp]).astype(int),
                 np.array([c for _, c in comp]).astype(int)] = False

            r0, c0, r1, c1 = bbox(comp)

            # Attach 2-cells that are within one cell of the piece bbox
            # and 4-connected to the piece through 2-cells.
            holes = set()
            seed = [(r, c) for r, c in comp]
            q = deque(seed)
            in_comp = set(comp)
            seen2 = set(comp)

            # We only allow 2-cells in an expanded box.
            er0, ec0, er1, ec1 = max(0, r0 - 1), max(0, c0 - 1), min(H - 1, r1 + 1), min(W - 1, c1 + 1)

            # BFS through 2-cells from colour cells
            q = deque(comp)
            seen = set(comp)
            while q:
                r, c = q.popleft()
                if grid[r, c] == 2:
                    holes.add((r, c))
                    continue
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (er0 <= nr <= er1 and ec0 <= nc <= ec1 and
                            grid[nr, nc] == 2 and (nr, nc) not in seen):
                        seen.add((nr, nc))
                        q.append((nr, nc))

            # If colour is 4, do not let stray 2s become holes.
            # This matches the examples where 4 is the fixed base without holes.
            if color == 4:
                holes = set()

            cells = [(r, c, int(color)) for r, c in comp]
            cells += [(r, c, 2) for r, c in holes]

            # Keep only cells inside the final piece bounding box.
            pr0, pc0, pr1, pc1 = bbox([(r, c) for r, c, _ in cells])
            cells = [(r - pr0, c - pc0, col) for (r, c, col) in cells if pr0 <= r <= pr1 and pc0 <= c <= pc1]

            pieces.append({
                "color": int(color),
                "top_in": (pr0, pc0),
                "h": pr1 - pr0 + 1,
                "w": pc1 - pc0 + 1,
                "cells": cells,
                "has_2": any(col == 2 for _, _, col in cells),
            })

    if not pieces:
        return np.zeros_like(grid)

    # Choose base piece: prefer colour 4, otherwise a piece without 2s,
    # finally the largest piece.
    base = None
    for p in pieces:
        if p["color"] == 4:
            base = p
            break
    if base is None:
        for p in pieces:
            if not p["has_2"]:
                base = p
                break
    if base is None:
        base = max(pieces, key=lambda p: len(p["cells"]))

    movable = [p for p in pieces if p is not base]

    # If there is nothing to move, simply return the input with external
    # 2-cells removed.
    if not movable:
        out = np.zeros_like(grid, dtype=int)
        for p in pieces:
            r0, c0 = p["top_in"]
            for dr, dc, col in p["cells"]:
                if col == 2:
                    continue
                out[r0 + dr, c0 + dc] = col
        return out

    # ------------------------------------------------------------------
    # Search for an arrangement.
    # The base is fixed at its input position.  We try to place all movable
    # pieces without overlap.  For efficiency, candidate placements are
    # generated in a window around the base.
    # ------------------------------------------------------------------
    br0, bc0 = base["top_in"]
    max_dim = max(p["h"] + p["w"] for p in pieces)
    window = max_dim + 6

    def placement_cells(piece, top):
        r0, c0 = top
        return [(r0 + dr, c0 + dc, col) for dr, dc, col in piece["cells"]]

    def overlaps(placed, top_cells):
        for r, c, _ in top_cells:
            if not (0 <= r < H and 0 <= c < W):
                return True
            if placed[r, c] != -1:
                return True
        return False

    def count_internal_2s(placed, piece_ids):
        """
        For each cell that is 2, check whether at least one 4-neighbor
        belongs to a different piece.
        """
        cnt = 0
        for r in range(H):
            for c in range(W):
                if piece_ids[r, c] == 2:
                    pid = None
                    # we need id of piece; encode in grid_id separately below
                    pass
        # The previous loop is unused; this function is replaced by evaluate().
        return 0

    def evaluate(assignment):
        """
        assignment: list of (piece_index, top) for all pieces.
        Returns a tuple:
            (-internal_2_count, top_left_score, extra_score)
        Lower is better.
        """
        occ = np.full((H, W), -1, dtype=int)   # -1 empty, piece index otherwise
        cell_col = np.zeros((H, W), dtype=int)

        for idx, (pidx, top) in enumerate(assignment):
            p = pieces[pidx]
            for dr, dc, col in p["cells"]:
                r, c = top[0] + dr, top[1] + dc
                if 0 <= r < H and 0 <= c < W:
                    occ[r, c] = pidx
                    cell_col[r, c] = col

        # Connectivity check using 4-neighbour over non-empty cells.
        if len(assignment) > 1:
            start = None
            for r in range(H):
                for c in range(W):
                    if occ[r, c] != -1:
                        start = (r, c)
                        break
                if start is not None:
                    break
            if start is None:
                return (10**9, 10**9, 10**9)

            seen = np.zeros((H, W), dtype=bool)
            q = deque([start])
            seen[start[0], start[1]] = True
            while q:
                r, c = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and occ[nr, nc] != -1 and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            # all occupied cells must be reached
            for r in range(H):
                for c in range(W):
                    if occ[r, c] != -1 and not seen[r, c]:
                        return (10**9, 10**9, 10**9)

        internal_2 = 0
        for r in range(H):
            for c in range(W):
                if cell_col[r, c] == 2 and occ[r, c] != -1:
                    pid = occ[r, c]
                    ok = False
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and occ[nr, nc] != -1 and occ[nr, nc] != pid:
                            ok = True
                            break
                    if ok:
                        internal_2 += 1

        # Top-left lexicographic score: sort occupied cells and compare.
        # To keep it cheap, use bounding box and then a small cap of earliest cells.
        cells = []
        for r in range(H):
            for c in range(W):
                if occ[r, c] != -1:
                    cells.append((r, c))
        cells.sort()
        # A lower tuple is better.  Use first up to 24 occupied cells.
        score_cells = tuple(cells[:24])
        # If lengths differ, shorter is not automatically better; append -1 pad.
        while len(score_cells) < 24:
            score_cells += (-1,)

        return (-internal_2, score_cells, len(cells))

    # Generate candidate placements for each piece near the base.
    candidates = {}
    for pidx, p in enumerate(pieces):
        if p is base:
            continue
        r0in, c0in = p["top_in"]
        min_r = max(0, br0 - window - p["h"] + 1)
        max_r = min(H - p["h"], br0 + window)
        min_c = max(0, bc0 - window - p["w"] + 1)
        max_c = min(W - p["w"], bc0 + window)

        cand = []
        for r0 in range(min_r, max_r + 1):
            for c0 in range(min_c, max_c + 1):
                cand.append((r0, c0))

        # Sort candidates top-to-bottom, left-to-right.
        cand.sort()
        candidates[pidx] = cand

    # Backtracking search.  We try all pieces, but order by fewer candidates
    # first.  The state is the current placement map.
    movable_idx = [i for i, p in enumerate(pieces) if p is not base]
    movable_idx.sort(key=lambda i: len(candidates[i]))

    base_top = base["top_in"]
    occupied = np.full((H, W), -1, dtype=int)
    # mark base
    for dr, dc, col in base["cells"]:
        r, c = br0 + dr, bc0 + dc
        if 0 <= r < H and 0 <= c < W:
            occupied[r, c] = pieces.index(base)

    best = None
    best_assign = None

    def dfs(pos, assignment):
        nonlocal best, best_assign

        if pos == len(movable_idx):
            val = evaluate(assignment)
            if best is None or val < best:
                best = val
                best_assign = assignment[:]
            return

        pidx = movable_idx[pos]
        p = pieces[pidx]

        # Try placements in lexicographic order.
        for top in candidates[pidx]:
            # quick overlap check
            ok = True
            cells = []
            for dr, dc, col in p["cells"]:
                r, c = top[0] + dr, top[1] + dc
                if not (0 <= r < H and 0 <= c < W) or occupied[r, c] != -1:
                    ok = False
                    break
                cells.append((r, c))

            if not ok:
                continue

            # optional pruning: the piece must touch the existing assembly
            # by at least one cell, otherwise final connectivity is impossible
            # if it has 2-holes.  If it has no holes, still require contact.
            touches = False
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and occupied[nr, nc] != -1:
                        touches = True
                        break
                if touches:
                    break
            if not touches:
                continue

            # place
            for r, c in cells:
                occupied[r, c] = pidx
            assignment.append((pidx, top))

            dfs(pos + 1, assignment)

            # unplace
            for r, c in cells:
                occupied[r, c] = -1
            assignment.pop()

            # If we already have a perfect score, stop early.
            # Perfect means: all possible internal 2s?  We use -1 as a strong
            # early-stop only if score is already better than any feasible
            # alternative for the first dimension and the top-left score is
            # minimal.  For safety, do not early-stop.

    dfs(0, [(pieces.index(base), base_top)])

    # Fallback: if search failed, just keep everything in input position.
    if best_assign is None:
        return np.array(grid, copy=True)

    # Build output grid.
    out = np.zeros((H, W), dtype=int)
    out_occ = np.full((H, W), -1, dtype=int)

    for pidx, top in best_assign:
        p = pieces[pidx]
        r0, c0 = top
        for dr, dc, col in p["cells"]:
            r, c = r0 + dr, c0 + dc
            if 0 <= r < H and 0 <= c < W:
                out[r, c] = col
                out_occ[r, c] = pidx

    # Final pass: keep a 2-cell only if it is adjacent to a different piece.
    for r in range(H):
        for c in range(W):
            if out[r, c] == 2:
                pid = out_occ[r, c]
                keep = False
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and out_occ[nr, nc] != -1 and out_occ[nr, nc] != pid:
                        keep = True
                        break
                if not keep:
                    out[r, c] = 0

    return out