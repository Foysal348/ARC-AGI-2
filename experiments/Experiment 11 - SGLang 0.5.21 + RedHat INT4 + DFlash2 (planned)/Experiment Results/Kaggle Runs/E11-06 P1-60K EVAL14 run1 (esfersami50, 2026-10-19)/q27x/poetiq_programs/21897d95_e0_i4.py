import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    H, W = grid.shape

    # Step 1: Clean the grid - replace 1s and 0s with the dominant surrounding color
    # First, identify the "real" blocks by treating 1s as holes
    cleaned = grid.copy()
    
    # Find connected components of non-1, non-0 cells
    comp_id = -np.ones((H, W), dtype=int)
    comp_colors = []
    for r in range(H):
        for c in range(W):
            if grid[r, c] in (0, 1) or comp_id[r, c] != -1:
                continue
            cid = len(comp_colors)
            comp_colors.append(int(grid[r, c]))
            q = deque([(r, c)])
            comp_id[r, c] = cid
            while q:
                rr, cc = q.popleft()
                for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    nr, nc = rr+dr, cc+dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if comp_id[nr, nc] == -1 and grid[nr, nc] not in (0, 1):
                            comp_id[nr, nc] = cid
                            q.append((nr, nc))
    
    # Fill 1s and 0s with the most common adjacent component color
    for r in range(H):
        for c in range(W):
            if grid[r, c] in (0, 1):
                adj_colors = []
                for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < H and 0 <= nc < W and comp_id[nr, nc] >= 0:
                        adj_colors.append(comp_colors[comp_id[nr, nc]])
                if adj_colors:
                    from collections import Counter
                    cnt = Counter(adj_colors)
                    cleaned[r, c] = cnt.most_common(1)[0][0]
                else:
                    cleaned[r, c] = 0
    
    # Step 2: Find T-shaped arrows of 1s
    def find_stem(arms):
        for (dr, dc) in arms:
            others = [a for a in arms if a != (dr, dc)]
            if len(others) == 2:
                (a1, a2) = others
                if a1[0] == -a2[0] and a1[1] == -a2[1]:
                    return (dr, dc)
        return None
    
    visited = np.zeros((H, W), dtype=bool)
    arrows = []  # (src_comp, tgt_comp)
    
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 1 or visited[r, c]:
                continue
            q = deque([(r, c)])
            cells = []
            visited[r, c] = True
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    nr, nc = rr+dr, cc+dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if not visited[nr, nc] and grid[nr, nc] == 1:
                            visited[nr, nc] = True
                            q.append((nr, nc))
            
            if len(cells) != 4:
                continue
            
            cell_set = set(cells)
            center = None
            for (rr, cc) in cells:
                cnt = sum(1 for dr, dc in ((1,0),(-1,0),(0,1),(0,-1))
                         if (rr+dr, cc+dc) in cell_set)
                if cnt == 3:
                    center = (rr, cc)
                    break
            if center is None:
                continue
            
            cr, cc = center
            arms = [(dr, dc) for dr, dc in ((1,0),(-1,0),(0,1),(0,-1))
                   if (cr+dr, cc+dc) in cell_set]
            stem = find_stem(arms)
            if stem is None:
                continue
            
            sr, sc = stem
            
            # Find source: look opposite to stem from center
            src = -1
            nr, nc = cr, cc
            while True:
                nr -= sr
                nc -= sc
                if not (0 <= nr < H and 0 <= nc < W):
                    break
                if comp_id[nr, nc] >= 0:
                    src = comp_id[nr, nc]
                    break
            
            # Find target: look in stem direction from center
            tgt = -1
            nr, nc = cr, cc
            while True:
                nr += sr
                nc += sc
                if not (0 <= nr < H and 0 <= nc < W):
                    break
                if comp_id[nr, nc] >= 0 and comp_id[nr, nc] != src:
                    tgt = comp_id[nr, nc]
                    break
            
            if src >= 0 and tgt >= 0:
                arrows.append((src, tgt))
    
    # Step 3: Apply the color mapping
    # For each target component, it gets the source's color
    # For components with no incoming arrow, they get the "remaining" color
    final_colors = list(comp_colors)
    
    # Count incoming arrows per target
    incoming = {}
    for src, tgt in arrows:
        if tgt not in incoming:
            incoming[tgt] = []
        incoming[tgt].append(src)
    
    # Assign target colors
    assigned = set()
    for tgt, srcs in incoming.items():
        if len(srcs) == 1:
            final_colors[tgt] = comp_colors[srcs[0]]
            assigned.add(tgt)
        else:
            # Multiple sources: use the one that's not the target's own color
            for s in srcs:
                if comp_colors[s] != comp_colors[tgt]:
                    final_colors[tgt] = comp_colors[s]
                    break
            assigned.add(tgt)
    
    # For unassigned components, find remaining colors
    used_colors = set(final_colors[t] for t in range(len(comp_colors)) if t in assigned)
    all_original_colors = set(comp_colors)
    
    # The unassigned components should get colors that "flow" to them
    # Heuristic: use the color that appears most among the sources pointing to assigned neighbors
    unassigned = [t for t in range(len(comp_colors)) if t not in assigned]
    
    if unassigned:
        # Determine which colors are "freed up" (their original component was reassigned)
        # and assign them to unassigned components based on spatial proximity to arrows
        for t in unassigned:
            # Find the arrow closest to this component's centroid
            best_color = comp_colors[t]  # default: keep own color
            best_dist = float('inf')
            comp_cells = [(r, c) for r in range(H) for c in range(W) if comp_id[r, c] == t]
            if comp_cells:
                cr_avg = sum(r for r, c in comp_cells) / len(comp_cells)
                cc_avg = sum(c for r, c in comp_cells) / len(comp_cells)
                for src, tgt in arrows:
                    src_cells = [(r, c) for r in range(H) for c in range(W) if comp_id[r, c] == src]
                    if src_cells:
                        sr_avg = sum(r for r, c in src_cells) / len(src_cells)
                        sc_avg = sum(c for r, c in src_cells) / len(src_cells)
                        dist = (sr_avg - cr_avg)**2 + (sc_avg - cc_avg)**2
                        if dist < best_dist:
                            best_dist = dist
                            best_color = comp_colors[src]
            final_colors[t] = best_color
    
    # Step 4: Build output grid
    out = np.empty((H, W), dtype=int)
    for r in range(H):
        for c in range(W):
            cid = comp_id[r, c]
            if cid >= 0:
                out[r, c] = final_colors[cid]
            else:
                # Fill with nearest component's final color
                val = 0
                for dr, dc in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)):
                    rr, cc = r, c
                    found = False
                    while 0 <= rr < H and 0 <= cc < W:
                        if comp_id[rr, cc] >= 0:
                            val = final_colors[comp_id[rr, cc]]
                            found = True
                            break
                        rr += dr
                        cc += dc
                    if found:
                        break
                out[r, c] = val
    
    # Step 5: Transpose if non-square
    if H != W:
        out = out.T
    
    return out