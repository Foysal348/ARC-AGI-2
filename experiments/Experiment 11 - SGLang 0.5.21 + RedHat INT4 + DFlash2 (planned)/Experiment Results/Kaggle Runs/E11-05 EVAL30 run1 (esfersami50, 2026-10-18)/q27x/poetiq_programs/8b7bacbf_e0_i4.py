import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill enclosed 0-regions with the rarest color (key).
    
    Rule:
    1. Background = most common color (0)
    2. Key = rarest non-background color
    3. Path color = most frequent non-bg, non-key color (typically 1)
    4. Treat path-color cells as walls; flood-fill from key through non-wall cells
    5. Fill enclosed 0-regions that are entirely within the key's flood-fill region
    """
    if grid.size == 0:
        return grid.copy()
    
    H, W = grid.shape
    out = grid.copy()
    
    # Color counts
    vals, cnts = np.unique(grid, return_counts=True)
    counts = {int(v): int(c) for v, c in zip(vals, cnts)}
    
    # Background
    bg = max(counts, key=lambda c: (counts[c], -c))
    
    # Key (rarest non-bg)
    nonbg = {c: n for c, n in counts.items() if c != bg}
    if not nonbg:
        return out
    min_c = min(nonbg.values())
    key_color = max(c for c, n in nonbg.items() if n == min_c)
    
    # Path color: most frequent non-bg, non-key
    path_candidates = {c: n for c, n in nonbg.items() if c != key_color}
    if not path_candidates:
        return out
    path_color = max(path_candidates, key=lambda c: (path_candidates[c], c))
    
    key_cells = list(zip(*np.where(grid == key_color)))
    if not key_cells:
        return out
    
    # Flood fill from key, treating path_color as wall
    key_reachable = np.zeros((H, W), dtype=bool)
    queue = deque()
    for r, c in key_cells:
        queue.append((r, c))
        key_reachable[r, c] = True
    
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and not key_reachable[nr, nc] and grid[nr, nc] != path_color:
                key_reachable[nr, nc] = True
                queue.append((nr, nc))
    
    # Find enclosed 0-regions and fill those inside key_reachable
    visited = np.zeros((H, W), dtype=bool)
    for sr in range(H):
        for sc in range(W):
            if visited[sr, sc] or grid[sr, sc] != bg:
                continue
            # BFS for this 0-region
            q = deque([(sr, sc)])
            visited[sr, sc] = True
            cells = []
            touches_border = (sr == 0 or sc == 0 or sr == H-1 or sc == W-1)
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                if r == 0 or c == 0 or r == H-1 or c == W-1:
                    touches_border = True
                for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and grid[nr, nc] == bg:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            
            if touches_border:
                continue
            
            # Check if all cells are within key_reachable
            if all(key_reachable[r, c] for r, c in cells):
                for r, c in cells:
                    out[r, c] = key_color
    
    return out