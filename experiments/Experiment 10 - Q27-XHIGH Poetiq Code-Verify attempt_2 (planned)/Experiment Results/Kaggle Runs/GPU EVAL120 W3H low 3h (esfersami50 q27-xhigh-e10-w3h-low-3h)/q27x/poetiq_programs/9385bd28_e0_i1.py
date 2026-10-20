import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Multi-source flood fill transformation:
    - Background = most common color
    - Non-background components of size >= 3 are walls (block flood, stay unchanged)  
    - Non-background components of size <= 2 are sources
    - BFS from sources fills background cells with source color
    - Lower color wins on distance ties
    """
    rows, cols = grid.shape
    original = grid.copy()
    
    # Find background color
    flat = original.flatten()
    bg_color = np.bincount(flat).argmax()
    
    # Find connected components of non-background pixels
    visited = np.zeros((rows, cols), dtype=bool)
    components = []
    
    for r in range(rows):
        for c in range(cols):
            if not visited[r, c] and original[r, c] != bg_color:
                color = original[r, c]
                comp = []
                queue = deque([(r, c)])
                visited[r, c] = True
                while queue:
                    cr, cc = queue.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if not visited[nr, nc] and original[nr, nc] == color:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                components.append((color, comp))
    
    # Classify components
    wall_set = set()
    source_positions = []  # (r, c, color)
    
    for color, comp in components:
        if len(comp) >= 3:
            for pos in comp:
                wall_set.add(pos)
        else:
            for pos in comp:
                source_positions.append((pos[0], pos[1], color))
    
    # Multi-source BFS with tie-breaking (lower color wins)
    # Use priority: (distance, color, r, c)
    import heapq
    
    result = original.copy()
    best_dist = np.full((rows, cols), 999999, dtype=int)
    best_color = np.full((rows, cols), 999999, dtype=int)
    
    heap = []
    for r, c, color in source_positions:
        best_dist[r, c] = 0
        best_color[r, c] = color
        heapq.heappush(heap, (0, color, r, c))
    
    while heap:
        d, color, r, c = heapq.heappop(heap)
        if d > best_dist[r, c] or (d == best_dist[r, c] and color > best_color[r, c]):
            continue
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if (nr, nc) in wall_set:
                    continue
                if original[nr, nc] != bg_color and (nr, nc) not in wall_set:
                    continue  # skip non-wall non-bg (other sources)
                nd = d + 1
                if nd < best_dist[nr, nc] or (nd == best_dist[nr, nc] and color < best_color[nr, nc]):
                    best_dist[nr, nc] = nd
                    best_color[nr, nc] = color
                    heapq.heappush(heap, (nd, color, nr, nc))
    
    # Apply results
    for r in range(rows):
        for c in range(cols):
            if (r, c) in wall_set:
                result[r, c] = original[r, c]
            elif original[r, c] != bg_color:
                result[r, c] = original[r, c]  # source stays
            else:
                if best_color[r, c] != 999999:
                    result[r, c] = best_color[r, c]
    
    return result