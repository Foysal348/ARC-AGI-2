import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform grid by:
    1. Finding 4-connected components of 1s.
    2. Computing 8-directional dilation (border) of all shapes.
    3. Flood-filling from grid boundary to identify 'outside' region.
    4. Coloring border pixels: outer→2, inner(holes)→6.
    5. For components with holes, changing their 1s to 8; otherwise keep as 1.
    """
    rows, cols = grid.shape
    shape_mask = (grid == 1)
    
    if not shape_mask.any():
        return grid.copy()
    
    # Step 1: Find connected components of 1s (4-connectivity)
    visited = np.zeros((rows, cols), dtype=bool)
    components = []
    
    for r in range(rows):
        for c in range(cols):
            if shape_mask[r, c] and not visited[r, c]:
                queue = deque([(r, c)])
                visited[r, c] = True
                component = []
                while queue:
                    cr, cc = queue.popleft()
                    component.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < cols and shape_mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                components.append(component)
    
    # Step 2: Compute 8-directional dilation (border)
    border_mask = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            if shape_mask[r, c]:
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not shape_mask[nr, nc]:
                            border_mask[nr, nc] = True
    
    # Step 3: Flood fill from grid boundary through non-occupied pixels
    occupied = shape_mask | border_mask
    outside = np.zeros((rows, cols), dtype=bool)
    queue = deque()
    
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows-1 or c == 0 or c == cols-1) and not occupied[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    
    while queue:
        cr, cc = queue.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cr+dr, cc+dc
            if 0 <= nr < rows and 0 <= nc < cols and not occupied[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                queue.append((nr, nc))
    
    # Step 4: Classify border pixels and build output
    output = grid.copy()
    inner_border = np.zeros((rows, cols), dtype=bool)
    
    for r in range(rows):
        for c in range(cols):
            if border_mask[r, c]:
                is_outer = (r == 0 or r == rows-1 or c == 0 or c == cols-1)
                if not is_outer:
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < rows and 0 <= nc < cols and outside[nr, nc]:
                                is_outer = True
                                break
                        if is_outer:
                            break
                
                if is_outer:
                    output[r, c] = 2
                else:
                    output[r, c] = 6
                    inner_border[r, c] = True
    
    # Step 5: Change 1s to 8 for components that have adjacent inner border pixels
    for component in components:
        has_hole = False
        for (r, c) in component:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols and inner_border[nr, nc]:
                        has_hole = True
                        break
                if has_hole:
                    break
            if has_hole:
                break
        
        if has_hole:
            for (r, c) in component:
                output[r, c] = 8
    
    return output