import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by:
    1. Finding the vertical divider (column of 5s)
    2. Right-aligning all shapes (0-components) on the left side
    3. Placing 2s on the right side at rows corresponding to the largest
       enclosed 6-group within each shape
    """
    rows, cols = grid.shape
    
    # Find the divider column (all 5s)
    divider_col = None
    for c in range(cols):
        if np.all(grid[:, c] == 5):
            divider_col = c
            break
    
    if divider_col is None:
        return grid.copy()
    
    left_cols = divider_col
    right_start = divider_col + 1
    
    left = grid[:, :left_cols].copy()
    
    # Find connected components of 0s on the left side
    visited = np.zeros((rows, left_cols), dtype=bool)
    shapes = []
    
    for r in range(rows):
        for c in range(left_cols):
            if left[r, c] == 0 and not visited[r, c]:
                component = []
                queue = deque([(r, c)])
                visited[r, c] = True
                while queue:
                    cr, cc = queue.popleft()
                    component.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < left_cols:
                            if left[nr, nc] == 0 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                shapes.append(component)
    
    # For each shape, find interior 6s and determine which rows get 2s
    two_rows = set()
    
    for comp in shapes:
        if not comp:
            continue
        r_min = min(r for r, c in comp)
        r_max = max(r for r, c in comp)
        c_min = min(c for r, c in comp)
        c_max = max(c for r, c in comp)
        
        bbox_h = r_max - r_min + 1
        bbox_w = c_max - c_min + 1
        
        # Create local grid for the bounding box
        local = np.zeros((bbox_h, bbox_w), dtype=int)
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                local[r - r_min, c - c_min] = left[r, c]
        
        # Flood fill from bbox boundary through 6s to find reachable 6s
        reachable = np.zeros((bbox_h, bbox_w), dtype=bool)
        queue = deque()
        
        for c in range(bbox_w):
            if local[0, c] == 6:
                reachable[0, c] = True
                queue.append((0, c))
            if local[bbox_h - 1, c] == 6:
                reachable[bbox_h - 1, c] = True
                queue.append((bbox_h - 1, c))
        for r in range(bbox_h):
            if local[r, 0] == 6:
                reachable[r, 0] = True
                queue.append((r, 0))
            if local[r, bbox_w - 1] == 6:
                reachable[r, bbox_w - 1] = True
                queue.append((r, bbox_w - 1))
        
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < bbox_h and 0 <= nc < bbox_w:
                    if local[nr, nc] == 6 and not reachable[nr, nc]:
                        reachable[nr, nc] = True
                        queue.append((nr, nc))
        
        # Collect interior 6s (not reachable from boundary)
        interior_6s = set()
        for r in range(bbox_h):
            for c in range(bbox_w):
                if local[r, c] == 6 and not reachable[r, c]:
                    interior_6s.add((r + r_min, c + c_min))
        
        # Group interior 6s into connected components
        if interior_6s:
            visited_int = set()
            int_groups = []
            
            for pos in interior_6s:
                if pos not in visited_int:
                    group = set()
                    queue = deque([pos])
                    visited_int.add(pos)
                    while queue:
                        cr, cc = queue.popleft()
                        group.add((cr, cc))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = cr + dr, cc + dc
                            if (nr, nc) in interior_6s and (nr, nc) not in visited_int:
                                visited_int.add((nr, nc))
                                queue.append((nr, nc))
                    int_groups.append(group)
            
            # Select largest group (tie-break: lowest = max row number)
            best_group = None
            best_size = -1
            best_max_row = -1
            
            for group in int_groups:
                size = len(group)
                max_row = max(r for r, c in group)
                if size > best_size or (size == best_size and max_row > best_max_row):
                    best_size = size
                    best_max_row = max_row
                    best_group = group
            
            if best_group is not None:
                for (r, c) in best_group:
                    two_rows.add(r)
    
    # Right-align all shapes on the left side
    new_left = np.full((rows, left_cols), 6, dtype=grid.dtype)
    for comp in shapes:
        if not comp:
            continue
        c_min = min(c for r, c in comp)
        c_max = max(c for r, c in comp)
        width = c_max - c_min + 1
        shift = left_cols - width - c_min
        
        for (r, c) in comp:
            new_left[r, c + shift] = 0
    
    # Construct output
    output = grid.copy()
    output[:, :left_cols] = new_left
    output[:, right_start:] = 6
    for r in two_rows:
        output[r, right_start:] = 2
    
    return output