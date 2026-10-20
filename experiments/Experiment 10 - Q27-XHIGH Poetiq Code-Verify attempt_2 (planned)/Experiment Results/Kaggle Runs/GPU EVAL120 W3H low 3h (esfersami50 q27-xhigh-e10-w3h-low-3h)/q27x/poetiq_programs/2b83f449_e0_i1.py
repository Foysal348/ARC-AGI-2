import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """Transform the grid:
    1. Replace consecutive 7s (groups of 3) with 8-6-8
    2. Project 6s from non-wall rows onto adjacent wall rows
    3. Mark structural corners with 3s at wall row edges
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Identify wall rows and non-wall rows
    wall_rows = [r for r in range(rows) if 8 in grid[r]]
    non_wall_rows = [r for r in range(rows) if 8 not in grid[r]]
    non_wall_set = set(non_wall_rows)
    
    # Step 1: Replace 7-groups with 8-6-8
    for r in non_wall_rows:
        c = 0
        while c < cols:
            if grid[r, c] == 7:
                start = c
                while c < cols and grid[r, c] == 7:
                    c += 1
                length = c - start
                if length == 3:
                    grid[r, start] = 8
                    grid[r, start + 1] = 6
                    grid[r, start + 2] = 8
            else:
                c += 1
    
    # Step 2: Project 6s onto wall rows
    for wr in wall_rows:
        for c in range(cols):
            if grid[wr, c] != 8:
                continue
            should_be_6 = False
            if wr - 1 in non_wall_set and grid[wr - 1, c] == 6:
                should_be_6 = True
            if wr + 1 in non_wall_set and grid[wr + 1, c] == 6:
                should_be_6 = True
            if should_be_6:
                grid[wr, c] = 6
    
    # Step 3: Determine 3s placement
    # Analyze: 3s appear at left edge (cols 0,1) and/or right edge (cols w-2,w-1)
    # of wall rows where the adjacent non-wall rows have narrow empty space at that edge
    
    for wr in wall_rows:
        adj = []
        if wr - 1 in non_wall_set:
            adj.append(wr - 1)
        if wr + 1 in non_wall_set:
            adj.append(wr + 1)
        if not adj:
            continue
        
        # Left edge analysis
        left_widths = []
        for ar in adj:
            w = 0
            while w < cols and grid[ar, w] == 0:
                w += 1
            left_widths.append(w)
        
        # Right edge analysis  
        right_widths = []
        for ar in adj:
            w = 0
            while w < cols and grid[ar, cols - 1 - w] == 0:
                w += 1
            right_widths.append(w)
        
        min_l = min(left_widths)
        min_r = min(right_widths)
        
        # Place 3s on left
        if min_l > 0:
            if len(adj) == 1:
                # Top/bottom wall row: only 1 three at edge
                if min_l >= 1 and grid[wr, 0] == 8:
                    grid[wr, 0] = 3
            else:
                # Interior wall row: up to 2 threes
                if min_l >= 2:
                    if grid[wr, 0] == 8:
                        grid[wr, 0] = 3
                    if grid[wr, 1] == 8:
                        grid[wr, 1] = 3
                elif min_l == 1:
                    if grid[wr, 0] == 8:
                        grid[wr, 0] = 3
        
        # Place 3s on right
        if min_r > 0:
            if len(adj) == 1:
                if min_r >= 1 and grid[wr, cols - 1] == 8:
                    grid[wr, cols - 1] = 3
            else:
                if min_r >= 2:
                    if grid[wr, cols - 1] == 8:
                        grid[wr, cols - 1] = 3
                    if grid[wr, cols - 2] == 8:
                        grid[wr, cols - 2] = 3
                elif min_r == 1:
                    if grid[wr, cols - 1] == 8:
                        grid[wr, cols - 1] = 3
    
    return grid