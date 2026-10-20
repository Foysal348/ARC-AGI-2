import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Move each 4 to the farthest reachable position along its corridor.
    
    For each 4, we check 6 possible directions (H, V, SE, SW, NE, NW).
    The corridor direction is the one with the maximum "run" (distance to 
    nearest wall). The 4 moves to the farthest cell in that direction 
    (just before hitting a wall).
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    fours = [(r, c) for r in range(rows) for c in range(cols) if grid[r, c] == 4]
    
    if not fours:
        return grid
    
    def run_length(r, c, dr, dc):
        """Number of empty steps before hitting a wall in direction (dr,dc)."""
        nr, nc = r + dr, c + dc
        dist = 0
        while 0 <= nr < rows and 0 <= nc < cols:
            if grid[nr, nc] != 0 and grid[nr, nc] != 4:
                break
            dist += 1
            nr += dr
            nc += dc
        return dist
    
    new_positions = []
    
    for (r, c) in fours:
        # Check all 6 corridor directions
        # H: left or right
        # V: up or down  
        # SE, SW, NE, NW: diagonals
        
        # For H: the "run" is max(left, right)
        left = run_length(r, c, 0, -1)
        right = run_length(r, c, 0, 1)
        h_run = max(left, right)
        
        # For V: the "run" is max(up, down)
        up = run_length(r, c, -1, 0)
        down = run_length(r, c, 1, 0)
        v_run = max(up, down)
        
        # For diagonals
        se = run_length(r, c, 1, 1)
        sw = run_length(r, c, 1, -1)
        ne = run_length(r, c, -1, 1)
        nw = run_length(r, c, -1, -1)
        
        # Choose direction with maximum run
        directions = [
            (h_run, 'h'),
            (v_run, 'v'),
            (se, 'se'),
            (sw, 'sw'),
            (ne, 'ne'),
            (nw, 'nw'),
        ]
        directions.sort(key=lambda x: -x[0])
        
        new_pos = None
        for run, d in directions:
            if run == 0:
                continue
            if d == 'h':
                if right >= left and right > 0:
                    new_pos = (r, c + right - 1)
                elif left > 0:
                    new_pos = (r, c - left + 1)
            elif d == 'v':
                if down >= up and down > 0:
                    new_pos = (r + down - 1, c)
                elif up > 0:
                    new_pos = (r - up + 1, c)
            elif d == 'se' and se > 0:
                new_pos = (r + se - 1, c + se - 1)
            elif d == 'sw' and sw > 0:
                new_pos = (r + sw - 1, c - sw + 1)
            elif d == 'ne' and ne > 0:
                new_pos = (r - ne + 1, c + ne - 1)
            elif d == 'nw' and nw > 0:
                new_pos = (r - nw + 1, c - nw + 1)
            
            if new_pos is not None:
                nr, nc = new_pos
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                    break
                new_pos = None
        
        if new_pos is None:
            new_pos = (r, c)
        new_positions.append(new_pos)
    
    # Apply changes
    grid[grid == 4] = 0
    for (r, c) in new_positions:
        grid[r, c] = 4
    
    return grid