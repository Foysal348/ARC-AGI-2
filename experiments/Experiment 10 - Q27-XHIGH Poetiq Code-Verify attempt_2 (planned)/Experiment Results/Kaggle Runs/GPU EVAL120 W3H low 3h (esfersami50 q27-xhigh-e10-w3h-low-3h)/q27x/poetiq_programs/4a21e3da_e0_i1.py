import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform: The 2 is at an edge. The 7-shape is split and moved to edges.
    - 2 at top/bottom: split 7s L/R of 2's col, move to L/R edges + anchor to 2's row edge
    - 2 at left/right: split 7s above/below 2's row, sub-split by grid center col,
      move to corresponding corners, draw cross of 2s
    """
    rows, cols = grid.shape
    new_grid = np.ones_like(grid)
    
    # Find the 2
    two_pos = np.argwhere(grid == 2)
    r2, c2 = int(two_pos[0][0]), int(two_pos[0][1])
    
    # Find all 7 positions
    seven_mask = (grid == 7)
    seven_pos = list(zip(*np.where(seven_mask)))
    
    if not seven_pos:
        new_grid[r2, c2] = 2
        return new_grid
    
    at_top = (r2 == 0)
    at_bottom = (r2 == rows - 1)
    at_left = (c2 == 0)
    at_right = (c2 == cols - 1)
    
    center_col = cols // 2
    
    if at_top or at_bottom:
        # Vertical line at c2
        # Split into left (c < c2) and right (c > c2)
        left_7s = [(r, c) for r, c in seven_pos if c < c2]
        right_7s = [(r, c) for r, c in seven_pos if c > c2]
        mid_7s = [(r, c) for r, c in seven_pos if c == c2]
        
        all_7s_r = [r for r, c in seven_pos]
        min_r_all, max_r_all = min(all_7s_r), max(all_7s_r)
        shape_h = max_r_all - min_r_all + 1
        
        if at_top:
            # Anchor top of each group at row 0
            if left_7s:
                min_c = min(c for r, c in left_7s)
                min_r = min(r for r, c in left_7s)
                for r, c in left_7s:
                    new_grid[r - min_r, c - min_c] = 7
            if right_7s:
                max_c = max(c for r, c in right_7s)
                min_r = min(r for r, c in right_7s)
                for r, c in right_7s:
                    new_grid[r - min_r, c - min_c + (cols - 1 - max_c)] = 7
            # Mid 7s go into the line
            min_r_mid = min(r for r, c in mid_7s) if mid_7s else 0
            for r, c in mid_7s:
                new_grid[r - min_r_mid, c2] = 7
            # Draw line
            for r in range(shape_h):
                if new_grid[r, c2] != 7:
                    new_grid[r, c2] = 2
        else:
            # Anchor bottom of each group at row rows-1
            if left_7s:
                min_c = min(c for r, c in left_7s)
                max_r = max(r for r, c in left_7s)
                for r, c in left_7s:
                    new_grid[r - max_r + (rows - 1), c - min_c] = 7
            if right_7s:
                max_c = max(c for r, c in right_7s)
                max_r = max(r for r, c in right_7s)
                for r, c in right_7s:
                    new_grid[r - max_r + (rows - 1), c - min_c + (cols - 1 - max_c)] = 7
            # Mid 7s
            max_r_mid = max(r for r, c in mid_7s) if mid_7s else rows-1
            for r, c in mid_7s:
                new_grid[r - max_r_mid + (rows - 1), c2] = 7
            # Draw line
            for r in range(rows - shape_h, rows):
                if new_grid[r, c2] != 7:
                    new_grid[r, c2] = 2
    
    elif at_left or at_right:
        # Horizontal line at r2, vertical line at center_col
        center_c = cols // 2
        top_7s = [(r, c) for r, c in seven_pos if r < r2]
        bot_7s = [(r, c) for r, c in seven_pos if r > r2]
        mid_7s = [(r, c) for r, c in seven_pos if r == r2]
        
        all_7s_r = [r for r, c in seven_pos]
        min_r_all, max_r_all = min(all_7s_r), max(all_r_all)
        shape_h = max_r_all - min_r_all + 1
        
        # Top group
        if top_7s:
            tl = [(r, c) for r, c in top_7s if c <= center_c]
            tr = [(r, c) for r, c in top_7s if c > center_c]
            if tl:
                min_r = min(r for r, c in tl)
                min_c = min(c for r, c in tl)
                for r, c in tl:
                    new_grid[r - min_r, c - min_c] = 7
            if tr:
                min_r = min(r for r, c in tr)
                max_c = max(c for r, c in tr)
                for r, c in tr:
                    new_grid[r - min_r, c - max_c + (cols - 1)] = 7
        
        # Bottom group
        if bot_7s:
            bl = [(r, c) for r, c in bot_7s if c <= center_c]
            br = [(r, c) for r, c in bot_7s if c > center_c]
            if at_right:
                if br:
                    max_r = max(r for r, c in br)
                    max_c = max(c for r, c in br)
                    for r, c in br:
                        new_grid[r - max_r + (rows - 1), c - max_c + (cols - 1)] = 7
            else:
                if bl:
                    max_r = max(r for r, c in bl)
                    min_c = min(c for r, c in bl)
                    for r, c in bl:
                        new_grid[r - max_r + (rows - 1), c - min_c] = 7
        
        # Mid 7s (at row r2) - place at center_col in vertical line
        for r, c in mid_7s:
            if abs(c - center_c) <= 1:
                new_grid[r2, center_c] = 7
        
        # Draw vertical line at center_c
        for r in range(max(0, r2 - shape_h + 1), min(rows, r2 + 1)):
            if new_grid[r, center_c] != 7:
                new_grid[r, center_c] = 2
        
        # Draw horizontal line at r2
        if at_right:
            for c in range(center_c, cols):
                if new_grid[r2, c] != 7:
                    new_grid[r2, c] = 2
        else:
            for c in range(0, center_c + 1):
                if new_grid[r2, c] != 7:
                    new_grid[r2, c] = 2
    
    return new_grid