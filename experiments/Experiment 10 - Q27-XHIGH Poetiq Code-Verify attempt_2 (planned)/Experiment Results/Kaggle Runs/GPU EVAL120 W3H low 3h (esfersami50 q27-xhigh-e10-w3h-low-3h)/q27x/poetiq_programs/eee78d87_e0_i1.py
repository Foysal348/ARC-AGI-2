import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a 6x6 input grid into a 16x16 output grid.
    
    The transformation:
    1. Extract the shape (non-7 cells) from the input
    2. Determine if the shape is 4-connected → period 3, else period 4
    3. Generate a periodic background pattern of 0s and 7s
    4. Generate a 9-region pattern in the 6x6 center at (5,5)
    """
    input_grid = np.array(grid)
    H, W = input_grid.shape  # 6x6
    
    # Find non-7 cells (the shape)
    shape_cells = set()
    for r in range(H):
        for c in range(W):
            if input_grid[r][c] != 7:
                shape_cells.add((r, c))
    
    # Find bounding box
    rows = sorted(set(r for r, c in shape_cells))
    cols = sorted(set(c for r, c in shape_cells))
    min_r, max_r = rows[0], rows[-1]
    min_c, max_c = cols[0], cols[-1]
    
    # Create local shape (within bounding box)
    box_h = max_r - min_r + 1
    box_w = max_c - min_c + 1
    local_shape = np.zeros((box_h, box_w), dtype=int)
    for r, c in shape_cells:
        local_shape[r - min_r][c - min_c] = 1
    
    # Check 4-connectivity
    def is_4_connected(cells, h, w):
        if not cells:
            return True
        cell_set = set(cells)
        start = next(iter(cell_set))
        visited = set()
        queue = [start]
        while queue:
            cur = queue.pop()
            if cur in visited:
                continue
            visited.add(cur)
            r, c = cur
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and (nr,nc) in cell_set and (nr,nc) not in visited:
                    queue.append((nr,nc))
        return len(visited) == len(cell_set)
    
    local_cells = [(r,c) for r in range(box_h) for c in range(box_w) if local_shape[r][c] == 1]
    four_connected = is_4_connected(local_cells, box_h, box_w)
    
    # Determine period
    period = 3 if four_connected else 4
    
    # Find full rows and columns in local shape
    full_rows = [r for r in range(box_h) if all(local_shape[r][c] == 1 for c in range(box_w))]
    full_cols = [c for c in range(box_w) if all(local_shape[r][c] == 1 for r in range(box_h))]
    
    output = np.zeros((16, 16), dtype=int)
    
    if period == 3:
        # Build 3x3 tile
        tile = np.zeros((3, 3), dtype=int)
        if full_rows and full_cols:
            wall_width = box_w - len(full_cols)
            wall_height = box_h - len(full_rows)
            start_row = full_rows[0]
            start_col = full_cols[0]
            for r in range(start_row, min(start_row + wall_height, 3)):
                for c in range(start_col, min(start_col + wall_width, 3)):
                    tile[r][c] = 7
        
        # Fill background with periodic tiling
        for r in range(16):
            for c in range(16):
                output[r][c] = tile[r % 3][c % 3]
        
        # 9-region: 6x6 at (5,5), which is 2x2 tiling of 3x3
        # The 9-region tile is the "dual" of the background tile
        # Where background has 0, 9-region has 9; where background has 7, 9-region has 7
        nine_tile = np.where(tile == 0, 9, 7)
        for r in range(6):
            for c in range(6):
                output[5+r][5+c] = nine_tile[r % 3][c % 3]
                
    else:  # period == 4
        # For non-4-connected shapes, use the "ring" pattern
        # 4x4 background tile: border (non-corner) = 7, corners and interior = 0
        tile = np.zeros((4, 4), dtype=int)
        for r in range(4):
            for c in range(4):
                # Edge but not corner
                if (r == 0 or r == 3 or c == 0 or c == 3) and not (r in [0,3] and c in [0,3]):
                    tile[r][c] = 7
        
        # Fill background with periodic tiling
        for r in range(16):
            for c in range(16):
                output[r][c] = tile[r % 4][c % 4]
        
        # 9-region: 6x6 at (5,5) with row period 4, col period 3
        # Fundamental domain is 4x3:
        nine_fund = np.array([
            [9, 7, 9],
            [7, 9, 7],
            [9, 7, 9],
            [9, 7, 9]
        ])
        for r in range(6):
            for c in range(6):
                output[5+r][5+c] = nine_fund[r % 4][c % 3]
    
    return output