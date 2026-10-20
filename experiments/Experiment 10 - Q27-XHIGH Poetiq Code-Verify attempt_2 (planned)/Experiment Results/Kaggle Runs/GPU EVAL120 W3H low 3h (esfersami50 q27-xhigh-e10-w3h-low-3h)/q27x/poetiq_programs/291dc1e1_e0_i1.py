import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid by extracting rectangular blocks from the interior
    and stacking them vertically in reading order (determined by 0 position).
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Find the 0 position to determine reading direction
    zero_positions = np.where(grid == 0)
    read_right_to_left = False
    if len(zero_positions[0]) > 0:
        zero_row, zero_col = zero_positions[0][0], zero_positions[1][0]
        read_right_to_left = (zero_col > cols // 2)
    
    # Find the bounding box of the 8-region (interior)
    eight_mask = (grid == 8)
    eight_rows = np.where(eight_mask.any(axis=1))[0]
    eight_cols = np.where(eight_mask.any(axis=0))[0]
    
    if len(eight_rows) == 0 or len(eight_cols) == 0:
        return np.zeros_like(grid)
    
    top, bottom = eight_rows[0], eight_rows[-1]
    left, right = eight_cols[0], eight_cols[-1]
    
    interior = grid[top:bottom+1, left:right+1]
    int_rows, int_cols = interior.shape
    non8_mask = (interior != 8)
    
    # Find connected components of non-8 pixels
    visited = np.zeros_like(non8_mask, dtype=bool)
    blocks = []
    
    for i in range(int_rows):
        for j in range(int_cols):
            if non8_mask[i, j] and not visited[i, j]:
                component = []
                queue = deque([(i, j)])
                visited[i, j] = True
                while queue:
                    r, c = queue.popleft()
                    component.append((r, c))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < int_rows and 0 <= nc < int_cols:
                            if non8_mask[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                
                min_r = min(r for r, c in component)
                max_r = max(r for r, c in component)
                min_c = min(c for r, c in component)
                max_c = max(c for r, c in component)
                
                if non8_mask[min_r:max_r+1, min_c:max_c+1].all():
                    block_data = interior[min_r:max_r+1, min_c:max_c+1].copy()
                    blocks.append({
                        'data': block_data,
                        'top': min_r,
                        'left': min_c,
                        'right': max_c,
                        'height': max_r - min_r + 1,
                        'width': max_c - min_c + 1
                    })
    
    if not blocks:
        return np.zeros_like(grid)
    
    # Sort blocks in reading order
    if read_right_to_left:
        blocks.sort(key=lambda b: (b['top'], -b['right']))
    else:
        blocks.sort(key=lambda b: (b['top'], b['left']))
    
    # Create output: stack blocks vertically, centered
    max_width = max(b['width'] for b in blocks)
    total_height = sum(b['height'] for b in blocks)
    
    output = np.full((total_height, max_width), 8, dtype=grid.dtype)
    
    current_row = 0
    for block in blocks:
        h, w = block['height'], block['width']
        col_offset = (max_width - w) // 2
        output[current_row:current_row+h, col_offset:col_offset+w] = block['data']
        current_row += h
    
    return output