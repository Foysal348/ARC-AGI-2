import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Connect adjacent blocks whose values are consecutive in the mapping sequence.
    The gap between connected blocks is filled with the value that appears first
    in the consecutive pair of the mapping sequence.
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Step 1: Find background color (color of corner)
    bg_color = int(grid[0, 0])
    
    # Step 2: Find the mapping row (scan from bottom for isolated non-bg values)
    mapping_row = None
    for r in range(rows - 1, -1, -1):
        non_bg_mask = grid[r] != bg_color
        if not np.any(non_bg_mask):
            continue
        non_bg_indices = np.where(non_bg_mask)[0]
        if len(non_bg_indices) < 2:
            continue
        # Check max consecutive run - mapping row has short runs (1-2 pixels)
        max_run = 1
        current_run = 1
        for i in range(1, len(non_bg_indices)):
            if non_bg_indices[i] - non_bg_indices[i-1] == 1:
                current_run += 1
            else:
                current_run = 1
            max_run = max(max_run, current_run)
        if max_run <= 2:
            mapping_row = r
            break
    
    if mapping_row is None:
        return grid
    
    # Step 3: Extract mapping sequence from the mapping row
    sequence = []
    for c in range(cols):
        if grid[mapping_row, c] != bg_color:
            sequence.append(int(grid[mapping_row, c]))
    
    if len(sequence) < 2:
        return grid
    
    # Step 4: Find blocks using connected component labeling
    mask = (grid != bg_color)
    mask[mapping_row, :] = False  # Remove mapping row from consideration
    
    # BFS-based connected component labeling (4-connectivity)
    labeled = np.zeros((rows, cols), dtype=int)
    num_blocks = 0
    for i in range(rows):
        for j in range(cols):
            if mask[i, j] and labeled[i, j] == 0:
                num_blocks += 1
                queue = deque([(i, j)])
                labeled[i, j] = num_blocks
                while queue:
                    ci, cj = queue.popleft()
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = ci+di, cj+dj
                        if 0 <= ni < rows and 0 <= nj < cols and mask[ni, nj] and labeled[ni, nj] == 0:
                            labeled[ni, nj] = num_blocks
                            queue.append((ni, nj))
    
    # Step 5: For each block, find its value and bounding box
    blocks = []
    for i in range(1, num_blocks + 1):
        block_mask = (labeled == i)
        block_pixels = grid[block_mask]
        unique_colors, counts = np.unique(block_pixels, return_counts=True)
        
        # The dominant color is the block's fill color
        dominant = unique_colors[np.argmax(counts)]
        # The value is the other color (the special number at center)
        other_colors = [c for c in unique_colors if c != dominant]
        if not other_colors:
            continue
        value = int(other_colors[0])
        
        rows_idx, cols_idx = np.where(block_mask)
        blocks.append({
            'r_min': int(rows_idx.min()),
            'r_max': int(rows_idx.max()),
            'c_min': int(cols_idx.min()),
            'c_max': int(cols_idx.max()),
            'value': value
        })
    
    if not blocks:
        return grid
    
    # Step 6: Helper - get the fill value if two values are consecutive in sequence
    def get_connection_value(val1, val2):
        """Return the first value if val1 and val2 are consecutive in sequence, else None."""
        if val1 == val2:
            return None
        for i in range(len(sequence) - 1):
            if sequence[i] == val1 and sequence[i+1] == val2:
                return val1
            if sequence[i] == val2 and sequence[i+1] == val1:
                return val2
        return None
    
    # Step 7: For each pair of blocks, check if adjacent and consecutive, fill gap
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            b1, b2 = blocks[i], blocks[j]
            
            fill_value = get_connection_value(b1['value'], b2['value'])
            if fill_value is None:
                continue
            
            filled = False
            
            # Check horizontal adjacency (row ranges overlap, column gap is background)
            r_ov_min = max(b1['r_min'], b2['r_min'])
            r_ov_max = min(b1['r_max'], b2['r_max'])
            
            if r_ov_min <= r_ov_max:
                if b1['c_max'] < b2['c_min']:
                    left, right = b1, b2
                elif b2['c_max'] < b1['c_min']:
                    left, right = b2, b1
                else:
                    left = right = None
                
                if left is not None:
                    gap_s = left['c_max'] + 1
                    gap_e = right['c_min'] - 1
                    if gap_s <= gap_e:
                        region = grid[r_ov_min:r_ov_max+1, gap_s:gap_e+1]
                        if np.all(region == bg_color):
                            grid[r_ov_min:r_ov_max+1, gap_s:gap_e+1] = fill_value
                            filled = True
            
            if filled:
                continue
            
            # Check vertical adjacency (column ranges overlap, row gap is background)
            c_ov_min = max(b1['c_min'], b2['c_min'])
            c_ov_max = min(b1['c_max'], b2['c_max'])
            
            if c_ov_min <= c_ov_max:
                if b1['r_max'] < b2['r_min']:
                    top, bottom = b1, b2
                elif b2['r_max'] < b1['r_min']:
                    top, bottom = b2, b1
                else:
                    top = bottom = None
                
                if top is not None:
                    gap_s = top['r_max'] + 1
                    gap_e = bottom['r_min'] - 1
                    if gap_s <= gap_e:
                        region = grid[gap_s:gap_e+1, c_ov_min:c_ov_max+1]
                        if np.all(region == bg_color):
                            grid[gap_s:gap_e+1, c_ov_min:c_ov_max+1] = fill_value
    
    return grid