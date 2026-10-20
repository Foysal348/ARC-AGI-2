import numpy as np
from scipy import ndimage
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform by:
    1. Identifying solid rectangular blocks and small irregular shapes
    2. Arranging solid blocks in their relative layout
    3. Applying the inverted pattern of the corresponding small shape to each block's center
    """
    # Step 1: Find background color (color of the border)
    bg_color = grid[0, 0]
    
    # Step 2: Find all non-background colors
    non_bg_colors = set(grid.flatten().tolist()) - {bg_color}
    
    # Step 3: For each non-background color, find 8-connected components
    solid_blocks = []  # List of (color, row_min, row_max, col_min, col_max)
    small_shapes = {}  # Dict: color -> pattern (2D array)
    
    for color in non_bg_colors:
        mask = (grid == color)
        
        # Find 8-connected components
        struct = np.ones((3, 3), dtype=int)
        labeled, num_features = ndimage.label(mask, structure=struct)
        
        for i in range(1, num_features + 1):
            component = (labeled == i)
            rows = np.where(component.any(axis=1))[0]
            cols = np.where(component.any(axis=0))[0]
            row_min, row_max = int(rows[0]), int(rows[-1])
            col_min, col_max = int(cols[0]), int(cols[-1])
            
            # Check if it's a solid rectangle (all cells in bounding box are this color)
            sub_grid = grid[row_min:row_max+1, col_min:col_max+1]
            if np.all(sub_grid == color):
                solid_blocks.append((color, row_min, row_max, col_min, col_max))
            else:
                # This is a small shape - store the pattern
                pattern = sub_grid.copy()
                if color not in small_shapes:
                    small_shapes[color] = pattern
    
    # Handle edge case: no solid blocks found
    if not solid_blocks:
        return grid.copy()
    
    # Step 4: Determine grid layout by grouping blocks by their row position
    row_map = defaultdict(list)
    for block in solid_blocks:
        row_map[block[1]].append(block)
    
    sorted_row_keys = sorted(row_map.keys())
    row_groups = [row_map[key] for key in sorted_row_keys]
    
    # Sort each row group by column position
    for group in row_groups:
        group.sort(key=lambda b: b[3])
    
    num_block_rows = len(row_groups)
    num_block_cols = max(len(group) for group in row_groups)
    
    # Determine block dimensions
    block_height = solid_blocks[0][2] - solid_blocks[0][1] + 1
    block_width = solid_blocks[0][4] - solid_blocks[0][3] + 1
    
    # Step 5: Create output grid with appropriate dimensions
    out_rows = num_block_rows * block_height + (num_block_rows - 1) + 2
    out_cols = num_block_cols * block_width + (num_block_cols - 1) + 2
    
    output = np.full((out_rows, out_cols), bg_color, dtype=grid.dtype)
    
    # Step 6: Fill in each block with color and apply inverted small shape pattern
    for i, group in enumerate(row_groups):
        for j, block in enumerate(group):
            color = block[0]
            out_row_start = 1 + i * (block_height + 1)
            out_col_start = 1 + j * (block_width + 1)
            
            # Fill the block with its color
            output[out_row_start:out_row_start + block_height,
                   out_col_start:out_col_start + block_width] = color
            
            # Apply inverted pattern from corresponding small shape
            if color in small_shapes:
                pattern = small_shapes[color]
                pattern_height = pattern.shape[0]
                pattern_width = pattern.shape[1]
                
                # Center the pattern within the block
                offset_row = (block_height - pattern_height) // 2
                offset_col = (block_width - pattern_width) // 2
                
                center_start_row = out_row_start + offset_row
                center_start_col = out_col_start + offset_col
                
                # Invert: where pattern has shape's color → bg_color (hole)
                #          where pattern has bg_color → shape's color (filled)
                for pr in range(pattern_height):
                    for pc in range(pattern_width):
                        if pattern[pr, pc] == color:
                            output[center_start_row + pr, center_start_col + pc] = bg_color
                        else:
                            output[center_start_row + pr, center_start_col + pc] = color
    
    return output