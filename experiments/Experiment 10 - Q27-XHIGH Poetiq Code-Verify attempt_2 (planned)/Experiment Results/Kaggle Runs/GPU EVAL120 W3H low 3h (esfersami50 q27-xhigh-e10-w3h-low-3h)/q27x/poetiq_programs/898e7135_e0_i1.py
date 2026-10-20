import numpy as np
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid by:
    1. Identifying the background color (most frequent non-zero color)
    2. Removing single-pixel components
    3. Extracting remaining shapes, rotating them 90° CCW
    4. Placing shapes in a new grid filled with background color
    """
    grid = np.array(grid, dtype=int)
    rows, cols = grid.shape
    
    # Find the background color (most frequent non-zero)
    non_zero = grid[grid != 0]
    if len(non_zero) == 0:
        return np.zeros_like(grid)
    
    unique, counts = np.unique(non_zero, return_counts=True)
    bg_color = unique[np.argmax(counts)]
    
    # Find all connected components (4-connectivity) of non-zero pixels
    non_zero_mask = (grid != 0)
    labeled, num_features = ndimage.label(non_zero_mask, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    
    # Collect all shapes (connected components with > 1 pixel)
    shapes = []
    for i in range(1, num_features + 1):
        component = (labeled == i)
        if component.sum() > 1:
            rows_idx = np.where(component.any(axis=1))[0]
            cols_idx = np.where(component.any(axis=0))[0]
            r_min, r_max = rows_idx[0], rows_idx[-1]
            c_min, c_max = cols_idx[0], cols_idx[-1]
            
            sub_grid = grid[r_min:r_max+1, c_min:c_max+1]
            sub_mask = component[r_min:r_max+1, c_min:c_max+1]
            
            # Get the color (should be uniform)
            color = sub_grid[sub_mask][0]
            
            # Create binary shape
            shape = sub_mask.astype(int)
            
            # Rotate 90° CCW
            rotated = np.rot90(shape, k=1)
            
            shapes.append({
                'color': int(color),
                'shape': shape,
                'rotated': rotated,
                'center_r': (r_min + r_max) / 2.0,
                'center_c': (c_min + c_max) / 2.0,
                'r_min': r_min, 'r_max': r_max,
                'c_min': c_min, 'c_max': c_max
            })
    
    if not shapes:
        return np.full((1, 1), bg_color, dtype=int)
    
    # Sort by vertical then horizontal position
    shapes.sort(key=lambda s: (s['center_r'], s['center_c']))
    
    margin = 2
    spacing = 2
    n = len(shapes)
    
    # Determine rows by vertical proximity
    row_groups = []
    current_row = [0]
    for i in range(1, n):
        gap = shapes[i]['center_r'] - shapes[i-1]['center_r']
        min_h = min(shapes[i]['rotated'].shape[0], shapes[i-1]['rotated'].shape[0])
        if gap > min_h / 2 + spacing:
            row_groups.append(current_row)
            current_row = [i]
        else:
            current_row.append(i)
    row_groups.append(current_row)
    
    # Calculate output dimensions
    total_h = 0
    max_w = 0
    for group in row_groups:
        group_sorted = sorted(group, key=lambda i: shapes[i]['center_c'])
        row_h = max(shapes[i]['rotated'].shape[0] for i in group_sorted)
        row_w = sum(shapes[i]['rotated'].shape[1] for i in group_sorted) + spacing * (len(group_sorted) - 1)
        total_h += row_h
        max_w = max(max_w, row_w)
    
    total_h += spacing * (len(row_groups) - 1)
    out_h = total_h + 2 * margin
    out_w = max_w + 2 * margin
    
    output = np.full((out_h, out_w), bg_color, dtype=int)
    
    # Place shapes
    row_offset = margin
    for group in row_groups:
        group_sorted = sorted(group, key=lambda i: shapes[i]['center_c'])
        col_offset = margin
        row_h = max(shapes[i]['rotated'].shape[0] for i in group_sorted)
        
        for i in group_sorted:
            s = shapes[i]
            h, w = s['rotated'].shape
            r_start = row_offset + (row_h - h) // 2
            c_start = col_offset
            
            for dr in range(h):
                for dc in range(w):
                    if s['rotated'][dr, dc] > 0:
                        output[r_start + dr, c_start + dc] = s['color']
            
            col_offset += w + spacing
        
        row_offset += row_h + spacing
    
    return output