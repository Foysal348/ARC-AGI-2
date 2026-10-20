import numpy as np
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform by finding the key (color 5) shape and using it to decode
    the main pattern in the input grid.
    
    Rule:
    - Find the key shape (color 5) to determine cell dimensions
    - Find the main pattern (non-zero, non-5 pixels)
    - If the pattern is a grid of cells: output the cell color grid
    - If the pattern is B-shapes: output the largest B-shape
    """
    grid = grid.copy()
    
    # Find key shape bounding box
    key_mask = (grid == 5)
    key_pixels = np.argwhere(key_mask)
    
    if len(key_pixels) > 0:
        key_min_r, key_min_c = key_pixels.min(axis=0)
        key_max_r, key_max_c = key_pixels.max(axis=0)
        key_h = key_max_r - key_min_r + 1
        key_w = key_max_c - key_min_c + 1
    else:
        key_h, key_w = 1, 1
    
    # Find main pattern (non-zero, non-5)
    main_mask = (grid > 0) & (grid != 5)
    main_pixels = np.argwhere(main_mask)
    
    if len(main_pixels) == 0:
        return np.zeros((1, 1), dtype=int)
    
    main_min_r, main_min_c = main_pixels.min(axis=0)
    main_max_r, main_max_c = main_pixels.max(axis=0)
    
    # Extract main region
    region = grid[main_min_r:main_max_r+1, main_min_c:main_max_c+1]
    region_mask = main_mask[main_min_r:main_max_r+1, main_min_c:main_max_c+1]
    
    # Find connected components of the main pattern to identify individual shapes
    labeled, num_features = ndimage.label(region_mask)
    
    # Check if shapes are B-like or grid-like
    # B-shapes: tall and narrow with specific pattern
    # Grid: many small uniform blocks
    
    # Determine if this is a "grid" pattern or "B-shape" pattern
    # Grid pattern: the region has a regular structure with gaps
    # B-pattern: individual shapes with the B structure
    
    # Check fill ratio and structure
    h, w = region.shape
    fill_ratio = region_mask.sum() / (h * w)
    
    # For B-patterns, the individual shapes are separated by large gaps
    # For grid patterns, the cells are closely packed
    
    # Check if there are multiple colors in the main pattern
    main_colors = set(region[region_mask].flatten().tolist())
    
    # If multiple colors and high fill ratio -> grid pattern
    # If single color per shape and low fill ratio -> B-pattern
    
    # Check if the pattern has a regular grid structure
    # Look for consistent spacing of non-zero regions
    
    if len(main_colors) > 1 and fill_ratio > 0.25:
        # Grid pattern: extract cell values
        # Find row and column boundaries of cells
        row_has_data = region_mask.any(axis=1)
        col_has_data = region_mask.any(axis=0)
        
        # Find groups of consecutive rows/cols with data
        def find_groups(mask_1d):
            groups = []
            in_group = False
            start = 0
            for i in range(len(mask_1d)):
                if mask_1d[i] and not in_group:
                    in_group = True
                    start = i
                elif not mask_1d[i] and in_group:
                    in_group = False
                    groups.append((start, i))
            if in_group:
                groups.append((start, len(mask_1d)))
            return groups
        
        row_groups = find_groups(row_has_data)
        col_groups = find_groups(col_has_data)
        
        # Build output: each cell gets the dominant color
        out = np.zeros((len(row_groups), len(col_groups)), dtype=int)
        
        for ri, (r1, r2) in enumerate(row_groups):
            for ci, (c1, c2) in enumerate(col_groups):
                cell = region[r1:r2, c1:c2]
                cell_m = region_mask[r1:r2, c1:c2]
                if cell_m.any():
                    vals = cell[cell_m]
                    unique_vals, counts = np.unique(vals, return_counts=True)
                    out[ri, ci] = unique_vals[np.argmax(counts)]
        
        return out
    else:
        # B-pattern: find and return the largest B-shape
        labeled, num_features = ndimage.label(region_mask)
        
        best_area = 0
        best_label = 0
        
        for i in range(1, num_features + 1):
            component = (labeled == i)
            area = component.sum()
            if area > best_area:
                best_area = area
                best_label = i
        
        # Extract the best component's bounding box
        best_pixels = np.argwhere(labeled == best_label)
        r1, c1 = best_pixels.min(axis=0)
        r2, c2 = best_pixels.max(axis=0)
        
        sub = region[r1:r2+1, c1:c2+1]
        sub_m = (labeled == best_label)[r1:r2+1, c1:c2+1]
        
        out = np.zeros((r2-r1+1, c2-c1+1), dtype=int)
        out[sub_m] = sub[sub_m]
        
        return out