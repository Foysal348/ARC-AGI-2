import numpy as np
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transformation rule:
    1. Identify the background color (most frequent color).
    2. In the bottom (last) row, find groups of non-background cells.
    3. For groups with color 2, note their column range.
    4. In the rest of the grid, find connected components of non-background cells.
       Any component whose column range matches a target range gets recolored to 5.
    5. Clear the bottom row to background.
    """
    result = grid.copy()
    
    # Step 1: Find background color (most common)
    colors, counts = np.unique(result, return_counts=True)
    bg_color = colors[np.argmax(counts)]
    
    # Step 2: Find non-background groups in the last row
    last_row = result[-1]
    non_bg_mask = (last_row != bg_color)
    
    # Find column ranges and colors of non-background groups
    groups = []
    in_group = False
    start = 0
    for i in range(len(non_bg_mask)):
        if non_bg_mask[i] and not in_group:
            in_group = True
            start = i
        elif not non_bg_mask[i] and in_group:
            in_group = False
            groups.append((start, i - 1, int(last_row[start])))
    if in_group:
        groups.append((start, len(non_bg_mask) - 1, int(last_row[start])))
    
    # Step 3: Find column ranges where color is 2
    target_cols = []
    for col_start, col_end, color in groups:
        if color == 2:
            target_cols.append((col_start, col_end))
    
    # Step 4: Find connected components in the main area (excluding last row)
    main_area = result[:-1]
    non_bg_main = (main_area != bg_color)
    
    labeled, num_features = ndimage.label(non_bg_main)
    
    # For each component, check if its column range matches any target
    for i in range(1, num_features + 1):
        component = (labeled == i)
        col_indices = np.where(component.any(axis=0))[0]
        if len(col_indices) == 0:
            continue
        col_start, col_end = col_indices[0], col_indices[-1]
        
        for tc_start, tc_end in target_cols:
            if col_start == tc_start and col_end == tc_end:
                # Recolor this component to 5 in the result grid
                result[component] = 5
                break
    
    # Step 5: Clear the last row
    result[-1] = bg_color
    
    return result