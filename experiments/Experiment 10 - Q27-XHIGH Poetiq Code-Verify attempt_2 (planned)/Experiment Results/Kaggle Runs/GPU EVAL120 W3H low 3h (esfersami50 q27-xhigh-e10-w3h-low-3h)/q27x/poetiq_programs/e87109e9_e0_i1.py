import numpy as np
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform by removing label section and drawing a lattice of lines
    from the target block to all regions.
    """
    grid = grid.copy()
    
    # Find and remove label section
    rows_all_5 = [i for i in range(grid.shape[0]) if np.all(grid[i] == 5)]
    main_start = rows_all_5[1] + 1 if len(rows_all_5) >= 2 else 0
    main_grid = grid[main_start:].copy()
    
    # Find background color
    colors, counts = np.unique(main_grid, return_counts=True)
    bg_color = colors[np.argmax(counts)]
    
    # Find all non-background regions
    non_bg_colors = [c for c in colors if c != bg_color and c != 0]
    
    regions = []
    for color in non_bg_colors:
        mask = (main_grid == color)
        labeled, num = ndimage.label(mask)
        for fid in range(1, num + 1):
            fm = (labeled == fid)
            rows_idx = np.where(fm.any(axis=1))[0]
            cols_idx = np.where(fm.any(axis=0))[0]
            regions.append({
                'color': color,
                'area': int(np.sum(fm)),
                'r_min': int(rows_idx[0]), 'r_max': int(rows_idx[-1]),
                'c_min': int(cols_idx[0]), 'c_max': int(cols_idx[-1])
            })
    
    if not regions:
        return main_grid
    
    # Target = smallest region
    target = min(regions, key=lambda r: r['area'])
    target_color = target['color']
    others = [r for r in regions if r is not target]
    
    t_height = target['r_max'] - target['r_min'] + 1
    t_width = target['c_max'] - target['c_min'] + 1
    
    # Determine line positions
    v_lines = [(target['c_min'], target['c_max'])]  # target's vertical
    h_lines = [(target['r_min'], target['r_max'])]  # target's horizontal
    
    for reg in others:
        if reg['c_max'] < target['c_min']:  # region to the left
            c = reg['c_max'] + 1
            v_lines.append((c, c + t_width - 1))
        elif reg['c_min'] > target['c_max']:  # region to the right
            c = reg['c_min'] - t_width
            v_lines.append((c, reg['c_min'] - 1))
        elif reg['r_max'] < target['r_min']:  # region above
            r = reg['r_max'] + 1
            h_lines.append((r, r + t_height - 1))
        elif reg['r_min'] > target['r_max']:  # region below
            r = reg['r_min'] - t_height
            h_lines.append((r, reg['r_min'] - 1))
    
    # Create output
    output = main_grid.copy()
    
    # Determine extents: vertical lines go from top to nearest region below
    # Horizontal lines go from left to nearest region to the right
    for c_min, c_max in v_lines:
        c_min = max(0, c_min)
        c_max = min(main_grid.shape[1]-1, c_max)
        bottom = main_grid.shape[0] - 1
        for reg in others:
            if reg['c_min'] <= c_max and reg['c_max'] >= c_min:
                if reg['r_min'] > target['r_min']:
                    bottom = min(bottom, reg['r_min'] - 1)
        output[0:bottom+1, c_min:c_max+1] = target_color
    
    for r_min, r_max in h_lines:
        r_min = max(0, r_min)
        r_max = min(main_grid.shape[0]-1, r_max)
        right = main_grid.shape[1] - 1
        for reg in others:
            if reg['r_min'] <= r_max and reg['r_max'] >= r_min:
                if reg['c_min'] > target['c_min']:
                    right = min(right, reg['c_min'] - 1)
        output[r_min:r_max+1, 0:right+1] = target_color
    
    # Restore regions
    for reg in others:
        output[reg['r_min']:reg['r_max']+1, reg['c_min']:reg['c_max']+1] = reg['color']
    
    return output