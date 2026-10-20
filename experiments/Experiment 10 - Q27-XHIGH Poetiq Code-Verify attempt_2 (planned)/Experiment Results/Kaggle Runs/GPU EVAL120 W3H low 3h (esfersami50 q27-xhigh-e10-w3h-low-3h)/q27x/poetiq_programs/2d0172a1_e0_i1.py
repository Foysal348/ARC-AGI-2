import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Extract the intermediate (middle) scale from nested self-similar shapes.
    
    The input contains nested '0'-like shapes (concentric rectangles with hollow centers).
    The output is the middle-sized shape from the nesting hierarchy.
    
    Algorithm:
    1. Identify background (most common color) and foreground colors
    2. Find the bounding box of the largest foreground component
    3. Determine the innermost detail size from the shape's interior structure
    4. Compute output dimensions using (max - min) // 2 + 1
    5. Extract the sub-region containing the middle shape
    """
    grid = grid.copy()
    h, w = grid.shape
    
    # Find background color (most common)
    colors, counts = np.unique(grid, return_counts=True)
    bg_color = colors[np.argmax(counts)]
    
    # Find foreground pixels
    fg_mask = (grid != bg_color)
    
    if not np.any(fg_mask):
        return grid
    
    # Find bounding box of all foreground
    rows = np.any(fg_mask, axis=1)
    cols = np.any(fg_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    max_h = rmax - rmin + 1
    max_w = cmax - cmin + 1
    
    # Analyze the interior to find the innermost detail
    # Look at the center region of the shape
    center_r = (rmin + rmax) // 2
    center_c = (cmin + cmax) // 2
    
    # Check if center pixel is foreground (indicates a dot/innermost detail)
    # and determine the minimum scale
    if fg_mask[center_r, center_c]:
        min_h = 1
        min_w = 1
    else:
        # Look for the smallest ring of foreground in the interior
        # Check rings from center outward
        found_min = False
        for dr in range(0, max_h // 2):
            for dc in range(0, max_w // 2):
                ring_h = 2 * dr + 1
                ring_w = 2 * dc + 1
                # Check if this ring position has foreground
                r1, r2 = center_r - dr, center_r + dr
                c1, c2 = center_c - dc, center_c + dc
                if r1 >= rmin and r2 <= rmax and c1 >= cmin and c2 <= cmax:
                    # Check if any pixel on this "ring" is foreground
                    ring_pixels = []
                    for r in range(r1, r2 + 1):
                        ring_pixels.append(grid[r, c1])
                        ring_pixels.append(grid[r, c2])
                    for c in range(c1, c2 + 1):
                        ring_pixels.append(grid[r1, c])
                        ring_pixels.append(grid[r2, c])
                    if any(grid[r, c] != bg_color for r in range(r1, r2+1) for c in [c1, c2]) or \
                       any(grid[r, c] != bg_color for r in [r1, r2] for c in range(c1, c2+1)):
                        min_h = ring_h
                        min_w = ring_w
                        found_min = True
                        break
            if found_min:
                break
        if not found_min:
            min_h = 1
            min_w = 1
    
    # Compute output dimensions
    out_h = (max_h - min_h) // 2 + 1
    out_w = (max_w - min_w) // 2 + 1
    
    # The output region is centered in the input shape
    # Start position to center the output in the bounding box
    start_r = rmin + (max_h - out_h) // 2
    start_c = cmin + (max_w - out_w) // 2
    
    # Extract the sub-region
    output = grid[start_r:start_r + out_h, start_c:start_c + out_w].copy()
    
    return output