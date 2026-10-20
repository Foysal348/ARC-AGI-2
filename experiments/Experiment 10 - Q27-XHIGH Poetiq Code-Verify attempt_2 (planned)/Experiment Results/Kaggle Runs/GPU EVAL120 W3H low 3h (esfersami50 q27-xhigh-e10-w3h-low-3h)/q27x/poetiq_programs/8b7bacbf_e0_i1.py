import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill the interior of square frames with the marker color.
    
    A 'frame' is a rectangular outline of a single color X where:
    - Top edge (excluding corners) is all X
    - Bottom edge (excluding corners) is all X  
    - Left edge (excluding corners) is all X
    - Right edge (excluding corners) is all X
    Frames can extend to the grid boundary (boundary implies a wall).
    
    The 'marker' color is the non-background color that appears exactly once.
    """
    grid = grid.copy()
    h, w = grid.shape
    
    # Find background color (most common)
    colors, counts = np.unique(grid, return_counts=True)
    bg_color = colors[np.argmax(counts)]
    
    # Find marker color (non-background, appears exactly once)
    marker_candidates = [c for i, c in enumerate(colors) if c != bg_color and counts[i] == 1]
    if not marker_candidates:
        return grid
    
    # If multiple candidates, pick the max value
    marker_color = max(marker_candidates)
    
    # Structural colors to check for frames
    structural_colors = [c for i, c in enumerate(colors) if c != bg_color and c != marker_color]
    
    result = grid.copy()
    
    for X in structural_colors:
        # Find all frames of color X
        for r1 in range(-1, h):
            ir1 = r1 + 1
            if ir1 >= h:
                continue
            for r2 in range(r1 + 2, h + 1):
                ir2 = r2 - 1
                if ir2 < 0 or ir2 < ir1:
                    continue
                for c1 in range(-1, w):
                    ic1 = c1 + 1
                    if ic1 >= w:
                        continue
                    # Check left edge
                    if 0 <= c1 < w:
                        if not np.all(grid[ir1:ir2+1, c1] == X):
                            continue
                    for c2 in range(c1 + 2, w + 1):
                        ic2 = c2 - 1
                        if ic2 < 0 or ic2 < ic1:
                            continue
                        # Check right edge
                        if 0 <= c2 < w:
                            if not np.all(grid[ir1:ir2+1, c2] == X):
                                continue
                        # Check top edge
                        if 0 <= r1 < h:
                            if not np.all(grid[r1, ic1:ic2+1] == X):
                                continue
                        # Check bottom edge
                        if 0 <= r2 < h:
                            if not np.all(grid[r2, ic1:ic2+1] == X):
                                continue
                        
                        # Require at least 3 explicit edges
                        explicit = (0 <= r1 < h) + (0 <= r2 < h) + (0 <= c1 < w) + (0 <= c2 < w)
                        if explicit < 3:
                            continue
                        
                        # Fill interior background cells with marker color
                        interior = result[ir1:ir2+1, ic1:ic2+1]
                        mask = grid[ir1:ir2+1, ic1:ic2+1] == bg_color
                        interior[mask] = marker_color
    
    return result