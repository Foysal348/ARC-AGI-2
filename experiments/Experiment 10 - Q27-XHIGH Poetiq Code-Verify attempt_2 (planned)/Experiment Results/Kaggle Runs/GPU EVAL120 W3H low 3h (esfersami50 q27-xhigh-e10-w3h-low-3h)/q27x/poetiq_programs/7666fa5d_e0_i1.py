import numpy as np
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill interior regions of diagonal line patterns with 2s.
    
    The colored pixels form diagonal lines in two directions. In the (u,v) 
    coordinate system where u=r+c and v=r-c, these become axis-aligned lines.
    For each pair of adjacent u-lines, the overlap of their v-ranges defines
    a fillable region. All background cells within these regions are set to 2.
    """
    result = grid.copy()
    rows, cols = grid.shape
    
    # Identify the background color (most common) and the "line" color
    colors, counts = np.unique(grid, return_counts=True)
    bg_color = colors[np.argmax(counts)]
    
    # Find the non-background, non-2 color (the "line" color)
    line_color = None
    for c in colors:
        if c != bg_color and c != 2:
            line_color = c
            break
    
    if line_color is None:
        return result
    
    # Collect all positions of line-colored pixels
    line_positions = np.argwhere(grid == line_color)
    
    if len(line_positions) == 0:
        return result
    
    # Group by u = r + c, tracking min and max v = r - c for each u
    u_v_ranges = {}  # u -> (min_v, max_v)
    for r, c in line_positions:
        u = r + c
        v = r - c
        if u not in u_v_ranges:
            u_v_ranges[u] = [v, v]
        else:
            u_v_ranges[u][0] = min(u_v_ranges[u][0], v)
            u_v_ranges[u][1] = max(u_v_ranges[u][1], v)
    
    # Get sorted u-line values
    u_lines = sorted(u_v_ranges.keys())
    
    # For each pair of adjacent u-lines, fill the overlap region
    for i in range(len(u_lines) - 1):
        u1 = u_lines[i]
        u2 = u_lines[i + 1]
        
        v1_min, v1_max = u_v_ranges[u1]
        v2_min, v2_max = u_v_ranges[u2]
        
        # Compute v-range overlap
        overlap_min = max(v1_min, v2_min)
        overlap_max = min(v1_max, v2_max)
        
        if overlap_min > overlap_max:
            continue
        
        # Fill all background cells in this region
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != bg_color:
                    continue
                u = r + c
                v = r - c
                if u1 < u < u2 and overlap_min <= v <= overlap_max:
                    result[r, c] = 2
    
    return result