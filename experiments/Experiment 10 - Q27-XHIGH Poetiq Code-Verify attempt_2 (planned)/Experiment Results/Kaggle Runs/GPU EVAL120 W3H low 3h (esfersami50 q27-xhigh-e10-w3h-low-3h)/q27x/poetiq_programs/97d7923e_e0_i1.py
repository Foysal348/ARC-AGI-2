import numpy as np
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    For vertical line segments where both endpoints have the same color 
    (different from the interior), fill the interior with the endpoint color.
    
    The segments that get filled are determined by: group segments by their 
    starting row, and fill the leftmost and rightmost segment in each group.
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Find all vertical segments with the border-interior-border pattern
    segments = []  # (start_row, end_row, col, border_color, interior_color)
    
    for c in range(cols):
        r = 0
        while r < rows:
            if grid[r, c] != 0:
                start = r
                while r < rows and grid[r, c] != 0:
                    r += 1
                end = r - 1
                height = end - start + 1
                
                if height >= 3:  # Must have at least one interior pixel
                    border_color = grid[start, c]
                    # Check all interior pixels are the same color and different from border
                    interior_colors = set(grid[start+1:end, c])
                    if len(interior_colors) == 1 and interior_colors.pop() != border_color:
                        interior_color = border_color  # placeholder
                        segments.append((start, end, c, border_color, grid[start+1, c]))
            else:
                r += 1
    
    # Group segments by starting row
    groups = defaultdict(list)
    for seg in segments:
        groups[seg[0]].append(seg)
    
    # For each group, identify leftmost and rightmost segments to fill
    segments_to_fill = set()
    for start_row, segs in groups.items():
        if len(segs) >= 1:
            # Sort by column
            segs_sorted = sorted(segs, key=lambda s: s[2])
            segments_to_fill.add(segs_sorted[0])   # leftmost
            segments_to_fill.add(segs_sorted[-1])  # rightmost
    
    # Fill the identified segments
    for (start, end, c, border_color, interior_color) in segments_to_fill:
        for r in range(start + 1, end):
            grid[r, c] = border_color
    
    return grid