import numpy as np
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform by:
    1. Finding 8-pattern with anchor at first 8 (row-major)
    2. For each non-8 color, remove outlier pixels (far from centroid)
    3. Stamp 8-pattern at remaining pixels
    4. Assign output colors by group size ranking
    """
    rows, cols = grid.shape
    new_grid = np.zeros_like(grid)
    
    # Find 8 positions
    eight_positions = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 8:
                eight_positions.append((r, c))
    
    if not eight_positions:
        return new_grid
    
    anchor_r, anchor_c = eight_positions[0]
    brush_offsets = set()
    for (r, c) in eight_positions:
        brush_offsets.add((r - anchor_r, c - anchor_c))
    
    # Find non-8, non-0 pixels by color
    color_pixels = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            val = grid[r, c]
            if val != 0 and val != 8:
                color_pixels[val].append((r, c))
    
    # For each color, remove outliers based on distance from centroid
    def filter_outliers(pixels):
        if len(pixels) <= 2:
            return set(pixels)
        rs = [p[0] for p in pixels]
        cs = [p[1] for p in pixels]
        cr, cc = np.mean(rs), np.mean(cs)
        dists = [np.sqrt((r-cr)**2 + (c-cc)**2) for r, c in pixels]
        median_d = np.median(dists)
        threshold = max(2.0 * median_d, 2.5)
        kept = set()
        for (p, d) in zip(pixels, dists):
            if d <= threshold:
                kept.add(p)
        return kept if kept else set(pixels)
    
    main_groups = {}
    for color, pixels in color_pixels.items():
        group = filter_outliers(pixels)
        if group:
            main_groups[color] = group
    
    # Rank by size, assign colors
    sorted_groups = sorted(main_groups.items(), key=lambda x: -len(x[1]))
    available_colors = sorted(color_pixels.keys())
    
    color_assignment = {}
    for i, (orig_color, group) in enumerate(sorted_groups):
        if i < len(available_colors):
            color_assignment[orig_color] = available_colors[i]
        else:
            color_assignment[orig_color] = available_colors[-1]
    
    # Stamp
    for orig_color, group in main_groups.items():
        out_color = color_assignment[orig_color]
        for (r, c) in group:
            for (dr, dc) in brush_offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    new_grid[nr, nc] = out_color
    
    return new_grid