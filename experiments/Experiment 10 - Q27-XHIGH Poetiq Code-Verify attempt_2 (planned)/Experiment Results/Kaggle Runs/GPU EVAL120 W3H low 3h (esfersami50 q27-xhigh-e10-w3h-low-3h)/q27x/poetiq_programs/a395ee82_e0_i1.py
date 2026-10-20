import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transformation:
    1. Find background (most common color), shape (largest non-bg group), markers (smaller non-bg group)
    2. Compute centers of shape and marker clusters
    3. Translate marker relative positions to shape center → new stamp positions
    4. Output: shape-color copies at new positions + marker-color shape at original position
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Identify colors
    unique_colors = np.unique(grid)
    color_counts = {c: np.sum(grid == c) for c in unique_colors}
    bg_color = max(color_counts, key=color_counts.get)
    
    non_bg = [(c, list(zip(*np.where(grid == c)))) for c in unique_colors if c != bg_color]
    
    if len(non_bg) < 2:
        return grid
    
    # Shape = more pixels, Markers = fewer pixels
    non_bg.sort(key=lambda x: len(x[1]), reverse=True)
    shape_color, shape_pixels = non_bg[0]
    marker_color, marker_pixels = non_bg[1]
    
    # Centers (as floats for precision)
    sr = np.mean([p[0] for p in shape_pixels])
    sc = np.mean([p[1] for p in shape_pixels])
    mr = np.mean([p[0] for p in marker_pixels])
    mc = np.mean([p[1] for p in marker_pixels])
    
    # Shape pattern (integer offsets from shape center)
    shape_offsets = [(int(round(r - sr)), int(round(c - sc))) for r, c in shape_pixels]
    
    # Marker offsets from marker center
    marker_offsets = [(int(round(r - mr)), int(round(c - mc))) for r, c in marker_pixels]
    
    # Build output
    output = np.full((rows, cols), bg_color, dtype=grid.dtype)
    
    # Place shape in marker_color at original shape position
    for dr, dc in shape_offsets:
        r, c = int(round(sr + dr)), int(round(sc + dc))
        if 0 <= r < rows and 0 <= c < cols:
            output[r, c] = marker_color
    
    # Place shape in shape_color at each translated marker position
    for dr, dc in marker_offsets:
        nr, nc = sr + dr, sc + dc
        for sdr, sdc in shape_offsets:
            r, c = int(round(nr + sdr)), int(round(nc + sdc))
            if 0 <= r < rows and 0 <= c < cols:
                output[r, c] = shape_color
    
    return output