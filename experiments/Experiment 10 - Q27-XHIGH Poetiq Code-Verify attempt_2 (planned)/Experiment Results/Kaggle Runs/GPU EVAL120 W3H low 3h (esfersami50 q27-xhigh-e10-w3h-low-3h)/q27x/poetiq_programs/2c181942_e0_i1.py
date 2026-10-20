import numpy as np
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform scattered colored shapes into vertically symmetric shapes
    centered in the grid.
    
    Steps:
    1. Find all non-background pixels for each color
    2. For each color, extract the shape within its bounding box
    3. Apply vertical symmetry (intersection of row and its mirror)
    4. For odd-height shapes, duplicate the middle row
    5. Sort colors by original horizontal center
    6. Place shapes left-to-right, vertically centered
    """
    bg = 8
    rows, cols = grid.shape
    out = np.full((rows, cols), bg, dtype=int)
    
    # Collect pixels per color
    color_pixels = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != bg:
                color_pixels[grid[r, c]].append((r, c))
    
    if not color_pixels:
        return out
    
    # Process each color: extract shape, make vertically symmetric
    shapes = {}
    centers = {}
    for color, pixels in color_pixels.items():
        rmin = min(p[0] for p in pixels)
        rmax = max(p[0] for p in pixels)
        cmin = min(p[1] for p in pixels)
        cmax = max(p[1] for p in pixels)
        
        h = rmax - rmin + 1
        w = cmax - cmin + 1
        
        # Extract shape
        shape = np.zeros((h, w), dtype=int)
        for (r, c) in pixels:
            shape[r - rmin, c - cmin] = 1
        
        # Apply vertical symmetry: intersection of row i and row H-1-i
        sym = np.zeros((h, w), dtype=int)
        for i in range(h):
            for j in range(w):
                sym[i, j] = shape[i, j] & shape[h - 1 - i, j]
        
        # For odd height, duplicate the middle row
        if h % 2 == 1:
            mid = h // 2
            new_h = h + 1
            new_sym = np.zeros((new_h, w), dtype=int)
            new_sym[:mid+1] = sym[:mid+1]
            new_sym[mid+1:] = sym[:mid][::-1]
            sym = new_sym
        
        shapes[color] = sym
        centers[color] = (cmin + cmax) / 2.0
    
    # Sort colors by original horizontal center
    sorted_colors = sorted(shapes.keys(), key=lambda c: centers[c])
    
    # Calculate total width
    total_w = sum(shapes[c].shape[1] for c in sorted_colors)
    
    # Find max height for vertical centering
    max_h = max(shapes[c].shape[0] for c in sorted_colors)
    
    # Start position
    start_col = (cols - total_w) // 2
    center_row = rows // 2
    
    # Place shapes
    col_offset = start_col
    for color in sorted_colors:
        sh = shapes[color]
        sh_h, sh_w = sh.shape
        row_offset = center_row - sh_h // 2
        
        for i in range(sh_h):
            for j in range(sh_w):
                if sh[i, j] == 1:
                    r = row_offset + i
                    c = col_offset + j
                    if 0 <= r < rows and 0 <= c < cols:
                        out[r, c] = color
        
        col_offset += sh_w
    
    return out