import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a grid containing pattern frames, separator lines, and markers
    into an output grid where patterns are placed at specified positions.
    
    The input has a 3-row grid of N×N cells separated by lines of 5s:
    - Top row: pattern templates (each has a dominant color and shape)
    - Middle row: placement maps (pixel positions indicate output locations)
    - Bottom row: marker colors (single non-zero pixel per cell)
    
    For each middle cell, its reference color matches a top cell's color.
    The matching top cell's pattern is colored with the corresponding bottom
    cell's marker color and placed at the positions indicated by the middle cell.
    """
    grid = np.array(grid)
    rows = grid.shape[0]
    cols = grid.shape[1]
    
    # Find separator rows (contain 5s, all non-5 values are 0)
    sep_rows = []
    for r in range(rows):
        row_vals = grid[r]
        if np.any(row_vals == 5) and not np.any((row_vals != 5) & (row_vals != 0)):
            sep_rows.append(r)
    
    # Find separator columns (contain 5s, all non-5 values are 0)
    sep_cols = []
    for c in range(cols):
        col_vals = grid[:, c]
        if np.any(col_vals == 5) and not np.any((col_vals != 5) & (col_vals != 0)):
            sep_cols.append(c)
    
    # Determine cell row boundaries
    row_bounds = []
    prev = 0
    for sr in sep_rows:
        if sr > prev:
            row_bounds.append((prev, sr))
        prev = sr + 1
    if prev < rows:
        row_bounds.append((prev, rows))
    
    # Determine cell column boundaries
    col_bounds = []
    prev = 0
    for sc in sep_cols:
        if sc > prev:
            col_bounds.append((prev, sc))
        prev = sc + 1
    if prev < cols:
        col_bounds.append((prev, cols))
    
    # Cell size N
    N = row_bounds[0][1] - row_bounds[0][0]
    n_rows = len(row_bounds)
    n_cols = len(col_bounds)
    
    def get_cell(r, c):
        return grid[row_bounds[r][0]:row_bounds[r][1], col_bounds[c][0]:col_bounds[c][1]]
    
    # Extract top row: patterns (boolean masks) and dominant colors
    top_patterns = []
    top_colors = []
    for c in range(n_cols):
        cell = get_cell(0, c)
        mask = (cell != 0) & (cell != 5)
        top_patterns.append(mask)
        if np.any(mask):
            vals = cell[mask]
            colors, counts = np.unique(vals, return_counts=True)
            top_colors.append(int(colors[np.argmax(counts)]))
        else:
            top_colors.append(0)
    
    # Extract bottom row: marker colors
    bottom_markers = []
    for c in range(n_cols):
        cell = get_cell(n_rows - 1, c)
        nz = cell[cell != 0]
        bottom_markers.append(int(nz[0]) if len(nz) > 0 else 0)
    
    # Extract middle row: reference colors and positions
    middle_refs = []
    middle_positions = []
    for c in range(n_cols):
        cell = get_cell(1, c)
        mask = (cell != 0) & (cell != 5)
        if np.any(mask):
            middle_refs.append(int(cell[mask][0]))
            middle_positions.append(np.argwhere(mask))
        else:
            middle_refs.append(0)
            middle_positions.append(np.array([], dtype=int).reshape(0, 2))
    
    # Build output grid
    out_size = N * N
    output = np.zeros((out_size, out_size), dtype=int)
    
    for c in range(n_cols):
        ref = middle_refs[c]
        positions = middle_positions[c]
        if ref == 0 or len(positions) == 0:
            continue
        
        # Find the top cell whose color matches the reference
        match = -1
        for tc in range(n_cols):
            if top_colors[tc] == ref:
                match = tc
                break
        if match == -1:
            continue
        
        pattern = top_patterns[match]
        color = bottom_markers[match]
        
        # Place the colored pattern at each specified position
        for (r, col) in positions:
            r, col = int(r), int(col)
            output[r*N:(r+1)*N, col*N:(col+1)*N] = np.where(pattern, color, 0)
    
    return output