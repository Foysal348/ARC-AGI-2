import numpy as np
from collections import Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform grid by:
    1. Finding the vertical separator (column of 4s)
    2. Extracting 3x3 shapes from the left side
    3. Extracting the special pixel from the right side
    4. Ordering shapes column-by-column (left col top-to-bottom, then right col top-to-bottom)
    5. Placing compressed shapes in a connected serpentine path
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Find separator column
    sep_col = None
    for c in range(cols):
        if np.all(grid[:, c] == 4):
            sep_col = c
            break
    if sep_col is None:
        return grid
    
    left = grid[:, :sep_col]
    right = grid[:, sep_col+1:]
    n_rows, w = left.shape
    
    # Find special pixel
    special_val, special_r, special_c = 0, 0, 0
    found = False
    for r in range(n_rows):
        for c in range(right.shape[1]):
            if right[r, c] != 0:
                special_val = int(right[r, c])
                special_r, special_c = r, c
                found = True
                break
        if found:
            break
    
    # Find shape blocks
    non_zero_rows = np.any(left != 0, axis=1)
    non_zero_cols = np.any(left != 0, axis=0)
    
    block_row_starts = []
    in_block = False
    for r in range(n_rows):
        if non_zero_rows[r] and not in_block:
            block_row_starts.append(r)
            in_block = True
        elif not non_zero_rows[r]:
            in_block = False
    
    block_col_starts = []
    in_block = False
    for c in range(w):
        if non_zero_cols[c] and not in_block:
            block_col_starts.append(c)
            in_block = True
        elif not non_zero_cols[c]:
            in_block = False
    
    n_br = len(block_row_starts)
    n_bc = len(block_col_starts)
    
    # Extract and classify shapes
    def classify(block):
        """Classify a 3x3 block and return (orientation, length)."""
        b = block.tolist()
        color = int(b[0][0]) if b[0][0] != 0 else int(b[1][1])
        nz = block[block != 0]
        if len(nz) == 0:
            return color, 'H', 0
        cnt = Counter(nz.tolist())
        color = cnt.most_common(1)[0][0]
        
        # Check patterns
        if b[0] == [color, 0, color] and b[1] == [0, color, 0] and b[2] == [0, color, 0]:
            return color, 'V', 2  # T-shape
        if b[0] == [color, 0, color] and b[1] == [color, 0, color] and b[2] == [color, color, color]:
            return color, 'H', 2  # U-shape
        if b[0] == [color, color, 0] and b[1] == [color, 0, color] and b[2] == [0, color, 0]:
            return color, 'H', 3  # Diagonal
        if b[0] == [color, color, color] and b[1] == [0, color, 0] and b[2] == [color, 0, color]:
            return color, 'H', 4  # Arch
        # Fallback
        n = int(np.sum(block != 0))
        return color, 'H', min(max(n, 1), 4)
    
    shape_grid = [[None]*n_bc for _ in range(n_br)]
    for bi in range(n_br):
        for bj in range(n_bc):
            r0 = block_row_starts[bi]
            c0 = block_col_starts[bj]
            if r0 + 3 <= n_rows and c0 + 3 <= w:
                block = left[r0:r0+3, c0:c0+3]
                if np.any(block != 0):
                    shape_grid[bi][bj] = classify(block)
    
    # Order: column by column
    ordered = []
    for bj in range(n_bc):
        for bi in range(n_br):
            if shape_grid[bi][bj] is not None:
                ordered.append(shape_grid[bi][bj])
    
    # Place in output
    out = np.zeros((n_rows, w), dtype=int)
    if found:
        out[special_r, special_c] = special_val
    
    if not ordered:
        return out
    
    # Serpentine path:
    # Start at (special_r + 1, special_c)
    # H-shapes: extend in current h_dir, then flip
    # V-shapes: extend downward
    # After each shape, pen moves to next available position
    
    pen_r = special_r + 1
    pen_c = special_c
    h_dir = 1 if special_c <= (w-1)//2 else -1
    
    for color, orient, length in ordered:
        if pen_r >= n_rows:
            break
        if orient == 'H':
            for i in range(length):
                c = pen_c + i * h_dir
                if 0 <= c < w:
                    out[pen_r, c] = color
            pen_c = pen_c + (length - 1) * h_dir
            pen_r += 1
            h_dir = -h_dir
        else:  # V
            for i in range(length):
                r = pen_r + i
                if r < n_rows:
                    out[r, pen_c] = color
            pen_r += length
    
    return out