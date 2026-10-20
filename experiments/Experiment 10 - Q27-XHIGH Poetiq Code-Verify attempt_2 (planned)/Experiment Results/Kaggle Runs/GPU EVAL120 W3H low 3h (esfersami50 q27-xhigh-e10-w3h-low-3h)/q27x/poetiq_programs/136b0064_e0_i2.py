import numpy as np
from collections import Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform grid by:
    1. Finding the vertical separator (column of 4s)
    2. Extracting 3x3 shapes from the left side in a grid pattern
    3. Extracting the special pixel from the right side
    4. Classifying each shape into (orientation, length)
    5. Placing compressed lines in a serpentine path from the special pixel
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
    
    # Find special pixel (only non-zero in right side)
    special_val, special_r, special_c = 0, 0, 0
    for r in range(n_rows):
        for c in range(right.shape[1]):
            if right[r, c] != 0:
                special_val = int(right[r, c])
                special_r, special_c = r, c
                break
        if special_val != 0:
            break
    
    # Find shape block boundaries
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
    
    # Classify a 3x3 block
    def classify(block):
        b = [row[:] for row in block.tolist()]
        nz = block[block != 0]
        if len(nz) == 0:
            return 0, 'H', 0
        color = int(Counter(nz.tolist()).most_common(1)[0][0])
        
        # Check patterns
        if b[0] == [color, 0, color] and b[1] == [0, color, 0] and b[2] == [0, color, 0]:
            return color, 'V', 2  # T-shape → vertical, length 2
        if b[0] == [color, 0, color] and b[1] == [color, 0, color] and b[2] == [color, color, color]:
            return color, 'H', 2  # U-shape → horizontal, length 2
        if b[0] == [color, color, 0] and b[1] == [color, 0, color] and b[2] == [0, color, 0]:
            return color, 'H', 3  # Cross → horizontal, length 3
        if b[0] == [color, color, color] and b[1] == [0, color, 0] and b[2] == [color, 0, color]:
            return color, 'H', 4  # Arch → horizontal, length 4
        # Fallback: count pixels
        n = int(np.sum(block != 0))
        return color, 'H', max(1, min(n, 4))
    
    # Extract shapes in column-by-column, top-to-bottom order
    ordered = []
    for bj in range(n_bc):
        for bi in range(n_br):
            r0 = block_row_starts[bi]
            c0 = block_col_starts[bj]
            if r0 + 3 <= n_rows and c0 + 3 <= w:
                block = left[r0:r0+3, c0:c0+3]
                if np.any(block != 0):
                    ordered.append(classify(block))
    
    # Create output
    out = np.zeros((n_rows, w), dtype=int)
    if special_val != 0:
        out[special_r, special_c] = special_val
    
    if not ordered:
        return out
    
    # Serpentine path placement
    pen_r = special_r
    pen_c = special_c
    
    for i, (color, orient, length) in enumerate(ordered):
        if orient == 'H':
            # Determine direction: alternate starting from left
            if i % 2 == 0:
                # Go right
                start_c = pen_c
                for j in range(length):
                    c = start_c + j
                    if 0 <= c < w and pen_r < n_rows:
                        out[pen_r, c] = color
                # Next position: end of this line, move down
                pen_c = start_c + length - 1
                pen_r += 1
            else:
                # Go left
                end_c = pen_c
                for j in range(length):
                    c = end_c - j
                    if 0 <= c < w and pen_r < n_rows:
                        out[pen_r, c] = color
                # Next position: end of this line, move down
                pen_c = end_c - (length - 1)
                pen_r += 1
        else:  # V
            for j in range(length):
                r = pen_r + j
                if r < n_rows:
                    out[r, pen_c] = color
            # Next position: end of vertical line
            pen_r += length
    
    return out