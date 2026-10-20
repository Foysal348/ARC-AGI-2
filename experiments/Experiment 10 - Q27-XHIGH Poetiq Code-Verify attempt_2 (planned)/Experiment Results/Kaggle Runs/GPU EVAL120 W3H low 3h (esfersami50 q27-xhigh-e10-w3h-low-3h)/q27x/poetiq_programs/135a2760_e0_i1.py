import numpy as np
from collections import Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fix corrupted cells in a grid of pattern blocks.
    
    The grid contains multiple blocks separated by borders (8s) and block frames (3s).
    Each block has a repeating 2D pattern with some cells corrupted.
    We identify the period of each block's pattern and use majority voting
    across all tiles to determine the correct pattern, then rebuild.
    """
    result = grid.copy()
    rows, cols = grid.shape
    
    # Find vertical separators (columns that are all 8s in the inner area)
    inner_rows = range(2, rows - 2)
    vertical_seps = set()
    for c in range(1, cols - 1):
        if all(grid[r, c] == 8 for r in inner_rows):
            vertical_seps.add(c)
    
    vsep_list = sorted(vertical_seps)
    
    # Find block column ranges (between consecutive vertical separators)
    boundaries = [1] + vsep_list + [cols - 1]
    block_cols = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end - start >= 4:  # minimum width for a block with 3-borders
            block_cols.append((start, end))
    
    content_rows = range(2, rows - 2)
    
    for (b_start, b_end) in block_cols:
        # Content area is between the 3-borders
        c_start = b_start + 1
        c_end = b_end - 1  # exclusive
        
        if c_end <= c_start:
            continue
        
        width = c_end - c_start
        height = len(list(content_rows))
        
        # Extract content
        content = grid[list(content_rows), c_start:c_end].copy()
        
        # Find row period using autocorrelation
        def find_period(values, max_period):
            for p in range(1, max_period + 1):
                # Check if shifting by p gives same values (with tolerance for errors)
                matches = 0
                total = 0
                for i in range(len(values) - p):
                    for j in range(len(values[0])):
                        total += 1
                        if values[i][j] == values[i + p][j]:
                            matches += 1
                if total > 0 and matches / total > 0.85:
                    return p
            return None
        
        # Find row period
        row_period = None
        for p in range(1, min(height, 12)):
            matches = 0
            total = 0
            for i in range(height - p):
                for j in range(width):
                    total += 1
                    if content[i, j] == content[i + p, j]:
                        matches += 1
            if total > 0 and matches / total > 0.85:
                row_period = p
                break
        
        if row_period is None or row_period >= height:
            continue
        
        # Find column period
        col_period = None
        for p in range(1, min(width + 1, 6)):
            matches = 0
            total = 0
            for i in range(height):
                for j in range(width - p):
                    total += 1
                    if content[i, j] == content[i, j + p]:
                        matches += 1
            if total > 0 and matches / total > 0.85:
                col_period = p
                break
        
        if col_period is None or col_period > width:
            # Try treating the full width as the period
            col_period = width
        
        # Determine the tile pattern using majority voting
        tile = np.zeros((row_period, col_period), dtype=grid.dtype)
        
        for dr in range(row_period):
            for dc in range(col_period):
                values = []
                for i in range(dr, height, row_period):
                    for j in range(dc, width, col_period):
                        values.append(content[i, j])
                if values:
                    tile[dr, dc] = Counter(values).most_common(1)[0][0]
        
        # Rebuild the content from the tile pattern
        for i in range(height):
            for j in range(width):
                result[2 + i, c_start + j] = tile[i % row_period, j % col_period]
    
    return result