import numpy as np
from collections import Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a grid of 3x3 cells by propagating special cell patterns
    to form a symmetric shape (X, triangle, or diamond).
    
    The rule: special cells (differing from background) propagate diagonally
    from the center of the special region. A cell becomes special if it
    has exactly one special diagonal neighbor (in the adjacent row).
    """
    grid = grid.copy()
    rows, cols = grid.shape
    n_cells = (rows - 1) // 4
    
    # Get border value
    border_vals = [grid[r, c] for r in range(rows) for c in range(cols) 
                   if r % 4 == 0 or c % 4 == 0]
    border_val = Counter(border_vals).most_common(1)[0][0]
    
    # Extract all cell patterns
    def get_cell(grid, cr, cc):
        sr, sc = cr * 4 + 1, cc * 4 + 1
        return tuple(grid[sr:sr+3, sc:sc+3].flatten())
    
    patterns = {}
    for cr in range(n_cells):
        for cc in range(n_cells):
            patterns[(cr, cc)] = get_cell(grid, cr, cc)
    
    # Find background (most common) and special patterns
    pat_counts = Counter(patterns.values())
    bg_pat = pat_counts.most_common(1)[0][0]
    
    special_cells = {(r, c) for (r, c), p in patterns.items() if p != bg_pat}
    
    if not special_cells:
        return grid
    
    # Get the special pattern to use for filling
    special_pat_key = max(
        (k for k, v in pat_counts.items() if k != bg_pat),
        key=lambda k: pat_counts[k]
    )
    special_pat = np.array(special_pat_key).reshape(3, 3)
    
    # Determine base row and column range
    row_counts = Counter(r for r, c in special_cells)
    base_row = row_counts.most_common(1)[0][0]
    
    cols_in_base = sorted(c for r, c in special_cells if r == base_row)
    all_cols = [c for r, c in special_cells]
    min_c, max_c = min(all_cols), max(all_cols)
    center_c = (min_c + max_c) // 2
    
    # Build output: start with input special cells
    out = set(special_cells)
    
    # Propagate: a cell (r,c) is added if it has exactly 1 special diagonal 
    # neighbor in row r±1, and c is within [min_c, max_c]
    for iteration in range(n_cells):
        new_cells = set()
        for r in range(n_cells):
            for c in range(min_c, max_c + 1):
                if (r, c) in out:
                    continue
                # Check diagonal neighbors above
                diag_above = [(r-1, c-1), (r-1, c+1)]
                # Check diagonal neighbors below
                diag_below = [(r+1, c-1), (r+1, c+1)]
                
                count_above = sum(1 for d in diag_above if d in out)
                count_below = sum(1 for d in diag_below if d in out)
                
                if count_above == 1 or count_below == 1:
                    new_cells.add((r, c))
        
        if not new_cells:
            break
        out.update(new_cells)
    
    # Apply special pattern to all output special cells
    for (cr, cc) in out:
        sr, sc = cr * 4 + 1, cc * 4 + 1
        grid[sr:sr+3, sc:sc+3] = special_pat
    
    return grid