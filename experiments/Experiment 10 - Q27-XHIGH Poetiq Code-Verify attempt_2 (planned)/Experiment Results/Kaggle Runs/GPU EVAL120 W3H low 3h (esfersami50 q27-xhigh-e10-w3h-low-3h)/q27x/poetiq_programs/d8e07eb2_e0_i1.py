import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform grid by:
    1. Finding 3x3 shapes in the top section and matching them in the middle section.
    2. Drawing 3-colored rectangular borders around matched shapes (merging adjacent).
    3. Filling top/bottom sections: with 3 if all matches align in one row/col, else bottom with 2.
    """
    result = grid.copy()
    rows, cols = grid.shape
    
    # Define 5x5 cell positions (row ranges, col ranges)
    row_cells = [(0, 4), (7, 11), (12, 16), (17, 21), (22, 26)]
    col_cells = [(1, 5), (6, 10), (11, 15), (16, 20)]
    
    def extract_shape(r_start, r_end, c_start, c_end):
        """Extract the 3x3 shape pattern from a 5x5 cell region."""
        best_pattern = None
        best_score = -1
        for dr in [1, 2]:
            for dc in [1, 2]:
                r = r_start + dr
                c = c_start + dc
                if r + 3 > r_end + 1 or c + 3 > c_end + 1:
                    continue
                block = grid[r:r+3, c:c+3]
                score = int(np.sum((block != 8) & (block != 6)))
                if score > best_score:
                    best_score = score
                    best_pattern = block.copy()
        if best_score == 0:
            return None
        return best_pattern
    
    # Extract top section patterns (row cell index 0)
    top_patterns = []
    for c_idx in range(4):
        cs, ce = col_cells[c_idx]
        top_patterns.append(extract_shape(row_cells[0][0], row_cells[0][1], cs, ce))
    
    # Find matched middle shapes
    matched_cells = set()
    for r_idx in range(1, 5):
        rs, re = row_cells[r_idx]
        for c_idx in range(4):
            cs, ce = col_cells[c_idx]
            pattern = extract_shape(rs, re, cs, ce)
            if pattern is None:
                continue
            for tp in top_patterns:
                if tp is not None and np.array_equal(pattern, tp):
                    matched_cells.add((r_idx, c_idx))
                    break
    
    # Find connected components of matched cells
    def get_component(start):
        component = set()
        queue = [start]
        while queue:
            cell = queue.pop()
            if cell in component:
                continue
            component.add(cell)
            ri, ci = cell
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nb = (ri + dr, ci + dc)
                if nb in matched_cells and nb not in component:
                    queue.append(nb)
        return component
    
    components = []
    visited = set()
    for cell in matched_cells:
        if cell not in visited:
            comp = get_component(cell)
            components.append(comp)
            visited.update(comp)
    
    # Draw merged rectangle borders with color 3
    for comp in components:
        r_indices = [c[0] for c in comp]
        c_indices = [c[1] for c in comp]
        r_top = row_cells[min(r_indices)][0]
        r_bot = row_cells[max(r_indices)][1]
        c_left = col_cells[min(c_indices)][0]
        c_right = col_cells[max(c_indices)][1]
        for r in range(r_top, r_bot + 1):
            for c in range(c_left, c_right + 1):
                if r == r_top or r == r_bot or c == c_left or c == c_right:
                    result[r, c] = 3
    
    # Determine fill color based on alignment
    all_same_row = len(set(r for r, c in matched_cells)) <= 1
    all_same_col = len(set(c for r, c in matched_cells)) <= 1
    
    if all_same_row or all_same_col:
        fill_color = 3
        # Fill top section (rows 0-4), preserving non-8 pixels
        for r in range(5):
            for c in range(cols):
                if grid[r, c] == 8:
                    result[r, c] = fill_color
    else:
        fill_color = 2
    
    # Fill bottom section (last 2 rows)
    for r in range(rows - 2, rows):
        for c in range(cols):
            result[r, c] = fill_color
    
    return result