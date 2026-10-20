import numpy as np
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete structural lines in the grid.
    
    The transformation:
    1. Identifies horizontal/vertical lines (>=3 same-color cells in a row/col)
    2. Fills gaps in those lines (0s and noise colors 2,4)
    3. Resolves conflicts at intersections using the 'gap' rule
    4. Trims extra cells beyond intersection points
    """
    grid = grid.copy()
    rows, cols = grid.shape
    noise_colors = {2, 4}
    
    # Find all lines for each color
    def get_lines(grid, color):
        h_lines = []  # (row, start_col, end_col)
        v_lines = []  # (col, start_row, end_row)
        for r in range(grid.shape[0]):
            pos = [c for c in range(grid.shape[1]) if grid[r, c] == color]
            if len(pos) >= 3:
                h_lines.append((r, min(pos), max(pos)))
        for c in range(grid.shape[1]):
            pos = [r for r in range(grid.shape[0]) if grid[r, c] == color]
            if len(pos) >= 3:
                v_lines.append((c, min(pos), max(pos)))
        return h_lines, v_lines
    
    colors = set(grid[grid > 0].flatten().tolist())
    
    all_h = {}  # color -> list of (row, start, end)
    all_v = {}  # color -> list of (col, start, end)
    
    for color in colors:
        h, v = get_lines(grid, color)
        all_h[color] = h
        all_v[color] = v
    
    # Build claim map: (r,c) -> set of colors that claim it
    claims = defaultdict(set)
    for color, hlines in all_h.items():
        for (r, c1, c2) in hlines:
            for c in range(c1, c2 + 1):
                claims[(r, c)].add(color)
    for color, vlines in all_v.items():
        for (c, r1, r2) in vlines:
            for r in range(r1, r2 + 1):
                claims[(r, c)].add(color)
    
    new_grid = grid.copy()
    
    # Process conflicts and fill gaps
    for (r, c), color_set in claims.items():
        current = grid[r, c]
        
        if len(color_set) == 1:
            color = list(color_set)[0]
            if current == 0 or current in noise_colors:
                new_grid[r, c] = color
        elif len(color_set) > 1:
            # Conflict: multiple colors claim this cell
            # The cell becomes the color for which it's currently a gap
            gap_colors = [cl for cl in color_set if current != cl]
            if gap_colors:
                # Choose the gap color with fewer total line cells
                def count_cells(color):
                    total = 0
                    for (rr, c1, c2) in all_h.get(color, []):
                        total += c2 - c1 + 1
                    for (cc, r1, r2) in all_v.get(color, []):
                        total += r2 - r1 + 1
                    return total
                gap_colors.sort(key=count_cells)
                new_grid[r, c] = gap_colors[0]
    
    # Handle trimming of extra cells at line ends
    # For vertical lines that intersect horizontal lines of same color:
    # if extra cells exist on one side of the intersection, trim them
    # and extend on the other side
    for color in colors:
        hlines = all_h.get(color, [])
        vlines = all_v.get(color, [])
        
        for (vcol, vr1, vr2) in vlines:
            # Find horizontal lines intersecting this vertical
            inter_rows = [hr for (hr, hc1, hc2) in hlines 
                         if hc1 <= vcol <= hc2 and vr1 <= hr <= vr2]
            if not inter_rows:
                continue
            
            min_inter = min(inter_rows)
            max_inter = max(inter_rows)
            
            # Count cells above and below intersections
            above_count = sum(1 for r in range(vr1, min_inter) if grid[r, vcol] == color)
            below_count = sum(1 for r in range(max_inter + 1, vr2 + 1) if grid[r, vcol] == color)
            
            if above_count > 0 and below_count == 0:
                # Extra cells above - trim and extend below
                for r in range(vr1, min_inter):
                    if new_grid[r, vcol] == color:
                        new_grid[r, vcol] = 0
                # Extend below by similar amount
                extend = min(above_count + 2, rows - 1 - max_inter)
                for r in range(max_inter + 1, max_inter + 1 + extend):
                    if r < rows and (new_grid[r, vcol] == 0 or new_grid[r, vcol] in noise_colors):
                        new_grid[r, vcol] = color
            elif below_count > 0 and above_count == 0:
                # Extra cells below - trim and extend above
                for r in range(max_inter + 1, vr2 + 1):
                    if new_grid[r, vcol] == color:
                        new_grid[r, vcol] = 0
                extend = min(below_count + 2, max_inter)
                for r in range(max_inter - extend, max_inter):
                    if r >= 0 and (new_grid[r, vcol] == 0 or new_grid[r, vcol] in noise_colors):
                        new_grid[r, vcol] = color
    
    return new_grid