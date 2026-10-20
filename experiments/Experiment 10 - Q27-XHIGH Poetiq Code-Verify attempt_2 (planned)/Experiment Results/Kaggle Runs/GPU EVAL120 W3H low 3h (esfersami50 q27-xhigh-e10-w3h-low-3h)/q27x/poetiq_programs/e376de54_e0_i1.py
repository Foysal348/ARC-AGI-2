import numpy as np
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Identify all line segments in the grid, determine the second-longest
    line length, and resize all lines to that length (anchored at their
    'bottom' - the endpoint with the highest row index).
    """
    bg = grid[0, 0]
    rows, cols = grid.shape
    
    # Collect all non-background cells
    cells = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != bg:
                cells[(r, c)] = grid[r, c]
    
    if not cells:
        return grid.copy()
    
    # Detect orientation by checking which grouping key gives consistent single-cell lines
    # Anti-diagonal: r + c = const
    # Diagonal: r - c = const
    # Horizontal: r = const
    # Vertical: c = const
    
    orientations = [
        ('anti_diag', lambda r, c: r + c),
        ('diag', lambda r, c: r - c),
        ('horiz', lambda r, c: r),
        ('vert', lambda r, c: c),
    ]
    
    def get_lines(orient_name, key_fn):
        groups = defaultdict(set)
        for (r, c) in cells:
            groups[key_fn(r, c)].add((r, c))
        
        # Check if each group forms a contiguous line
        lines = []
        for key, group in groups.items():
            if len(group) < 2:
                continue
            # Check contiguity
            sorted_cells = sorted(group, key=lambda x: (x[0], x[1]))
            contiguous = True
            for i in range(len(sorted_cells) - 1):
                r1, c1 = sorted_cells[i]
                r2, c2 = sorted_cells[i + 1]
                if orient_name == 'anti_diag':
                    if not (r2 - r1 == 1 and c1 - c2 == 1):
                        contiguous = False
                        break
                elif orient_name == 'diag':
                    if not (r2 - r1 == 1 and c2 - c1 == 1):
                        contiguous = False
                        break
                elif orient_name == 'horiz':
                    if not (r1 == r2 and c2 - c1 == 1):
                        contiguous = False
                        break
                elif orient_name == 'vert':
                    if not (c1 == c2 and r2 - r1 == 1):
                        contiguous = False
                        break
            if contiguous:
                lines.append((key, sorted_cells))
        return lines
    
    # Find the correct orientation
    chosen_lines = None
    for orient_name, key_fn in orientations:
        lines = get_lines(orient_name, key_fn)
        # Check that all non-bg cells are covered by lines
        covered = set()
        for _, line_cells in lines:
            covered.update(line_cells)
        if covered == set(cells.keys()) and len(lines) >= 2:
            chosen_lines = lines
            break
    
    if chosen_lines is None:
        return grid.copy()
    
    # Get lengths and find second longest
    lengths = [len(line_cells) for _, line_cells in chosen_lines]
    sorted_lengths = sorted(set(lengths), reverse=True)
    
    if len(sorted_lengths) >= 2:
        target = sorted_lengths[1]
    else:
        target = sorted_lengths[0]
    
    # Create output grid
    output = np.full_like(grid, bg)
    
    for key, line_cells in chosen_lines:
        # line_cells is sorted by (row, col)
        # The "bottom" is the last cell (highest row)
        bottom = line_cells[-1]
        br, bc = bottom
        
        new_cells = []
        for i in range(target):
            if 'diag' in orient_name or orient_name == 'anti_diag':
                # For anti-diagonal: going up means r decreases, c increases
                # For diagonal: going up means r decreases, c decreases
                if orient_name == 'anti_diag':
                    nr = br - i
                    nc = bc + i
                else:
                    nr = br - i
                    nc = bc - i
            elif orient_name == 'horiz':
                nr = br
                nc = bc - i
            else:  # vert
                nr = br - i
                nc = bc
            
            if 0 <= nr < rows and 0 <= nc < cols:
                new_cells.append((nr, nc))
        
        # Get color from original line
        color = cells[bottom]
        for (nr, nc) in new_cells:
            output[nr, nc] = color
    
    return output