import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Recolor connected components of 5s by matching them to reference shapes
    based on the number of enclosed holes in their bounding box pattern.
    
    A 'hole' is a 0-cell in the bounding box that is completely surrounded
    by 1-cells in all 4 cardinal directions (within the bounding box).
    """
    result = grid.copy()
    rows, cols = grid.shape
    
    # Find separator line
    sep_row = -1
    for r in range(rows):
        if np.all(grid[r] == 1):
            sep_row = r
            break
    
    sep_col = -1
    if sep_row == -1:
        for c in range(cols):
            if np.all(grid[:, c] == 1):
                sep_col = c
                break
    
    # Determine reference area
    if sep_row != -1:
        ref_area = grid[:sep_row, :]
    else:
        ref_area = grid[:, :sep_col]
    
    # Find reference colors
    used_colors = set()
    for r in range(ref_area.shape[0]):
        for c in range(ref_area.shape[1]):
            v = ref_area[r, c]
            if v not in [0, 1, 5]:
                used_colors.add(v)
    
    # Extract reference shape properties
    ref_info = {}
    for color in used_colors:
        mask = (ref_area == color).astype(int)
        rows_idx = np.where(mask.any(axis=1))[0]
        cols_idx = np.where(mask.any(axis=0))[0]
        if len(rows_idx) == 0:
            continue
        r_min, r_max = rows_idx[0], rows_idx[-1]
        c_min, c_max = cols_idx[0], cols_idx[-1]
        pattern = mask[r_min:r_max+1, c_min:c_max+1]
        holes = count_holes(pattern)
        empty_count = int((pattern == 0).sum())
        ref_info[color] = (holes, empty_count)
    
    # Group by hole count
    holes_to_colors = {}
    for color, (holes, empty_count) in ref_info.items():
        if holes not in holes_to_colors:
            holes_to_colors[holes] = []
        holes_to_colors[holes].append((color, empty_count))
    
    # Process 5-components
    visited = np.zeros_like(grid, dtype=bool)
    
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 5 and not visited[r, c]:
                component = []
                queue = deque([(r, c)])
                visited[r, c] = True
                while queue:
                    cr, cc = queue.popleft()
                    component.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if grid[nr, nc] == 5 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                
                comp_rows = [p[0] for p in component]
                comp_cols = [p[1] for p in component]
                r_min, r_max = min(comp_rows), max(comp_rows)
                c_min, c_max = min(comp_cols), max(comp_cols)
                
                pattern = np.zeros((r_max - r_min + 1, c_max - c_min + 1), dtype=int)
                for cr, cc in component:
                    pattern[cr - r_min, cc - c_min] = 1
                
                holes = count_holes(pattern)
                empty_count = int((pattern == 0).sum())
                
                # Find best matching reference color
                target_color = None
                if holes in holes_to_colors:
                    candidates = holes_to_colors[holes]
                    if len(candidates) == 1:
                        target_color = candidates[0][0]
                    else:
                        min_diff = float('inf')
                        for color, ec in candidates:
                            diff = abs(ec - empty_count)
                            if diff < min_diff:
                                min_diff = diff
                                target_color = color
                else:
                    min_diff = float('inf')
                    for h, candidates in holes_to_colors.items():
                        diff = abs(h - holes)
                        if diff < min_diff:
                            min_diff = diff
                            target_color = candidates[0][0]
                
                if target_color is not None:
                    for cr, cc in component:
                        result[cr, cc] = target_color
    
    return result


def count_holes(pattern):
    """Count 0-cells fully surrounded by 1-cells in 4 directions within the pattern."""
    if pattern.size == 0:
        return 0
    h, w = pattern.shape
    holes = 0
    for r in range(h):
        for c in range(w):
            if pattern[r, c] == 0:
                surrounded = True
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        surrounded = False
                        break
                    if pattern[nr, nc] == 0:
                        surrounded = False
                        break
                if surrounded:
                    holes += 1
    return holes