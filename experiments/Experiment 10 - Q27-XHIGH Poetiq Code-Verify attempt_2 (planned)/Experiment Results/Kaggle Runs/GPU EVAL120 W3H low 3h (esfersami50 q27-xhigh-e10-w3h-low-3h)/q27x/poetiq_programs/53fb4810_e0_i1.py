import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Find crosses of 1s with adjacent non-1, non-background patterns and extend
    those patterns to the nearest grid edge.
    
    Algorithm:
    1. Find connected components of 1s (4-connectivity)
    2. Identify cross shapes (plus-shaped with both vertical and horizontal extent)
    3. For each cross, find the vertical stem (columns with 1s in both top and bottom rows)
    4. Find contiguous non-1, non-background patterns adjacent to the cross in stem columns
    5. Extend the pattern by repeating it to the nearest grid edge
    """
    result = grid.copy()
    rows, cols = grid.shape
    background = 8
    
    # Find connected components of 1s using BFS
    mask_1 = (grid == 1)
    visited = np.zeros((rows, cols), dtype=bool)
    
    for start_r in range(rows):
        for start_c in range(cols):
            if not mask_1[start_r, start_c] or visited[start_r, start_c]:
                continue
            
            # BFS to find the component
            component = []
            queue = deque([(start_r, start_c)])
            visited[start_r, start_c] = True
            while queue:
                r, c = queue.popleft()
                component.append((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and mask_1[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            
            # Analyze this component for cross shape
            comp_rows = np.array([p[0] for p in component])
            comp_cols = np.array([p[1] for p in component])
            
            if len(comp_rows) < 5:
                continue
            
            r_min, r_max = comp_rows.min(), comp_rows.max()
            c_min, c_max = comp_cols.min(), comp_cols.max()
            
            # Must have both vertical and horizontal extent
            if (r_max - r_min < 2) or (c_max - c_min < 2):
                continue
            
            # Find stem columns: columns with 1s in both top and bottom rows
            top_cols = set(comp_cols[comp_rows == r_min].tolist())
            bottom_cols = set(comp_cols[comp_rows == r_max].tolist())
            stem_cols = sorted(top_cols & bottom_cols)
            
            if not stem_cols:
                continue
            
            # Check above the cross for a pattern to extend
            if r_min > 0:
                pattern_rows = []
                for r in range(r_min - 1, -1, -1):
                    if all(grid[r, c] != 1 and grid[r, c] != background for c in stem_cols):
                        pattern_rows.append(r)
                    else:
                        break
                if pattern_rows:
                    pattern_rows.reverse()  # top to bottom order
                    p_top = pattern_rows[0]
                    p_bottom = pattern_rows[-1]
                    h = p_bottom - p_top + 1
                    pattern_values = [[int(grid[r, c]) for c in stem_cols] for r in pattern_rows]
                    
                    # Fill from grid top to pattern top with repeated pattern
                    for r in range(p_top):
                        d = p_top - r
                        idx = d % h
                        for j, c in enumerate(stem_cols):
                            result[r, c] = pattern_values[idx][j]
            
            # Check below the cross for a pattern to extend
            if r_max < rows - 1:
                pattern_rows = []
                for r in range(r_max + 1, rows):
                    if all(grid[r, c] != 1 and grid[r, c] != background for c in stem_cols):
                        pattern_rows.append(r)
                    else:
                        break
                if pattern_rows:
                    p_top = pattern_rows[0]
                    p_bottom = pattern_rows[-1]
                    h = p_bottom - p_top + 1
                    pattern_values = [[int(grid[r, c]) for c in stem_cols] for r in pattern_rows]
                    
                    # Fill from pattern bottom to grid bottom with repeated pattern
                    for r in range(p_bottom + 1, rows):
                        d = r - p_bottom
                        idx = d % h
                        for j, c in enumerate(stem_cols):
                            result[r, c] = pattern_values[idx][j]
    
    return result