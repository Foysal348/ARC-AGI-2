import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Identifies objects (connected components) in the grid and emits diagonal rays
    from each object's tip in a direction determined by the object's shape,
    orientation, and position relative to the grid center.
    
    - Bars: ray from the end closest to center, perpendicular to bar axis
    - L-shapes: ray from corner into the empty quadrant  
    - Single pixels: rays in all 4 diagonal directions
    """
    if grid.size == 0:
        return grid.copy()
    
    rows, cols = grid.shape
    center_r, center_c = rows / 2.0, cols / 2.0
    result = grid.astype(int).copy()
    
    # Find all connected components (4-connected)
    visited = np.zeros_like(grid, dtype=bool)
    objects = []
    
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0 and not visited[r, c]:
                color = grid[r, c]
                component = []
                queue = deque([(r, c)])
                visited[r, c] = True
                while queue:
                    cr, cc = queue.popleft()
                    component.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr,nc] and grid[nr,nc] == color:
                            visited[nr,nc] = True
                            queue.append((nr, nc))
                objects.append((color, component))
    
    def emit_ray(sr, sc, dr, dc, color):
        """Emit a diagonal ray from (sr,sc) in direction (dr,dc)."""
        nr, nc = sr + dr, sc + dc
        while 0 <= nr < rows and 0 <= nc < cols and result[nr, nc] == 0:
            result[nr, nc] = color
            nr += dr
            nc += dc
    
    for color, component in objects:
        n = len(component)
        
        if n == 1:
            # Single pixel: emit in all 4 diagonal directions
            (r, c) = component[0]
            for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                emit_ray(r, c, dr, dc, color)
        
        else:
            r_min = min(r for r,c in component)
            r_max = max(r for r,c in component)
            c_min = min(c for r,c in component)
            c_max = max(c for r,c in component)
            
            if r_min == r_max:
                # Horizontal bar
                tip_c = c_min if abs(c_min - center_c) <= abs(c_max - center_c) else c_max
                tip_r = r_min
                if tip_c == c_min:
                    dc = 1
                    dr = 1 if tip_r >= center_r else -1
                else:
                    dc = -1
                    dr = 1 if tip_r >= center_r else -1
                emit_ray(tip_r, tip_c, dr, dc, color)
                
            elif c_min == c_max:
                # Vertical bar
                tip_r = r_min if abs(r_min - center_r) <= abs(r_max - center_r) else r_max
                tip_c = c_min
                if tip_r == r_min:
                    dr = 1
                    dc = 1 if tip_c >= center_c else -1
                else:
                    dr = -1
                    dc = 1 if tip_c >= center_c else -1
                emit_ray(tip_r, tip_c, dr, dc, color)
                
            else:
                # L-shape: find corner
                comp_set = set(component)
                corner = None
                for r, c in component:
                    if ((r-1,c) in comp_set and (r,c-1) in comp_set or
                        (r-1,c) in comp_set and (r,c+1) in comp_set or
                        (r+1,c) in comp_set and (r,c-1) in comp_set or
                        (r+1,c) in comp_set and (r,c+1) in comp_set):
                        corner = (r, c)
                        break
                if corner is None:
                    corner = min(component, key=lambda p: (p[0]-center_r)**2 + (p[1]-center_c)**2)
                
                cr, cc = corner
                up = (cr-1, cc) in comp_set
                down = (cr+1, cc) in comp_set
                left = (cr, cc-1) in comp_set
                right = (cr, cc+1) in comp_set
                
                if up and left:
                    emit_ray(cr, cc, 1, 1, color)
                elif up and right:
                    emit_ray(cr, cc, 1, -1, color)
                elif down and left:
                    emit_ray(cr, cc, -1, 1, color)
                elif down and right:
                    emit_ray(cr, cc, -1, -1, color)
    
    return result