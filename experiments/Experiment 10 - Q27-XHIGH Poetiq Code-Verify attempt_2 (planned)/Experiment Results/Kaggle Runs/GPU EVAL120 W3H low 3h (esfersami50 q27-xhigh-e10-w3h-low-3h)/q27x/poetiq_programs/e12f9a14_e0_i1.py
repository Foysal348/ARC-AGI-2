import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    For each shape (border of 3s with a fill color inside), identify gaps in the
    border (perimeter positions of the bounding box containing background color).
    From each gap, extend a ray of the fill color outward in the direction away
    from the shape (determined by which edge/corner the gap is on).
    Rays extend until they hit the grid boundary or a non-background pixel.
    """
    result = grid.copy()
    rows, cols = grid.shape
    
    # Identify background color (most common on grid border)
    border_pixels = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
    unique, counts = np.unique(border_pixels, return_counts=True)
    bg_color = int(unique[np.argmax(counts)])
    
    BORDER = 3
    visited = np.zeros((rows, cols), dtype=bool)
    
    # Find connected components of border pixels
    components = []
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == BORDER and not visited[i, j]:
                comp = []
                queue = deque([(i, j)])
                visited[i, j] = True
                while queue:
                    r, c = queue.popleft()
                    comp.append((r, c))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and grid[nr, nc] == BORDER:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                components.append(comp)
    
    for comp in components:
        rs = [r for r, c in comp]
        cs = [c for r, c in comp]
        r_min, r_max = min(rs), max(rs)
        c_min, c_max = min(cs), max(cs)
        
        # Find fill color
        fill = None
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                if grid[r, c] != bg_color and grid[r, c] != BORDER:
                    fill = int(grid[r, c])
                    break
            if fill is not None:
                break
        if fill is None:
            continue
        
        # Find gaps on the perimeter of the bounding box
        gaps = set()
        for c in range(c_min, c_max + 1):
            if grid[r_min, c] == bg_color:
                gaps.add((r_min, c))
            if grid[r_max, c] == bg_color:
                gaps.add((r_max, c))
        for r in range(r_min, r_max + 1):
            if grid[r, c_min] == bg_color:
                gaps.add((r, c_min))
            if grid[r, c_max] == bg_color:
                gaps.add((r, c_max))
        
        # Extend rays from each gap
        for (gr, gc) in gaps:
            top = (gr == r_min)
            bot = (gr == r_max)
            left = (gc == c_min)
            right = (gc == c_max)
            
            if top and left:
                dr, dc = -1, -1
            elif top and right:
                dr, dc = -1, 1
            elif bot and left:
                dr, dc = 1, -1
            elif bot and right:
                dr, dc = 1, 1
            elif top:
                dr, dc = -1, 0
            elif bot:
                dr, dc = 1, 0
            elif left:
                dr, dc = 0, -1
            elif right:
                dr, dc = 0, 1
            else:
                continue
            
            result[gr, gc] = fill
            r, c = gr + dr, gc + dc
            while 0 <= r < rows and 0 <= c < cols:
                if result[r, c] != bg_color:
                    break
                result[r, c] = fill
                r += dr
                c += dc
    
    return result