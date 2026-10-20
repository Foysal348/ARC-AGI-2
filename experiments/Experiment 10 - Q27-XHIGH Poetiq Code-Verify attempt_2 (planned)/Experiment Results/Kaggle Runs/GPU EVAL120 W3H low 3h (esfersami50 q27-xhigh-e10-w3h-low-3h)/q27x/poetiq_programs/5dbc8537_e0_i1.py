import numpy as np
from collections import deque, Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform input grid to compact output:
    1. Identify wall color (dominant border color)
    2. Identify floor color (dominant non-wall color)  
    3. Extract objects (connected components of other colors)
    4. Pack objects vertically in reading order, centered horizontally
    """
    grid = grid.astype(int)
    rows, cols = grid.shape
    
    # Identify wall color from border
    border_pixels = []
    border_pixels.extend(grid[0, :].tolist())
    border_pixels.extend(grid[-1, :].tolist())
    border_pixels.extend(grid[:, 0].tolist())
    border_pixels.extend(grid[:, -1].tolist())
    wall_color = Counter(border_pixels).most_common(1)[0][0]
    
    # Identify floor color (most common non-wall color)
    non_wall = grid[grid != wall_color]
    if len(non_wall) > 0:
        floor_color = Counter(non_wall.tolist()).most_common(1)[0][0]
    else:
        floor_color = 0
    
    # Find objects
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects = []
    
    for i in range(h):
        for j in range(w):
            if not visited[i, j] and grid[i, j] != wall_color and grid[i, j] != floor_color:
                queue = deque([(i, j)])
                visited[i, j] = True
                cells = []
                while queue:
                    r, c = queue.popleft()
                    cells.append((r, c))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] != wall_color and grid[nr, nc] != floor_color:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                
                min_r = min(r for r, c in cells)
                max_r = max(r for r, c in cells)
                min_c = min(c for r, c in cells)
                max_c = max(c for r, c in cells)
                
                sub = grid[min_r:max_r+1, min_c:max_c+1].copy()
                msk = np.zeros(sub.shape, dtype=bool)
                for r, c in cells:
                    msk[r - min_r, c - min_c] = True
                
                objects.append({
                    'sub': sub, 'mask': msk,
                    'top': min_r, 'left': min_c,
                    'h': max_r - min_r + 1, 'w': max_c - min_c + 1
                })
    
    if not objects:
        return np.full_like(grid, wall_color)
    
    # Sort in reading order
    objects.sort(key=lambda o: (o['top'], o['left']))
    
    # Determine output size
    max_w = max(o['w'] for o in objects)
    out_w = max_w + 2
    out_h = sum(o['h'] for o in objects)
    
    # Place objects
    out = np.full((out_h, out_w), wall_color, dtype=int)
    row_offset = 0
    for obj in objects:
        col_offset = (out_w - obj['w']) // 2
        for r in range(obj['h']):
            for c in range(obj['w']):
                if obj['mask'][r, c]:
                    out[row_offset + r, col_offset + c] = obj['sub'][r, c]
        row_offset += obj['h']
    
    return out