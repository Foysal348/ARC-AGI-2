import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    grid = grid.copy()
    rows, cols = grid.shape
    
    def is_wall(r, c):
        return grid[r, c] == 7 or grid[r, c] == 3
    
    visited = np.zeros((rows, cols), dtype=bool)
    
    for r in range(rows):
        for c in range(cols):
            if not visited[r, c] and not is_wall(r, c):
                room = []
                queue = deque([(r, c)])
                visited[r, c] = True
                while queue:
                    cr, cc = queue.popleft()
                    room.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if not visited[nr, nc] and not is_wall(nr, nc):
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                
                area = len(room)
                non_zero = sum(1 for (r2, c2) in room if grid[r2, c2] != 0)
                zeros = area - non_zero
                
                if area >= 15 and zeros >= 12 and non_zero >= 3 and area <= 5 * non_zero:
                    for (r2, c2) in room:
                        grid[r2, c2] = 7
    
    return grid