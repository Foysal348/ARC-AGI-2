import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Flood fill transformation:
    1. Identify background (corner color), border (most common non-bg), 
       and source (non-bg, non-border color adjacent to background).
    2. BFS flood fill from source through background cells,
       blocked by border color and other non-bg/non-source colors.
    3. Painted background cells become the source color.
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Step 1: Determine background color (from corners)
    bg = grid[0, 0]
    
    # Step 2: Find all non-background colors and their counts
    color_counts = {}
    for i in range(rows):
        for j in range(cols):
            c = grid[i, j]
            if c != bg:
                color_counts[c] = color_counts.get(c, 0) + 1
    
    if not color_counts:
        return grid
    
    # Step 3: Border = most frequent non-background color
    border = max(color_counts, key=lambda c: color_counts[c])
    
    # Step 4: Source = non-bg, non-border color with adjacent background cell
    source = None
    for c in color_counts:
        if c == border:
            continue
        # Check if any cell of this color has an adjacent background cell
        for i in range(rows):
            for j in range(cols):
                if grid[i, j] == c:
                    neighbors = [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
                    for ni, nj in neighbors:
                        if 0 <= ni < rows and 0 <= nj < cols and grid[ni, nj] == bg:
                            source = c
                            break
                if source:
                    break
            if source:
                break
        if source:
            break
    
    if source is None:
        return grid
    
    # Step 5: BFS flood fill
    # Passable: background (paint it) or source color (pass through)
    # Blocked: border color and all other non-bg, non-source colors
    result = grid.copy()
    visited = set()
    queue = deque()
    
    # Find all source cells adjacent to background as starting points
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == source:
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < rows and 0 <= nj < cols and grid[ni, nj] == bg:
                        queue.append((i, j))
                        visited.add((i, j))
                        break
    
    while queue:
        i, j = queue.popleft()
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i+di, j+dj
            if 0 <= ni < rows and 0 <= nj < cols and (ni, nj) not in visited:
                cell = grid[ni, nj]
                if cell == bg:
                    result[ni, nj] = source
                    visited.add((ni, nj))
                    queue.append((ni, nj))
                elif cell == source:
                    visited.add((ni, nj))
                    queue.append((ni, nj))
                # else: blocked (border or other color)
    
    return result