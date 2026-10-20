import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform grid by identifying connected components of non-background pixels
    that contain 2x2 blocks adjacent to different non-background colors.
    The 2x2 block color becomes 3, the adjacent color becomes 5.
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Find background color (most common on borders)
    border_colors = {}
    for i in range(rows):
        for j in range(cols):
            if i == 0 or i == rows-1 or j == 0 or j == cols-1:
                c = grid[i, j]
                border_colors[c] = border_colors.get(c, 0) + 1
    bg = max(border_colors, key=border_colors.get)
    
    # Find non-background colors
    non_bg_colors = set()
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] != bg:
                non_bg_colors.add(grid[i, j])
    
    # Count 2x2 blocks for each color
    def count_2x2(color):
        count = 0
        for i in range(rows - 1):
            for j in range(cols - 1):
                if (grid[i,j] == color and grid[i,j+1] == color and
                    grid[i+1,j] == color and grid[i+1,j+1] == color):
                    count += 1
        return count
    
    color_2x2 = {c: count_2x2(c) for c in non_bg_colors}
    
    # Sort colors by 2x2 count: most → 3, least → 5, middle → stays
    sorted_colors = sorted(non_bg_colors, key=lambda c: color_2x2[c], reverse=True)
    color_3 = sorted_colors[0] if len(sorted_colors) > 0 else None
    color_5 = sorted_colors[-1] if len(sorted_colors) > 1 else None
    color_stay = sorted_colors[1] if len(sorted_colors) == 3 else None
    
    # Find connected components of non-bg pixels
    visited = np.zeros((rows, cols), dtype=bool)
    components = []
    
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] != bg and not visited[i, j]:
                # BFS
                comp = []
                queue = deque([(i, j)])
                visited[i, j] = True
                while queue:
                    r, c = queue.popleft()
                    comp.append((r, c))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if not visited[nr, nc] and grid[nr, nc] != bg:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                components.append(comp)
    
    # For each component, check if it has a 2x2 block of color_3
    # and contains color_5 cells
    new_grid = grid.copy()
    
    for comp in components:
        comp_set = set(comp)
        comp_colors = set(grid[r, c] for r, c in comp)
        
        # Check if component contains color_3 and has a 2x2 block of color_3
        has_2x2_of_3 = False
        for (r, c) in comp:
            if (grid[r,c] == color_3 and 
                (r+1, c) in comp_set and grid[r+1,c] == color_3 and
                (r, c+1) in comp_set and grid[r,c+1] == color_3 and
                (r+1, c+1) in comp_set and grid[r+1,c+1] == color_3):
                has_2x2_of_3 = True
                break
        
        if has_2x2_of_3 and color_5 in comp_colors:
            # Change 2x2 block cells of color_3 to 3
            for (r, c) in comp:
                if grid[r, c] == color_3:
                    new_grid[r, c] = 3
                elif grid[r, c] == color_5:
                    new_grid[r, c] = 5
    
    return new_grid