import numpy as np
from collections import deque, Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Find all 2s, group them into 4-connected components.
    For each group, determine the mirror axis:
      - Single 2: 180-degree rotation around that point
      - Horizontal line of 2s: reflect vertically (up-down)
      - Vertical line of 2s: reflect horizontally (left-right)
    Find 4-connected components of non-background, non-2 pixels adjacent to the 2-group,
    and add their reflections to the output.
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Determine background color from the border
    border = list(grid[0, :]) + list(grid[-1, :]) + list(grid[:, 0]) + list(grid[:, -1])
    bg_color = Counter(border).most_common(1)[0][0]
    
    # Find all 2 pixels
    twos = [(r, c) for r in range(rows) for c in range(cols) if grid[r, c] == 2]
    if not twos:
        return grid
    
    # Group 2s into 4-connected components
    two_set = set(twos)
    visited_twos = set()
    two_groups = []
    
    for two in twos:
        if two in visited_twos:
            continue
        group = []
        queue = deque([two])
        visited_twos.add(two)
        while queue:
            r, c = queue.popleft()
            group.append((r, c))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in two_set and (nr, nc) not in visited_twos:
                    visited_twos.add((nr, nc))
                    queue.append((nr, nc))
        two_groups.append(group)
    
    # For each group, determine mirror and reflect adjacent non-bg, non-2 components
    for group in two_groups:
        if len(group) == 1:
            # Single point: 180-degree rotation
            mr, mc = group[0]
            mirror_type = 'point'
        else:
            rows_in_group = set(r for r, c in group)
            cols_in_group = set(c for r, c in group)
            if len(rows_in_group) == 1:
                mirror_type = 'horizontal'
                mirror_row = list(rows_in_group)[0]
            elif len(cols_in_group) == 1:
                mirror_type = 'vertical'
                mirror_col = list(cols_in_group)[0]
            else:
                # Complex shape - skip
                continue
        
        # Find non-bg, non-2 pixels directly adjacent to any 2 in this group
        adjacent = set()
        for r, c in group:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if grid[nr, nc] != bg_color and grid[nr, nc] != 2:
                        adjacent.add((nr, nc))
        
        if not adjacent:
            continue
        
        # Find 4-connected components starting from adjacent pixels
        visited = set()
        components = []
        for pixel in adjacent:
            if pixel in visited:
                continue
            comp = []
            queue = deque([pixel])
            visited.add(pixel)
            while queue:
                r, c = queue.popleft()
                comp.append((r, c))
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if (nr, nc) not in visited and grid[nr, nc] != bg_color and grid[nr, nc] != 2:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
            components.append(comp)
        
        # Reflect each component across the mirror
        for comp in components:
            for r, c in comp:
                if mirror_type == 'point':
                    nr = 2 * mr - r
                    nc = 2 * mc - c
                elif mirror_type == 'horizontal':
                    nr = 2 * mirror_row - r
                    nc = c
                else:  # vertical
                    nr = r
                    nc = 2 * mirror_col - c
                
                if 0 <= nr < rows and 0 <= nc < cols:
                    grid[nr, nc] = grid[r, c]
    
    return grid