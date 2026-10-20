import numpy as np
from collections import Counter, deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform grid by identifying wall-separated regions and filling them
    with nested rectangle patterns based on seed colors.
    
    Algorithm:
    1. Background = most common color, Wall = second most common color
    2. Find connected components of non-wall pixels (regions)
    3. For each region:
       a. Find seed pixels (non-bg, non-wall)
       b. Compute ring number for each seed based on bounding box
       c. Determine period = max_seed_ring + 1
       d. Fill each pixel: color = seeds[ring % period] or background
    4. Restore wall pixels
    """
    H, W = grid.shape
    
    # Edge case: single color or empty grid
    flat = grid.flatten().tolist()
    color_counts = Counter(flat)
    if len(color_counts) <= 1:
        return grid.copy()
    
    sorted_colors = color_counts.most_common()
    background = sorted_colors[0][0]
    wall = sorted_colors[1][0]
    
    # Create wall mask
    wall_mask = (grid == wall)
    
    # Find connected components of non-wall pixels using BFS
    visited = np.zeros((H, W), dtype=bool)
    regions = []
    
    for i in range(H):
        for j in range(W):
            if not wall_mask[i, j] and not visited[i, j]:
                component = []
                queue = deque([(i, j)])
                visited[i, j] = True
                while queue:
                    r, c = queue.popleft()
                    component.append((r, c))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and not wall_mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                regions.append(component)
    
    # Initialize result with background color
    result = np.full((H, W), background, dtype=grid.dtype)
    
    # Process each region
    for component in regions:
        if not component:
            continue
        
        rows = [r for r, c in component]
        cols = [c for r, c in component]
        r_min, r_max = min(rows), max(rows)
        c_min, c_max = min(cols), max(cols)
        
        # Find seeds (non-background, non-wall pixels) and their ring numbers
        seeds = {}  # ring -> color
        for r, c in component:
            color = grid[r, c]
            if color != background and color != wall:
                ring = min(r - r_min, c - c_min, r_max - r, c_max - c)
                if ring not in seeds:
                    seeds[ring] = color
        
        if not seeds:
            continue  # No seeds, stays as background
        
        max_ring = max(seeds.keys())
        period = max_ring + 1
        
        # Fill the region with the nested rectangle pattern
        for r, c in component:
            ring = min(r - r_min, c - c_min, r_max - r, c_max - c)
            lookup_ring = ring % period
            if lookup_ring in seeds:
                result[r, c] = seeds[lookup_ring]
    
    # Restore wall pixels
    result[wall_mask] = wall
    
    return result