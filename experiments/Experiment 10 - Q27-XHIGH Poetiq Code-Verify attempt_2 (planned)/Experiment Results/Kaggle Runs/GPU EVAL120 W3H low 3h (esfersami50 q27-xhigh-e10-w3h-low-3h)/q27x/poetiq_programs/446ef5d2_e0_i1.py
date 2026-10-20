import numpy as np
from collections import defaultdict
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Combine multiple objects into a single centered object.
    
    The transformation:
    1. Find background color
    2. Find connected components
    3. Remove colors unique to a single component
    4. Create a single centered frame with border = most common boundary color
    5. Fill interior with combined non-border content from all objects
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Find background color
    colors, counts = np.unique(grid, return_counts=True)
    bg_color = colors[np.argmax(counts)]
    
    # Find connected components
    non_bg = (grid != bg_color).astype(int)
    labeled, num_features = ndimage.label(non_bg)
    
    if num_features == 0:
        return grid
    
    # Determine colors per component
    comp_colors = {}
    for i in range(1, num_features + 1):
        mask = labeled == i
        comp_colors[i] = set(grid[mask].tolist()) - {bg_color}
    
    # Find unique colors (in only one component)
    color_comps = defaultdict(set)
    for cid, clist in comp_colors.items():
        for c in clist:
            color_comps[c].add(cid)
    
    unique_colors = {c for c, comps in color_comps.items() if len(comps) == 1}
    
    # Remove unique colors
    for c in unique_colors:
        grid[grid == c] = bg_color
    
    # Re-find components
    non_bg = (grid != bg_color).astype(int)
    labeled, num_features = ndimage.label(non_bg)
    
    if num_features == 0:
        return np.full_like(grid, bg_color)
    
    # Find border color (most frequent on component boundaries)
    boundary_colors = defaultdict(int)
    for i in range(1, num_features + 1):
        mask = labeled == i
        for r, c in zip(*np.where(mask)):
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols or labeled[nr,nc] != i:
                    boundary_colors[grid[r,c]] += 1
                    break
    
    border_color = max(boundary_colors, key=boundary_colors.get)
    
    # Get bounding box of all remaining non-bg pixels
    all_mask = grid != bg_color
    coords = np.argwhere(all_mask)
    min_r, min_c = coords.min(axis=0)
    max_r, max_c = coords.max(axis=0)
    
    content_h = max_r - min_r + 1
    content_w = max_c - min_c + 1
    
    # Output frame dimensions
    frame_h = content_h + 2
    frame_w = content_w + 2
    
    # Create output
    output = np.full((rows, cols), bg_color, dtype=grid.dtype)
    
    # Center the frame
    sr = (rows - frame_h) // 2
    sc = (cols - frame_w) // 2
    
    # Draw border
    output[sr, sc:sc+frame_w] = border_color
    output[sr+frame_h-1, sc:sc+frame_w] = border_color
    output[sr:sr+frame_h, sc] = border_color
    output[sr:sr+frame_h, sc+frame_w-1] = border_color
    
    # Fill interior
    content = grid[min_r:max_r+1, min_c:max_c+1]
    for i in range(content_h):
        for j in range(content_w):
            if content[i, j] != bg_color:
                output[sr+1+i, sc+1+j] = content[i, j]
            else:
                output[sr+1+i, sc+1+j] = border_color
    
    return output