import numpy as np
from collections import Counter, deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reassembles scattered tiles into a compact output grid.
    
    The input contains multiple rectangular tiles bounded by a 'border' color
    on a 'background' color. One tile serves as the 'frame' (structure) and
    other tiles contain 'fill' colors. The output is the frame's bounding box
    with enclosed cavities filled by matching content from the fill tiles.
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Identify background color
    border_pixels = np.concatenate([grid[0,:], grid[-1,:], grid[:,0], grid[:,-1]])
    bg_color = Counter(border_pixels.flatten()).most_common(1)[0][0]
    
    # Identify border color
    non_bg_mask = grid != bg_color
    if not non_bg_mask.any():
        return np.zeros_like(grid)
    non_bg_pixels = grid[non_bg_mask]
    border_color = Counter(non_bg_pixels.flatten()).most_common(1)[0][0]
    
    # Find all connected components of border color
    visited = np.zeros((rows, cols), dtype=bool)
    components = []
    
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == border_color and not visited[r, c]:
                comp = []
                queue = deque([(r, c)])
                visited[r, c] = True
                while queue:
                    cr, cc = queue.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and grid[nr, nc] == border_color:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                components.append(comp)
    
    if not components:
        return np.zeros_like(grid)
    
    # Frame = largest component
    frame_comp = max(components, key=len)
    fr_min = min(r for r, c in frame_comp)
    fr_max = max(r for r, c in frame_comp)
    fc_min = min(c for r, c in frame_comp)
    fc_max = max(c for r, c in frame_comp)
    
    # Extract frame
    frame = grid[fr_min:fr_max+1, fc_min:fc_max+1].copy()
    frows, fcols = frame.shape
    
    # Find enclosed cavities in frame (non-border pixels not connected to frame edge)
    is_cavity = frame != border_color
    non_enclosed = np.zeros((frows, fcols), dtype=bool)
    queue = deque()
    for r in range(frows):
        for c in range(fcols):
            if (r == 0 or r == frows-1 or c == 0 or c == fcols-1) and is_cavity[r, c]:
                non_enclosed[r, c] = True
                queue.append((r, c))
    while queue:
        cr, cc = queue.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cr+dr, cc+dc
            if 0 <= nr < frows and 0 <= nc < fcols and is_cavity[nr, nc] and not non_enclosed[nr, nc]:
                non_enclosed[nr, nc] = True
                queue.append((nr, nc))
    
    enclosed = is_cavity & ~non_enclosed
    
    # Label enclosed cavities
    enc_labels = np.zeros((frows, fcols), dtype=int)
    cavities = []
    lid = 0
    for r in range(frows):
        for c in range(fcols):
            if enclosed[r, c] and enc_labels[r, c] == 0:
                lid += 1
                pixels = []
                q = deque([(r, c)])
                enc_labels[r, c] = lid
                while q:
                    cr, cc = q.popleft()
                    pixels.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < frows and 0 <= nc < fcols and enclosed[nr, nc] and enc_labels[nr, nc] == 0:
                            enc_labels[nr, nc] = lid
                            q.append((nr, nc))
                cavities.append(pixels)
    
    # Find fill colors from content tiles (other components)
    fill_colors = []
    for comp in components:
        if comp == frame_comp:
            continue
        cr_min = min(r for r, c in comp)
        cr_max = max(r for r, c in comp)
        cc_min = min(c for r, c in comp)
        cc_max = max(c for r, c in comp)
        sub = grid[cr_min:cr_max+1, cc_min:cc_max+1]
        fill_mask = (sub != bg_color) & (sub != border_color)
        fill_colors.extend(set(sub[fill_mask].flatten()))
    
    # Assign fill colors to cavities by position (top-to-bottom ordering)
    # Sort cavities by their top-most position
    cavities_with_pos = [(min(r for r, c in cav), cav) for cav in cavities]
    cavities_with_pos.sort()
    
    # Assign colors in order
    unique_fills = sorted(set(fill_colors))
    output = frame.copy()
    
    for i, (_, cav_pixels) in enumerate(cavities_with_pos):
        if i < len(unique_fills):
            color = unique_fills[i % len(unique_fills)]
        else:
            color = unique_fills[-1]
        for (r, c) in cav_pixels:
            output[r, c] = color
    
    return output