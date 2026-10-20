import numpy as np
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    """
    For each box border (connected component of 0s forming rectangular/L-shaped borders),
    find numbered pixels inside. Each number projects to its nearest corner:
    - 2x2 block at the corner (extending outward)
    - Diagonal step from number toward corner
    """
    result = grid.copy()
    rows, cols = grid.shape
    
    border_vals = np.concatenate([grid[0,:], grid[-1,:], grid[:,0], grid[:,-1]])
    bg_color = np.bincount(border_vals).argmax()
    
    zero_mask = (grid == 0).astype(int)
    labeled, num_features = ndimage.label(zero_mask)
    
    for i in range(1, num_features + 1):
        component = (labeled == i)
        comp_ys, comp_xs = np.where(component)
        r_min, r_max = comp_ys.min(), comp_ys.max()
        c_min, c_max = comp_xs.min(), comp_xs.max()
        
        # Find horizontal segments (consecutive 0s in a row, length >= 2)
        h_segments = []  # (row, c_start, c_end)
        for r in range(r_min, r_max + 1):
            c = c_min
            while c <= c_max:
                if component[r, c]:
                    c_start = c
                    while c <= c_max and component[r, c]:
                        c += 1
                    c_end = c - 1
                    if c_end - c_start + 1 >= 2:
                        h_segments.append((r, c_start, c_end))
                else:
                    c += 1
        
        # Find vertical segments (consecutive 0s in a col, length >= 2)
        v_segments = []  # (col, r_start, r_end)
        for c in range(c_min, c_max + 1):
            r = r_min
            while r <= r_max:
                if component[r, c]:
                    r_start = r
                    while r <= r_max and component[r, c]:
                        r += 1
                    r_end = r - 1
                    if r_end - r_start + 1 >= 2:
                        v_segments.append((c, r_start, r_end))
                else:
                    r += 1
        
        # Find corners: cells where a horizontal and vertical segment intersect
        # and the corner cell is at an endpoint of at least one segment
        corners = []  # (r, c, inside_dr, inside_dc)
        for (hr, hs, he) in h_segments:
            for (vc, vs, ve) in v_segments:
                if hs <= vc <= he and vs <= hr <= ve:
                    # Determine inside direction
                    # h_dir: which way the h segment extends from the corner
                    if vc == hs:
                        h_dir = 1
                    elif vc == he:
                        h_dir = -1
                    else:
                        h_dir = 1 if (vc - hs) >= (he - vc) else -1
                    # v_dir: which way the v segment extends from the corner
                    if hr == vs:
                        v_dir = 1
                    elif hr == ve:
                        v_dir = -1
                    else:
                        v_dir = 1 if (hr - vs) >= (ve - hr) else -1
                    
                    # Check if this is a true corner (endpoint of at least one segment)
                    is_h_end = (vc == hs) or (vc == he)
                    is_v_end = (hr == vs) or (hr == ve)
                    if is_h_end or is_v_end:
                        corners.append((hr, vc, v_dir, h_dir))
        
        if not corners:
            continue
        
        # Remove duplicate corners
        unique_corners = []
        seen = set()
        for corner in corners:
            key = (corner[0], corner[1])
            if key not in seen:
                seen.add(key)
                unique_corners.append(corner)
        corners = unique_corners
        
        # For each numbered cell, find its nearest corner and apply the rule
        for (cr, cc, v_dir, h_dir) in corners:
            # Determine inside region for this corner
            if v_dir == 1:
                r_start, r_end = cr + 1, r_max
            else:
                r_start, r_end = r_min, cr - 1
            
            if h_dir == 1:
                c_start, c_end = cc + 1, c_max
            else:
                c_start, c_end = c_min, cc - 1
            
            if r_start > r_end or c_start > c_end:
                continue
            
            # Find numbered cells in this inside region
            for r in range(max(r_start, 0), min(r_end, rows-1) + 1):
                for c in range(max(c_start, 0), min(c_end, cols-1) + 1):
                    val = grid[r, c]
                    if val == 0 or val == bg_color:
                        continue
                    
                    # Place 2x2 block outside the corner
                    out_v = -v_dir
                    out_h = -h_dir
                    block = [(cr, cc), (cr+out_v, cc), (cr, cc+out_h), (cr+out_v, cc+out_h)]
                    for (br, bc) in block:
                        if 0 <= br < rows and 0 <= bc < cols:
                            result[br, bc] = val
                    
                    # Diagonal step from number toward corner
                    dr = np.sign(cr - r)
                    dc = np.sign(cc - c)
                    sr, sc = r + dr, c + dc
                    if 0 <= sr < rows and 0 <= sc < cols:
                        result[sr, sc] = val
    
    return result