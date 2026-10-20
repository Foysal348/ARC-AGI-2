import numpy as np
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Extracts colored objects (shapes + their 2-markers) from the input grid
    and rearranges them into a compact centered layout.
    
    The solid object (color 4) is placed at the center.
    Other objects are arranged in a 2x2 grid around it if they have
    varying widths, or in a vertical column if widths are uniform.
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Find connected components of colored pixels (non-zero, non-2)
    color_mask = (grid != 0) & (grid != 2)
    labeled, num_features = ndimage.label(color_mask)
    
    objects = []
    used_2s = np.zeros_like(grid, dtype=bool)
    
    for i in range(1, num_features + 1):
        mask = (labeled == i)
        color = int(grid[mask][0])
        
        rows_idx, cols_idx = np.where(mask)
        r_min, r_max = rows_idx.min(), rows_idx.max()
        c_min, c_max = cols_idx.min(), cols_idx.max()
        
        # Find attached 2-pixels
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                if mask[r, c]:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if grid[nr, nc] == 2 and not used_2s[nr, nc]:
                                    mask[nr, nc] = True
                                    used_2s[nr, nc] = True
        
        rows_idx, cols_idx = np.where(mask)
        r_min_f, r_max_f = rows_idx.min(), rows_idx.max()
        c_min_f, c_max_f = cols_idx.min(), cols_idx.max()
        
        h = r_max_f - r_min_f + 1
        w = c_max_f - c_min_f + 1
        patch = np.zeros((h, w), dtype=int)
        for r in range(r_min_f, r_max_f + 1):
            for c in range(c_min_f, c_max_f + 1):
                if mask[r, c]:
                    patch[r - r_min_f, c - c_min_f] = grid[r, c]
        
        objects.append({'color': color, 'patch': patch, 'h': h, 'w': w})
    
    if not objects:
        return np.zeros_like(grid)
    
    # Separate center object (color 4) from others
    center_obj = None
    others = []
    for obj in objects:
        if obj['color'] == 4:
            center_obj = obj
        else:
            others.append(obj)
    
    if center_obj is None:
        center_obj = others.pop()
    
    # Determine layout based on object dimensions
    if len(others) == 4:
        # Check for 2x2 grid vs column
        by_h = sorted(others, key=lambda o: o['h'])
        top = sorted(by_h[:2], key=lambda o: o['w'])
        bot = sorted(by_h[2:], key=lambda o: o['w'])
        tl, tr, bl, br = top[0], top[1], bot[0], bot[1]
        
        top_h = max(tl['h'], tr['h'])
        bot_h = max(bl['h'], br['h'])
        left_w = max(tl['w'], bl['w'])
        right_w = max(tr['w'], br['w'])
        
        total_h = top_h + bot_h
        total_w = left_w + right_w
        
        cr = max(0, (total_h - center_obj['h']) // 2)
        cc = max(0, (total_w - center_obj['w']) // 2)
        
        placements = [
            (tl, (0, 0)),
            (tr, (0, left_w)),
            (bl, (top_h, 0)),
            (br, (top_h, left_w)),
            (center_obj, (cr, cc))
        ]
    else:
        others.sort(key=lambda o: o['h'])
        max_w = max([o['w'] for o in others] + [center_obj['w']])
        y = 0
        placements = []
        for obj in others:
            x = (max_w - obj['w']) // 2
            placements.append((obj, (y, x)))
            y += obj['h']
        cr = y // 2
        cc = (max_w - center_obj['w']) // 2
        placements.append((center_obj, (cr, cc)))
        total_h = y
        total_w = max_w
    
    # Build output canvas
    canvas = np.zeros((total_h, total_w), dtype=int)
    for obj, (r, c) in placements:
        h, w = obj['h'], obj['w']
        if r + h <= total_h and c + w <= total_w:
            canvas[r:r+h, c:c+w] = obj['patch']
    
    # Center in output grid
    out = np.zeros((rows, cols), dtype=int)
    sr = (rows - total_h) // 2
    sc = (cols - total_w) // 2
    out[sr:sr+total_h, sc:sc+total_w] = canvas
    
    return out