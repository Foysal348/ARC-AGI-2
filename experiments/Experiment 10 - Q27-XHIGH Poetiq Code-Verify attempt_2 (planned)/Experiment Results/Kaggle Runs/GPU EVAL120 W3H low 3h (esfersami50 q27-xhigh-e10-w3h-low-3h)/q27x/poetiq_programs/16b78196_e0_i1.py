import numpy as np
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    """Transform by stacking non-background shapes on sides of the background band."""
    grid = grid.copy()
    rows, cols = grid.shape
    
    binary = (grid > 0).astype(int)
    labeled, num_features = ndimage.label(binary)
    
    if num_features == 0:
        return grid
    
    # Find background (largest connected component)
    areas = ndimage.sum(binary, labeled, range(1, num_features + 1))
    bg_label = int(np.argmax(areas)) + 1
    bg_mask = (labeled == bg_label)
    bg_idx = np.where(bg_mask)
    br0, br1 = bg_idx[0].min(), bg_idx[0].max()
    bc0, bc1 = bg_idx[1].min(), bg_idx[1].max()
    bg_h, bg_w = br1 - br0 + 1, bc1 - bc0 + 1
    bg_crow, bg_ccol = (br0 + br1) / 2.0, (bc0 + bc1) / 2.0
    is_vertical = bg_h > bg_w
    
    # Extract shapes
    shapes = []
    for lbl in range(1, num_features + 1):
        if lbl == bg_label:
            continue
        idx = np.where(labeled == lbl)
        r0, r1 = idx[0].min(), idx[0].max()
        c0, c1 = idx[1].min(), idx[1].max()
        h, w = r1 - r0 + 1, c1 - c0 + 1
        pix = grid[r0:r1+1, c0:c1+1].copy()
        
        if is_vertical:
            if (c0 + c1) / 2.0 < bg_ccol:
                side, dist = 'left', bc0 - c1
            else:
                side, dist = 'right', c0 - bc1
        else:
            if (r0 + r1) / 2.0 < bg_crow:
                side, dist = 'above', br0 - r1
            else:
                side, dist = 'below', r0 - br1
        
        color = int(np.unique(pix[pix > 0])[0]) if (pix > 0).any() else 0
        shapes.append({'side': side, 'dist': dist, 'h': h, 'w': w, 'pix': pix, 'color': color})
    
    # Result: start with background only
    result = np.zeros_like(grid)
    result[bg_mask] = grid[bg_mask]
    
    for side in set(s['side'] for s in shapes):
        ss = sorted([s for s in shapes if s['side'] == side], key=lambda x: (x['dist'], x['color']))
        
        if is_vertical:
            # Horizontal stacking, perpendicular = row
            max_h = max(s['h'] for s in ss)
            row0 = rows // 2 - max_h // 2
            
            if side == 'left':
                # Rightmost of first shape is at bc0 - 1
                prev_rm = None
                for s in ss:
                    if prev_rm is None:
                        rm = bc0 - 1
                    else:
                        rm = prev_rm - (s['w'] - 1)
                    lm = rm - s['w'] + 1
                    prev_rm = rm
                    for dr in range(s['h']):
                        for dc in range(s['w']):
                            r, c = row0 + dr, lm + dc
                            if 0 <= r < rows and 0 <= c < cols and s['pix'][dr, dc] > 0:
                                result[r, c] = s['pix'][dr, dc]
            else:
                prev_lm = None
                for s in ss:
                    if prev_lm is None:
                        lm = bc1 + 1
                    else:
                        lm = prev_lm + (s['w'] - 1)
                    prev_lm = lm
                    for dr in range(s['h']):
                        for dc in range(s['w']):
                            r, c = row0 + dr, lm + dc
                            if 0 <= r < rows and 0 <= c < cols and s['pix'][dr, dc] > 0:
                                result[r, c] = s['pix'][dr, dc]
        else:
            # Vertical stacking, perpendicular = col
            max_w = max(s['w'] for s in ss)
            col0 = cols // 2 - max_w // 2
            
            if side == 'above':
                prev_bt = None
                for s in ss:
                    if prev_bt is None:
                        bt = br0 - 1
                    else:
                        bt = prev_bt - (s['h'] - 1)
                    tp = bt - s['h'] + 1
                    prev_bt = bt
                    for dr in range(s['h']):
                        for dc in range(s['w']):
                            r, c = tp + dr, col0 + dc
                            if 0 <= r < rows and 0 <= c < cols and s['pix'][dr, dc] > 0:
                                result[r, c] = s['pix'][dr, dc]
            else:
                prev_tp = None
                for s in ss:
                    if prev_tp is None:
                        tp = br1 + 1
                    else:
                        tp = prev_tp + (s['h'] - 1)
                    prev_tp = tp
                    for dr in range(s['h']):
                        for dc in range(s['w']):
                            r, c = tp + dr, col0 + dc
                            if 0 <= r < rows and 0 <= c < cols and s['pix'][dr, dc] > 0:
                                result[r, c] = s['pix'][dr, dc]
    
    return result