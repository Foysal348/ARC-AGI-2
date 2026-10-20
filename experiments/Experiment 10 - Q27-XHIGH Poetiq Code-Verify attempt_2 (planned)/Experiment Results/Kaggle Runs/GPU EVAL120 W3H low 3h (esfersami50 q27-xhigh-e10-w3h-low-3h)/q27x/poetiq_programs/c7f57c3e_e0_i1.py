import numpy as np
from scipy import ndimage
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Identify cross-shaped objects, group them by their decoration pattern
    (extension type, extension color, center color), and swap decorations
    between the two groups.
    """
    result = grid.copy()
    h, w = grid.shape
    
    # Find background color (most common on border)
    border = np.concatenate([grid[0,:].flatten(), grid[-1,:].flatten(),
                             grid[:,0].flatten(), grid[:,-1].flatten()])
    bg = int(np.bincount(border).argmax())
    
    # Create mask of non-background pixels
    non_bg = (grid != bg)
    
    # Find connected components (8-connectivity)
    struct = np.ones((3,3), dtype=int)
    labeled, num = ndimage.label(non_bg, structure=struct)
    
    if num == 0:
        return result
    
    # Analyze each component
    crosses = []
    loose_pixels = []
    
    for i in range(1, num + 1):
        comp_mask = (labeled == i)
        comp_pixels = np.argwhere(comp_mask)
        
        # Check if it contains 1s (is a cross)
        has_ones = any(grid[r, c] == 1 for r, c in comp_pixels)
        
        if not has_ones:
            for r, c in comp_pixels:
                loose_pixels.append((r, c))
            continue
        
        # It's a cross
        one_pixels = [(r, c) for r, c in comp_pixels if grid[r, c] == 1]
        special_pixels = [(r, c) for r, c in comp_pixels if grid[r, c] != 1 and grid[r, c] != bg]
        
        r_min = min(r for r, c in one_pixels)
        r_max = max(r for r, c in one_pixels)
        
        # Top extension: special pixels above r_min
        top_ext = [(r, c) for r, c in special_pixels if r < r_min]
        # Bottom extension: special pixels below r_max
        bot_ext = [(r, c) for r, c in special_pixels if r > r_max]
        # Center: special pixels within [r_min, r_max]
        center = [(r, c) for r, c in special_pixels if r_min <= r <= r_max]
        
        top_color = int(grid[top_ext[0][0], top_ext[0][1]]) if top_ext else None
        bot_color = int(grid[bot_ext[0][0], bot_ext[0][1]]) if bot_ext else None
        center_color = int(grid[center[0][0], center[0][1]]) if center else None
        
        top_height = len(set(r for r, c in top_ext)) if top_ext else 0
        bot_height = len(set(r for r, c in bot_ext)) if bot_ext else 0
        
        top_row_cols = sorted(set(c for r, c in one_pixels if r == r_min))
        bot_row_cols = sorted(set(c for r, c in one_pixels if r == r_max))
        
        crosses.append({
            'r_min': r_min, 'r_max': r_max,
            'top_ext': top_ext, 'top_color': top_color, 'top_height': top_height,
            'bot_ext': bot_ext, 'bot_color': bot_color, 'bot_height': bot_height,
            'center': center, 'center_color': center_color,
            'top_row_cols': top_row_cols,
            'bot_row_cols': bot_row_cols,
        })
    
    if len(crosses) < 2:
        return result
    
    # Determine pattern for each cross: (ext_type, ext_color, center_color)
    def get_pattern(cross):
        if cross['top_ext']:
            return ('top', cross['top_color'], cross['center_color'])
        elif cross['bot_ext']:
            return ('bottom', cross['bot_color'], cross['center_color'])
        else:
            return (None, None, cross['center_color'])
    
    patterns = [get_pattern(c) for c in crosses]
    
    # Group by pattern
    groups = defaultdict(list)
    for i, p in enumerate(patterns):
        groups[p].append(i)
    
    if len(groups) != 2:
        return result
    
    (pat_a, idx_a), (pat_b, idx_b) = list(groups.items())
    
    # Remove all old decorations
    for cross in crosses:
        for r, c in cross['top_ext']:
            result[r, c] = bg
        for r, c in cross['bot_ext']:
            result[r, c] = bg
        for r, c in cross['center']:
            result[r, c] = bg
    
    # Remove loose pixels
    for r, c in loose_pixels:
        result[r, c] = bg
    
    # Apply new decorations
    def apply_new(cross, new_pat):
        new_ext_type, new_ext_color, new_center_color = new_pat
        
        # Apply new center
        for r, c in cross['center']:
            result[r, c] = new_center_color
        
        # Apply new extension
        if new_ext_type == 'top':
            ext_h = cross['bot_height'] if cross['bot_height'] > 0 else cross['top_height']
            if ext_h == 0: ext_h = 1
            cs, ce = cross['top_row_cols'][0], cross['top_row_cols'][-1]
            for r in range(cross['r_min'] - ext_h, cross['r_min']):
                if 0 <= r < h:
                    for c in range(cs, ce + 1):
                        if 0 <= c < w:
                            result[r, c] = new_ext_color
        elif new_ext_type == 'bottom':
            ext_h = cross['top_height'] if cross['top_height'] > 0 else cross['bot_height']
            if ext_h == 0: ext_h = 1
            cs, ce = cross['bot_row_cols'][0], cross['bot_row_cols'][-1]
            for r in range(cross['r_max'] + 1, cross['r_max'] + 1 + ext_h):
                if 0 <= r < h:
                    for c in range(cs, ce + 1):
                        if 0 <= c < w:
                            result[r, c] = new_ext_color
    
    for idx in idx_a:
        apply_new(crosses[idx], pat_b)
    for idx in idx_b:
        apply_new(crosses[idx], pat_a)
    
    return result