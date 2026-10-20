import numpy as np
from collections import Counter
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    result = grid.copy()
    rows, cols = grid.shape
    
    # Find background
    border = np.concatenate([grid[0,:], grid[-1,:], grid[:,0], grid[:,-1]])
    bg = Counter(border.flatten()).most_common(1)[0][0]
    
    # Find connected components of non-bg
    mask = (grid != bg).astype(int)
    labeled, num = ndimage.label(mask)
    
    components = []
    for i in range(1, num + 1):
        pos = np.argwhere(labeled == i)
        color = grid[pos[0][0], pos[0][1]]
        r0, r1 = pos[:,0].min(), pos[:,0].max()
        c0, c1 = pos[:,1].min(), pos[:,1].max()
        h, w = r1-r0+1, c1-c0+1
        solid = (len(pos) == h*w)
        components.append({'color': color, 'r0': r0, 'r1': r1, 'c0': c0, 'c1': c1, 'h': h, 'w': w, 'solid': solid})
    
    blocks = [c for c in components if c['solid'] and c['h'] > 1]
    singles = [c for c in components if c['h'] == 1 and c['w'] == 1]
    
    if not blocks or not singles:
        return result
    
    # Block dimensions
    bh = max(b['h'] for b in blocks)
    bw = max(b['w'] for b in blocks)
    
    # Sort blocks by column
    blocks.sort(key=lambda x: x['c0'])
    h_gap = blocks[1]['c0'] - blocks[0]['c1'] - 1 if len(blocks) > 1 else 0
    
    # Template position
    tmpl_top = min(b['r0'] for b in blocks)
    first_c = blocks[0]['c0']
    
    # Parse key
    key_rows_data = {}
    for s in singles:
        r, c, color = s['r0'], s['c0'], s['color']
        if r not in key_rows_data:
            key_rows_data[r] = {}
        key_rows_data[r][c] = color
    
    key_row_list = sorted(key_rows_data.keys())
    all_kcols = sorted(set(c for d in key_rows_data.values() for c in d))
    
    # Find anchor (key row matching template colors)
    tmpl_colors = [b['color'] for b in blocks]
    anchor_idx = 0
    for i, kr in enumerate(key_row_list):
        d = key_rows_data[kr]
        if all(i2 < len(tmpl_colors) and all_kcols[i2] in d and d[all_kcols[i2]] == tmpl_colors[i2] for i2 in range(min(len(all_kcols), len(tmpl_colors)))):
            anchor_idx = i
            break
    
    # Spacing
    v_sp = bh + h_gap
    h_sp = bw + h_gap
    
    # Output positions
    anchor_out = tmpl_top
    key_out_pos = {kr: anchor_out + (i - anchor_idx) * v_sp for i, kr in enumerate(key_row_list)}
    
    # Protect key pixels and original blocks
    protect = set()
    for s in singles:
        protect.add((s['r0'], s['c0']))
    for b in blocks:
        for r in range(b['r0'], b['r1']+1):
            for c in range(b['c0'], b['c1']+1):
                protect.add((r, c))
    
    # Place blocks
    for kr in key_row_list:
        out_r = key_out_pos[kr]
        d = key_rows_data[kr]
        for i, kc in enumerate(all_kcols):
            if kc not in d:
                continue
            color = d[kc]
            out_c = first_c + i * h_sp
            for dr in range(bh):
                for dc in range(bw):
                    r, c = out_r + dr, out_c + dc
                    if 0 <= r < rows and 0 <= c < cols and (r,c) not in protect:
                        result[r, c] = color
    
    return result