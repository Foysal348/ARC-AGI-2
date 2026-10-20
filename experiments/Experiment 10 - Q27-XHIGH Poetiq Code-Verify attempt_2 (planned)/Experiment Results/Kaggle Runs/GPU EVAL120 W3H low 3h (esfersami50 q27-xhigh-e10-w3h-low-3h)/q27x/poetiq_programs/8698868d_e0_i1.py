import numpy as np
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Identify rectangular frames in the input, pair inner (smaller) frames with
    outer (larger) frames by nearest center, and produce a compact output where
    each outer frame contains its paired inner frame centered inside.
    Inner frame background cells are replaced with the outer frame's color.
    """
    grid = np.array(grid, dtype=int)
    rows, cols = grid.shape
    
    # Find background color
    border_vals = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
    background = np.bincount(border_vals).argmax()
    
    # Find all rectangular frames
    frames = []
    for color in range(10):
        if color == background:
            continue
        mask = (grid == color).astype(int)
        labeled, num = ndimage.label(mask, structure=np.ones((3, 3)))
        for i in range(1, num + 1):
            comp = (labeled == i)
            if comp.sum() < 4:
                continue
            rows_idx, cols_idx = np.where(comp)
            r_min, r_max = int(rows_idx.min()), int(rows_idx.max())
            c_min, c_max = int(cols_idx.min()), int(cols_idx.max())
            h = r_max - r_min + 1
            w = c_max - c_min + 1
            
            # Verify rectangular border
            sub = grid[r_min:r_max+1, c_min:c_max+1]
            border_mask = np.zeros((h, w), dtype=bool)
            border_mask[0, :] = True
            border_mask[-1, :] = True
            border_mask[:, 0] = True
            border_mask[:, -1] = True
            if np.all(sub[border_mask] == color):
                frames.append({
                    'color': color,
                    'r_min': r_min, 'r_max': r_max,
                    'c_min': c_min, 'c_max': c_max,
                    'h': h, 'w': w,
                    'sub': sub.copy()
                })
    
    if not frames:
        return np.zeros_like(grid)
    
    max_h = max(f['h'] for f in frames)
    min_h = min(f['h'] for f in frames)
    
    outer_frames = [f for f in frames if f['h'] == max_h]
    inner_frames = [f for f in frames if f['h'] == min_h]
    
    outer_frames.sort(key=lambda f: (f['r_min'], f['c_min']))
    
    # Pair by nearest center
    pairs = []
    used_inners = set()
    for outer in outer_frames:
        oc_r = (outer['r_min'] + outer['r_max']) / 2.0
        oc_c = (outer['c_min'] + outer['c_max']) / 2.0
        best_i = -1
        best_d = float('inf')
        for ii, inner in enumerate(inner_frames):
            if ii in used_inners:
                continue
            ic_r = (inner['r_min'] + inner['r_max']) / 2.0
            ic_c = (inner['c_min'] + inner['c_max']) / 2.0
            d = (oc_r - ic_r)**2 + (oc_c - ic_c)**2
            if d < best_d:
                best_d = d
                best_i = ii
        if best_i >= 0:
            pairs.append((outer, inner_frames[best_i]))
            used_inners.add(best_i)
    
    # Determine output layout
    fh = max_h
    mid_r = (outer_frames[0]['r_min'] + outer_frames[-1]['r_min']) / 2.0
    top = [f for f in outer_frames if (f['r_min']+f['r_max'])/2.0 <= mid_r]
    bot = [f for f in outer_frames if (f['r_min']+f['r_max'])/2.0 > mid_r]
    
    if top and bot:
        grid_rows, grid_cols = 2, len(top)
    else:
        grid_rows, grid_cols = 1, len(outer_frames)
    
    out_h = grid_rows * fh
    out_w = grid_cols * fh
    output = np.zeros((out_h, out_w), dtype=int)
    
    for idx, (outer, inner) in enumerate(pairs):
        if grid_rows == 2:
            if (outer['r_min']+outer['r_max'])/2.0 <= mid_r:
                gr = 0
                gc = sum(1 for f in top if f['c_min'] <= outer['c_min'])
            else:
                gr = 1
                gc = sum(1 for f in bot if f['c_min'] <= outer['c_min'])
        else:
            gr, gc = 0, idx
        
        br, bc = gr * fh, gc * fh
        
        # Draw outer frame
        output[br, bc:bc+fh] = outer['color']
        output[br+fh-1, bc:bc+fh] = outer['color']
        for i in range(fh):
            output[br+i, bc] = outer['color']
            output[br+i, bc+fh-1] = outer['color']
        # Fill interior with outer color
        for i in range(1, fh-1):
            for j in range(1, fh-1):
                output[br+i, bc+j] = outer['color']
        
        # Place inner frame
        ih, iw = inner['h'], inner['w']
        off_r = (fh - ih) // 2
        off_c = (fh - iw) // 2
        for i in range(ih):
            for j in range(iw):
                val = inner['sub'][i, j]
                if val == background:
                    val = outer['color']
                output[br+off_r+i, bc+off_c+j] = val
    
    return output