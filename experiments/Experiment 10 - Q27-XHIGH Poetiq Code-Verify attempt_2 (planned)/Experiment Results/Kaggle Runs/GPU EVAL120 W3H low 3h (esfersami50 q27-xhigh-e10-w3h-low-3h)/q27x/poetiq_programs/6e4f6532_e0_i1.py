import numpy as np
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Identifies objects (connected components of non-background pixels) within rooms
    defined by wall lines. Transforms each object based on its aspect ratio:
    - Wider than tall: vertical flip
    - Taller than wide: 90° CCW rotation
    - Square: 90° CW rotation
    Repositions objects to the opposite horizontal side of the room,
    stacked from bottom to top (tallest at bottom).
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Detect walls (uniform rows or columns)
    is_wall = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        if len(np.unique(grid[r])) <= 2:
            is_wall[r] = True
    for c in range(cols):
        if len(np.unique(grid[:, c])) <= 2:
            is_wall[:, c] = True
    
    non_wall = ~is_wall
    if not non_wall.any():
        return grid
    
    # Background = most common non-wall color
    bg = np.bincount(grid[non_wall].astype(int)).argmax()
    
    # Object pixels
    is_obj = (grid != bg) & non_wall
    
    labeled, n = ndimage.label(is_obj)
    if n == 0:
        return grid
    
    # Room bounds
    nw = np.argwhere(non_wall)
    r0, c0 = nw.min(axis=0)
    r1, c1 = nw.max(axis=0)
    
    # Extract objects
    objs = []
    for i in range(1, n + 1):
        m = labeled == i
        coords = np.argwhere(m)
        if len(coords) == 0:
            continue
        mr, mc = coords.min(axis=0)
        xR, xC = coords.max(axis=0)
        h, w = xR - mr + 1, xC - mc + 1
        shape = grid[mr:xR+1, mc:xC+1].copy()
        sm = m[mr:xR+1, mc:xC+1]
        objs.append(dict(mr=mr, mc=mc, xR=xR, xC=xC, h=h, w=w,
                         shape=shape, sm=sm, cc=(mc+xC)/2.0, cr=(mr+xR)/2.0))
    
    # Transform each object
    for o in objs:
        s = o['shape']
        if o['w'] > o['h']:
            o['ns'] = s[::-1]
            o['nm'] = o['sm'][::-1]
        elif o['h'] > o['w']:
            o['ns'] = np.rot90(s, 1)
            o['nm'] = np.rot90(o['sm'], 1)
        else:
            o['ns'] = np.rot90(s, -1)
            o['nm'] = np.rot90(o['sm'], -1)
        o['nh'], o['nw'] = o['ns'].shape
    
    # Sort tallest first for bottom-up stacking
    objs.sort(key=lambda x: -x['nh'])
    
    # Clear originals
    grid[is_obj] = bg
    
    # Place transformed objects
    rc = (c0 + c1) / 2.0
    for i, o in enumerate(objs):
        bot = r1
        for j in range(i):
            bot -= objs[j]['nh'] + 1
        top = bot - o['nh'] + 1
        
        left = c1 - o['nw'] + 1 if o['cc'] < rc else c0
        
        for dr in range(o['nh']):
            for dc in range(o['nw']):
                if o['nm'][dr, dc]:
                    r, c = top + dr, left + dc
                    if 0 <= r < rows and 0 <= c < cols and not is_wall[r, c]:
                        grid[r, c] = o['ns'][dr, dc]
    
    return grid