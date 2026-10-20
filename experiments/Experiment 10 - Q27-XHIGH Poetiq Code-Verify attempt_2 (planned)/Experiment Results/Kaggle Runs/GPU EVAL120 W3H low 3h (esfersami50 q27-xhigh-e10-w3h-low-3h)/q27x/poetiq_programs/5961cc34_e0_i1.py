import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform grid by identifying shapes (1s+3s), finding the ray (4+2s column),
    determining connectivity, and rendering shapes + connection lines in color 2.
    """
    rows, cols = grid.shape
    output = np.full((rows, cols), 8, dtype=int)
    
    # Find ray column
    ray_col = None
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 4:
                ray_col = c
                break
        if ray_col is not None:
            break
    
    if ray_col is None:
        return output
    
    # Find connected components of 1s and 3s
    visited = np.zeros((rows, cols), dtype=bool)
    shapes = []
    
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] in (1, 3) and not visited[r, c]:
                comp = set()
                queue = deque([(r, c)])
                visited[r, c] = True
                while queue:
                    cr, cc = queue.popleft()
                    comp.add((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                            if grid[nr, nc] in (1, 3):
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                shapes.append(comp)
    
    # Compute shape info
    shape_info = []
    for comp in shapes:
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        threes = [(r, c) for (r, c) in comp if grid[r, c] == 3]
        
        if not threes:
            shape_info.append({'pixels': comp, 'min_r': min_r, 'max_r': max_r,
                               'min_c': min_c, 'max_c': max_c,
                               'spine_type': None, 'spine_cols': None,
                               'spine_rows': None, 'spine_dir': None})
            continue
        
        t_rows = sorted(set(r for r, c in threes))
        t_cols = sorted(set(c for r, c in threes))
        
        if len(t_rows) == 1:
            spine_type = 'vertical'
            spine_cols = t_cols
            t_row = t_rows[0]
            if t_row <= min_r + 1:
                spine_dir = 'up'
            else:
                spine_dir = 'down'
        elif len(t_cols) == 1:
            spine_type = 'horizontal'
            spine_rows = t_rows
            t_col = t_cols[0]
            if t_col <= min_c + 1:
                spine_dir = 'left'
            else:
                spine_dir = 'right'
        else:
            spine_type = 'vertical'
            spine_cols = t_cols
            spine_dir = 'down'
        
        shape_info.append({'pixels': comp, 'min_r': min_r, 'max_r': max_r,
                           'min_c': min_c, 'max_c': max_c,
                           'spine_type': spine_type,
                           'spine_cols': spine_cols if spine_type == 'vertical' else None,
                           'spine_rows': spine_rows if spine_type == 'horizontal' else None,
                           'spine_dir': spine_dir})
    
    # Determine connectivity to ray via BFS
    n = len(shape_info)
    connected = [False] * n
    for i, info in enumerate(shape_info):
        if info['min_c'] <= ray_col <= info['max_c']:
            connected[i] = True
    
    for _ in range(n + 1):
        changed = False
        for i in range(n):
            if not connected[i]:
                continue
            for j in range(n):
                if connected[j]:
                    continue
                a, b = shape_info[i], shape_info[j]
                row_ov = a['min_r'] <= b['max_r'] and b['min_r'] <= a['max_r']
                col_ov = a['min_c'] <= b['max_c'] and b['min_c'] <= a['max_c']
                if row_ov or col_ov:
                    connected[j] = True
                    changed = True
        if not changed:
            break
    
    # Render shapes
    for i, info in enumerate(shape_info):
        if connected[i]:
            for (r, c) in info['pixels']:
                output[r, c] = 2
    
    # Draw connections between connected shapes
    for i in range(n):
        if not connected[i]:
            continue
        for j in range(i + 1, n):
            if not connected[j]:
                continue
            a, b = shape_info[i], shape_info[j]
            row_ov = a['min_r'] <= b['max_r'] and b['min_r'] <= a['max_r']
            col_ov = a['min_c'] <= b['max_c'] and b['min_c'] <= a['max_c']
            
            if row_ov and not col_ov:
                # Horizontal line
                lc = min(a['min_c'], b['min_c'])
                rc = max(a['max_c'], b['max_c'])
                h_rows = None
                if a['spine_type'] == 'horizontal':
                    h_rows = a['spine_rows']
                elif b['spine_type'] == 'horizontal':
                    h_rows = b['spine_rows']
                if h_rows is None:
                    or_min = max(a['min_r'], b['min_r'])
                    or_max = min(a['max_r'], b['max_r'])
                    mid = (or_min + or_max) // 2
                    h_rows = [mid] if (or_max - or_min) < 2 else [mid, mid + 1]
                for r in h_rows:
                    for c in range(lc, rc + 1):
                        if 0 <= r < rows and 0 <= c < cols:
                            output[r, c] = 2
            elif col_ov and not row_ov:
                # Vertical line
                tr = min(a['min_r'], b['min_r'])
                br = max(a['max_r'], b['max_r'])
                v_cols = None
                if a['spine_type'] == 'vertical':
                    v_cols = a['spine_cols']
                elif b['spine_type'] == 'vertical':
                    v_cols = b['spine_cols']
                if v_cols is None:
                    oc_min = max(a['min_c'], b['min_c'])
                    oc_max = min(a['max_c'], b['max_c'])
                    mid = (oc_min + oc_max) // 2
                    v_cols = [mid] if (oc_max - oc_min) < 2 else [mid, mid + 1]
                for r in range(tr, br + 1):
                    for c in v_cols:
                        if 0 <= r < rows and 0 <= c < cols:
                            output[r, c] = 2
    
    # Ray line: extend from bottom of ray-connected shapes to grid bottom
    ray_shape_bottoms = []
    for i, info in enumerate(shape_info):
        if connected[i] and info['min_c'] <= ray_col <= info['max_c']:
            ray_shape_bottoms.append(info['max_r'])
    if ray_shape_bottoms:
        start_r = max(ray_shape_bottoms) + 1
        for r in range(start_r, rows):
            output[r, ray_col] = 2
    
    # Spine extensions
    for i, info in enumerate(shape_info):
        if not connected[i]:
            continue
        if info['spine_type'] == 'vertical' and info['spine_cols']:
            if info['spine_dir'] == 'up':
                for r in range(0, info['min_r']):
                    for c in info['spine_cols']:
                        if 0 <= c < cols:
                            output[r, c] = 2
            elif info['spine_dir'] == 'down':
                for r in range(info['max_r'] + 1, rows):
                    for c in info['spine_cols']:
                        if 0 <= c < cols:
                            output[r, c] = 2
        elif info['spine_type'] == 'horizontal' and info['spine_rows']:
            if info['spine_dir'] == 'left':
                for r in info['spine_rows']:
                    for c in range(0, info['min_c']):
                        if 0 <= r < rows:
                            output[r, c] = 2
            elif info['spine_dir'] == 'right':
                for r in info['spine_rows']:
                    for c in range(info['max_c'] + 1, cols):
                        if 0 <= r < rows:
                            output[r, c] = 2
    
    return output