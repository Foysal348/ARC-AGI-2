import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Cluster all shapes around the largest (anchor) shape.
    The anchor stays in place; other shapes are moved adjacent to it.
    Shapes may be rotated 90° CCW to fit in anchor's empty bounding box cells.
    Anchor pixels take priority in overlaps.
    """
    if grid.size == 0:
        return grid
    
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Find background
    flat = grid.flatten()
    unique, counts = np.unique(flat, return_counts=True)
    bg = unique[np.argmax(counts)]
    
    # Find connected components
    visited = np.zeros((rows, cols), dtype=bool)
    shapes = []
    
    for r in range(rows):
        for c in range(cols):
            if not visited[r, c] and grid[r, c] != bg:
                color = int(grid[r, c])
                component = set()
                queue = deque([(r, c)])
                visited[r, c] = True
                while queue:
                    cr, cc = queue.popleft()
                    component.add((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr,nc] and grid[nr,nc] == color:
                            visited[nr,nc] = True
                            queue.append((nr,nc))
                shapes.append((color, component))
    
    if not shapes:
        return grid
    
    # Anchor = largest shape
    anchor_i = max(range(len(shapes)), key=lambda i: len(shapes[i][1]))
    anchor_color, anchor_px = shapes[anchor_i]
    
    a_min_r = min(p[0] for p in anchor_px)
    a_max_r = max(p[0] for p in anchor_px)
    a_min_c = min(p[1] for p in anchor_px)
    a_max_c = max(p[1] for p in anchor_px)
    
    # Output grid
    output = np.full((rows, cols), bg, dtype=grid.dtype)
    for (r, c) in anchor_px:
        output[r, c] = anchor_color
    
    other = [(c, p) for i, (c, p) in enumerate(shapes) if i != anchor_i]
    other.sort(key=lambda x: len(x[1]), reverse=True)
    
    def shape_mat(pixels):
        rs = [p[0] for p in pixels]
        cs = [p[1] for p in pixels]
        h = max(rs) - min(rs) + 1
        w = max(cs) - min(cs) + 1
        m = np.zeros((h, w), dtype=int)
        for (r, c) in pixels:
            m[r - min(rs), c - min(cs)] = 1
        return m
    
    occupied = set(anchor_px)
    
    for color, pixels in other:
        mat = shape_mat(pixels)
        h, w = mat.shape
        placed = False
        
        # Try original and 90° CCW rotation
        variants = [mat, np.rot90(mat, 1)]
        
        for vm in variants:
            vh, vw = vm.shape
            # Try positions: inside anchor BB, then adjacent
            positions = []
            # Inside/overlapping anchor BB
            for sr in range(a_min_r, a_max_r - vh + 2):
                for sc in range(a_min_c, a_max_c - vw + 2):
                    positions.append((sr, sc))
            # Adjacent: right, below, left, above
            positions.append((a_min_r, a_max_c + 1))
            positions.append((a_max_r + 1, a_min_c))
            positions.append((a_min_r - vh, a_min_c))
            positions.append((a_min_r, a_min_c - vw))
            
            for (sr, sc) in positions:
                # Check bounds and conflicts
                valid = True
                for i in range(vh):
                    for j in range(vw):
                        if vm[i,j] == 1:
                            r, c = sr+i, sc+j
                            if r < 0 or r >= rows or c < 0 or c >= cols:
                                valid = False
                                break
                            if (r,c) in anchor_px or (r,c) in occupied:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    for i in range(vh):
                        for j in range(vw):
                            if vm[i,j] == 1:
                                r, c = sr+i, sc+j
                                output[r,c] = color
                                occupied.add((r,c))
                    placed = True
                    break
            if placed:
                break
        
        if not placed:
            for (r, c) in pixels:
                if (r,c) not in anchor_px and (r,c) not in occupied:
                    output[r,c] = color
                    occupied.add((r,c))
    
    return output