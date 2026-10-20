import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform grid using TL quadrant as template.
    TR = horizontal mirror of TL (scaled by seed)
    BL = vertical flip of TL (scaled by seed)  
    BR = 180° rotation of TL (scaled by seed)
    Scale determined by seed size relative to TL's 2-block size.
    """
    result = grid.copy()
    rows, cols = grid.shape
    mid_r, mid_c = rows // 2, cols // 2
    h, w = mid_r, mid_c  # 13x13 quadrants
    
    tl = grid[:h, :w].copy()
    
    def get_blocks(quadrant):
        hq, wq = quadrant.shape
        visited = np.zeros((hq, wq), dtype=bool)
        blocks = []
        for r in range(hq):
            for c in range(wq):
                if quadrant[r, c] != 4 and not visited[r, c]:
                    color = quadrant[r, c]
                    min_r, max_r, min_c, max_c = r, r, c, c
                    stack = [(r, c)]
                    while stack:
                        cr, cc = stack.pop()
                        if 0 <= cr < hq and 0 <= cc < wq and not visited[cr, cc] and quadrant[cr, cc] == color:
                            visited[cr, cc] = True
                            min_r = min(min_r, cr); max_r = max(max_r, cr)
                            min_c = min(min_c, cc); max_c = max(max_c, cc)
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = cr+dr, cc+dc
                                if 0 <= nr < hq and 0 <= nc < wq and not visited[nr, nc] and quadrant[nr, nc] == color:
                                    stack.append((nr, nc))
                    blocks.append((color, min_r, max_r, min_c, max_c))
        return blocks
    
    tl_blocks = get_blocks(tl)
    
    # Find 2s block in TL for scale reference
    tl_2s = [b for b in tl_blocks if b[0] == 2]
    if not tl_2s:
        return result
    tl_2s_h = tl_2s[0][2] - tl_2s[0][1] + 1
    tl_2s_w = tl_2s[0][4] - tl_2s[0][3] + 1
    
    def process_quadrant(qr, qc, ttype):
        quad = grid[qr:qr+h, qc:qc+w].copy()
        blocks = get_blocks(quad)
        seed = [b for b in blocks if b[0] == 2]
        if not seed:
            return
        seed = seed[0]
        seed_h = seed[2] - seed[1] + 1
        seed_w = seed[4] - seed[3] + 1
        sr = seed_h / tl_2s_h
        sc = seed_w / tl_2s_w
        
        def xform(b):
            color, r1, r2, c1, c2 = b
            if ttype == 'h':
                return color, r1, r2, w-1-c2, w-1-c1
            elif ttype == 'v':
                return color, h-1-r2, h-1-r1, c1, c2
            else:
                return color, h-1-r2, h-1-r1, w-1-c2, w-1-c1
        
        tblocks = [xform(b) for b in tl_blocks]
        t2s = [b for b in tblocks if b[0] == 2][0]
        
        # Align: 2s top-left in pattern = seed top-left
        or_ = seed[1] - t2s[1]
        oc = seed[3] - t2s[3]
        
        for color, r1, r2, c1, c2 in tblocks:
            br = r2 - r1 + 1
            bc = c2 - c1 + 1
            nr = max(1, round(br * sr))
            nc = max(1, round(bc * sc))
            ar1 = r1 + or_
            ac1 = c1 + oc
            for dr in range(nr):
                for dc in range(nc):
                    rr, cc = ar1+dr, ac1+dc
                    if 0 <= rr < h and 0 <= cc < w:
                        result[qr+rr, qc+cc] = color
    
    process_quadrant(0, mid_c+1, 'h')
    process_quadrant(mid_r+1, 0, 'v')
    process_quadrant(mid_r+1, mid_c+1, 'r')
    
    return result