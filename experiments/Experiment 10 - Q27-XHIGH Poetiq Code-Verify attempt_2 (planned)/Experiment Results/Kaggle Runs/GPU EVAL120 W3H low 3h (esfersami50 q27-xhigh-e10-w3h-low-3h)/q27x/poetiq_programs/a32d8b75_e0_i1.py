import numpy as np
from collections import Counter

def transform(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    wall = 6
    out_H, out_W = 20, 24
    
    # Extract pattern from top-left cell and swap colors
    interior = grid[1:4, 1:4].copy().astype(int)
    pc = sorted(set(interior.flatten()) - {0})
    if len(pc) == 2:
        a, b = pc
        pat = np.where(interior == a, b, np.where(interior == b, a, 0)).astype(int)
    else:
        pat = interior.copy()
    
    # Build output
    out = np.zeros((out_H, out_W), dtype=int)
    
    # Fill each 3x3 block with background color from input
    for rb in range(out_H // 3 + 1):
        for cb in range(out_W // 3):
            r0, r1 = rb * 3, min(rb * 3 + 3, out_H)
            c0, c1 = cb * 3, cb * 3 + 3
            ir0, ir1 = min(rb * 3, H), min(rb * 3 + 3, H)
            ic0, ic1 = cb * 3 + 6, min(cb * 3 + 9, W)
            if ic0 < W and ir0 < H:
                reg = grid[ir0:ir1, ic0:ic1].flatten()
                reg = reg[reg != wall]
                out[r0:r1, c0:c1] = Counter(reg).most_common(1)[0][0] if len(reg) > 0 else 0
    
    # Place patterns where input left side indicates
    for r in range(0, out_H, 3):
        if r + 3 > H:
            break
        lreg = grid[r:r+3, 1:4]
        lcolors = set(lreg.flatten()) - {0, wall}
        if len(lcolors) >= 2:
            # Pattern goes in col blocks 0 and 2
            out[r:r+3, 0:3] = pat
            out[r:r+3, 6:9] = pat
        elif len(lcolors) == 1:
            # Single color - might indicate middle placement
            out[r:r+3, 3:6] = pat
    
    return out