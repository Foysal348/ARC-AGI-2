import numpy as np
from collections import Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform grid by identifying blocks, finding their 'seed' colors,
    and producing a clean (possibly transposed) output.
    """
    rows, cols = grid.shape
    noise_val = 1
    
    # Find blocks by identifying rectangular regions of same color (ignoring noise)
    # Replace noise for analysis
    analysis = grid.copy()
    analysis[analysis == noise_val] = -1
    
    # Fill noise with neighbor majority
    for _ in range(3):  # multiple passes
        for i in range(rows):
            for j in range(cols):
                if analysis[i, j] == -1:
                    neighbors = []
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < rows and 0 <= nj < cols and analysis[ni, nj] != -1:
                            neighbors.append(analysis[ni, nj])
                    if neighbors:
                        analysis[i, j] = Counter(neighbors).most_common(1)[0][0]
    
    # Find rectangular blocks
    visited = np.zeros((rows, cols), dtype=bool)
    blocks = []
    
    for i in range(rows):
        for j in range(cols):
            if not visited[i, j]:
                color = analysis[i, j]
                r1, c1 = i, j
                r2, c2 = i, j
                
                # Greedily extend to form a rectangle
                changed = True
                while changed:
                    changed = False
                    # Extend right
                    while c2 + 1 < cols and all(analysis[r, c2+1] == color for r in range(r1, r2+1)):
                        c2 += 1
                        changed = True
                    # Extend down
                    while r2 + 1 < rows and all(analysis[r2+1, c] == color for c in range(c1, c2+1)):
                        r2 += 1
                        changed = True
                
                for r in range(r1, r2+1):
                    for c in range(c1, c2+1):
                        visited[r, c] = True
                
                # Determine output color from original grid
                region = grid[r1:r2+1, c1:c2+1]
                non_noise = region[region != noise_val].flatten()
                cc = Counter(non_noise.tolist())
                
                if len(cc) > 1:
                    sorted_cc = cc.most_common()
                    dom = sorted_cc[0][0]
                    minorities = [(c, n) for c, n in sorted_cc if c != dom]
                    if minorities:
                        out_color = minorities[0][0]
                    else:
                        out_color = dom
                else:
                    out_color = color
                
                blocks.append((r1, r2, c1, c2, out_color))
    
    # Build output
    if rows == cols:
        output = np.zeros((rows, cols), dtype=int)
        for r1, r2, c1, c2, oc in blocks:
            output[r1:r2+1, c1:c2+1] = oc
    else:
        output = np.zeros((cols, rows), dtype=int)
        for r1, r2, c1, c2, oc in blocks:
            output[c1:c2+1, r1:r2+1] = oc
    
    return output