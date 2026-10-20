import numpy as np
from collections import Counter

def transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid)
    H, W = g.shape
    
    def find_boxes(size):
        """Find all size x size boxes with uniform border color."""
        boxes = []
        for r in range(H - size + 1):
            for c in range(W - size + 1):
                bc = g[r, c]
                # Check all four borders
                if not np.all(g[r, c:c+size] == bc):
                    continue
                if not np.all(g[r+size-1, c:c+size] == bc):
                    continue
                if not np.all(g[r:r+size, c] == bc):
                    continue
                if not np.all(g[r:r+size, c+size-1] == bc):
                    continue
                # Interior must not be all border color
                interior = g[r+1:r+size-1, c+1:c+size-1]
                if np.all(interior == bc):
                    continue
                boxes.append((r, c, size, bc))
        # Remove duplicates: keep only maximal (no larger same-border box contains it)
        # For same size, just deduplicate by position
        seen = set()
        unique = []
        for b in boxes:
            key = (b[0], b[1], b[2])
            if key not in seen:
                seen.add(key)
                unique.append(b)
        return unique
    
    # Try different box sizes, prefer the most common interior size
    best_boxes = None
    best_isz = 0
    for size in range(3, min(H, W) + 1):
        boxes = find_boxes(size)
        if len(boxes) >= 2:
            isz = size - 2
            if isz > best_isz:
                best_isz = isz
                best_boxes = boxes
    
    if best_boxes is None or len(best_boxes) < 2:
        # Fallback: try all sizes, pick the one with most boxes
        for size in range(min(H, W), 2, -1):
            boxes = find_boxes(size)
            if len(boxes) >= 2:
                best_boxes = boxes
                best_isz = size - 2
                break
    
    if best_boxes is None:
        return np.zeros((1, 1), dtype=int)
    
    isz = best_isz
    interiors = []
    for r, c, size, bc in best_boxes:
        interior = g[r+1:r+size-1, c+1:c+size-1].copy()
        interiors.append(interior)
    
    n_boxes = len(interiors)
    
    # For each cell, collect all values across boxes
    out = np.zeros((isz, isz), dtype=int)
    for i in range(isz):
        for j in range(isz):
            vals = [interiors[k][i, j] for k in range(n_boxes)]
            non_zero = [v for v in vals if v != 0]
            if not non_zero:
                out[i, j] = 0
            else:
                # Count occurrences of each value
                cnt = Counter(non_zero)
                most_common = cnt.most_common()
                max_count = most_common[0][1]
                # If a single value dominates (appears in >= half the boxes), use it
                # Otherwise use 2
                if len(most_common) == 1 and max_count >= max(2, n_boxes // 2):
                    out[i, j] = most_common[0][0]
                elif len(most_common) == 1:
                    out[i, j] = most_common[0][0]
                else:
                    # Check if there's a value that appears in a clear majority
                    top_val, top_cnt = most_common[0]
                    if top_cnt >= n_boxes - 1 and top_cnt > most_common[1][1]:
                        out[i, j] = top_val
                    else:
                        out[i, j] = 2
    return out