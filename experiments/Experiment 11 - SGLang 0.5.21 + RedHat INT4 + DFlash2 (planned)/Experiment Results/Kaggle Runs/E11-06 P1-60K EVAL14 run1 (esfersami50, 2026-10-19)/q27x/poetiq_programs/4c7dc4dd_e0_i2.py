import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid)
    H, W = g.shape
    
    # Find all square boxes with consistent border color
    def find_boxes():
        boxes = []
        for size in range(3, min(H, W) + 1):
            for r in range(H - size + 1):
                for c in range(W - size + 1):
                    # Determine border color from top-left corner
                    bc = g[r, c]
                    # Check border
                    if g[r, c:c+size].any() != bc:
                        continue
                    if g[r+r, c:c+size].any() != bc:
                        continue
                    if g[r:r+size+1, c].any() != bc:
                        continue
                    if g[r:r+size+1, c+c].any() != bc:
                        continue
                    # Check interior is not border color
                    interior = g[r+1:r+size, c+1:c+size]
                    if (interior == bc).all():
                        continue
                    boxes.append((r, c, size, bc))
        # Remove duplicates and overlapping boxes
        # Keep only maximal boxes (a box is maximal if no larger box contains it with same border)
        # For simplicity, deduplicate: keep unique (r,c,size,bc)
        seen = set()
        unique = []
        for r, c, size, bc in boxes:
            key = (r, c, size, bc)
            if key not in seen:
                seen.add(key)
                unique.append((r, c, size, bc))
        return unique
    
    boxes = find_boxes()
    if not boxes:
        return np.zeros_like(g)
    
    # Determine the interior size (should be consistent across all boxes)
    interior_sizes = [size - 2 for _, _, size, _ in boxes]
    # Use the most common interior size
    from collections import Counter
    size_counts = Counter(interior_sizes)
    isz = size_counts.most_common(1)[0][0]
    
    # Filter boxes to those with this interior size
    boxes = [(r, c, s, bc) for r, c, s, bc in boxes if s - 2 == isz]
    
    if not boxes:
        return np.zeros((isz, isz), dtype=int)
    
    # Extract interiors and XOR binary presence
    xor_pattern = np.zeros((isz, isz), dtype=int)
    value_lists = [[[] for _ in range(isz)] for _ in range(isz)]
    
    for r, c, size, bc in boxes:
        interior = g[r+1:r+size, c+1:c+size]
        for i in range(isz):
            for j in range(isz):
                v = interior[i, j]
                if v != 0:
                    xor_pattern[i, j] ^= 1
                    value_lists[i][j].append(v)
    
    # Build output
    out = np.zeros((isz, isz), dtype=int)
    for i in range(isz):
        for j in range(isz):
            if xor_pattern[i, j] == 0:
                out[i, j] = 0
            else:
                vals = value_lists[i][j]
                if not vals:
                    out[i, j] = 2
                elif len(set(vals)) == 1:
                    # All non-zero values are the same
                    out[i, j] = vals[0]
                else:
                    # Mixed values: use 2 as default, unless all are 6
                    out[i, j] = 2
    
    return out