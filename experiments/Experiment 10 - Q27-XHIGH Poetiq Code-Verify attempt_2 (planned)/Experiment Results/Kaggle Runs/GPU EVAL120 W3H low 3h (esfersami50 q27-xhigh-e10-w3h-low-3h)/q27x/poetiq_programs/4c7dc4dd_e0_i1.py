import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Find rectangular boxes with uniform borders containing zeros,
    identify the special content in each, and create an output grid
    showing lines extending right and down from the source cell of each box.
    """
    rows, cols = grid.shape
    
    def is_background(r, c, val):
        return val == ((3 * r + c) % 4) + 1
    
    # Find all rectangular boxes
    boxes = []
    all_colors = sorted(set(int(x) for x in grid.flatten()))
    
    for border_color in all_colors:
        if border_color == 0:
            continue
        mask = (grid == border_color)
        
        for r in range(rows):
            for c in range(cols):
                if not mask[r, c]:
                    continue
                c_end = c
                while c_end + 1 < cols and mask[r, c_end + 1]:
                    c_end += 1
                if c_end - c + 1 < 3:
                    continue
                for r_end in range(r + 2, rows):
                    if not np.all(mask[r_end, c:c_end+1]):
                        break
                    if not np.all(mask[r:r_end+1, c]) or not np.all(mask[r:r_end+1, c_end]):
                        continue
                    interior = grid[r+1:r_end, c+1:c_end]
                    if interior.size > 0 and np.any(interior == 0):
                        boxes.append((r, c, r_end, c_end, border_color))
    
    # Remove contained boxes
    boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    final_boxes = []
    for box in boxes:
        r1, c1, r2, c2, _ = box
        if not any(fb[0] <= r1 and fb[1] <= c1 and fb[2] >= r2 and fb[3] >= c2 for fb in final_boxes):
            final_boxes.append(box)
    
    if not final_boxes:
        return np.zeros((1, 1), dtype=int)
    
    # Output size from interior dimensions
    sizes = [(b[2]-b[0]-1, b[3]-b[1]-1) for b in final_boxes]
    out_h, out_w = max(s[0] for s in sizes), max(s[1] for s in sizes)
    
    output = np.zeros((out_h, out_w), dtype=int)
    
    for r1, c1, r2, c2, bc in final_boxes:
        ir1, ic1 = r1 + 1, c1 + 1
        ir2, ic2 = r2 - 1, c2 - 1
        interior = grid[ir1:ir2+1, ic1:ic2+1]
        
        if interior.size == 0:
            continue
        
        # Find special content cells
        content = []
        for i in range(interior.shape[0]):
            for j in range(interior.shape[1]):
                val = int(interior[i, j])
                if val == 0 or val == bc:
                    continue
                if is_background(ir1 + i, ic1 + j, val):
                    continue
                content.append((i, j, val))
        
        if not content:
            continue
        
        content.sort(key=lambda x: (x[0], x[1]))
        sr, sc, sv = content[0]
        
        # Draw horizontal line right and vertical line down
        for j in range(sc, out_w):
            if output[sr, j] == 0:
                output[sr, j] = 2
        for i in range(sr, out_h):
            if output[i, sc] == 0:
                output[i, sc] = 2
        output[sr, sc] = sv
    
    return output