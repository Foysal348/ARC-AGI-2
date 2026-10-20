import numpy as np
from collections import Counter
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transforms the grid by moving 2-blocks into adjacent shapes.
    
    Rules:
    1. Identify background (most common color), shapes (non-bg, non-2), and 2-blocks
    2. For each 2-block aligned with a shape (overlapping row/col with shape's bbox):
       - Move the 2-block into the shape's bbox until its leading edge is adjacent to a shape cell
       - Fill the path (trail) between original and new position with 0s
    3. Unaligned 2-blocks are removed (replaced with background)
    """
    if grid.size == 0:
        return grid
    
    # Step 1: Identify background color
    flat = grid.flatten()
    color_counts = Counter(flat.tolist())
    background = color_counts.most_common(1)[0][0]
    
    # Step 2: Create masks
    shape_mask = (grid != background) & (grid != 2)
    two_mask = (grid == 2)
    
    # Step 3: Find connected components
    shape_labels, num_shapes = ndimage.label(shape_mask)
    two_labels, num_twos = ndimage.label(two_mask)
    
    # Step 4: Compute bounding boxes for shapes
    shape_bboxes = []
    for i in range(1, num_shapes + 1):
        rows, cols = np.where(shape_labels == i)
        if len(rows) > 0:
            shape_bboxes.append((rows.min(), cols.min(), rows.max(), cols.max()))
    
    # Step 5: Process each 2-block
    result = grid.copy()
    
    def is_adjacent_to_shape(r, c, shape_mask):
        """Check if cell (r,c) is adjacent to at least one shape cell."""
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < shape_mask.shape[0] and 0 <= nc < shape_mask.shape[1]:
                if shape_mask[nr, nc]:
                    return True
        return False
    
    def block_adjacent_to_shape(r1, c1, r2, c2, shape_mask):
        """Check if any cell in the block is adjacent to a shape cell."""
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if is_adjacent_to_shape(r, c, shape_mask):
                    return True
        return False
    
    for t in range(1, num_twos + 1):
        two_rows, two_cols = np.where(two_labels == t)
        tr1, tr2 = int(two_rows.min()), int(two_rows.max())
        tc1, tc2 = int(two_cols.min()), int(two_cols.max())
        h = tr2 - tr1 + 1
        w = tc2 - tc1 + 1
        
        moved = False
        for (sr1, sc1, sr2, sc2) in shape_bboxes:
            if moved:
                break
            
            # Check vertical alignment (column overlap)
            if tc1 <= sc2 and tc2 >= sc1:
                # Check if columns fit within shape bbox
                if tc1 >= sc1 and tc2 <= sc2:
                    if tr2 < sr1:
                        # 2-block is above, approach from above (moving down)
                        # Leading edge is the bottom of the block
                        # Target: bottom row of block at position where it's adjacent to shape
                        # Block must fit: top >= sr1, bottom <= sr2
                        for bottom in range(sr2, sr1 + h - 2, -1):
                            top = bottom - h + 1
                            if top < sr1:
                                continue
                            if block_adjacent_to_shape(top, tc1, bottom, tc2, shape_mask):
                                # Move the 2-block
                                # Clear original position
                                for r in range(tr1, tr2 + 1):
                                    for c in range(tc1, tc2 + 1):
                                        result[r, c] = background
                                # Fill trail with 0s
                                for r in range(tr2 + 1, top):
                                    for c in range(tc1, tc2 + 1):
                                        result[r, c] = 0
                                # Place 2-block
                                for r in range(top, bottom + 1):
                                    for c in range(tc1, tc2 + 1):
                                        result[r, c] = 2
                                moved = True
                                break
                    elif tr1 > sr2:
                        # 2-block is below, approach from below (moving up)
                        # Leading edge is the top of the block
                        for top in range(sr1, sr2 - h + 2):
                            bottom = top + h - 1
                            if bottom > sr2:
                                continue
                            if block_adjacent_to_shape(top, tc1, bottom, tc2, shape_mask):
                                # Move the 2-block
                                for r in range(tr1, tr2 + 1):
                                    for c in range(tc1, tc2 + 1):
                                        result[r, c] = background
                                # Fill trail with 0s
                                for r in range(bottom + 1, tr1):
                                    for c in range(tc1, tc2 + 1):
                                        result[r, c] = 0
                                # Place 2-block
                                for r in range(top, bottom + 1):
                                    for c in range(tc1, tc2 + 1):
                                        result[r, c] = 2
                                moved = True
                                break
            
            # Check horizontal alignment (row overlap)
            if not moved and tr1 <= sr2 and tr2 >= sr1:
                # Check if rows fit within shape bbox
                if tr1 >= sr1 and tr2 <= sr2:
                    if tc2 < sc1:
                        # 2-block is to the left, approach from left (moving right)
                        # Leading edge is the right side of the block
                        for right in range(sc2, sc1 + w - 2, -1):
                            left = right - w + 1
                            if left < sc1:
                                continue
                            if block_adjacent_to_shape(tr1, left, tr2, right, shape_mask):
                                # Move the 2-block
                                for r in range(tr1, tr2 + 1):
                                    for c in range(tc1, tc2 + 1):
                                        result[r, c] = background
                                # Fill trail with 0s
                                for c in range(tc2 + 1, left):
                                    for r in range(tr1, tr2 + 1):
                                        result[r, c] = 0
                                # Place 2-block
                                for r in range(tr1, tr2 + 1):
                                    for c in range(left, right + 1):
                                        result[r, c] = 2
                                moved = True
                                break
                    elif tc1 > sc2:
                        # 2-block is to the right, approach from right (moving left)
                        # Leading edge is the left side of the block
                        for left in range(sc1, sc2 - w + 2):
                            right = left + w - 1
                            if right > sc2:
                                continue
                            if block_adjacent_to_shape(tr1, left, tr2, right, shape_mask):
                                # Move the 2-block
                                for r in range(tr1, tr2 + 1):
                                    for c in range(tc1, tc2 + 1):
                                        result[r, c] = background
                                # Fill trail with 0s
                                for c in range(right + 1, tc1):
                                    for r in range(tr1, tr2 + 1):
                                        result[r, c] = 0
                                # Place 2-block
                                for r in range(tr1, tr2 + 1):
                                    for c in range(left, right + 1):
                                        result[r, c] = 2
                                moved = True
                                break
        
        # If not moved, remove the 2-block
        if not moved:
            for r in range(tr1, tr2 + 1):
                for c in range(tc1, tc2 + 1):
                    result[r, c] = background
    
    return result