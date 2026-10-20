import numpy as np
from typing import List, Tuple, Dict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by:
    1. Finding all 3x3 frames (uniform border, different center)
    2. Building a claim graph (frame A claims B if A.center == B.border)
    3. Determining kept frames (sources, or those claiming pure targets not targeted by kept frames)
    4. Kept frames get their target's center; removed frames become zeros
    5. All non-frame cells are preserved
    """
    grid = grid.copy()
    rows, cols = grid.shape
    
    # Step 1: Find all 3x3 frames
    frames = []  # List of (top_row, left_col, border_color, center_color)
    frame_cells = set()  # All cell positions belonging to frames
    
    for r in range(rows - 2):
        for c in range(cols - 2):
            # Extract 3x3 subgrid
            sub = grid[r:r+3, c:c+3]
            border_color = sub[0, 0]
            # Check border is uniform
            border_cells = [sub[0,0], sub[0,1], sub[0,2],
                           sub[1,0], sub[1,2],
                           sub[2,0], sub[2,1], sub[2,2]]
            center_color = sub[1, 1]
            
            if border_color == 0:
                continue
            if all(b == border_color for b in border_cells) and center_color != border_color and center_color != 0:
                frames.append((r, c, border_color, center_color))
                for dr in range(3):
                    for dc in range(3):
                        frame_cells.add((r + dr, c + dc))
    
    if not frames:
        return grid
    
    # Step 2: Build claim graph
    # claim_map[i] = j means frame i claims frame j (frame i's center == frame j's border)
    n = len(frames)
    claim_map = {}  # i -> j (frame i claims frame j)
    claimed_by = {}  # j -> i (frame j is claimed by frame i)
    
    for i in range(n):
        center_i = frames[i][3]
        for j in range(n):
            if i != j and frames[j][2] == center_i:
                claim_map[i] = j
                claimed_by[j] = i
                break
    
    # Step 3: Determine kept frames
    # Sources: frames that claim others but are not claimed by anyone
    sources = set()
    for i in claim_map:
        if i not in claimed_by:
            sources.add(i)
    
    kept = set(sources)
    
    # Non-source frames: kept if their target is pure (claims no one) AND not targeted by a kept frame
    for i in range(n):
        if i in kept:
            continue
        if i in claim_map:
            target = claim_map[i]
            # Target is pure if it doesn't claim anyone
            if target not in claim_map:
                # Check if i is targeted by a kept frame
                if i in claimed_by and claimed_by[i] in kept:
                    continue  # Not kept
                else:
                    kept.add(i)
    
    # Step 4: Create output
    # Start with copy of input
    output = grid.copy()
    
    # Set removed frames to 0
    for i in range(n):
        if i not in kept:
            r, c, _, _ = frames[i]
            output[r:r+3, c:c+3] = 0
    
    # Set kept frames' centers to their target's center
    for i in kept:
        if i in claim_map:
            target = claim_map[i]
            r, c, _, _ = frames[i]
            target_center = frames[target][3]
            output[r+1, c+1] = target_center
    
    return output