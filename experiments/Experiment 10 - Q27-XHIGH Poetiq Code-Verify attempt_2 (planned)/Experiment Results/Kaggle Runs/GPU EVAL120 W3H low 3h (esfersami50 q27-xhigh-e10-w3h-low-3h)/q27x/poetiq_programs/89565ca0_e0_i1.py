import numpy as np
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform input grid with colored boxes and noise into a staircase pattern.
    
    Steps:
    1. Identify noise color (scattered pixels, not forming a rectangular frame)
    2. Find all rectangular boxes and their bounding boxes
    3. Sort boxes by their vertical center (top to bottom)
    4. Output a staircase grid: row i has min(i+1, 4) cells of box color, rest noise
    """
    h, w = grid.shape
    colors = set(int(grid[r, c]) for r in range(h) for c in range(w) if grid[r, c] != 0)
    
    # For each color, find bounding box and check if it forms a frame
    def get_bounding_box(grid, color):
        mask = (grid == color)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return None
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax
    
    def frame_completeness(grid, rmin, rmax, cmin, cmax, color):
        """Fraction of border pixels that match the color."""
        total = 0
        matching = 0
        for c in range(cmin, cmax + 1):
            total += 1
            if grid[rmin, c] == color: matching += 1
        for c in range(cmin, cmax + 1):
            total += 1
            if grid[rmax, c] == color: matching += 1
        for r in range(rmin + 1, rmax):
            total += 1
            if grid[r, cmin] == color: matching += 1
        for r in range(rmin + 1, rmax):
            total += 1
            if grid[r, cmax] == color: matching += 1
        return matching / total if total > 0 else 0
    
    # Identify boxes vs noise
    boxes = []
    noise_color = None
    
    for color in colors:
        bb = get_bounding_box(grid, color)
        if bb is None:
            continue
        rmin, rmax, cmin, cmax = bb
        comp = frame_completeness(grid, rmin, rmax, cmin, cmax, color)
        if comp > 0.7:
            center_row = (rmin + rmax) / 2.0
            center_col = (cmin + cmax) / 2.0
            boxes.append((color, center_row, center_col, rmin, rmax, cmin, cmax))
        else:
            noise_color = color
    
    if noise_color is None:
        # Fallback: pick the color with lowest frame completeness
        scores = []
        for color in colors:
            bb = get_bounding_box(grid, color)
            if bb:
                rmin, rmax, cmin, cmax = bb
                scores.append((frame_completeness(grid, rmin, rmax, cmin, cmax, color), color))
        scores.sort()
        noise_color = scores[0][1]
        boxes = [b for b in boxes if b[0] != noise_color]
    
    # Sort boxes by center row
    boxes.sort(key=lambda x: x[1])
    
    # Build output staircase
    n_boxes = len(boxes)
    num_cols = 4
    output = np.full((n_boxes, num_cols), noise_color, dtype=int)
    
    for i, box in enumerate(boxes):
        fill = min(i + 1, num_cols)
        output[i, :fill] = box[0]
    
    return output