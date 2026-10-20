import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Identify rectangular frames in the input, select the smallest as output,
    and overlay special pixels from all frames onto it.
    """
    grid = np.array(grid)
    rows, cols = grid.shape
    
    def find_frames():
        """Find all valid rectangular frames."""
        frames = []
        visited = np.zeros((rows, cols), dtype=bool)
        
        for r in range(rows):
            for c in range(cols):
                if visited[r, c]:
                    continue
                color = grid[r, c]
                
                # Try to find a frame with top-left at (r,c)
                # Find right extent of top edge
                right = c
                while right + 1 < cols and grid[r, right + 1] == color:
                    right += 1
                
                if right - c < 2:
                    continue
                
                # Find bottom extent of left edge
                bottom = r
                while bottom + 1 < rows and grid[bottom + 1, c] == color:
                    bottom += 1
                
                if bottom - r < 2:
                    continue
                
                # Verify it's a complete rectangle
                valid = True
                for rr in range(r, bottom + 1):
                    if grid[rr, right] != color:
                        valid = False
                        break
                if valid:
                    for cc in range(c, right + 1):
                        if grid[bottom, cc] != color:
                            valid = False
                            break
                if not valid:
                    continue
                
                # Check interior is uniform
                interior = set()
                for rr in range(r + 1, bottom):
                    for cc in range(c + 1, right):
                        interior.add(grid[rr, cc])
                
                if len(interior) != 1:
                    continue
                interior_color = list(interior)[0]
                if interior_color == color:
                    continue
                
                frames.append({
                    'top': r, 'left': c, 'bottom': bottom, 'right': right,
                    'border': color, 'fill': interior_color,
                    'h': bottom - r + 1, 'w': right - c + 1,
                    'area': (bottom - r + 1) * (right - c + 1)
                })
                for rr in range(r, bottom + 1):
                    for cc in range(c, right + 1):
                        visited[rr, cc] = True
        
        return frames
    
    frames = find_frames()
    if not frames:
        return np.zeros_like(grid)
    
    # Target = smallest frame
    target = min(frames, key=lambda f: f['area'])
    
    out_h, out_w = target['h'], target['w']
    output = np.full((out_h, out_w), target['border'], dtype=grid.dtype)
    output[1:-1, 1:-1] = target['fill']
    
    # Overlay special pixels from all frames
    for frame in frames:
        for rr in range(frame['top'] + 1, frame['bottom']):
            for cc in range(frame['left'] + 1, frame['right']):
                px = grid[rr, cc]
                if px != frame['fill'] and px != frame['border']:
                    # Relative position in this frame's interior
                    ir = rr - frame['top'] - 1
                    ic = cc - frame['left'] - 1
                    # Place in output at same relative position
                    orr = ir + 1
                    ocl = ic + 1
                    if 1 <= orr < out_h - 1 and 1 <= ocl < out_w - 1:
                        output[orr, ocl] = px
    
    return output