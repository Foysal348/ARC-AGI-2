import numpy as np
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Extract all connected non-zero components, place each at top-left (0,0)
    of the output. Priority: shapes lower in input override shapes higher up.
    Output size = bounding box of the union of all placed shapes.
    """
    nonzero = (grid != 0).astype(int)
    labeled, num_features = ndimage.label(nonzero, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    
    if num_features == 0:
        return np.zeros_like(grid)
    
    shapes = []
    for label_id in range(1, num_features + 1):
        mask = (labeled == label_id)
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        r_min, r_max = rows[0], rows[-1]
        c_min, c_max = cols[0], cols[-1]
        box = grid[r_min:r_max+1, c_min:c_max+1].copy()
        shapes.append((r_min, box))
    
    # Sort by top row ascending (bottom shapes placed last = highest priority)
    shapes.sort(key=lambda x: x[0])
    
    max_h = max(box.shape[0] for _, box in shapes)
    max_w = max(box.shape[1] for _, box in shapes)
    
    output = np.zeros((max_h, max_w), dtype=grid.dtype)
    for _, box in shapes:
        h, w = box.shape
        output[:h, :w] = np.where(box != 0, box, output[:h, :w])
    
    return output