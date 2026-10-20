import numpy as np
from scipy import ndimage

def transform(grid: np.ndarray) -> np.ndarray:
    """Remove outlier pixels from geometric outline shapes to produce clean outlines."""
    grid = grid.copy()
    rows, cols = grid.shape
    
    colors, counts = np.unique(grid, return_counts=True)
    bg_color = colors[np.argmax(counts)]
    non_bg_colors = [c for c in colors if c != bg_color]
    
    result = grid.copy()
    
    for color in non_bg_colors:
        mask = (grid == color).astype(np.uint8)
        structure = np.ones((3, 3), int)
        labeled, num_features = ndimage.label(mask, structure=structure)
        
        for comp_id in range(1, num_features + 1):
            comp_mask = (labeled == comp_id).astype(np.uint8)
            coords = np.argwhere(comp_mask)
            if len(coords) < 4:
                continue
            
            r_min, c_min = coords.min(axis=0)
            r_max, c_max = coords.max(axis=0)
            h = r_max - r_min + 1
            w = c_max - c_min + 1
            
            best_score = -np.inf
            best_ideal = None
            
            # Try hollow rectangles (rings)
            for rh in range(3, h + 1):
                for rw in range(3, w + 1):
                    for dr in range(0, h - rh + 1):
                        for dc in range(0, w - rw + 1):
                            r0, c0 = r_min + dr, c_min + dc
                            ideal = np.zeros((rows, cols), dtype=np.uint8)
                            ideal[r0, c0:c0+rw] = 1
                            ideal[r0+rh-1, c0:c0+rw] = 1
                            ideal[r0:r0+rh, c0] = 1
                            ideal[r0:r0+rh, c0+rw-1] = 1
                            s = _score(comp_mask, ideal)
                            if s > best_score:
                                best_score = s
                                best_ideal = ideal.copy()
            
            # Try hollow diamonds - center near bounding box center
            center_r = (r_min + r_max) // 2
            center_c = (c_min + c_max) // 2
            
            for radius in range(2, max(h, w)):
                # Center should be near the bounding box center
                for cr in range(max(0, center_r - 2), min(rows, center_r + 3)):
                    for cc in range(max(0, center_c - 2), min(cols, center_c + 3)):
                        if cr - radius < 0 or cr + radius >= rows:
                            continue
                        if cc - radius < 0 or cc + radius >= cols:
                            continue
                        ideal = np.zeros((rows, cols), dtype=np.uint8)
                        for dr in range(-radius, radius + 1):
                            dc = radius - abs(dr)
                            ideal[cr + dr, cc + dc] = 1
                            ideal[cr + dr, cc - dc] = 1
                        s = _score(comp_mask, ideal)
                        if s > best_score:
                            best_score = s
                            best_ideal = ideal.copy()
            
            if best_ideal is not None and best_score > 0:
                remove_mask = (comp_mask == 1) & (best_ideal == 0)
                result[remove_mask] = bg_color
    
    return result


def _score(comp_mask, ideal):
    matches = np.sum((comp_mask == 1) & (ideal == 1))
    extra = np.sum((comp_mask == 1) & (ideal == 0))
    missing = np.sum((comp_mask == 0) & (ideal == 1))
    return matches - extra * 3 - missing * 2