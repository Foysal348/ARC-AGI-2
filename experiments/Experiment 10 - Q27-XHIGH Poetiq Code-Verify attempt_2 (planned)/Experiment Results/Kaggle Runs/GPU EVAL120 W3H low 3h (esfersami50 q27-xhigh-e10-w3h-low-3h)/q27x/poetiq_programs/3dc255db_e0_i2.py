import numpy as np
from scipy.ndimage import binary_fill_holes, label

def transform(grid: np.ndarray) -> np.ndarray:
    """
    For each ring/container shape, identify inner pixels (pixels of another color
    within the ring's bounding box that are enclosed by the ring).
    Move those inner pixels to a line on the most 'closed' side of the ring.
    
    The 'closed' side is determined by which edge of the ring's bounding box
    has the highest density of ring pixels.
    """
    result = np.array(grid, dtype=int)
    rows, cols = grid.shape
    colors = set(grid[grid != 0].flatten().tolist())
    
    color_positions = {}
    for c in colors:
        color_positions[c] = np.argwhere(grid == c)
    
    def get_bbox(positions):
        return (int(positions[:,0].min()), int(positions[:,1].min()),
                int(positions[:,0].max()), int(positions[:,1].max()))
    
    def edge_density(mask, bbox):
        """Calculate density of ring pixels on each edge of the bounding box."""
        rmin, cmin, rmax, cmax = bbox
        height = rmax - rmin + 1
        width = cmax - cmin + 1
        
        # Count pixels on each edge
        top = np.sum(mask[rmin, cmin:cmax+1])
        bottom = np.sum(mask[rmax, cmin:cmax+1])
        left = np.sum(mask[rmin:rmax+1, cmin])
        right = np.sum(mask[rmin:rmax+1, cmax])
        
        return {
            'top': top / max(width, 1),
            'bottom': bottom / max(width, 1),
            'left': left / max(height, 1),
            'right': right / max(height, 1)
        }
    
    def find_enclosed_pixels(container_mask, content_positions, bbox):
        """Find which content pixels are enclosed by the container ring."""
        rmin, cmin, rmax, cmax = bbox
        enclosed = []
        outside = []
        
        # Use flood fill from the content pixel to see if it can reach outside the bbox
        # A pixel is "inside" if it's within the bbox and surrounded by the ring
        # Simple check: pixel is within bbox
        for pos in content_positions:
            r, c = int(pos[0]), int(pos[1])
            if rmin <= r <= rmax and cmin <= c <= cmax:
                enclosed.append((r, c))
            else:
                outside.append((r, c))
        
        return enclosed, outside
    
    # Identify container-inner pairs
    # A container is a color that forms a ring (has holes when filled)
    def is_ring(color):
        mask = (grid == color)
        filled = binary_fill_holes(mask)
        # Check if there are pixels that get filled (i.e., there's a hole)
        return bool(np.any(filled & ~mask))
    
    ring_colors = [c for c in colors if is_ring(c)]
    
    # If no ring detected by hole-filling, use size-based heuristic
    if not ring_colors:
        sorted_colors = sorted(colors, key=lambda c: -len(color_positions[c]))
        ring_colors = sorted_colors[:2]
    
    pairs = []
    for rc in ring_colors:
        rc_bbox = get_bbox(color_positions[rc])
        for c in colors:
            if c == rc:
                continue
            c_bbox = get_bbox(color_positions[c])
            # Content bbox should be mostly inside container bbox
            if rc_bbox[0] <= c_bbox[0] and rc_bbox[1] <= c_bbox[1] and \
               rc_bbox[2] >= c_bbox[2] and rc_bbox[3] >= c_bbox[3]:
                pairs.append((rc, c))
    
    # Fallback: if no pairs found, try looser criteria
    if not pairs:
        sorted_colors = sorted(colors, key=lambda c: -len(color_positions[c]))
        for i, rc in enumerate(sorted_colors):
            rc_bbox = get_bbox(color_positions[rc])
            for c in sorted_colors:
                if c == rc:
                    continue
                c_bbox = get_bbox(color_positions[c])
                # Overlap check
                overlap_r = max(0, min(rc_bbox[2], c_bbox[2]) - max(rc_bbox[0], c_bbox[0]) + 1)
                overlap_c = max(0, min(rc_bbox[3], c_bbox[3]) - max(rc_bbox[1], c_bbox[1]) + 1)
                content_area = (c_bbox[2] - c_bbox[0] + 1) * (c_bbox[3] - c_bbox[1] + 1)
                if content_area > 0 and (overlap_r * overlap_c) / content_area > 0.3:
                    pairs.append((rc, c))
    
    for container_color, inner_color in pairs:
        inner_pos = color_positions[inner_color]
        cont_mask = (grid == container_color)
        c_bbox = get_bbox(color_positions[container_color])
        rmin, cmin, rmax, cmax = c_bbox
        
        # Separate inside and outside pixels
        enclosed, outside = find_enclosed_pixels(cont_mask, inner_pos, c_bbox)
        
        if not enclosed:
            continue
        
        # Determine the most "closed" side
        densities = edge_density(cont_mask, c_bbox)
        
        # Find the side with maximum density (most closed)
        max_density = max(densities.values())
        closed_sides = [side for side, d in densities.items() if d == max_density]
        
        # If tied, prefer the side that is opposite to where the enclosed pixels tend to be
        enc_arr = np.array(enclosed)
        enc_center_r = enc_arr[:, 0].mean()
        enc_center_c = enc_arr[:, 1].mean()
        cont_center_r = (rmin + rmax) / 2.0
        cont_center_c = (cmin + cmax) / 2.0
        
        # Determine preferred direction based on where enclosed pixels are
        # relative to container center
        dr = enc_center_r - cont_center_r
        dc = enc_center_c - cont_center_c
        
        if abs(dr) > abs(dc):
            # Enclosed pixels are above or below → prefer moving to the opposite vertical side
            if dr < 0:  # pixels are above center → move down (to bottom)
                preferred = ['bottom', 'top', 'left', 'right']
            else:  # pixels are below center → move up (to top)
                preferred = ['top', 'bottom', 'left', 'right']
        else:
            # Enclosed pixels are left or right → prefer moving to the opposite horizontal side
            if dc < 0:  # pixels are left of center → move right
                preferred = ['right', 'left', 'top', 'bottom']
            else:  # pixels are right of center → move left
                preferred = ['left', 'right', 'top', 'bottom']
        
        # Pick the closed side that's also in our preferred list
        target_side = None
        for side in preferred:
            if side in closed_sides:
                target_side = side
                break
        if target_side is None:
            # Just pick the most closed side
            target_side = max(densities, key=densities.get)
        
        # Determine placement
        n_pixels = len(enclosed)
        
        if target_side == 'top':
            center_col = int(round(cont_center_c))
            new_pos = [(rmin - 1 - i, center_col) for i in range(n_pixels)]
            new_pos = [p for p in new_pos if p[0] >= 0]
        elif target_side == 'bottom':
            center_col = int(round(cont_center_c))
            new_pos = [(rmax + 1 + i, center_col) for i in range(n_pixels)]
            new_pos = [p for p in new_pos if p[0] < rows]
        elif target_side == 'left':
            center_row = int(round(cont_center_r))
            new_pos = [(center_row, cmin - 1 - i) for i in range(n_pixels)]
            new_pos = [p for p in new_pos if p[1] >= 0]
        elif target_side == 'right':
            center_row = int(round(cont_center_r))
            new_pos = [(center_row, cmax + 1 + i) for i in range(n_pixels)]
            new_pos = [p for p in new_pos if p[1] < cols]
        
        # Apply: clear all inner pixels, restore outside, place new
        for pos in inner_pos:
            r, c = int(pos[0]), int(pos[1])
            result[r, c] = 0
        for r, c in outside:
            result[r, c] = inner_color
        for r, c in new_pos:
            if 0 <= r < rows and 0 <= c < cols:
                result[r, c] = inner_color
    
    return result