import numpy as np
from collections import Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """Transform by extracting the frame shape and filling its holes with other shapes."""
    rows, cols = grid.shape
    background = 1  # Most common background color in these problems
    
    # Find all non-background pixels
    non_bg_mask = (grid != background)
    
    # Find connected components (4-connectivity)
    def get_connected_components(mask):
        visited = np.zeros_like(mask, dtype=bool)
        components = []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] and not visited[i, j]:
                    # BFS
                    component = []
                    queue = [(i, j)]
                    visited[i, j] = True
                    while queue:
                        r, c = queue.pop(0)
                        component.append((r, c))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]:
                                if mask[nr, nc] and not visited[nr, nc]:
                                    visited[nr, nc] = True
                                    queue.append((nr, nc))
                    components.append(component)
        return components
    
    components = get_connected_components(non_bg_mask)
    
    # Find the largest component (the frame)
    largest = max(components, key=len)
    frame_color = grid[largest[0][0], largest[0][1]]
    
    # Get frame bounding box
    frame_rows = [p[0] for p in largest]
    frame_cols = [p[1] for p in largest]
    r_min, r_max = min(frame_rows), max(frame_rows)
    c_min, c_max = min(frame_cols), max(frame_cols)
    
    # Extract frame pattern
    frame_height = r_max - r_min + 1
    frame_width = c_max - c_min + 1
    frame_pattern = np.zeros((frame_height, frame_width), dtype=int)
    for r, c in largest:
        frame_pattern[r - r_min, c - c_min] = 1
    
    # Create output grid (frame color as background)
    output = np.full((frame_height, frame_width), frame_color, dtype=int)
    
    # Find holes (cells within bounding box not part of frame)
    holes = (frame_pattern == 0)
    
    # Group holes into connected regions (4-connectivity)
    hole_mask = holes.copy()
    hole_regions = []
    visited_h = np.zeros_like(hole_mask, dtype=bool)
    for i in range(frame_height):
        for j in range(frame_width):
            if hole_mask[i, j] and not visited_h[i, j]:
                region = []
                queue = [(i, j)]
                visited_h[i, j] = True
                while queue:
                    r, c = queue.pop(0)
                    region.append((r, c))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < frame_height and 0 <= nc < frame_width:
                            if hole_mask[nr, nc] and not visited_h[nr, nc]:
                                visited_h[nr, nc] = True
                                queue.append((nr, nc))
                hole_regions.append(region)
    
    # Classify hole regions by shape
    def classify_region(region):
        """Classify a region by its geometric pattern."""
        size = len(region)
        if size == 1:
            return 'single'
        if size == 2:
            return 'line'
        if size == 4:
            r_set = set(p[0] for p in region)
            c_set = set(p[1] for p in region)
            if len(r_set) == 2 and len(c_set) == 2:
                return 'block_2x2'
            return 'other_4'
        if size == 5:
            r_set = set(p[0] for p in region)
            c_set = set(p[1] for p in region)
            if len(r_set) == 3 and len(c_set) == 3:
                return 'cross'
            return 'other_5'
        # Check for rectangular outline
        r_set = sorted(set(p[0] for p in region))
        c_set = sorted(set(p[1] for p in region))
        if len(r_set) >= 2 and len(c_set) >= 2:
            r0, r1 = r_set[0], r_set[-1]
            c0, c1 = c_set[0], c_set[-1]
            # Check if it's a rectangular outline
            expected = set()
            for c in range(c0, c1+1):
                expected.add((r0, c))
                expected.add((r1, c))
            for r in range(r0, r1+1):
                expected.add((r, c0))
                expected.add((r, c1))
            if set(region) == expected:
                return f'rect_outline_{r1-r0+1}x{c1-c0+1}'
        return 'other'
    
    # Classify all hole regions
    hole_types = {}
    for region in hole_regions:
        htype = classify_region(region)
        if htype not in hole_types:
            hole_types[htype] = []
        hole_types[htype].append(region)
    
    # Find all non-frame shapes and classify them
    non_frame_components = []
    for comp in components:
        if comp[0] in largest or any(p in largest for p in comp):
            continue
        color = grid[comp[0][0], comp[0][1]]
        if color == frame_color:
            continue
        non_frame_components.append((comp, color))
    
    # Classify non-frame shapes
    shape_types = {}
    for comp, color in non_frame_components:
        stype = classify_region(comp)
        if stype not in shape_types:
            shape_types[stype] = Counter()
        shape_types[stype][color] += 1
    
    # For each hole type, find the best matching shape color
    # Matching: exact type match first, then 'other' as fallback
    color_assignment = {}
    
    # Define compatibility groups
    compat = {
        'block_2x2': ['block_2x2'],
        'cross': ['cross'],
        'line': ['line'],
        'single': ['single'],
        'other_4': ['other_4', 'block_2x2'],
        'other_5': ['other_5', 'cross'],
    }
    
    # For rect outlines, match with 'other' shapes
    for htype, regions in hole_types.items():
        if htype.startswith('rect_outline'):
            # Find 'other' shaped non-frame shapes
            candidates = []
            for stype, colors in shape_types.items():
                if stype.startswith('other'):
                    candidates.extend(colors.items())
            if candidates:
                best_color = max(candidates, key=lambda x: x[1])[0]
                color_assignment[htype] = best_color
            continue
        
        # Try exact match, then compatible types
        if htype in compat:
            for compat_type in compat[htype]:
                if compat_type in shape_types:
                    best_color = max(shape_types[compat_type].items(), key=lambda x: x[1])[0]
                    color_assignment[htype] = best_color
                    break
        else:
            # Fallback: use most common non-frame shape
            all_colors = Counter()
            for comp, color in non_frame_components:
                all_colors[color] += len(comp)
            if all_colors:
                color_assignment[htype] = all_colors.most_common(1)[0][0]
    
    # Fill holes with assigned colors
    for region in hole_regions:
        htype = classify_region(region)
        if htype in color_assignment:
            color = color_assignment[htype]
            for r, c in region:
                output[r, c] = color
    
    return output