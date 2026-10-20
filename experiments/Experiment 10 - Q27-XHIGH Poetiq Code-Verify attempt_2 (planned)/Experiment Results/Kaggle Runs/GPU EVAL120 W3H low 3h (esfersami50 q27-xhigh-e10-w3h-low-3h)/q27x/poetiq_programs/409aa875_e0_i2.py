import numpy as np
from collections import deque, defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform grid by finding connected components of non-background pixels,
    projecting marks from each component 5 rows above, handling collisions
    (recolored to 9) and middle-mark rules (color 1).
    
    L-shapes (2 pixels in top row): mark at (top_row-5, leftmost_col-5)
    V-shapes (1 pixel in top row): mark at (top_row-5, center_of_component)
    """
    result = grid.copy()
    rows, cols = grid.shape
    bg = 7

    non_bg = np.argwhere(result != bg)
    if len(non_bg) == 0:
        return result

    # Find connected components using 8-connectivity (BFS)
    visited = np.zeros((rows, cols), dtype=bool)
    components = []

    for r, c in non_bg:
        if visited[r, c]:
            continue
        component = []
        queue = deque([(r, c)])
        visited[r, c] = True
        while queue:
            cr, cc = queue.popleft()
            component.append((cr, cc))
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and result[nr, nc] != bg:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
        components.append(component)

    # For each component, determine type and mark position
    comp_info = []
    for comp in components:
        min_row = min(r for r, c in comp)
        top_pixels = sorted([c for r, c in comp if r == min_row])
        all_cols = [c for r, c in comp]
        color = result[min_row, top_pixels[0]]
        
        num_top = len(top_pixels)
        if num_top == 2:
            # L-shape: mark at (top-5, leftmost-5)
            mark_row = min_row - 5
            mark_col = top_pixels[0] - 5
        else:
            # V-shape: mark at (top-5, center of all columns)
            mark_row = min_row - 5
            mark_col = min(all_cols) + (max(all_cols) - min(all_cols)) // 2
        
        comp_info.append({
            'pixels': comp,
            'color': color,
            'mark_row': mark_row,
            'mark_col': mark_col,
        })

    # Detect collisions: mark lands on another component's pixel
    recolored = set()
    for i, info in enumerate(comp_info):
        mr, mc = info['mark_row'], info['mark_col']
        if 0 <= mr < rows and 0 <= mc < cols:
            for j, other in enumerate(comp_info):
                if j != i and (mr, mc) in other['pixels']:
                    recolored.add(j)

    # Apply recoloring
    for j in recolored:
        for r, c in comp_info[j]['pixels']:
            result[r, c] = 9

    # Place marks (only valid ones)
    valid_marks = []
    for i, info in enumerate(comp_info):
        mr, mc = info['mark_row'], info['mark_col']
        if 0 <= mr < rows and 0 <= mc < cols:
            valid_marks.append((mr, mc, info['color'], i))

    # Group by row, apply middle-mark rule
    marks_by_row = defaultdict(list)
    for mr, mc, color, idx in valid_marks:
        marks_by_row[mr].append((mc, color, idx))

    for mr, mark_list in marks_by_row.items():
        mark_list.sort(key=lambda x: x[0])
        n = len(mark_list)
        if n >= 3:
            mid = n // 2
            for i, (mc, color, idx) in enumerate(mark_list):
                if i == mid:
                    result[mr, mc] = 1
                else:
                    result[mr, mc] = color
        else:
            for mc, color, idx in mark_list:
                result[mr, mc] = color

    return result