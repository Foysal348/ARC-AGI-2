import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform grid by finding connected components of non-background pixels,
    projecting marks from each component, and handling collisions and middle-mark rules.
    """
    result = grid.copy()
    rows, cols = grid.shape
    bg = 7
    
    # Find all non-background pixels
    non_bg = np.argwhere(result != bg)
    if len(non_bg) == 0:
        return result
    
    # Find connected components using 8-connectivity (BFS)
    visited = np.zeros((rows, cols), dtype=bool)
    components = []
    
    for r, c in non_bg:
        if visited[r, c]:
            continue
        # BFS
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
    
    # For each component, find the tip and shape type
    comp_info = []  # (tip_row, tip_col, shape_type, color, pixels)
    for comp in components:
        # Find topmost row, then leftmost col in that row
        min_row = min(r for r, c in comp)
        top_pixels = [(r, c) for r, c in comp if r == min_row]
        top_pixels.sort(key=lambda x: x[1])
        tip_row, tip_col = top_pixels[0]
        num_top = len(top_pixels)
        color = result[tip_row, tip_col]
        shape_type = 'V' if num_top == 1 else 'L'
        comp_info.append({
            'tip_row': tip_row,
            'tip_col': tip_col,
            'shape_type': shape_type,
            'color': color,
            'pixels': comp
        })
    
    # Calculate mark positions
    marks = []  # (mark_row, mark_col, color, source_comp_index)
    for i, info in enumerate(comp_info):
        mr = info['tip_row'] - 5
        if info['shape_type'] == 'V':
            mc = info['tip_col']
        else:  # L
            mc = info['tip_col'] - 5
        marks.append((mr, mc, info['color'], i))
    
    # Determine which components get recolored (collision)
    # A mark collides with a component if it lands on one of the component's pixels
    recolored_components = set()
    for mr, mc, color, src_idx in marks:
        if 0 <= mr < rows and 0 <= mc < cols:
            for j, info in enumerate(comp_info):
                if j != src_idx and (mr, mc) in info['pixels']:
                    recolored_components.add(j)
    
    # Recolored components become 9
    for j in recolored_components:
        for r, c in comp_info[j]['pixels']:
            result[r, c] = 9
    
    # Group marks by row to find middle ones
    from collections import defaultdict
    marks_by_row = defaultdict(list)
    for mr, mc, color, src_idx in marks:
        if 0 <= mr < rows and 0 <= mc < cols:
            marks_by_row[mr].append((mc, color, src_idx))
    
    # Determine mark colors (middle gets 1 if 3+ marks on same row)
    mark_colors = {}  # (mr, mc) -> color
    for mr, mark_list in marks_by_row.items():
        mark_list.sort(key=lambda x: x[0])  # sort by column
        n = len(mark_list)
        if n >= 3:
            mid_idx = n // 2
            for i, (mc, color, src_idx) in enumerate(mark_list):
                if i == mid_idx:
                    mark_colors[(mr, mc)] = 1
                else:
                    # Use the source component's color
                    mark_colors[(mr, mc)] = comp_info[src_idx]['color']
        else:
            for mc, color, src_idx in mark_list:
                mark_colors[(mr, mc)] = comp_info[src_idx]['color']
    
    # Place marks
    for (mr, mc), color in mark_colors.items():
        result[mr, mc] = color
    
    return result