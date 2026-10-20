import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    For each color in the grid, compute the L1 (Manhattan) convex hull of all 
    positions of that color, and draw the boundary of that hull.
    
    The L1 convex hull is bounded by 4 lines:
      r+c = max(r+c), r+c = min(r+c), r-c = max(r-c), r-c = min(r-c)
    
    The boundary consists of 4 edges, each along one of these lines,
    which are always horizontal, vertical, or 45-degree diagonals.
    
    For single points, a 3-step diagonal is drawn down-left.
    """
    rows, cols = grid.shape
    result = np.zeros_like(grid)
    
    # Collect positions for each color
    color_points = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0:
                color = int(grid[r, c])
                if color not in color_points:
                    color_points[color] = []
                color_points[color].append((r, c))
    
    for color, points in color_points.items():
        n = len(points)
        
        if n == 1:
            # Single point: draw 3-step diagonal down-left
            r, c = points[0]
            for i in range(4):
                rr, cc = r + i, c - i
                if 0 <= rr < rows and 0 <= cc < cols:
                    result[rr, cc] = color
        
        elif n == 2:
            (r1, c1), (r2, c2) = points
            dr, dc = r2 - r1, c2 - c1
            if dr == 0:
                for cc in range(min(c1, c2), max(c1, c2) + 1):
                    result[r1, cc] = color
            elif dc == 0:
                for rr in range(min(r1, r2), max(r1, r2) + 1):
                    result[rr, c1] = color
            elif abs(dr) == abs(dc):
                steps = abs(dr)
                sr = 1 if dr > 0 else -1
                sc = 1 if dc > 0 else -1
                for i in range(steps + 1):
                    result[r1 + i * sr, c1 + i * sc] = color
            else:
                # Draw 45-degree diagonal from topmost toward other
                top, bot = (points[0], points[1]) if r1 <= r2 else (points[1], points[0])
                dr2 = bot[0] - top[0]
                dc2 = bot[1] - top[1]
                steps = min(abs(dr2), abs(dc2))
                sr = 1
                sc = 1 if dc2 > 0 else -1
                for i in range(steps + 1):
                    result[top[0] + i * sr, top[1] + i * sc] = color
        
        else:
            # Compute L1 convex hull boundary
            rc_vals = [r + c for r, c in points]
            rd_vals = [r - c for r, c in points]
            rc_max, rc_min = max(rc_vals), min(rc_vals)
            rd_max, rd_min = max(rd_vals), min(rd_vals)
            
            # 4 vertices of the L1 convex hull
            v1_r = (rc_max + rd_max) / 2.0
            v1_c = (rc_max - rd_max) / 2.0
            v2_r = (rc_min + rd_max) / 2.0
            v2_c = (rc_min - rd_max) / 2.0
            v3_r = (rc_min + rd_min) / 2.0
            v3_c = (rc_min - rd_min) / 2.0
            v4_r = (rc_max + rd_min) / 2.0
            v4_c = (rc_max - rd_min) / 2.0
            
            # Edge 1: V4 -> V1 along r-c = rd_max
            r_start, r_end = min(v4_r, v1_r), max(v4_r, v1_r)
            r = np.ceil(r_start)
            while r <= r_end:
                c = r - rd_max
                if 0 <= int(r) < rows and 0 <= int(c) < cols and c == int(c):
                    result[int(r), int(c)] = color
                r += 1
            
            # Edge 2: V1 -> V2 along r+c = rc_max
            r_start, r_end = min(v1_r, v2_r), max(v1_r, v2_r)
            r = np.ceil(r_start)
            while r <= r_end:
                c = rc_max - r
                if 0 <= int(r) < rows and 0 <= int(c) < cols and c == int(c):
                    result[int(r), int(c)] = color
                r += 1
            
            # Edge 3: V2 -> V3 along r-c = rd_min
            r_start, r_end = min(v2_r, v3_r), max(v2_r, v3_r)
            r = np.ceil(r_start)
            while r <= r_end:
                c = r - rd_min
                if 0 <= int(r) < rows and 0 <= int(c) < cols and c == int(c):
                    result[int(r), int(c)] = color
                r += 1
            
            # Edge 4: V3 -> V4 along r+c = rc_min
            r_start, r_end = min(v3_r, v4_r), max(v3_r, v4_r)
            r = np.ceil(r_start)
            while r <= r_end:
                c = rc_min - r
                if 0 <= int(r) < rows and 0 <= int(c) < cols and c == int(c):
                    result[int(r), int(c)] = color
                r += 1
    
    return result