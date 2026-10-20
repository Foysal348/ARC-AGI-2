import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transformation rules:
    1. Identify background color (most common).
    2. For each non-bg color, find pairs of pixels on the same 45° diagonal.
       Draw line segments between paired pixels.
    3. For each non-bg pixel that lies on the same diagonal as a pair of a
       different color, draw the full perpendicular diagonal through it.
    4. Pairs take priority over perpendiculars; originals always preserved.
    """
    if grid.size == 0:
        return grid.copy()
    
    rows, cols = grid.shape
    
    # Find background color
    colors, counts = np.unique(grid, return_counts=True)
    bg = colors[np.argmax(counts)]
    
    # Collect non-background pixels by color
    color_pixels = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != bg:
                color_pixels.setdefault(grid[r, c], []).append((r, c))
    
    result = np.full_like(grid, bg)
    
    # Step 1: Find valid pairs for each color
    pair_segments = {}  # color -> list of ((r1,c1),(r2,c2))
    paired_pixels = {}  # color -> set of (r,c)
    
    for color, pixels in color_pixels.items():
        paired = set()
        segments = []
        used = set()
        for i in range(len(pixels)):
            if i in used:
                continue
            r1, c1 = pixels[i]
            for j in range(i + 1, len(pixels)):
                if j in used:
                    continue
                r2, c2 = pixels[j]
                if abs(r1 - r2) == abs(c1 - c2) and (r1 != r2 or c1 != c2):
                    segments.append(((r1, c1), (r2, c2)))
                    paired.add((r1, c1))
                    paired.add((r2, c2))
                    used.add(i)
                    used.add(j)
                    break
        pair_segments[color] = segments
        paired_pixels[color] = paired
    
    # Step 2: Identify which diagonals have pairs (for perpendicular rule)
    # diag_info: list of (family, constant, color)
    diag_info = []
    for color, segments in pair_segments.items():
        for (r1, c1), (r2, c2) in segments:
            if r1 - c1 == r2 - c2:
                diag_info.append(('A', r1 - c1, color))
            else:
                diag_info.append(('B', r1 + c2, color))
    
    # Step 3: Determine which non-bg pixels need perpendicular lines
    perp_pixels = []  # list of (r, c, color, perp_family)
    for color, pixels in color_pixels.items():
        for (r, c) in pixels:
            found = False
            for family, const, pair_color in diag_info:
                if pair_color == color:
                    continue
                if family == 'A' and r - c == const:
                    perp_pixels.append((r, c, color, 'B'))
                    found = True
                    break
                elif family == 'B' and r + c == const:
                    perp_pixels.append((r, c, color, 'A'))
                    found = True
                    break
    
    # Step 4: Draw perpendicular lines (edge to edge)
    for r, c, color, family in perp_pixels:
        if family == 'A':  # r-c = const
            const = r - c
            # Find all valid (rr, cc) where rr-cc=const
            for rr in range(rows):
                cc = rr - const
                if 0 <= cc < cols:
                    result[rr, cc] = color
        else:  # r+c = const
            const = r + c
            for rr in range(rows):
                cc = const - rr
                if 0 <= cc < cols:
                    result[rr, cc] = color
    
    # Step 5: Draw pair line segments (overrides perpendiculars)
    for color, segments in pair_segments.items():
        for (r1, c1), (r2, c2) in segments:
            dr = np.sign(r2 - r1)
            dc = np.sign(c2 - c1)
            r, c = r1, c1
            while True:
                result[r, c] = color
                if r == r2 and c == c2:
                    break
                r += dr
                c += dc
    
    # Step 6: Place all original non-bg pixels (preserved)
    for color, pixels in color_pixels.items():
        for (r, c) in pixels:
            result[r, c] = color
    
    return result