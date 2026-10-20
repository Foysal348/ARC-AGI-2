import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Expand isolated marker pixels into 4x4 glyphs of the paired color.

    Color/glyph pairs:
      1/3 -> edge segments without corners
      2/4 -> full 4x4 border
      5/6 -> two opposite 2x2 blocks
      7/8 -> four corners plus central 2x2 block

    The input may contain full or nearly full 4x4 glyphs as distractors.
    Those glyph pixels are not treated as seeds.

    If two or more seeds would produce overlapping 4x4 glyphs, keep the
    most bottom-right seed (larger row first, then larger column) and
    discard the others.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2:
        return np.zeros_like(grid, dtype=int)

    H, W = grid.shape
    out = np.zeros((H, W), dtype=int)

    if H < 4 or W < 4:
        return out

    # 4x4 glyph patterns.
    FULL = [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0),                 (1, 3),
        (2, 0),                 (2, 3),
        (3, 0), (3, 1), (3, 2), (3, 3),
    ]

    MISSING = [
        (0, 1), (0, 2),
        (1, 0),                 (1, 3),
        (2, 0),                 (2, 3),
        (3, 1), (3, 2),
    ]

    DIAGONAL = [
        (0, 0), (0, 1),
        (1, 0), (1, 1),
        (2, 2), (2, 3),
        (3, 2), (3, 3),
    ]

    CORNER_CENTER = [
        (0, 0),                 (0, 3),
        (1, 1), (1, 2),
        (2, 1), (2, 2),
        (3, 0),                 (3, 3),
    ]

    glyph_coords = {
        1: MISSING,
        2: FULL,
        3: MISSING,
        4: FULL,
        5: DIAGONAL,
        6: DIAGONAL,
        7: CORNER_CENTER,
        8: CORNER_CENTER,
    }

    paired_color = {
        1: 3, 3: 1,
        2: 4, 4: 2,
        5: 6, 6: 5,
        7: 8, 8: 7,
    }

    all_44 = [(r, c) for r in range(4) for c in range(4)]
    non_glyph_coords = {}
    for color, coords in glyph_coords.items():
        coord_set = set(coords)
        non_glyph_coords[color] = [(r, c) for (r, c) in all_44 if (r, c) not in coord_set]

    # Mark pixels belonging to full or nearly full 4x4 distractor glyphs.
    # "Nearly full" allows one expected glyph pixel to be missing/overwritten,
    # which happens when a seed replaces one pixel of a distractor glyph.
    is_template = np.zeros((H, W), dtype=bool)

    for r in range(H - 3):
        for c in range(W - 3):
            for color, coords in glyph_coords.items():
                # If the distractor color appears outside the glyph pattern,
                # this is not a clean/nearly clean glyph.
                extra = False
                for dr, dc in non_glyph_coords[color]:
                    if grid[r + dr, c + dc] == color:
                        extra = True
                        break
                if extra:
                    continue

                matches = []
                missing = 0
                for dr, dc in coords:
                    if grid[r + dr, c + dc] == color:
                        matches.append((r + dr, c + dc))
                    else:
                        missing += 1

                # Exact or missing at most one expected pixel.
                if missing <= 1:
                    for pr, pc in matches:
                        is_template[pr, pc] = True

    # Find single-pixel seed components that are not part of template glyphs.
    candidates = []
    visited = np.zeros((H, W), dtype=bool)
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for r in range(H):
        for c in range(W):
            if grid[r, c] == 0 or is_template[r, c] or visited[r, c]:
                continue

            color = grid[r, c]
            if color not in paired_color:
                visited[r, c] = True
                continue

            stack = [(r, c)]
            visited[r, c] = True
            size = 0

            while stack:
                i, j = stack.pop()
                size += 1

                for di, dj in dirs:
                    ni, nj = i + di, j + dj
                    if (
                        0 <= ni < H
                        and 0 <= nj < W
                        and not visited[ni, nj]
                        and not is_template[ni, nj]
                        and grid[ni, nj] == color
                    ):
                        visited[ni, nj] = True
                        stack.append((ni, nj))

            # Only a lone pixel can be a seed, and it must have room for a 4x4 glyph.
            if size == 1 and r + 3 < H and c + 3 < W:
                candidates.append((r, c, color))

    # Resolve overlaps globally. Higher priority = lower row, then righter column.
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

    selected = []
    for r, c, color in candidates:
        conflict = False
        for sr, sc, _ in selected:
            if abs(r - sr) < 4 and abs(c - sc) < 4:
                conflict = True
                break
        if not conflict:
            selected.append((r, c, color))

    # Draw the selected glyphs using the paired output color.
    for r, c, color in selected:
        out_color = paired_color[color]
        for dr, dc in glyph_coords[color]:
            out[r + dr, c + dc] = out_color

    return out