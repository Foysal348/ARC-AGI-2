import numpy as np
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Expand isolated marker pixels into 4x4 glyphs.

    The four color/glyph pairs are:
      1/3 -> hollow square without corners
      2/4 -> full 4x4 border
      5/6 -> two diagonal 2x2 blocks
      7/8 -> four corners plus central 2x2 block

    Multi-pixel exact 4x4 glyphs in the input are treated as templates
    and are removed unless a corresponding seed produces that glyph.
    If same-color seeds would create overlapping 4x4 glyphs, the
    bottom-right one is kept.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2:
        return np.zeros_like(grid, dtype=int)

    H, W = grid.shape
    out = np.zeros((H, W), dtype=int)
    if H < 4 or W < 4:
        return out

    # Relative coordinates for each 4x4 glyph.
    FULL = [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 3),
        (2, 0), (2, 3),
        (3, 0), (3, 1), (3, 2), (3, 3),
    ]

    MISSING = [
        (0, 1), (0, 2),
        (1, 0), (1, 3),
        (2, 0), (2, 3),
        (3, 1), (3, 2),
    ]

    DIAGONAL = [
        (0, 0), (0, 1),
        (1, 0), (1, 1),
        (2, 2), (2, 3),
        (3, 2), (3, 3),
    ]

    CORNER_CENTER = [
        (0, 0), (0, 3),
        (1, 1), (1, 2),
        (2, 1), (2, 2),
        (3, 0), (3, 3),
    ]

    all_44 = [(r, c) for r in range(4) for c in range(4)]

    def complement(coords):
        s = set(coords)
        return [(r, c) for (r, c) in all_44 if (r, c) not in s]

    non_full = complement(FULL)
    non_missing = complement(MISSING)
    non_diagonal = complement(DIAGONAL)
    non_corner_center = complement(CORNER_CENTER)

    glyph_coords = {
        1: MISSING, 3: MISSING,
        2: FULL, 4: FULL,
        5: DIAGONAL, 6: DIAGONAL,
        7: CORNER_CENTER, 8: CORNER_CENTER,
    }

    non_glyph_coords = {
        1: non_missing, 3: non_missing,
        2: non_full, 4: non_full,
        5: non_diagonal, 6: non_diagonal,
        7: non_corner_center, 8: non_corner_center,
    }

    paired_color = {
        1: 3, 3: 1,
        2: 4, 4: 2,
        5: 6, 6: 5,
        7: 8, 8: 7,
    }

    # Mark clean 4x4 template glyphs so their pixels are not treated as seeds.
    is_template = np.zeros((H, W), dtype=bool)
    glyph_colors = (1, 2, 3, 4, 5, 6, 7, 8)

    for r in range(H - 3):
        for c in range(W - 3):
            for color in glyph_colors:
                ok = True

                # All glyph positions must be this color.
                for dr, dc in glyph_coords[color]:
                    if grid[r + dr, c + dc] != color:
                        ok = False
                        break

                if not ok:
                    continue

                # Non-glyph positions must not also be this color,
                # so solid blocks or noisy blobs are not partially treated as templates.
                for dr, dc in non_glyph_coords[color]:
                    if grid[r + dr, c + dc] == color:
                        ok = False
                        break

                if ok:
                    for dr, dc in glyph_coords[color]:
                        is_template[r + dr, c + dc] = True
                    break

    # Find isolated non-template pixels: these are the seeds.
    candidates = defaultdict(list)
    visited = np.zeros((H, W), dtype=bool)
    dirs4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for r in range(H):
        for c in range(W):
            if grid[r, c] == 0 or is_template[r, c] or visited[r, c]:
                continue

            color = grid[r, c]
            stack = [(r, c)]
            visited[r, c] = True
            size = 0

            while stack:
                i, j = stack.pop()
                size += 1

                for di, dj in dirs4:
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

            # A seed must be a single pixel and must have room for a 4x4 glyph.
            if size == 1 and color in paired_color and r + 3 < H and c + 3 < W:
                candidates[color].append((r, c))

    # Select non-overlapping seeds for each color.
    # Process from bottom-right to top-left so the lower-right seed wins conflicts.
    selected = []
    for color in sorted(candidates):
        positions = candidates[color]
        positions.sort(key=lambda p: (p[0], p[1]), reverse=True)

        kept = []
        for r, c in positions:
            blocked = False
            for kr, kc in kept:
                if abs(r - kr) < 4 and abs(c - kc) < 4:
                    blocked = True
                    break
            if not blocked:
                kept.append((r, c))

        for r, c in kept:
            selected.append((r, c, color))

    # Draw selected glyphs on a clean output grid.
    for r, c, color in selected:
        out_color = paired_color[color]
        for dr, dc in glyph_coords[color]:
            out[r + dr, c + dc] = out_color

    return out