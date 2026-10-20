import numpy as np
from typing import List, Tuple

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform an ARC grid by extracting the four intended square "frame"
    interiors, XOR-ing their occupied pixels, and drawing the result with
    color 2.  If the only odd-parity color at a cell is 6, color it 6
    instead of 2.

    The intended frames are closed 1-pixel-thick rectangular borders whose:
      * size is small and consistent across the frame set,
      * interior does not contain the border color,
      * interiors are mostly "quiet" (few background-like colors), and
      * a consistent square interior size is shared by at least two frames.

    This robustly avoids large background rectangles formed by the repeating
    filler pattern.
    """
    grid = np.asarray(grid)
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape

    # Colors that are not used for the final generic output.
    FINAL_COLOR = 2
    SPECIAL_COLOR = 6

    def uniform_border(r0: int, c0: int, r1: int, c1: int, color: int) -> bool:
        """All pixels on the rectangle border are exactly `color`."""
        if not np.all(grid[r0, c0:c1 + 1] == color):
            return False
        if not np.all(grid[r1, c0:c1 + 1] == color):
            return False
        if not np.all(grid[r0:r1 + 1, c0] == color):
            return False
        if not np.all(grid[r0:r1 + 1, c1] == color):
            return False
        return True

    def interior_penalty(r0: int, c0: int, r1: int, c1: int, border_color: int) -> int:
        """
        Lower is better.
        Penalize interiors that look like ordinary background rather than the
        small puzzle panels:
          * large interior,
          * too many distinct colors,
          * border color inside,
          * too many zeros (large empty false frames).
        """
        if r1 - r0 < 2 or c1 - c0 < 2:
            return 10**9

        interior = grid[r0 + 1:r1, c0 + 1:c1]
        ih, iw = interior.shape
        area = ih * iw

        if area == 0:
            return 10**9

        # Intended interiors in the visible tasks are small (roughly <= 9x9).
        size_pen = max(0, max(ih, iw) - 9) * 100

        # Interior should not contain the border color.
        if np.any(interior == border_color):
            size_pen += 10000

        uniq = np.unique(interior)
        distinct = len(uniq)

        # Usually 2-5 colors, including 0.
        if distinct <= 2:
            distinct_pen = 0
        elif distinct <= 6:
            distinct_pen = 0
        else:
            distinct_pen = (distinct - 6) * 100

        zero_count = int(np.sum(interior == 0))
        zero_frac = zero_count / area
        if zero_frac < 0.15:
            zero_pen = 100
        else:
            zero_pen = 0

        return size_pen + distinct_pen + zero_pen

    # Candidate closed frames.
    # Keep all plausible small square/rectangular frames, then score.
    candidates = []
    colors = np.unique(grid)
    # Frames are not usually very large in these tasks.
    max_side = min(18, max(h, w))

    for color in colors:
        if color == 0:
            continue

        for r0 in range(h):
            for c0 in range(w):
                if grid[r0, c0] != color:
                    continue

                # Search for the right edge along the top border.
                for c1 in range(c0 + 2, min(w, c0 + max_side + 1)):
                    if grid[r0, c1] != color:
                        continue
                    if not np.all(grid[r0, c0:c1 + 1] == color):
                        continue

                    # Search for the bottom edge.
                    for r1 in range(r0 + 2, min(h, r0 + max_side + 1)):
                        if grid[r1, c0] != color or grid[r1, c1] != color:
                            continue

                        if not uniform_border(r0, c0, r1, c1, color):
                            continue

                        penalty = interior_penalty(r0, c0, r1, c1, color)
                        if penalty >= 10000:
                            continue

                        candidates.append((penalty, r0, c0, r1, c1, int(color)))

    if not candidates:
        return np.zeros_like(grid)

    # Prefer low-penalty frames.
    candidates.sort(key=lambda x: x[0])

    # Choose the best consistent interior size.
    # Count how many of the best frames have each interior shape.
    shape_counts = {}
    shape_frames = {}
    for pen, r0, c0, r1, c1, color in candidates:
        shape = (r1 - r0 - 1, c1 - c0 - 1)
        # Only consider reasonable small interior shapes.
        if shape[0] < 1 or shape[1] < 1:
            continue
        if max(shape) > 9:
            continue
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
        shape_frames.setdefault(shape, []).append((pen, r0, c0, r1, c1, color))

    if not shape_counts:
        return np.zeros_like(grid)

    # Use the shape that occurs most often; break ties by smaller total penalty.
    def shape_key(shape):
        frames = shape_frames[shape]
        total_pen = sum(f[0] for f in frames)
        return (-len(frames), total_pen, shape)

    target_shape = min(shape_counts.keys(), key=shape_key)

    # Take the lowest-penalty frames of that shape.  The visible tasks have
    # four, but use up to four to avoid noisy duplicates.
    frames = shape_frames[target_shape]
    frames.sort(key=lambda x: x[0])
    frames = frames[:4]

    if not frames:
        return np.zeros_like(grid)

    # If there are fewer than two frames, still attempt to output the XOR
    # of what was found.
    interiors = []
    for _, r0, c0, r1, c1, _ in frames:
        interiors.append(grid[r0 + 1:r1, c0 + 1:c1].copy())

    target_h, target_w = target_shape

    # Per-cell parity by color.
    odd_counts = np.zeros((target_h, target_w), dtype=np.int32)
    odd_color = np.zeros((target_h, target_w), dtype=np.int32)

    for inter in interiors:
        ih, iw = inter.shape
        if ih != target_h or iw != target_w:
            # Should not happen because frames were grouped by shape.
            tmp = np.zeros((target_h, target_w), dtype=grid.dtype)
            tmp[:ih, :iw] = inter[:target_h, :target_w]
            inter = tmp
        else:
            inter = inter.copy()

        flat = inter.reshape(-1)
        colors_here = np.unique(flat)
        for col in colors_here:
            if col == 0:
                continue
            mask = (inter == col)
            odd_counts ^= mask.astype(np.int32)
            odd_color = np.where(mask, col, odd_color)

    occupied = odd_counts % 2 == 1

    out = np.zeros((target_h, target_w), dtype=grid.dtype)

    # Generic occupied pixels become color 2.
    out[occupied] = FINAL_COLOR

    # Preserve color 6 when it is the only odd-parity color at that cell.
    special = occupied & (odd_color == SPECIAL_COLOR)
    out[special] = SPECIAL_COLOR

    return out