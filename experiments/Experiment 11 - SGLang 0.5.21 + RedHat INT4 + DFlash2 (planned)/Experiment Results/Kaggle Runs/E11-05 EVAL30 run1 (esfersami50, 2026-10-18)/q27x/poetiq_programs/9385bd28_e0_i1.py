import numpy as np
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform grid according to the ARC rule inferred from the examples.

    Rule summary
    ------------
    The input contains:
      - a "legend"/palette made of a vertical stack of horizontal 2-cell pairs.
        Each row gives a mapping:  target_color -> fill_color.
      - one or more pairs of horizontal dominoes of the same color.
        These dominoes are usually the top/bottom edges of L-shaped pieces.

    For each legend row (a, b):
      - find all horizontal dominoes of color a outside the legend block.
      - if there are exactly two such dominoes, fill the axis-aligned
        bounding box of the two dominoes with color b.
      - fill operations are applied in descending order of target color,
        so lower target colors overwrite higher ones where boxes overlap.
      - if b == 0, paint with the grid's background color instead.
        (0 behaves as "clear to background" in these examples.)

    The legend block is located by finding a 2-column region containing
    several horizontally adjacent pairs that are not target dominoes.
    """

    if grid.size == 0:
        return grid

    H, W = grid.shape
    out = grid.copy()

    # Determine background: most common color, tie broken by smaller color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    # If multiple with same count, choose smallest.
    maxc = counts.max()
    candidates = vals[counts == maxc]
    if len(candidates) > 1:
        bg = int(candidates.min())

    # ------------------------------------------------------------------
    # Detect legend block.
    #
    # A legend row is a horizontal adjacent pair (x, x+1) in the same row
    # where the two cells are not both part of the same "target domino"
    # structure that will be painted. In the examples, legend blocks are
    # narrow 2-column vertical stacks, often near the left/bottom.
    #
    # We search for the best 2-column window that contains at least 2 rows
    # of non-background horizontal pairs and is relatively compact.
    # ------------------------------------------------------------------

    def is_nonzero_or_nonbg(c):
        return c != 0 or c != bg  # always true, kept for clarity

    # Candidate legend columns: find columns c where many rows have
    # non-background at c and c+1.
    best = None  # (score, r0, r1, c, score_details)
    for c in range(W - 1):
        # rows where both cells are non-background
        rows = [r for r in range(H) if grid[r, c] != bg and grid[r, c + 1] != bg]
        if len(rows) < 2:
            continue

        # Prefer compact vertical runs.
        # Score: number of rows, compactness, leftness, bottomness.
        # We will pick a run of rows in this column window.
        for i in range(len(rows)):
            for j in range(i, len(rows)):
                r0, r1 = rows[i], rows[j]
                # require contiguous row coverage in the window
                if not all(
                    grid[r, c] != bg and grid[r, c + 1] != bg
                    for r in range(r0, r1 + 1)
                ):
                    continue
                height = r1 - r0 + 1
                if height < 2:
                    continue
                # Score: maximize height, compactness, prefer left & lower
                score = (height, -r0, c, H - r1)
                if best is None or score > best[0]:
                    best = (score, r0, r1, c)

    if best is not None:
        legend_r0, legend_r1, legend_c = best[1], best[2], best[3]
        legend_c1 = legend_c + 1
        # mappings: target -> fill
        mapping = {}
        for r in range(legend_r0, legend_r1 + 1):
            a = int(grid[r, legend_c])
            b = int(grid[r, legend_c1])
            # Keep last if duplicates
            mapping[a] = b
    else:
        # Fallback: no clear legend. Use identity mapping for colors that
        # have exactly two horizontal dominoes.
        mapping = {}

    # ------------------------------------------------------------------
    # Find horizontal dominoes of each color.
    # A domino is two adjacent cells in the same row with same non-bg color.
    # We use domino top-left coordinates as representatives.
    # Exclude dominoes inside the legend block.
    # ------------------------------------------------------------------

    dominoes_by_color = defaultdict(list)
    for r in range(H - 0):  # rows only
        for c in range(W - 1):
            v = int(grid[r, c])
            if v == bg:
                continue
            if v != int(grid[r, c + 1]):
                continue

            # Exclude legend cells.
            if (legend_r0 <= r <= legend_r1 and
                legend_c <= c < legend_c1 + 1):
                # If the domino starts or overlaps the legend block, skip it.
                if c == legend_c and c + 1 == legend_c1:
                    continue
                if legend_c <= c <= legend_c1:
                    continue

            dominoes_by_color[v].append((r, c))

    # If no legend mapping was found, make an identity mapping for colors
    # that have exactly two dominoes. This helps handle variations.
    if not mapping:
        for color, doms in dominoes_by_color.items():
            if len(doms) == 2:
                mapping[color] = color

    # ------------------------------------------------------------------
    # Build fills. For each mapping entry with exactly two target dominoes,
    # compute bbox. Apply in descending target color order.
    # ------------------------------------------------------------------

    fills = []
    for target, fill in mapping.items():
        doms = dominoes_by_color.get(target, [])
        if len(doms) != 2:
            continue
        r0a, c0a = doms[0]
        r1a, c1a = (doms[0][0], doms[0][1] + 1)
        r0b, c0b = doms[1]
        r1b, c1b = (doms[1][0], doms[1][1] + 1)

        r0 = min(r0a, r0b)
        r1 = max(r1a, r1b)
        c0 = min(c0a, c0b)
        c1 = max(c1a, c1b)

        actual_fill = bg if fill == 0 else fill
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        fills.append((target, area, r0, r1, c0, c1, actual_fill))

    # Apply in descending target color, then descending area for stability.
    fills.sort(key=lambda x: (x[0], x[1]), reverse=True)

    for target, area, r0, r1, c0, c1, fill_color in fills:
        # Paint the rectangle. Preserve original non-background pixels
        # that are not background, unless they are part of the target color
        # or already painted by a later (lower target) fill.
        # Simpler: assign fill_color to all cells in rectangle, except
        # keep original non-background cells that are not background and
        # not target color? Examples show smaller fills overwrite larger.
        # We do direct overwrite, then preserve original non-bg cells of
        # colors not in current fill set? To match examples, direct
        # overwrite by ordered fills is sufficient.
        out[r0:r1 + 1, c0:c1 + 1] = fill_color

        # Restore target domino cells (and any original non-bg pixels that
        # should remain as objects) for this target color inside its box.
        # This keeps source shapes visible when fill color differs.
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                orig = int(grid[r, c])
                if orig != bg and orig != fill_color:
                    # Preserve original foreground unless it is target color
                    # and the fill is meant to cover it? In examples,
                    # target shapes are preserved.
                    if orig == target:
                        out[r, c] = orig
                    else:
                        # Preserve other pre-existing foreground pixels that
                        # are not part of a later smaller fill.
                        # We keep them if they were not background.
                        # This matches preservation of nested objects.
                        # However, larger fills later may overwrite them
                        # if that later fill has lower target color.
                        # Since we process descending target, current fill
                        # is larger/higher priority; preserve existing.
                        out[r, c] = orig

    return out