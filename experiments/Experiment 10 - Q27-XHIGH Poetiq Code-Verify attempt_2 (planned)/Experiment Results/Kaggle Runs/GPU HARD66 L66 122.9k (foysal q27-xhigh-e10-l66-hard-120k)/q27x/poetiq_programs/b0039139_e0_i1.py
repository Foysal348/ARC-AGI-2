import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a panel-based ARC grid.

    The grid is divided by full lines of 1 into four panels:
      1) a template shape,
      2) a set of connected components that determines the repetition count,
      3) the foreground color,
      4) the background color.

    The template is cropped to its non-zero bounding box. Its non-zero cells
    become the foreground color, and its zero cells become the background color.
    The colored template is then repeated as many times as there are 4-connected
    components in the second panel. The copies are arranged horizontally if the
    input panels are side-by-side, or vertically if the panels are stacked.
    """
    grid = np.asarray(grid, dtype=int)

    def group_consecutive(indices):
        """Group consecutive indices into (start, end) tuples."""
        groups = []
        if not indices:
            return groups
        start = prev = indices[0]
        for x in indices[1:]:
            if x == prev + 1:
                prev = x
            else:
                groups.append((start, prev))
                start = prev = x
        groups.append((start, prev))
        return groups

    def make_blocks(sep_indices, axis):
        """
        Split the grid along axis using separator lines.
        axis=1: vertical separator columns -> blocks left-to-right.
        axis=0: horizontal separator rows -> blocks top-to-bottom.
        """
        h, w = grid.shape
        groups = group_consecutive(sep_indices)
        blocks = []

        if axis == 1:
            bounds = [0]
            for s, e in groups:
                bounds.append(s)
                bounds.append(e + 1)
            bounds.append(w)
            for a, b in zip(bounds[0::2], bounds[1::2]):
                if b > a:
                    blocks.append(grid[:, a:b])
        else:
            bounds = [0]
            for s, e in groups:
                bounds.append(s)
                bounds.append(e + 1)
            bounds.append(h)
            for a, b in zip(bounds[0::2], bounds[1::2]):
                if b > a:
                    blocks.append(grid[a:b, :])

        return blocks

    # Detect full separator rows/columns of color 1.
    row_seps = [r for r in range(grid.shape[0]) if np.all(grid[r] == 1)]
    col_seps = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == 1)]

    row_groups = group_consecutive(row_seps)
    col_groups = group_consecutive(col_seps)

    orientation = None
    blocks = None

    if col_groups and not row_groups:
        blocks = make_blocks(col_seps, axis=1)
        orientation = "horizontal"
    elif row_groups and not col_groups:
        blocks = make_blocks(row_seps, axis=0)
        orientation = "vertical"
    elif col_groups and row_groups:
        # Fallback: choose the more likely separator axis.
        if len(col_groups) >= len(row_groups):
            blocks = make_blocks(col_seps, axis=1)
            orientation = "horizontal"
        else:
            blocks = make_blocks(row_seps, axis=0)
            orientation = "vertical"

    if not blocks or len(blocks) < 4:
        return np.zeros((1, 1), dtype=int)

    if len(blocks) > 4:
        blocks = blocks[:4]

    template_block = blocks[0]
    selector_block = blocks[1]
    foreground_block = blocks[2]
    background_block = blocks[3]

    def pattern_mask(block):
        """Mask of relevant non-background, non-separator pixels."""
        return (block != 0) & (block != 1)

    def crop_to_content(mask):
        """Crop a binary mask to its bounding box."""
        if not mask.any():
            return np.zeros((1, 1), dtype=int)

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        r0 = int(np.argmax(rows))
        r1 = int(len(rows) - np.argmax(rows[::-1])) - 1
        c0 = int(np.argmax(cols))
        c1 = int(len(cols) - np.argmax(cols[::-1])) - 1

        return mask[r0:r1 + 1, c0:c1 + 1].astype(int)

    def count_components(mask):
        """Count 4-connected components in a binary mask."""
        if not mask.any():
            return 0

        h, w = mask.shape
        seen = np.zeros((h, w), dtype=bool)
        components = 0

        for r in range(h):
            for c in range(w):
                if not mask[r, c] or seen[r, c]:
                    continue

                components += 1
                stack = [(r, c)]
                seen[r, c] = True

                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if mask[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True
                                stack.append((nr, nc))

        return components

    def dominant_solid_color(block):
        """
        Get the solid color of a panel.
        Prefer non-zero, non-1 values, then non-1 values, then any value.
        """
        if block.size == 0:
            return 0

        values, counts = np.unique(block, return_counts=True)

        for candidate_mask in (
            (values != 0) & (values != 1),
            values != 1,
            np.ones(values.shape, dtype=bool)
        ):
            if np.any(candidate_mask):
                values = values[candidate_mask]
                counts = counts[candidate_mask]
                return int(values[np.argmax(counts)])

        return 0

    template = crop_to_content(pattern_mask(template_block))
    repetitions = count_components(pattern_mask(selector_block))

    foreground = dominant_solid_color(foreground_block)
    background = dominant_solid_color(background_block)

    if repetitions == 0:
        return np.full((1, 1), background, dtype=int)

    template = np.where(template == 1, foreground, background).astype(int)

    if orientation == "horizontal":
        pieces = []
        separator = np.full((template.shape[0], 1), background, dtype=int)
        for i in range(repetitions):
            if i > 0:
                pieces.append(separator)
            pieces.append(template)
        return np.hstack(pieces)
    else:
        pieces = []
        separator = np.full((1, template.shape[1]), background, dtype=int)
        for i in range(repetitions):
            if i > 0:
                pieces.append(separator)
            pieces.append(template)
        return np.vstack(pieces)