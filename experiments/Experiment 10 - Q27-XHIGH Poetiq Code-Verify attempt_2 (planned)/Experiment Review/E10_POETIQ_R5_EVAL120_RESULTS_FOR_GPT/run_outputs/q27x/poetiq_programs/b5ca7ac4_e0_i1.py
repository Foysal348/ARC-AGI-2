import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Move framed boxes horizontally.

    Observed rule:
      - Boxes with border color 8 are packed to the left.
      - Boxes with border color 2 are packed to the right.
      - Boxes keep their vertical position, size, border color, and fill color.
      - Left-packed boxes are processed in increasing original column order.
      - Right-packed boxes are processed in decreasing original column order.
      - Each box is placed in the leftmost/rightmost column that does not
        overlap an already placed box of the same border color.

    The examples use 5x5 framed boxes; the code first detects 5x5 boxes and
    falls back to detecting arbitrary square frames if no 5x5 boxes exist.
    """
    grid = np.asarray(grid)

    if grid.size == 0 or grid.ndim != 2:
        return grid.copy()

    H, W = grid.shape

    # Background is the most frequent color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    def detect_5x5_boxes():
        """Detect 5x5 boxes with a uniform border and uniform interior."""
        boxes = []
        seen = set()

        if H < 5 or W < 5:
            return boxes

        for r in range(H - 4):
            for c in range(W - 4):
                b = grid[r, c]
                if b == bg:
                    continue

                # Quick corner check.
                if not (
                    grid[r, c + 4] == b
                    and grid[r + 4, c] == b
                    and grid[r + 4, c + 4] == b
                ):
                    continue

                sub = grid[r:r + 5, c:c + 5]
                border = np.concatenate((
                    sub[0, :],
                    sub[4, :],
                    sub[:, 0],
                    sub[:, -1]
                ))

                if not np.all(border == b):
                    continue

                interior = sub[1:4, 1:4]
                f = interior[0, 0]

                if f == b or not np.all(interior == f):
                    continue

                key = (r, c, 5, 5, int(b), int(f))
                if key not in seen:
                    seen.add(key)
                    boxes.append(key)

        return boxes

    def detect_square_frames():
        """Fallback: detect square frames of any size >= 3."""
        boxes = []
        seen = set()
        max_s = min(H, W)

        for s in range(3, max_s + 1):
            for r in range(H - s + 1):
                for c in range(W - s + 1):
                    b = grid[r, c]
                    if b == bg:
                        continue

                    # Quick corner check.
                    if not (
                        grid[r, c + s - 1] == b
                        and grid[r + s - 1, c] == b
                        and grid[r + s - 1, c + s - 1] == b
                    ):
                        continue

                    if not (
                        np.all(grid[r, c:c + s] == b)
                        and np.all(grid[r + s - 1, c:c + s] == b)
                        and np.all(grid[r:r + s, c] == b)
                        and np.all(grid[r:r + s, c + s - 1] == b)
                    ):
                        continue

                    interior = grid[r + 1:r + s - 1, c + 1:c + s - 1]
                    if interior.size == 0:
                        continue

                    f = interior[0, 0]
                    if f == b or not np.all(interior == f):
                        continue

                    key = (r, c, s, s, int(b), int(f))
                    if key not in seen:
                        seen.add(key)
                        boxes.append(key)

        return boxes

    # Preferred detector for the given task.
    boxes = detect_5x5_boxes()
    if not boxes:
        boxes = detect_square_frames()

    if not boxes:
        return grid.copy()

    # Determine which boxes move left. In the examples, color 8 moves left
    # and color 2 moves right. Fallbacks handle other palettes reasonably.
    border_colors = set(b[4] for b in boxes)
    has_8 = 8 in border_colors
    has_2 = 2 in border_colors

    if has_8:
        def is_left_box(box):
            return box[4] == 8
    else:
        def is_left_box(box):
            # If color 2 is present, treat it as the right-moving color.
            return not (has_2 and box[4] == 2)

    left_boxes = [b for b in boxes if is_left_box(b)]
    right_boxes = [b for b in boxes if not is_left_box(b)]

    def overlaps(a, b):
        """Inclusive rectangle overlap test."""
        r1, l1, h1, w1 = a[0], a[1], a[2], a[3]
        r2, l2, h2, w2 = b[0], b[1], b[2], b[3]
        return (
            r1 < r2 + h2
            and r2 < r1 + h1
            and l1 < l2 + w2
            and l2 < l1 + w1
        )

    final_boxes = []

    # Left-moving boxes: process closer-to-left boxes first, then pack left.
    left_boxes.sort(key=lambda b: (b[1], b[0]))
    placed_left = []

    for b in left_boxes:
        top, orig_left, h, w, bc, fc = b
        chosen = None
        max_left = W - w

        if max_left >= 0:
            for l in range(0, max_left + 1):
                candidate = (top, l, h, w, bc, fc)
                if not any(overlaps(candidate, p) for p in placed_left):
                    chosen = l
                    break

        if chosen is None:
            chosen = orig_left

        new_box = (top, chosen, h, w, bc, fc)
        final_boxes.append(new_box)
        placed_left.append(new_box)

    # Right-moving boxes: process closer-to-right boxes first, then pack right.
    right_boxes.sort(key=lambda b: (-b[1], b[0]))
    placed_right = []

    for b in right_boxes:
        top, orig_left, h, w, bc, fc = b
        chosen = None
        max_left = W - w

        if max_left >= 0:
            for l in range(max_left, -1, -1):
                candidate = (top, l, h, w, bc, fc)
                if not any(overlaps(candidate, p) for p in placed_right):
                    chosen = l
                    break

        if chosen is None:
            chosen = orig_left

        new_box = (top, chosen, h, w, bc, fc)
        final_boxes.append(new_box)
        placed_right.append(new_box)

    # Start from the original grid, erase the original box locations,
    # then draw the relocated boxes.
    out = grid.copy()

    for top, left, h, w, _, _ in boxes:
        out[top:top + h, left:left + w] = bg

    for top, left, h, w, bc, fc in final_boxes:
        # Clamp just in case.
        if top < 0:
            top = 0
        if left < 0:
            left = 0
        if top + h > H:
            h = H - top
        if left + w > W:
            w = W - left

        if h <= 0 or w <= 0:
            continue

        # Interior fill.
        if h > 2 and w > 2:
            out[top + 1:top + h - 1, left + 1:left + w - 1] = fc

        # Border.
        out[top, left:left + w] = bc
        out[top + h - 1, left:left + w] = bc
        out[top:top + h, left] = bc
        out[top:top + h, left + w - 1] = bc

    return out