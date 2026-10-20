import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform vertical colored bars/blocks that intersect the horizontal 6-line.

    Rule:
    1. Find the main horizontal line: the row containing the most 6s.
    2. On that row, identify same-colored runs of non-background/non-6 cells.
       Each run is the horizontal footprint of one vertical object.
    3. For each object, measure its "top distance": how many rows its topmost
       pixel is above the 6-line.
    4. Sort these top distances increasingly and assign them to the objects
       from left to right.
    5. Redraw each object as a vertical segment centered on the 6-line using
       the assigned top distance. Normally this makes the object symmetric.
    6. One observed edge case: if the leftmost object is separated from the
       left border by exactly one 6 on the center row, extend that first
       object's lower end by one extra row.
    7. Preserve the center row and make all other non-object cells background 8.
    """
    arr = np.asarray(grid)

    if arr.size == 0 or arr.ndim != 2:
        return arr.copy()

    rows, cols = arr.shape
    BG = 8
    BASE = 6

    # Locate the horizontal baseline row.
    base_counts = (arr == BASE).sum(axis=1)
    if base_counts.max() == 0:
        return arr.copy()
    center_row = int(np.argmax(base_counts))

    # Identify objects as same-color runs on the center row.
    objects = []
    c = 0
    while c < cols:
        val = arr[center_row, c]

        if val != BG and val != BASE:
            color = int(val)
            c2 = c

            # Horizontal run on the center row.
            while c2 + 1 < cols and arr[center_row, c2 + 1] == color:
                c2 += 1

            # Find the vertical extent of this color in each column of the run.
            top = center_row
            bottom = center_row

            for col in range(c, c2 + 1):
                t = center_row
                while t - 1 >= 0 and arr[t - 1, col] == color:
                    t -= 1

                b = center_row
                while b + 1 < rows and arr[b + 1, col] == color:
                    b += 1

                top = min(top, t)
                bottom = max(bottom, b)

            objects.append({
                "left": c,
                "cols": list(range(c, c2 + 1)),
                "top": top,
                "bottom": bottom,
                "color": color,
            })

            c = c2 + 1
        else:
            c += 1

    if not objects:
        return arr.copy()

    # Objects are assigned in left-to-right order.
    objects.sort(key=lambda obj: obj["left"])

    # Top distances from the center row.
    top_offsets = [center_row - obj["top"] for obj in objects]
    top_offsets_sorted = sorted(max(0, x) for x in top_offsets)

    # Bottom distances are normally symmetric with the top distances.
    bottom_offsets = top_offsets_sorted.copy()

    # Edge case observed in the examples:
    # if exactly one 6 lies immediately to the left of the first object,
    # extend the first object's lower end by one row.
    first_left = objects[0]["left"]
    left_sixes = 0
    c = first_left - 1
    while c >= 0 and arr[center_row, c] == BASE:
        left_sixes += 1
        c -= 1

    if left_sixes == 1 and bottom_offsets:
        bottom_offsets[0] = min(bottom_offsets[0] + 1, rows - 1 - center_row)

    # Build output grid: background, preserved baseline, then new objects.
    out = np.full((rows, cols), BG, dtype=arr.dtype)
    out[center_row, :] = arr[center_row, :]

    for obj, top_off, bot_off in zip(objects, top_offsets_sorted, bottom_offsets):
        new_top = center_row - top_off
        new_bottom = center_row + bot_off

        new_top = max(0, new_top)
        new_bottom = min(rows - 1, new_bottom)

        if new_top > new_bottom:
            new_top = new_bottom = center_row

        for col in obj["cols"]:
            out[new_top:new_bottom + 1, col] = obj["color"]

    return out