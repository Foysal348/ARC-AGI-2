import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Move each color class to the top or bottom edge while preserving its columns.

    Rule:
    1. Background is the most frequent color.
    2. Each non-background color forms one object, even if its pixels are disconnected.
    3. Initial side:
         - odd color  -> top
         - even color -> bottom
    4. If two objects overlap horizontally by at least 2 columns, the wider object
       is preferred on top.
    5. If same-side horizontal overlap remains, the narrower object is moved to
       the other side.
    """
    if grid.size == 0:
        return grid

    h, w = grid.shape

    flat = grid.ravel()
    vals, counts = np.unique(flat, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    out = np.full_like(grid, bg)

    # Collect all cells for each non-background color.
    colors = [int(v) for v in vals if v != bg]

    objects = []
    for color in colors:
        rows, cols = np.where(grid == color)
        if len(rows) == 0:
            continue

        rmin, rmax = int(rows.min()), int(rows.max())
        cmin, cmax = int(cols.min()), int(cols.max())
        height = rmax - rmin + 1
        width = cmax - cmin + 1

        # Preserve the exact pattern relative to the object's bounding box.
        pattern = np.zeros((height, width), dtype=bool)
        for r, c in zip(rows, cols):
            pattern[r - rmin, c - cmin] = True

        objects.append({
            "color": color,
            "rmin": rmin,
            "rmax": rmax,
            "cmin": cmin,
            "cmax": cmax,
            "height": height,
            "width": width,
            "pattern": pattern,
        })

    if not objects:
        return out

    # Default assignment by color parity.
    for obj in objects:
        obj["side"] = 0 if obj["color"] % 2 == 1 else 1  # 0=top, 1=bottom

    def overlap_width(a, b):
        return max(0, min(a["cmax"], b["cmax"]) - max(a["cmin"], b["cmin"]) + 1)

    # Resolve substantial horizontal overlaps.
    # Prefer the wider object on top when two objects overlap by >= 2 columns.
    n = len(objects)
    changed = True
    guard = 0
    while changed and guard < 100:
        changed = False
        guard += 1

        for i in range(n):
            for j in range(i + 1, n):
                a, b = objects[i], objects[j]
                ov = overlap_width(a, b)
                if ov < 2:
                    continue

                # The wider object should be on top.
                if a["width"] > b["width"]:
                    preferred_top = a
                    preferred_bottom = b
                elif b["width"] > a["width"]:
                    preferred_top = b
                    preferred_bottom = a
                else:
                    # Tie-break by color parity / smaller color to top if needed.
                    if (a["color"] % 2 == 1 and b["color"] % 2 == 0):
                        preferred_top, preferred_bottom = a, b
                    elif (b["color"] % 2 == 1 and a["color"] % 2 == 0):
                        preferred_top, preferred_bottom = b, a
                    else:
                        preferred_top, preferred_bottom = (a, b) if a["color"] < b["color"] else (b, a)

                # Enforce top/bottom preference if possible.
                if preferred_top["side"] != 0 and preferred_bottom["side"] != 1:
                    preferred_top["side"] = 0
                    preferred_bottom["side"] = 1
                    changed = True
                elif preferred_top["side"] != 0 and preferred_bottom["side"] == 1:
                    # Only top needs to change, but avoid forcing if it creates
                    # a same-side overlap with another object of greater width.
                    preferred_top["side"] = 0
                    changed = True
                elif preferred_top["side"] == 0 and preferred_bottom["side"] != 1:
                    preferred_bottom["side"] = 1
                    changed = True

    # Final pass: remove same-side horizontal overlaps.
    # If two objects on the same side overlap substantially, move the narrower one
    # to the other side.
    changed = True
    guard = 0
    while changed and guard < 100:
        changed = False
        guard += 1

        for i in range(n):
            for j in range(i + 1, n):
                a, b = objects[i], objects[j]
                if a["side"] != b["side"]:
                    continue
                if overlap_width(a, b) < 2:
                    continue

                if a["width"] < b["width"]:
                    a["side"] ^= 1
                elif b["width"] < a["width"]:
                    b["side"] ^= 1
                else:
                    # Tie: move the later color to the other side.
                    if a["color"] < b["color"]:
                        b["side"] ^= 1
                    else:
                        a["side"] ^= 1
                changed = True

    # Place objects.
    for obj in objects:
        cmin = obj["cmin"]
        pattern = obj["pattern"]
        height = obj["height"]

        if obj["side"] == 0:
            rstart = 0
        else:
            rstart = h - height

        for dr, row in enumerate(pattern):
            r = rstart + dr
            if not (0 <= r < h):
                continue
            for dc, val in enumerate(row):
                if not val:
                    continue
                c = cmin + dc
                if 0 <= c < w:
                    out[r, c] = obj["color"]

    return out