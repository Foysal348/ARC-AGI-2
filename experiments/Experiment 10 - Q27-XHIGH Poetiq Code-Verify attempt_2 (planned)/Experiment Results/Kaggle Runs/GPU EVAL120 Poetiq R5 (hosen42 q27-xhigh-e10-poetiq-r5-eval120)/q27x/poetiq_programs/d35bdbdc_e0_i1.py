import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Detect frame-like objects, link them by their center/border colors,
    then repeatedly erase unpaired frames while keeping and updating
    frames that point to them.
    """
    g = np.array(grid, dtype=int, copy=True)

    if g.ndim != 2:
        return g

    H, W = g.shape
    if H == 0 or W == 0:
        return g

    objects = []

    # Detect frames:
    # A frame center is a non-zero cell whose in-bounds non-zero neighbors
    # are all the same other color. The surrounding non-zero cells are the border.
    for r in range(H):
        for c in range(W):
            C = int(g[r, c])
            if C == 0:
                continue

            neigh = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        v = int(g[nr, nc])
                        if v != 0:
                            neigh.append((nr, nc, v))

            if not neigh:
                continue

            vals = {v for _, _, v in neigh}
            if len(vals) != 1:
                continue

            B = int(next(iter(vals)))
            if B == C:
                continue

            border = [(nr, nc) for nr, nc, _ in neigh]

            # Require a reasonably closed frame, not just a stray line.
            if len(border) < 3:
                continue

            in_bound_cardinals = 0
            present_cardinals = 0
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    in_bound_cardinals += 1
                    if int(g[nr, nc]) == B:
                        present_cardinals += 1

            if in_bound_cardinals >= 3:
                if present_cardinals < 3:
                    continue
            else:
                if present_cardinals < 2:
                    continue

            # The border cells should form a small standalone component,
            # avoiding large distractor shapes.
            sr, sc = border[0]
            comp = {(sr, sc)}
            stack = [(sr, sc)]
            while stack:
                cr, cc = stack.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < H
                            and 0 <= nc < W
                            and g[nr, nc] == B
                            and (nr, nc) not in comp
                        ):
                            comp.add((nr, nc))
                            stack.append((nr, nc))

            if len(comp) != len(border):
                continue

            objects.append({
                "r": r,
                "c": c,
                "B": B,
                "C": C,
                "border": border,
            })

    if not objects:
        return g

    # Map border color -> object indices.
    border_map = {}
    for i, obj in enumerate(objects):
        border_map.setdefault(obj["B"], []).append(i)

    def get_target(idx, active):
        """Return the active object whose border color equals objects[idx]['C']."""
        C = objects[idx]["C"]
        candidates = border_map.get(C, [])
        best = None
        for j in candidates:
            if j in active:
                if best is None or (
                    objects[j]["r"], objects[j]["c"]
                ) < (
                    objects[best]["r"], objects[best]["c"]
                ):
                    best = j
        return best

    active = set(range(len(objects)))
    decision = {}  # idx -> ("keep", new_center_color) or ("remove", None)

    while active:
        # Count active incoming links.
        indeg = {i: 0 for i in active}
        for i in active:
            t = get_target(i, active)
            if t is not None and t != i:
                indeg[t] += 1

        # Free frames: no active frame points to them.
        heads = [i for i in active if indeg[i] == 0]

        # Fallback for cyclic inputs: break the cycle deterministically.
        if not heads:
            heads = [min(active, key=lambda i: (objects[i]["r"], objects[i]["c"]))]

        remove = set()
        keep_new_center = {}

        for i in heads:
            t = get_target(i, active)
            if t is not None and t != i:
                # Keep this frame, use the target's original center color.
                keep_new_center[i] = objects[t]["C"]
                remove.add(t)
            else:
                # Unpaired terminal frame: erase it.
                remove.add(i)

        for i in heads:
            if i in remove:
                decision[i] = ("remove", None)
            elif i in keep_new_center:
                decision[i] = ("keep", keep_new_center[i])
            else:
                decision[i] = ("remove", None)

        for i in remove:
            decision[i] = ("remove", None)

        active.difference_update(heads)
        active.difference_update(remove)

    out = g.copy()

    # Erase removed frames.
    for idx, (status, _) in decision.items():
        if status == "remove":
            obj = objects[idx]
            out[obj["r"], obj["c"]] = 0
            for rr, cc in obj["border"]:
                out[rr, cc] = 0

    # Restore kept frames with their updated centers.
    for idx, obj in enumerate(objects):
        status, new_c = decision.get(idx, ("keep", obj["C"]))
        if status == "remove":
            continue

        if new_c is None:
            new_c = obj["C"]

        for rr, cc in obj["border"]:
            out[rr, cc] = obj["B"]
        out[obj["r"], obj["c"]] = int(new_c)

    return out