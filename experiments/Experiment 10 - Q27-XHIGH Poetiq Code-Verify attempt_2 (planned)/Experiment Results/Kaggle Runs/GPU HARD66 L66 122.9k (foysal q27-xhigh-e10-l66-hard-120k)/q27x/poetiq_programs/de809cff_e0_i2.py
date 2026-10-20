import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid by:
      - keeping the two dominant non-zero colors,
      - removing small color noise by a local 3x3 majority filter on non-zero cells,
      - finding zero holes with at least 3 orthogonal neighbors of one main color,
      - replacing the 3x3 neighborhood of each hole by the other main color,
        with the hole center becoming 8.
    """
    arr = np.array(grid, dtype=int, copy=True)

    if arr.ndim != 2:
        return arr

    h, w = arr.shape
    if h == 0 or w == 0:
        return arr

    # Main colors: two most frequent non-zero, non-8 values.
    main_mask = (arr != 0) & (arr != 8)
    if not np.any(main_mask):
        return arr

    colors, counts = np.unique(arr[main_mask], return_counts=True)
    order = np.argsort(-counts)
    main_colors = colors[order[:2]].tolist()
    main_set = set(main_colors)

    def other_color(color: int) -> int:
        for c in main_colors:
            if c != color:
                return int(c)
        return 0

    # ------------------------------------------------------------------
    # 1) Denoise non-zero cells using a 3x3 majority vote.
    #    Zeros are not filled here; they may become centers later.
    # ------------------------------------------------------------------
    base = arr.copy()

    for r in range(h):
        for c in range(w):
            v = int(arr[r, c])

            # Keep background and 8 as-is.
            if v == 0 or v == 8:
                continue

            counts_local = {0: 0}
            for col in main_colors:
                counts_local[col] = 0

            # Count only main colors and background. Other colors are noise.
            for nr in range(max(0, r - 1), min(h, r + 2)):
                for nc in range(max(0, c - 1), min(w, c + 2)):
                    x = int(arr[nr, nc])
                    if x == 0 or x == 8 or x not in main_set:
                        counts_local[0] += 1
                    else:
                        counts_local[x] += 1

            if v in main_set:
                # Keep the original color on ties.
                best_val, best_count = max(
                    counts_local.items(),
                    key=lambda kv: (kv[1], -kv[0])
                )
                if best_count > counts_local[v]:
                    base[r, c] = int(best_val)
                else:
                    base[r, c] = v
            else:
                # Non-main color: replace by local majority.
                # Prefer background on ties.
                best_val, _ = max(
                    counts_local.items(),
                    key=lambda kv: (
                        kv[1],
                        1 if kv[0] == 0 else 0,
                        -kv[0]
                    )
                )
                base[r, c] = int(best_val)

    # ------------------------------------------------------------------
    # 2) Find centers: zero cells with at least 3 orthogonal neighbors
    #    of the same main color.
    # ------------------------------------------------------------------
    center_set = set()
    centers = []
    orthogonal = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(h):
        for c in range(w):
            if base[r, c] != 0:
                continue

            neigh_counts = {col: 0 for col in main_colors}

            for dr, dc in orthogonal:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    val = int(base[nr, nc])
                    if val in main_set:
                        neigh_counts[val] += 1

            if not any(neigh_counts.values()):
                continue

            base_color = max(
                neigh_counts,
                key=lambda col: (neigh_counts[col], -col)
            )

            if neigh_counts[base_color] >= 3:
                if (r, c) not in center_set:
                    center_set.add((r, c))
                    centers.append((r, c, base_color))

    # ------------------------------------------------------------------
    # 3) Stamp 3x3 patterns around centers.
    #    Ring cells get the other main color; center becomes 8.
    # ------------------------------------------------------------------
    ring = {}
    ring_dirs = [
        (dr, dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]

    for r, c, base_color in centers:
        stamp_color = other_color(base_color)
        for dr, dc in ring_dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in center_set:
                cell_counts = ring.setdefault((nr, nc), {})
                cell_counts[stamp_color] = cell_counts.get(stamp_color, 0) + 1

    out = base.copy()

    # Apply ring colors. If two stamps conflict, use the more frequent one.
    for (nr, nc), cell_counts in ring.items():
        chosen = max(
            cell_counts.items(),
            key=lambda kv: (
                kv[1],
                1 if int(out[nr, nc]) == kv[0] else 0,
                -kv[0]
            )
        )
        out[nr, nc] = int(chosen[0])

    # Finally mark all centers as 8.
    for r, c, _ in centers:
        out[r, c] = 8

    return out