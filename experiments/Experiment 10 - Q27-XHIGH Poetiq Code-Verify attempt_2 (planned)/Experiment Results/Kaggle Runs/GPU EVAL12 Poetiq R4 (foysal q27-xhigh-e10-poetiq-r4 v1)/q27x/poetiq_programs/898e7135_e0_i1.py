import numpy as np
from collections import deque, defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule
    ==============
    The input contains:
      - one "base" object, whose bounding box has holes,
      - several "piece" objects,
      - some isolated noise pixels.

    The base object determines:
      - the output background color,
      - the output size:  k * base_bbox_height  x  k * base_bbox_width,
    where k is the common scale factor between the pieces and the holes.

    Each hole in the base bounding box is filled by the color of a piece whose
    shape, after downsampling by k and allowing rotations/reflections, matches
    the hole shape.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape

    # ------------------------------------------------------------------
    # Connected components for all non-zero colors
    # ------------------------------------------------------------------
    def nonzero_components():
        visited = np.zeros((H, W), dtype=bool)
        comps = []

        for i in range(H):
            for j in range(W):
                if grid[i, j] == 0 or visited[i, j]:
                    continue

                color = int(grid[i, j])
                q = deque([(i, j)])
                visited[i, j] = True
                coords = []
                min_r = max_r = i
                min_c = max_c = j

                while q:
                    r, c = q.popleft()
                    coords.append((r, c))
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)

                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if not visited[nr, nc] and grid[nr, nc] == color:
                                visited[nr, nc] = True
                                q.append((nr, nc))

                mask = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=bool)
                for r, c in coords:
                    mask[r - min_r, c - min_c] = True

                comps.append({
                    "color": color,
                    "mask": mask,
                    "bbox": (min_r, max_r, min_c, max_c),
                    "size": len(coords),
                    "bbox_area": (max_r - min_r + 1) * (max_c - min_c + 1),
                })

        return comps

    comps = nonzero_components()

    # ------------------------------------------------------------------
    # Ignore noise: colors that appear only as multiple isolated pixels
    # ------------------------------------------------------------------
    sizes_by_color = defaultdict(list)
    for comp in comps:
        sizes_by_color[comp["color"]].append(comp["size"])

    noise_colors = set()
    for color, sizes in sizes_by_color.items():
        if len(sizes) > 1 and all(s == 1 for s in sizes):
            noise_colors.add(color)

    # ------------------------------------------------------------------
    # Small helpers for binary masks
    # ------------------------------------------------------------------
    def crop_mask(m):
        if m.size == 0 or not m.any():
            return m
        rows = np.any(m, axis=1)
        cols = np.any(m, axis=0)
        r0 = int(np.argmax(rows))
        r1 = int(len(rows) - 1 - np.argmax(rows[::-1]))
        c0 = int(np.argmax(cols))
        c1 = int(len(cols) - 1 - np.argmax(cols[::-1]))
        return m[r0:r1 + 1, c0:c1 + 1]

    def canonical(m):
        """
        Canonical form under the 8 dihedral transformations.
        """
        m = crop_mask(m)
        if m.size == 0:
            return (0,)

        keys = set()
        for flip in (False, True):
            t = m[:, ::-1] if flip else m
            for k in range(4):
                tt = np.rot90(t, k)
                tt = crop_mask(tt)
                keys.add(tuple(tuple(int(x) for x in row) for row in tt))
        return min(keys)

    def downsample(m, k):
        """
        Downsample by integer factor k using "any" pooling in k x k blocks.
        Partial blocks at the bottom/right are included.
        """
        if k <= 1:
            return m

        h, w = m.shape
        oh = (h + k - 1) // k
        ow = (w + k - 1) // k
        out = np.zeros((oh, ow), dtype=bool)

        for i in range(oh):
            r0 = i * k
            r1 = min(r0 + k, h)
            for j in range(ow):
                c0 = j * k
                c1 = min(c0 + k, w)
                if m[r0:r1, c0:c1].any():
                    out[i, j] = True
        return out

    def bool_components(mask):
        h, w = mask.shape
        visited = np.zeros((h, w), dtype=bool)
        comps = []

        for i in range(h):
            for j in range(w):
                if not mask[i, j] or visited[i, j]:
                    continue

                q = deque([(i, j)])
                visited[i, j] = True
                coords = []
                min_r = max_r = i
                min_c = max_c = j

                while q:
                    r, c = q.popleft()
                    coords.append((r, c))
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)

                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if mask[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))

                cmask = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=bool)
                for r, c in coords:
                    cmask[r - min_r, c - min_c] = True

                comps.append({
                    "mask": cmask,
                    "bbox": (min_r, max_r, min_c, max_c),
                    "size": len(coords),
                })

        return comps

    def piece_patterns(comp, k, allow_offsets):
        """
        All canonical downsampled patterns for this piece at scale k.
        Offset 0 is always used; offsets are optional fallback.
        """
        m = comp["mask"]
        pats = {canonical(downsample(m, k))}

        if allow_offsets:
            h, w = m.shape
            for off_r in range(min(k, h)):
                for off_c in range(min(k, w)):
                    if off_r == 0 and off_c == 0:
                        continue
                    sub = m[off_r:, off_c:]
                    if sub.any():
                        pats.add(canonical(downsample(sub, k)))
        return pats

    def max_matching(adj, n_holes):
        """
        Maximum bipartite matching from pieces (rows in adj) to holes.
        Returns match_hole[h] = piece index, or -1.
        """
        match = [-1] * n_holes

        def dfs(p, seen):
            for h in adj[p]:
                if seen[h]:
                    continue
                seen[h] = True
                if match[h] == -1 or dfs(match[h], seen):
                    match[h] = p
                    return True
            return False

        for p in range(len(adj)):
            dfs(p, [False] * n_holes)
        return match

    # ------------------------------------------------------------------
    # Try a base component and find the best scale factor / assignment
    # ------------------------------------------------------------------
    def evaluate_base(base, allow_offsets):
        base_color = base["color"]
        r0, r1, c0, c1 = base["bbox"]

        sub = grid[r0:r1 + 1, c0:c1 + 1]
        hole_mask = (sub == 0)
        if not hole_mask.any():
            return None

        holes = bool_components(hole_mask)
        if not holes:
            return None

        holes.sort(key=lambda x: (x["bbox"][0], x["bbox"][2], x["bbox"][1], x["bbox"][3]))

        pieces = []
        for comp in comps:
            if comp is base:
                continue
            if comp["color"] == base_color:
                continue
            if comp["color"] in noise_colors and comp["size"] == 1:
                continue
            pieces.append(comp)

        if not pieces:
            return None

        pieces.sort(key=lambda x: (x["color"], x["bbox"][0], x["bbox"][2]))

        max_piece_dim = max(max(p["mask"].shape) for p in pieces)
        max_k = min(max(H, W), max(5, max_piece_dim), 20)

        hole_pats = [canonical(h["mask"]) for h in holes]
        best = None

        for k in range(1, max_k + 1):
            adj = []
            for p in pieces:
                pats = piece_patterns(p, k, allow_offsets)
                adj.append([hi for hi, hp in enumerate(hole_pats) if hp in pats])

            match = max_matching(adj, len(holes))
            matched = sum(1 for m in match if m != -1)
            if matched == 0:
                continue

            perfect_holes = (matched == len(holes))
            all_pieces = (matched == len(pieces))

            score = (
                perfect_holes,
                all_pieces,
                matched,
                base["bbox_area"],
                -k,
            )

            cand = {
                "base": base,
                "k": k,
                "match": match,
                "holes": holes,
                "pieces": pieces,
                "matched": matched,
                "score": score,
            }

            if best is None or cand["score"] > best["score"]:
                best = cand

        return best

    # ------------------------------------------------------------------
    # Search over possible bases. Prefer strict downsample; fallback offsets.
    # ------------------------------------------------------------------
    base_candidates = [c for c in comps if c["size"] > 1]
    base_candidates.sort(key=lambda x: (x["bbox_area"], x["size"]), reverse=True)

    best_solution = None

    # Strict pass
    for base in base_candidates:
        cand = evaluate_base(base, allow_offsets=False)
        if cand is not None:
            if best_solution is None or cand["score"] > best_solution["score"]:
                best_solution = cand

    # If no perfect strict solution, try fallback with offsets.
    if best_solution is None or best_solution["matched"] < len(best_solution["holes"]):
        for base in base_candidates:
            cand = evaluate_base(base, allow_offsets=True)
            if cand is not None:
                if best_solution is None or cand["score"] > best_solution["score"]:
                    best_solution = cand

    # ------------------------------------------------------------------
    # If nothing was found, return input gracefully.
    # ------------------------------------------------------------------
    if best_solution is None:
        return grid.copy()

    base = best_solution["base"]
    k = best_solution["k"]
    holes = best_solution["holes"]
    pieces = best_solution["pieces"]
    match = best_solution["match"]

    base_color = base["color"]
    r0, r1, c0, c1 = base["bbox"]
    bh = r1 - r0 + 1
    bw = c1 - c0 + 1

    out = np.full((bh * k, bw * k), base_color, dtype=int)

    for h_idx, p_idx in enumerate(match):
        if p_idx == -1:
            continue

        color = pieces[p_idx]["color"]
        h = holes[h_idx]
        hr0, hr1, hc0, hc1 = h["bbox"]
        hm = h["mask"]

        for dr, dc in zip(*np.where(hm)):
            r = hr0 + dr
            c = hc0 + dc
            out[r * k:(r + 1) * k, c * k:(c + 1) * k] = color

    return out