import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform thick stroke components into thin centerlines.

    The inferred rule is:
    - The border color is the background.
    - Small separate 2x2 colored squares are markers:
        * the topmost group defines the repeating paint sequence;
        * the remaining marker defines the endpoint color.
    - All other non-background components are treated as thick strokes.
    - For each stroke:
        * compute a morphological skeleton;
        * keep the part supported by the thick core, while allowing short
          end caps to survive;
        * paint the skeleton with the repeating sequence, using graph distance
          from the top-left-most skeleton pixel as the phase;
        * mark skeleton leaves by extending one step outward with the endpoint
          color when that extension is outside the stroke.
    """
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------
    def most_frequent_border_color(g):
        border = np.concatenate([
            g[0, :], g[-1, :], g[:, 0], g[:, -1]
        ])
        vals, counts = np.unique(border, return_counts=True)
        best = counts.max()
        # deterministic tie-break: smaller color id
        candidates = vals[counts == best]
        return int(candidates[0])

    def label_components(mask, connectivity=4):
        if not mask.any():
            return []
        h2, w2 = mask.shape
        labels = np.zeros((h2, w2), dtype=int)
        comps = []
        dirs = [
            ((1, 0), (-1, 0), (0, 1), (0, -1)) if connectivity == 4 else
            ((1, 0), (-1, 0), (0, 1), (0, -1),
             (1, 1), (1, -1), (-1, 1), (-1, -1))
        ]
        drs, dcs = dirs[0]

        for r in range(h2):
            for c in range(w2):
                if not mask[r, c] or labels[r, c]:
                    continue
                lab = len(comps) + 1
                comp = np.zeros((h2, w2), dtype=bool)
                q = deque([(r, c)])
                labels[r, c] = lab
                comp[r, c] = True
                while q:
                    x, y = q.popleft()
                    for dr, dc in zip(drs, dcs):
                        nx, ny = x + dr, y + dc
                        if 0 <= nx < h2 and 0 <= ny < w2:
                            if mask[nx, ny] and not labels[nx, ny]:
                                labels[nx, ny] = lab
                                comp[nx, ny] = True
                                q.append((nx, ny))
                comps.append(comp)
        return comps

    def thin_zhang_suen(mask):
        """Standard Zhang-Suen thinning, preserving 8-connectivity."""
        if not mask.any():
            return mask.copy()

        img = mask.astype(np.uint8).copy()
        H, W = img.shape

        while True:
            removed_any = False

            # Step 1
            to_remove = set()
            for x in range(1, H - 1):
                for y in range(1, W - 1):
                    if img[x, y] == 0:
                        continue
                    P2 = img[x, y + 1]
                    P3 = img[x - 1, y + 1]
                    P4 = img[x - 1, y]
                    P5 = img[x - 1, y - 1]
                    P6 = img[x, y - 1]
                    P7 = img[x + 1, y - 1]
                    P8 = img[x + 1, y]
                    P9 = img[x + 1, y + 1]

                    B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
                    A = 0
                    if P2 == 0 and P3 == 1: A += 1
                    if P3 == 0 and P4 == 1: A += 1
                    if P4 == 0 and P5 == 1: A += 1
                    if P5 == 0 and P6 == 1: A += 1
                    if P6 == 0 and P7 == 1: A += 1
                    if P7 == 0 and P8 == 1: A += 1
                    if P8 == 0 and P9 == 1: A += 1
                    if P9 == 0 and P2 == 1: A += 1

                    if (2 <= B <= 6 and A == 1 and
                            P2 * P4 * P6 == 0 and P4 * P6 * P8 == 0):
                        to_remove.add((x, y))

            for x, y in to_remove:
                img[x, y] = 0
                removed_any = True

            # Step 2
            to_remove = set()
            for x in range(1, H - 1):
                for y in range(1, W - 1):
                    if img[x, y] == 0:
                        continue
                    P2 = img[x, y + 1]
                    P3 = img[x - 1, y + 1]
                    P4 = img[x - 1, y]
                    P5 = img[x - 1, y - 1]
                    P6 = img[x, y - 1]
                    P7 = img[x + 1, y - 1]
                    P8 = img[x + 1, y]
                    P9 = img[x + 1, y + 1]

                    B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
                    A = 0
                    if P2 == 0 and P3 == 1: A += 1
                    if P3 == 0 and P4 == 1: A += 1
                    if P4 == 0 and P5 == 1: A += 1
                    if P5 == 0 and P6 == 1: A += 1
                    if P6 == 0 and P7 == 1: A += 1
                    if P7 == 0 and P8 == 1: A += 1
                    if P8 == 0 and P9 == 1: A += 1
                    if P9 == 0 and P2 == 1: A += 1

                    if (2 <= B <= 6 and A == 1 and
                            P2 * P4 * P8 == 0 and P2 * P6 * P8 == 0):
                        to_remove.add((x, y))

            for x, y in to_remove:
                img[x, y] = 0
                removed_any = True

            if not removed_any:
                break

        return img.astype(bool)

    def distance_to_background(mask):
        """4-neighbor Manhattan distance from mask pixels to outside/background."""
        H, W = mask.shape
        dist = np.full((H, W), -1, dtype=int)
        q = deque()

        for r in range(H):
            for c in range(W):
                if not mask[r, c]:
                    continue
                touch = False
                if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                    touch = True
                else:
                    if not mask[r - 1, c] or not mask[r + 1, c] or \
                       not mask[r, c - 1] or not mask[r, c + 1]:
                        touch = True
                if touch:
                    dist[r, c] = 1
                    q.append((r, c))

        while q:
            r, c = q.popleft()
            nd = dist[r, c] + 1
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and dist[nr, nc] == -1:
                    dist[nr, nc] = nd
                    q.append((nr, nc))

        # Any unreachable mask cells (should not happen for finite masks) get 1.
        dist[mask & (dist == -1)] = 1
        return dist

    def prune_skeleton_by_core(skel, core, max_noncore_dist=3):
        """
        Keep skeleton branches that are supported by the thick core.
        Short terminal caps (up to max_noncore_dist pixels) are retained.
        """
        if skel.sum() == 0:
            return skel

        kept = np.zeros_like(skel, dtype=bool)
        skel_comps = label_components(skel, connectivity=8)

        for comp in skel_comps:
            comp_core = comp & core
            if not comp_core.any():
                continue

            # BFS inside this skeleton component, starting from all core pixels.
            d = np.full(skel.shape, -1, dtype=int)
            q = deque()
            crs, ccs = np.where(comp_core)
            for r, c in zip(crs, ccs):
                d[r, c] = 0
                q.append((r, c))

            while q:
                r, c = q.popleft()
                nd = d[r, c] + 1
                for dr, dc in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < skel.shape[0] and 0 <= nc < skel.shape[1]:
                        if comp[nr, nc] and d[nr, nc] == -1:
                            d[nr, nc] = nd
                            q.append((nr, nc))

            keep = comp & (d >= 0) & (d <= max_noncore_dist)
            kept |= keep

        return kept

    def bfs_distance_on_component(comp, start, connectivity=8):
        H, W = comp.shape
        dist = np.full((H, W), -1, dtype=int)
        dist[start[0], start[1]] = 0
        q = deque([start])
        dirs = [
            ((1, 0), (-1, 0), (0, 1), (0, -1)) if connectivity == 4 else
            ((1, 0), (-1, 0), (0, 1), (0, -1),
             (1, 1), (1, -1), (-1, 1), (-1, -1))
        ]
        drs, dcs = dirs[0]

        while q:
            r, c = q.popleft()
            nd = dist[r, c] + 1
            for dr, dc in zip(drs, dcs):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and comp[nr, nc] and dist[nr, nc] == -1:
                    dist[nr, nc] = nd
                    q.append((nr, nc))
        return dist

    def find_8_neighbors(coords_set, r, c):
        res = []
        H, W = len(coords_set)  # placeholder, not used
        return res

    def leaf_extensions(skel_comp, H, W):
        """
        Return (leaf, outward_candidate) for 8-connected leaves of skel_comp.
        """
        coords = [(r, c) for r in range(H) for c in range(W) if skel_comp[r, c]]
        coord_set = set(coords)
        res = []

        for r, c in coords:
            neighs = []
            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) in coord_set:
                    neighs.append((nr, nc))

            if len(neighs) == 1:
                nr, nc = neighs[0]
                dr = r - nr
                dc = c - nc
                er, ec = r + dr, c + dc
                res.append(((r, c), (er, ec)))

        return res

    # ---------------------------------------------------------------
    # 1. Background and components
    # ---------------------------------------------------------------
    bg = most_frequent_border_color(grid)
    nonbg = (grid != bg)

    if not nonbg.any():
        return grid.copy()

    comps = label_components(nonbg, connectivity=4)

    markers = []
    stroke_masks = []
    stroke_union = np.zeros((h, w), dtype=bool)

    for comp in comps:
        area = int(comp.sum())
        if area == 0:
            continue

        ys, xs = np.where(comp)
        r0, r1 = int(ys.min()), int(ys.max())
        c0, c1 = int(xs.min()), int(xs.max())
        bh, bw = r1 - r0 + 1, c1 - c0 + 1

        vals = np.unique(grid[comp])
        uniform = len(vals) == 1

        # Small separate 2x2 squares are marker squares.
        if area == 4 and bh == 2 and bw == 2 and uniform:
            markers.append((r0, c0, int(vals[0])))
            continue

        # Large non-background components are thick strokes.
        if area > 4:
            stroke_masks.append(comp)
            stroke_union |= comp
        # Very tiny non-marker components are ignored/erased.

    # Fallback: if no large component was found, use the largest non-bg component.
    if not stroke_masks:
        largest = max(comps, key=lambda m: int(m.sum())) if comps else nonbg
        stroke_masks = [largest]
        stroke_union = largest.copy()

    # ---------------------------------------------------------------
    # 2. Determine paint sequence and endpoint color from markers
    # ---------------------------------------------------------------
    seq = []
    endpoint = None

    if markers:
        min_r = min(m[0] for m in markers)
        top_markers = [m for m in markers if m[0] == min_r]

        # If everything is in the top row, treat all as sequence and no endpoint.
        if len(top_markers) == len(markers):
            seq = [m[2] for m in sorted(top_markers, key=lambda x: (x[0], x[1]))]
            endpoint = None
        else:
            seq = [m[2] for m in sorted(top_markers, key=lambda x: (x[0], x[1]))]
            others = [m for m in markers if m[0] != min_r]
            # Lowest / farthest marker is the endpoint marker.
            endpoint_marker = max(others, key=lambda x: (x[0], x[0] + x[1]))
            endpoint = endpoint_marker[2]

    # Fallbacks if markers are missing.
    if not seq:
        # Use the first two distinct non-background, non-stroke colors if present.
        extra_colors = []
        for color in np.unique(grid):
            ci = int(color)
            if ci == bg or ci in stroke_union:
                continue
            extra_colors.append(ci)
        seq = extra_colors[:3] if extra_colors else [1]

    if endpoint is None:
        # Use a non-background color not in seq if possible.
        endpoint = None
        for color in np.unique(grid):
            ci = int(color)
            if ci != bg and ci not in seq:
                endpoint = ci
                break
        if endpoint is None:
            endpoint = int(seq[0])

    # ---------------------------------------------------------------
    # 3. Erase non-stroke non-background pixels
    # ---------------------------------------------------------------
    out = grid.copy()
    out[nonbg & ~stroke_union] = bg

    # ---------------------------------------------------------------
    # 4. Process each stroke component
    # ---------------------------------------------------------------
    for stroke_mask in stroke_masks:
        if not stroke_mask.any():
            continue

        # Raw skeleton of the whole stroke.
        skel = thin_zhang_suen(stroke_mask)
        if not skel.any():
            continue

        # Prune long thin branches that are not supported by the thick core.
        dist = distance_to_background(stroke_mask)
        if dist.max() >= 2:
            core = skel & (dist >= 2)
            if core.any():
                pruned = prune_skeleton_by_core(skel, core, max_noncore_dist=3)
                if pruned.any():
                    skel = pruned

        # Process each 8-connected skeleton piece independently.
        for skel_comp in label_components(skel, connectivity=8):
            if not skel_comp.any():
                continue

            coords = [(r, c) for r in range(h) for c in range(w) if skel_comp[r, c]]
            root = min(coords, key=lambda x: (x[0] + x[1], x[0], x[1]))

            dmap = bfs_distance_on_component(skel_comp, root, connectivity=8)

            # Paint pattern by distance modulo sequence length.
            for r, c in coords:
                d = dmap[r, c]
                if d < 0:
                    d = 0
                out[r, c] = seq[d % len(seq)]

            # Add endpoint markers at outward extensions of leaves.
            for leaf, ext in leaf_extensions(skel_comp, h, w):
                er, ec = ext
                if 0 <= er < h and 0 <= ec < w:
                    # Put endpoint only where it lands outside the stroke.
                    if not stroke_union[er, ec]:
                        out[er, ec] = endpoint

    return out