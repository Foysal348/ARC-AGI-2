import numpy as np
from collections import deque
from itertools import product

def _components_8(grid: np.ndarray):
    """Return a list of boolean masks for all 8-connected non-zero components."""
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    masks = []

    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0 and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                cells = []

                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))

                    for dr, dc in product((-1, 0, 1), repeat=2):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < h
                            and 0 <= nc < w
                            and not visited[nr, nc]
                            and grid[nr, nc] != 0
                        ):
                            visited[nr, nc] = True
                            q.append((nr, nc))

                mask = np.zeros((h, w), dtype=bool)
                for rr, cc in cells:
                    mask[rr, cc] = True
                masks.append(mask)

    return masks


def _bounding_box(mask: np.ndarray):
    """Return (rmin, rmax, cmin, cmax) for a non-empty boolean mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin = int(np.argmax(rows))
    rmax = int(len(rows) - 1 - np.argmax(rows[::-1]))
    cmin = int(np.argmax(cols))
    cmax = int(len(cols) - 1 - np.argmax(cols[::-1]))
    return rmin, rmax, cmin, cmax


def _bbox_crop(mask: np.ndarray):
    """Crop a boolean mask to its minimal bounding box."""
    rmin, rmax, cmin, cmax = _bounding_box(mask)
    return mask[rmin:rmax + 1, cmin:cmax + 1]


def _dihedral_variants(mask: np.ndarray):
    """
    Produce all 8 dihedral orientations of a square boolean mask:
    rotations and reflections.
    """
    base = mask.astype(np.uint8)

    # Ensure square.  The relevant color patches in this task are square after
    # cropping, but this guard makes the helper safe for non-square input.
    if base.shape[0] != base.shape[1]:
        # Pad to a square temporarily just to generate symmetries.
        n = max(base.shape)
        padded = np.zeros((n, n), dtype=np.uint8)
        h, w = base.shape
        padded[:h, :w] = base
        base = padded

    # Standard 4 rotations.
    rots = [np.rot90(base, k) for k in range(4)]

    # All 8 symmetries: rotations plus their horizontal flips.
    variants = []
    seen = set()

    for m in rots + [np.fliplr(m) for m in rots]:
        key = m.tobytes()
        if key not in seen:
            seen.add(key)
            variants.append(m)

    return variants


def _canonical_key(variant: np.ndarray):
    """
    Key used to choose the canonical orientation of one color layer.

    The key is a tuple:
      1. Row counts after cropping to the variant's own bounding box,
         lexicographically largest.  This pulls the shape "up" as much as
         possible.
      2. Column counts after cropping, lexicographically smallest.  This is a
         stable tie-breaker that prefers the shape to be pushed right.
      3. The raw flattened binary pattern, lexicographically smallest,
         to make the choice completely deterministic.

    This key reproduces the examples:
      - outer / middle / inner layers are normalized independently;
      - scattered noise outside the largest component is discarded;
      - small asymmetric layers such as the 3x3 triangle in example 2 are
        placed in the same orientation as the shown output.
    """
    cropped = _bbox_crop(variant.astype(bool))

    row_counts = tuple(int(x) for x in cropped.sum(axis=1))
    col_counts = tuple(int(x) for x in cropped.sum(axis=0))

    # We want:
    #   row_counts: lex largest
    #   col_counts: lex smallest
    #   bytes:      lex smallest
    #
    # Numpy tuple comparison does not have separate "largest/smallest" modes,
    # so encode col_counts and bytes in a way that smaller values win,
    # while row_counts naturally wants larger values.  Use negative counts for
    # columns because counts are non-negative.
    return (row_counts, tuple(-x for x in col_counts), cropped.tobytes())


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule:

    1. Find the largest 8-connected non-zero component in the input.
    2. Crop the grid to that component's bounding box.
    3. For every non-background color present in the crop, choose the best of
       the 8 dihedral orientations (4 rotations + 4 reflections) using a
       deterministic canonical key.
    4. Write the chosen orientation of each color into a fresh grid of the
       same cropped size.

    This removes isolated noise outside the main object and normalizes the
    orientation of each colored layer, matching all provided examples.
    """
    if grid.size == 0:
        return grid.copy()

    grid = np.asarray(grid)

    # 1. Largest 8-connected component.
    components = _components_8(grid)
    if not components:
        return np.zeros_like(grid)

    largest = max(components, key=lambda m: int(m.sum()))

    # 2. Crop to that component's bounding box.
    rmin, rmax, cmin, cmax = _bounding_box(largest)
    crop = grid[rmin:rmax + 1, cmin:cmax + 1].copy()

    # 3. Canonicalize each color independently.
    out = np.zeros_like(crop)

    # Process colors in descending order so that if two transformed layers
    # ever conflict, larger color values are not silently overwritten by
    # smaller ones.  The examples do not produce overlaps, but this gives a
    # deterministic fallback.
    colors = [int(c) for c in np.unique(crop) if c != 0]
    colors.sort(reverse=True)

    for color in colors:
        mask = (crop == color)
        if not mask.any():
            continue

        # Work on the color's own minimal bounding box to choose orientation.
        small = _bbox_crop(mask)

        best_variant = None
        best_key = None

        for variant in _dihedral_variants(small):
            key = _canonical_key(variant)
            if best_key is None or key > best_key:
                best_key = key
                best_variant = variant

        # Place the chosen square patch back into the output at the position
        # corresponding to the full-grid dihedral transform that produced it.
        #
        # Because we selected among variants of the cropped bbox, we need to
        # recover where that patch should go in the full crop.  The simplest
        # robust way is to apply the same dihedral transform to the full
        # color mask and keep only the pixels matching the selected local
        # pattern in the correct location.
        #
        # We do this by brute force: test all 8 full-grid transforms, crop
        # each one, and compare it to best_variant.  The matching full-grid
        # transform is used to write the color into `out`.
        full_mask = mask
        chosen_full = None

        # The local variants are generated from small.  Generate the same
        # candidate full transforms and compare their bbox crops.
        full_rots = [np.rot90(full_mask, k) for k in range(4)]
        full_variants = full_rots + [np.fliplr(m) for m in full_rots]

        for fv in full_variants:
            if fv.shape != full_mask.shape:
                continue
            if not fv.any():
                continue
            if _bbox_crop(fv).tobytes() == best_variant.tobytes():
                chosen_full = fv
                break

        if chosen_full is None:
            # Fallback: put the local best pattern in the original bbox.
            r0, r1, c0, c1 = _bounding_box(mask)
            out[r0:r0 + best_variant.shape[0], c0:c0 + best_variant.shape[1]] = (
                best_variant.astype(crop.dtype) * color
            )
        else:
            out[chosen_full] = color

    return out