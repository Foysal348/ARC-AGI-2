import numpy as np
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform grid by:
    1. Finding connected components (objects) of non-background pixels
    2. Splitting objects with multiple color groups into sub-objects
    3. Removing duplicate objects
    4. For same-sized objects, simplifying the less-diverse one to its core color pattern
    5. Padding all to max height and placing side by side
    """
    # Determine background color from borders
    border = np.concatenate([grid[0,:], grid[-1,:], grid[:,0], grid[:,-1]])
    vals, cnts = np.unique(border, return_counts=True)
    bg = int(vals[np.argmax(cnts)])
    
    R, C = grid.shape
    visited = np.zeros((R, C), dtype=bool)
    raw_objects = []
    
    for r in range(R):
        for c in range(C):
            if grid[r,c] != bg and not visited[r,c]:
                comp = []
                queue = [(r,c)]
                visited[r,c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    comp.append((cr,cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc = cr+dr, cc+dc
                        if 0<=nr<R and 0<=nc<C and not visited[nr,nc] and grid[nr,nc]!=bg:
                            visited[nr,nc] = True
                            queue.append((nr,nc))
                r0 = min(p[0] for p in comp); r1 = max(p[0] for p in comp)
                c0 = min(p[1] for p in comp); c1 = max(p[1] for p in comp)
                obj = np.full((r1-r0+1, c1-c0+1), bg, dtype=grid.dtype)
                for pr,pc in comp:
                    obj[pr-r0, pc-c0] = grid[pr,pc]
                raw_objects.append(obj)
    
    # Split objects with distinct color regions
    def split(obj):
        h,w = obj.shape
        if w <= 3 and h <= 3:
            return [obj]
        results = []
        # Vertical split
        if w > 3:
            for sc in range(1, w):
                L, Rt = obj[:,:sc], obj[:,sc:]
                lc = set(L[L!=bg].flatten()); rc = set(Rt[Rt!=bg].flatten())
                if lc and rc and len(lc & rc) < min(len(lc),len(rc)):
                    results = [L, Rt]
                    break
        # Horizontal split
        if not results and h > 3:
            for sr in range(1, h):
                T, B = obj[:sr,:], obj[sr:,:]
                tc = set(T[T!=bg].flatten()); bc = set(B[B!=bg].flatten())
                if tc and bc and len(tc & bc) < min(len(tc),len(bc)):
                    results = [T, B]
                    break
        return results if results else [obj]
    
    subs = []
    for o in raw_objects:
        subs.extend(split(o))
    
    # Deduplicate
    unique = []
    for o in subs:
        if not any(u.shape==o.shape and np.array_equal(u,o) for u in unique):
            unique.append(o)
    
    # Simplify less-diverse objects when same-shaped objects exist
    groups = defaultdict(list)
    for i,o in enumerate(unique):
        groups[o.shape].append(i)
    
    final = list(unique)
    for shape, idxs in groups.items():
        if len(idxs) > 1:
            diversity = [len(set(unique[i].flatten())-{bg}) for i in idxs]
            primary = idxs[np.argmax(diversity)]
            for i in idxs:
                if i != primary:
                    o = unique[i]
                    nb = o[o!=bg]
                    if len(nb)>0:
                        v,c = np.unique(nb, return_counts=True)
                        mc = v[np.argmax(c)]
                        mask = o==mc
                        if mask.any():
                            ri = np.where(mask.any(axis=1))[0]
                            ci = np.where(mask.any(axis=0))[0]
                            no = np.full_like(o, bg)
                            no[ri[0]:ri[-1]+1, ci[0]:ci[-1]+1] = mc
                            final[i] = no
    
    # Pad and concatenate
    mh = max(o.shape[0] for o in final)
    padded = []
    for o in final:
        h,w = o.shape
        if h < mh:
            pt = (mh-h)//2
            no = np.full((mh,w), bg, dtype=grid.dtype)
            no[pt:pt+h,:] = o
            padded.append(no)
        else:
            padded.append(o.copy())
    
    tw = sum(o.shape[1] for o in padded)
    res = np.full((mh, tw), bg, dtype=grid.dtype)
    co = 0
    for o in padded:
        res[:, co:co+o.shape[1]] = o
        co += o.shape[1]
    return res