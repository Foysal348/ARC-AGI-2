import os
import sys
from pathlib import Path
source_root = Path(os.environ["PROGRAM063_LEGACY_ROOT"]).resolve()
overlay_root = Path(os.environ["PROGRAM063_OVERLAY_ROOT"]).resolve()
clean = []
for item in sys.path:
    if not item:
        clean.append(item)
        continue
    try:
        resolved = Path(item).resolve()
    except OSError:
        continue
    if resolved == source_root or source_root in resolved.parents or resolved == overlay_root:
        continue
    clean.append(item)
sys.path[:] = [str(overlay_root)] + clean

import torch
import triton
import triton.language as tl

@triton.jit
def copy_kernel(x_ptr, y_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, values, mask=mask)

x = torch.arange(16, device="cuda:0", dtype=torch.float32)
y = torch.empty_like(x)
copy_kernel[(1,)](x, y, n_elements=16, BLOCK_SIZE=16)
torch.cuda.synchronize()
if not torch.equal(x, y):
    raise RuntimeError("Triton probe produced an incorrect result")
print("PROGRAM063_TRITON_PROBE_OK")
