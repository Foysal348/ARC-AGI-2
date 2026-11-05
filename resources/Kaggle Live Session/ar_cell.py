import subprocess, textwrap, pathlib
pathlib.Path('/kaggle/working/ar.py').write_text(textwrap.dedent('''
import os, time, torch, torch.distributed as dist
dist.init_process_group("nccl"); r = dist.get_rank(); torch.cuda.set_device(r)
for n in (16*1024, 256*1024, 4*1024*1024, 64*1024*1024):   # bytes
    x = torch.ones(n // 2, dtype=torch.bfloat16, device="cuda")
    for _ in range(5): dist.all_reduce(x)
    torch.cuda.synchronize(); t = time.perf_counter(); it = 50
    for _ in range(it): dist.all_reduce(x)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) / it
    if r == 0: print(f"all_reduce {n/1024:>8.0f} KiB bf16 x4: {dt*1e6:8.1f} us  busbw {2*3/4*n/dt/1e9:6.2f} GB/s", flush=True)
dist.destroy_process_group()
'''))
p = subprocess.run(["torchrun", "--nproc_per_node=4", "/kaggle/working/ar.py"], capture_output=True, text=True, timeout=300)
print(p.stdout[-3000:]); print(p.stderr[-1500:] if p.returncode else "rc 0")
