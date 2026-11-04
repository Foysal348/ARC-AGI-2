"""Bounded Flash-Next startup I/O for the pinned Kaggle SGLang runtime.

Only the separate draft is copied in full. Rank zero stages at most the current
and next target GPU shard, copying each with parallel range reads. All TP ranks
finish reading before it unlinks a shard.
Each rank fills its final pinned PLE rows directly from the attachment in parallel,
with parallel range reads rather than mmap page faults.
No checkpoint values, quantization, forward code, or evaluation policy change.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import time

PLAN_ENV = "FLASH_NEXT_LOADING_PLAN"
# Kaggle's dataset mount serves about 50-65 MB/s per read stream and scales with parallel streams.
# v12 copied each shard as one stream: 65.7 GB in 1,290 s, while four PLE readers reached ~254 MB/s.
# Around 2026-10-20 attachments moved from NFS to GCSFUSE, where every read is a high-latency
# Cloud Storage request: more concurrent streams keep more bytes in flight. Each reader opens its
# own handle and reads its range sequentially, and PLE no longer goes through mmap page faults.
COPY_READERS = 16
COPY_CHUNK = 16 * 2**20
PLE_READERS = 8  # Per TP rank, so 32 PLE streams in flight beside the shard copy.
SAFETENSORS_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}
# Stock SGLang 0.5.20; qwen4_exp.py must instead be flash_next_serving's patch of its stock original.
WEIGHT_UTILS_SHA256 = "64795609f90e997d7c303d372444e432be2ffd25351ecb5694420b243f0b092e"


def event(stage, **fields):
    print("[weight-loading] " + json.dumps({"stage": stage, **fields}), flush=True)


def prepare_plan(source, cache, destination):
    source, cache, destination = map(Path, (source, cache, destination))
    mapping = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    groups = {}
    for name, filename in mapping.items():
        if Path(filename).name != filename or not (source / filename).is_file():
            raise ValueError(f"invalid or missing checkpoint shard: {filename}")
        kind = "ple" if ".ngram_embedding.shard_" in name else "unused" if "mtp" in name else "gpu"
        groups.setdefault(filename, set()).add(kind)
    if any(len(kinds) != 1 for kinds in groups.values()):
        raise ValueError("PLE/MTP/GPU tensors must be in separate shard files")
    ple = sorted(name for name, kinds in groups.items() if kinds == {"ple"})
    gpu = sorted(name for name, kinds in groups.items() if kinds == {"gpu"})
    if not ple or not gpu:
        raise ValueError("expected both PLE and GPU shards")
    cache.mkdir(parents=True, exist_ok=False)
    sizes = {name: (source / name).stat().st_size for name in groups}
    # Bound our own writes rather than trusting an overlay filesystem's free space.
    window_bytes = sum(sorted((sizes[name] for name in gpu), reverse=True)[:2])
    if window_bytes > 12 * 2**30:
        raise ValueError("two-shard window exceeds the 12 GiB staging budget")
    plan = {"source": str(source.resolve()), "cache": str(cache.resolve()),
            "gpu": gpu, "ple": ple, "sizes": sizes,
            "window_bytes": window_bytes,
            "index_sha256": hashlib.sha256((source / "model.safetensors.index.json").read_bytes()).hexdigest()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan, indent=2) + "\n")
    event("plan", gpu_shards=len(gpu), ple_shards=len(ple), window_bytes=window_bytes,
          ple_bytes=sum(sizes[name] for name in ple), source=str(source))
    return plan


def get_plan():
    path = os.environ.get(PLAN_ENV)
    return json.loads(Path(path).read_text()) if path else None


def is_streamed_target(files):
    plan = get_plan()
    return bool(plan and files and all(Path(p).parent.resolve() == Path(plan["source"]) for p in files))


def copy_range(source, temporary, start, end):
    """Copy bytes [start, end) of source into the same offsets of the preallocated temporary."""
    with open(source, "rb", buffering=0) as reader, open(temporary, "r+b", buffering=0) as writer:
        offset = start
        while offset < end:
            data = os.pread(reader.fileno(), min(COPY_CHUNK, end - offset), offset)
            if not data:
                raise IOError(f"short checkpoint read at byte {offset} of {source.name}")
            view = memoryview(data)
            while view:
                written = os.pwrite(writer.fileno(), view, offset)
                offset += written
                view = view[written:]


def copy_shard(plan, name):
    source, destination = Path(plan["source"]) / name, Path(plan["cache"]) / name
    temporary = destination.with_suffix(".copying")
    size = plan["sizes"][name]
    event("copy_begin", file=name, bytes=size, readers=COPY_READERS)
    started = time.monotonic()
    try:
        with open(temporary, "wb") as handle:
            handle.truncate(size)  # Each reader fills its own range; the window still counts the full shard.
        bounds = [size * part // COPY_READERS for part in range(COPY_READERS + 1)]
        with ThreadPoolExecutor(max_workers=COPY_READERS) as readers:
            jobs = [readers.submit(copy_range, source, temporary, start, end)
                    for start, end in zip(bounds, bounds[1:]) if end > start]
            for job in jobs:
                job.result()
        if temporary.stat().st_size != size:
            raise IOError(f"short checkpoint copy: {name}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    seconds = time.monotonic() - started
    event("copy_end", file=name, seconds=seconds, readers=COPY_READERS,
          megabytes_per_second=size / max(seconds, 1e-9) / 1e6)
    return destination


def staged_paths(plan, rank, publish, barrier, synchronize):
    """The callbacks use the TP CPU group; PLE workers never use collectives."""
    with ThreadPoolExecutor(max_workers=1) as copier:
        pending = copier.submit(copy_shard, plan, plan["gpu"][0]) if rank == 0 else None
        for index, name in enumerate(plan["gpu"]):
            error = None
            if rank == 0:
                try:
                    pending.result()
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            error = publish(error)
            if error:
                raise RuntimeError(f"checkpoint staging failed: {error}")
            if rank == 0 and index + 1 < len(plan["gpu"]):
                pending = copier.submit(copy_shard, plan, plan["gpu"][index + 1])
            yield Path(plan["cache"]) / name
            synchronize()
            barrier()
            if rank == 0:
                (Path(plan["cache"]) / name).unlink()
                event("shard_released", file=name)
            # No third shard may be copied until rank zero has deleted this one.
            barrier()


def staged_target_weights(files):
    import torch
    from safetensors import safe_open
    from sglang.srt.distributed import get_tp_group

    plan = get_plan()
    if not is_streamed_target(files):
        raise ValueError("target staging called for a different checkpoint")
    group = get_tp_group()

    def publish(error):
        values = [error]
        torch.distributed.broadcast_object_list(values, src=group.first_rank, group=group.cpu_group)
        return values[0]

    for path in staged_paths(
        plan, group.rank_in_group, publish,
        lambda: torch.distributed.barrier(group=group.cpu_group), torch.cuda.synchronize,
    ):
        with safe_open(str(path), framework="pt", device="cpu") as tensors:
            for name in tensors.keys():
                yield name, tensors.get_tensor(name)


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safetensors_header(path):
    """(offset of the data section, header) of a safetensors file."""
    with open(path, "rb", buffering=0) as stream:
        size = int.from_bytes(stream.read(8), "little")
        header = json.loads(stream.read(size))
    return 8 + size, header


def read_range_into(source, view, offset):
    """Fill the writable memoryview from bytes [offset, offset + len(view)) of source."""
    with open(source, "rb", buffering=0) as reader:
        done = 0
        while done < len(view):
            count = os.preadv(reader.fileno(), [view[done:done + COPY_CHUNK]], offset + done)
            if count <= 0:
                raise IOError(f"short checkpoint read at byte {offset + done} of {Path(source).name}")
            done += count


def read_tensor(source, entry, data_offset, readers):
    """One safetensors tensor read with ``readers`` parallel range reads instead of mmap."""
    import torch

    if entry["dtype"] not in SAFETENSORS_DTYPES:
        raise ValueError(f"unsupported PLE dtype {entry['dtype']} in {Path(source).name}")
    start, end = entry["data_offsets"]
    buffer = torch.empty(end - start, dtype=torch.uint8)
    view = memoryview(buffer.numpy())
    bounds = [len(view) * part // readers for part in range(readers + 1)]
    with ThreadPoolExecutor(max_workers=readers) as pool:
        jobs = [pool.submit(read_range_into, source, view[low:high], data_offset + start + low)
                for low, high in zip(bounds, bounds[1:]) if high > low]
        for job in jobs:
            job.result()
    return buffer.view(getattr(torch, SAFETENSORS_DTYPES[entry["dtype"]])).reshape(entry["shape"])


def load_ple(plan, load_shard, modules, num_parts):
    import torch

    started, count, copied_bytes, read_bytes = time.monotonic(), 0, 0, 0
    with torch.no_grad():
        for filename in plan["ple"]:
            path = Path(plan["source"]) / filename
            data_offset, header = safetensors_header(path)
            for original, entry in header.items():
                if original == "__metadata__":
                    continue
                name = original.replace("model.language_model.", "model.")
                match = re.search(r"\.ngram_embedding\.shard_(\d+)\.weight$", name)
                if match is None:
                    raise ValueError(f"unexpected tensor in PLE-only shard: {original}")
                embedding = modules[name[:name.index(".ngram_embedding.shard_")]].ngram_embedding
                rows = entry["shape"][0]
                start = int(match[1]) * ((embedding.org_vocab_size + num_parts - 1) // num_parts)
                overlap = min(start + rows, embedding.shard_indices.org_vocab_end_index) - max(start, embedding.shard_indices.org_vocab_start_index)
                if overlap <= 0:
                    continue
                if not load_shard(name, read_tensor(path, entry, data_offset, PLE_READERS)):
                    raise RuntimeError(f"PLE loader rejected {name}")
                count += 1
                copied_bytes += overlap * embedding.embedding_dim * embedding.weight.element_size()
                read_bytes += entry["data_offsets"][1] - entry["data_offsets"][0]
            elapsed = time.monotonic() - started
            event("ple_progress", shards=count, bytes=copied_bytes, seconds=elapsed,
                  read_megabytes_per_second=read_bytes / max(elapsed, 1e-9) / 1e6)
    elapsed = time.monotonic() - started
    event("ple_complete", shards=count, bytes=copied_bytes, seconds=elapsed, readers=PLE_READERS,
          read_megabytes_per_second=read_bytes / max(elapsed, 1e-9) / 1e6)


def concurrent_ple_weights(weights, load_shard, modules, num_parts):
    plan = get_plan()
    if not plan or not modules:
        yield from weights
        return
    with ThreadPoolExecutor(max_workers=1) as reader:
        pending = reader.submit(load_ple, plan, load_shard, modules, num_parts)
        try:
            for item in weights:
                if pending.done():
                    pending.result()  # Propagate a PLE error while GPU loading continues.
                yield item
        finally:
            pending.result()  # Never permit post-processing/warmup before PLE completes.


def patch_sglang(site):
    """Two explicit hooks in the pinned sources, after flash_next_serving; save originals for inspection."""
    from flash_next_serving import replace_text, sha256_text, transform

    site = Path(site)
    qwen4_exp = "srt/models/qwen4_exp.py"
    replacements = [
        ("sglang/srt/model_loader/weight_utils.py", WEIGHT_UTILS_SHA256,
         'def safetensors_weights_iterator(\n'
         '    hf_weights_files: List[str],\n'
         '    disable_mmap: bool = False,\n'
         '    prefetch: bool = False,\n'
         '    prefetch_num_threads: int = 4,\n'
         '    drop_cache_after_load: bool = False,\n'
         ') -> Generator[Tuple[str, torch.Tensor], None, None]:\n'
         '    """Iterate over the weights in the model safetensor files."""\n',
         '    from flash_next_loading import is_streamed_target, staged_target_weights\n'
         '    if is_streamed_target(hf_weights_files):\n'
         '        yield from staged_target_weights(hf_weights_files)\n'
         '        return\n'),
        ("sglang/" + qwen4_exp,
         sha256_text(transform(qwen4_exp, (site / "sglang" / qwen4_exp).with_suffix(".py.before_serving").read_text())),
         '        for name, loaded_weight in weights:\n',
         '        from flash_next_loading import concurrent_ple_weights\n'
         '        weights = concurrent_ple_weights(\n'
         '            weights, load_qwen4_exp_ple_shard, ple_modules, ple_num_sync_shards\n'
         '        )\n'),
    ]
    prepared = []
    for relative, expected, anchor, addition in replacements:
        path = site / relative
        source = path.read_text()
        if sha256_text(source) != expected or source.count(anchor) != 1:
            raise ValueError(f"unexpected pinned loader source: {path}")
        changed = source.replace(anchor, anchor + addition if relative.endswith("weight_utils.py") else addition + anchor)
        compile(changed, str(path), "exec")
        prepared.append((path, source, changed))
    for path, source, changed in prepared:
        replace_text(path.with_suffix(".py.before_streaming"), source)
        replace_text(path, changed)
        event("source_patch", file=str(path), sha256=hashlib.sha256(changed.encode()).hexdigest())
