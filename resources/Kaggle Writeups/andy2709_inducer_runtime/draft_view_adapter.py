"""Metadata-only INT4 draft view from Daniel Franzen's pinned solution.

Source: https://github.com/da-fr/arc-agi-3-solution
Retained byte-for-byte below from the validated v29 payload; model tensors
are referenced through symlinks and are never rewritten.
"""
import copy, hashlib, json
from pathlib import Path

EXPERT_TARGET = r"re:^mtp\.layers\.0\.mlp\.experts\.[0-9]+\.(gate_proj|up_proj|down_proj)$"

DENSE_IGNORE = r"re:^(?!mtp\.layers\.0\.mlp\.experts(?:\.|$)).*"

def indexed_shards(root):
    root = Path(root)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()))
    if not names: raise RuntimeError(f"Empty weight index: {root}")
    for name in names:
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"Invalid shard path: {name}")
        p = root / rel
        if not p.is_file() or p.stat().st_size < 8:
            raise RuntimeError(f"Missing/empty shard: {p}")
    return index, names

def prepare_draft_view(target, source_root, work_root):
    """Adapt metadata only; link original dataset weights without copying them."""
    target, source_root, work_root = map(Path, (target, source_root, work_root))
    configs = [source_root / "config.json"] if (source_root / "config.json").is_file() else []
    configs += sorted(p for p in source_root.rglob("config.json") if p not in configs)
    candidates = []
    for path in configs:
        cfg = json.loads(path.read_text())
        q = cfg.get("quantization_config", {})
        if q.get("quant_method") == "compressed-tensors" and "mtp_routed_experts" in q.get("config_groups", {}):
            candidates.append((path.parent, cfg))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one INT4 g32 MTP checkpoint under DRAFT_MODEL_DIR={source_root}; found {len(candidates)}")
    source, cfg = candidates[0]
    index, shards = indexed_shards(source)
    if not any(k.startswith("mtp.layers.0.mlp.experts.") for k in index["weight_map"]):
        raise RuntimeError("Draft index does not contain the expected MTP experts")
    cfg = copy.deepcopy(cfg)
    q = cfg["quantization_config"]
    group = q["config_groups"]["mtp_routed_experts"]
    w = group["weights"]
    if (w["num_bits"], w["group_size"], w["symmetric"]) != (4, 32, True):
        raise RuntimeError("Expected symmetric INT4 group32 draft experts")
    if group["targets"] not in (["RoutedExperts"], [EXPERT_TARGET]):
        raise RuntimeError(f"Unexpected draft target rules: {group['targets']}")
    if q.get("ignore", []) not in ([], [DENSE_IGNORE]):
        raise RuntimeError("Unexpected draft ignore rules")
    group["targets"] = [EXPERT_TARGET]
    q["ignore"] = [DENSE_IGNORE]
    # Content/path-specific directory makes notebook reruns idempotent without
    # overwriting an unrelated view. We never modify the read-only datasets.
    identity = json.dumps({"source": str(source.resolve()), "target": str(target.resolve()), "config": cfg}, sort_keys=True)
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:16]
    view = work_root / f"draft-view-{suffix}"
    view.mkdir(parents=True, exist_ok=True)
    config_path = view / "config.json"
    config_text = json.dumps(cfg, indent=2) + "\n"
    if config_path.exists() or config_path.is_symlink():
        if config_path.is_symlink() or config_path.read_text() != config_text:
            raise RuntimeError(f"Unexpected existing draft-view config: {config_path}")
    else:
        config_path.write_text(config_text)
    links = {name: source / name for name in shards}
    links["model.safetensors.index.json"] = source / "model.safetensors.index.json"
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json", "generation_config.json"):
        if (target / name).is_file(): links[name] = target / name
    for name, src in links.items():
        dest = view / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            if not dest.is_symlink() or dest.resolve() != src.resolve():
                raise RuntimeError(f"Unexpected existing draft-view file: {dest}")
        else:
            dest.symlink_to(src.resolve())
    print(f"Draft source: {source}\nSGLang draft view: {view} ({len(shards)} linked weight shards)")
    return view
