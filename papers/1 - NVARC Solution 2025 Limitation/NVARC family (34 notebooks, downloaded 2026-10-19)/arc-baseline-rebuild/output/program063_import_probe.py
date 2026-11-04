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

import importlib
import json
from pathlib import Path

# Match production import order: native Torch/Triton first, then Unsloth before
# Transformers/PEFT/TRL so Unsloth can apply its patches deterministically.
modules = {}
for name in ["torch", "triton"]:
    module = importlib.import_module(name)
    modules[name] = {
        "version": getattr(module, "__version__", "unknown"),
        "file": str(Path(module.__file__).resolve()) if getattr(module, "__file__", None) else None,
    }
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
import unsloth
for name in ["transformers", "datasets", "peft", "accelerate", "bitsandbytes", "trl"]:
    module = importlib.import_module(name)
    modules[name] = {
        "version": getattr(module, "__version__", "unknown"),
        "file": str(Path(module.__file__).resolve()) if getattr(module, "__file__", None) else None,
    }
import torch
source_root = Path(os.environ["PROGRAM063_LEGACY_ROOT"]).resolve()
overlay_root = Path(os.environ["PROGRAM063_OVERLAY_ROOT"]).resolve()
for anchor in ["torch", "triton"]:
    anchor_path = Path(modules[anchor]["file"]).resolve()
    if anchor_path == source_root or source_root in anchor_path.parents or anchor_path == overlay_root or overlay_root in anchor_path.parents:
        raise RuntimeError(f"{anchor} was imported from a legacy overlay: {anchor_path}")
visible = torch.cuda.device_count()
if visible < 4:
    raise RuntimeError(f"Expected four visible GPUs, found {visible}")
optional = {}
for name in ["flash_attn", "xformers", "cut_cross_entropy", "hf_transfer", "msgspec", "sentencepiece"]:
    try:
        module = importlib.import_module(name)
        optional[name] = {"ok": True, "version": getattr(module, "__version__", "unknown"), "file": str(Path(module.__file__).resolve()) if getattr(module, "__file__", None) else None}
    except Exception as exc:
        optional[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
print("PROGRAM063_IMPORT_PROBE=" + json.dumps({
    "python": str(Path(sys.executable).resolve()),
    "python_version": sys.version.split()[0],
    "modules": modules,
    "unsloth_version": getattr(unsloth, "__version__", "unknown"),
    "unsloth_file": str(Path(unsloth.__file__).resolve()),
    "optional": optional,
    "cuda_version": torch.version.cuda,
    "device_count": visible,
    "device_names": [torch.cuda.get_device_name(i) for i in range(visible)],
    "required_symbols": {
        "FastLanguageModel": FastLanguageModel is not None,
        "UnslothTrainer": UnslothTrainer is not None,
        "UnslothTrainingArguments": UnslothTrainingArguments is not None,
    },
}, sort_keys=True))
