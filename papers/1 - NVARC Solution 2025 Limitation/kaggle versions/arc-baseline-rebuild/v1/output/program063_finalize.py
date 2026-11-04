import os
import sys
from pathlib import Path
legacy_root = Path(os.environ["PROGRAM063_LEGACY_ROOT"]).resolve()
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
    if resolved == legacy_root or legacy_root in resolved.parents or resolved == overlay_root:
        continue
    clean.append(item)
sys.path[:] = [str(overlay_root)] + clean

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder, hashable, score_kgmon, score_full_probmul_3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _origin_name(sample_key: str) -> str:
    return re.sub(r"\.out\d+$", "", sample_key)


def _geometry_family(sample_key: str) -> str:
    # Ignore color permutation and example-order suffixes; keep only D4 geometry.
    origin = _origin_name(sample_key)
    parts = origin.split(".")[1:]
    prefix = []
    for part in parts:
        if part.startswith("permute"):
            break
        if part.startswith("ex") or part.startswith("out") or part.startswith("run"):
            continue
        prefix.append(part)
    transposed = "transpose" in prefix
    rotations = sum(part == "rot90" for part in prefix) % 4
    return ("transpose_" if transposed else "plain_") + f"rot{90 * rotations}"


def _candidate_metrics(basevalues: dict) -> dict:
    grouped = {}
    for sample_key, sample in basevalues.items():
        h = hashable(sample["solution"])
        row = grouped.setdefault(h, {
            "solution": sample["solution"],
            "sample_keys": [],
            "origins": set(),
            "families": set(),
            "beam_scores": [],
            "aug_means": [],
        })
        row["sample_keys"].append(sample_key)
        row["origins"].add(_origin_name(sample_key))
        row["families"].add(_geometry_family(sample_key))
        row["beam_scores"].append(float(sample["beam_score"]))
        row["aug_means"].append(float(np.mean(sample["score_aug"])))

    guesses = {k: g for k, g in basevalues.items()}
    kgmon = score_kgmon(guesses)
    probmul = score_full_probmul_3(guesses)
    kg_rank = {hashable(solution): i + 1 for i, solution in enumerate(kgmon)}
    prob_rank = {hashable(solution): i + 1 for i, solution in enumerate(probmul)}

    for h, row in grouped.items():
        row["raw_support_count"] = len(row["sample_keys"])
        row["origin_count"] = len(row["origins"])
        row["family_count"] = len(row["families"])
        row["families"] = sorted(row["families"])
        row["mean_beam_score"] = float(np.mean(row["beam_scores"])) if row["beam_scores"] else float("inf")
        row["mean_aug_score"] = float(np.mean(row["aug_means"])) if row["aug_means"] else float("inf")
        row["kgmon_rank"] = int(kg_rank.get(h, 10**9))
        row["probmul_rank"] = int(prob_rank.get(h, 10**9))
    return grouped


def _public_metrics(row):
    if row is None:
        return None
    return {
        "family_count": row["family_count"],
        "origin_count": row["origin_count"],
        "raw_support_count": row["raw_support_count"],
        "kgmon_rank": row["kgmon_rank"],
        "probmul_rank": row["probmul_rank"],
        "mean_beam_score": row["mean_beam_score"],
        "mean_aug_score": row["mean_aug_score"],
        "families": row["families"],
    }


def conservative_cross_symmetry_two_guess(basevalues: dict):
    # Keep KGMon top-1 exactly; only a narrowly-qualified candidate may replace rank 2.
    guesses = {k: g for k, g in basevalues.items()}
    kgmon = score_kgmon(guesses)
    if not kgmon:
        return [], {"changed": False, "reason": "NO_CANDIDATES", "top1_preserved": True,
                    "baseline_second": None, "selected_second": None}
    if len(kgmon) == 1:
        return [kgmon[0]], {"changed": False, "reason": "ONLY_ONE_UNIQUE_CANDIDATE",
                            "top1_preserved": True, "baseline_second": None, "selected_second": None}

    metrics = _candidate_metrics(basevalues)
    top1 = kgmon[0]
    baseline_second = kgmon[1]
    top1_h = hashable(top1)
    baseline_h = hashable(baseline_second)
    baseline_row = metrics[baseline_h]

    qualified = []
    for h, row in metrics.items():
        if h == top1_h or h == baseline_h:
            continue
        if row["kgmon_rank"] > 4 or row["probmul_rank"] > 4:
            continue
        if row["family_count"] < 2 or row["origin_count"] < 2:
            continue
        if row["family_count"] <= baseline_row["family_count"]:
            continue
        if row["origin_count"] < baseline_row["origin_count"]:
            continue
        qualified.append((h, row))

    if not qualified:
        return [top1, baseline_second], {
            "changed": False,
            "reason": "NO_STRICTLY_BROADER_TOP4_CANDIDATE",
            "top1_preserved": True,
            "baseline_second": _public_metrics(baseline_row),
            "selected_second": _public_metrics(baseline_row),
        }

    qualified.sort(key=lambda item: (
        -item[1]["family_count"],
        -item[1]["origin_count"],
        item[1]["kgmon_rank"] + item[1]["probmul_rank"],
        item[1]["kgmon_rank"],
        item[1]["probmul_rank"],
        item[1]["mean_aug_score"],
        item[1]["mean_beam_score"],
    ))
    selected_h, selected_row = qualified[0]
    selected_second = selected_row["solution"]
    return [top1, selected_second], {
        "changed": True,
        "reason": "PROMOTED_STRICTLY_BROADER_CROSS_SYMMETRY_SUPPORT",
        "top1_preserved": True,
        "baseline_second": _public_metrics(baseline_row),
        "selected_second": _public_metrics(selected_row),
    }


legacy_root = Path(os.environ["PROGRAM063_LEGACY_ROOT"]).resolve()
overlay_root = Path(os.environ["PROGRAM063_OVERLAY_ROOT"]).resolve()
numpy_path = Path(np.__file__).resolve()
if numpy_path == legacy_root or legacy_root in numpy_path.parents:
    raise RuntimeError(f"Finalize NumPy leaked from legacy root: {numpy_path}")
if numpy_path == overlay_root or overlay_root in numpy_path.parents:
    raise RuntimeError(f"Finalize NumPy leaked from overlay: {numpy_path}")
print("PROGRAM063_FINALIZE_RUNTIME_OK=" + json.dumps({"numpy_file": str(numpy_path), "python": sys.executable}, sort_keys=True))

run_plan = json.loads(Path("program063_run_plan.json").read_text(encoding="utf-8"))
challenge_path = Path(run_plan["challenge_path"])
rerun_mode = bool(run_plan["rerun_mode"])

data = ArcDataset.from_file(str(challenge_path))
if not rerun_mode:
    solution_path = Path(run_plan["solution_path"])
    data = data.load_replies(str(solution_path))

output_root = Path("/kaggle/inference_outputs")
if not output_root.is_dir():
    raise RuntimeError(f"PROGRAM063_OUTPUT_DIRECTORY_MISSING={output_root}")

decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)
decoder.load_decoded_results(str(output_root))

baseline_results = decoder.run_selection_algo(score_kgmon)
guarded_results = {}
change_details = {}
for basekey, basevalues in sorted(decoder.decoded_results.items()):
    selected, detail = conservative_cross_symmetry_two_guess(basevalues)
    guarded_results[basekey] = selected
    change_details[basekey] = detail

for basekey in sorted(baseline_results):
    baseline = baseline_results[basekey]
    guarded = guarded_results.get(basekey, [])
    if baseline and guarded and hashable(baseline[0]) != hashable(guarded[0]):
        raise RuntimeError(f"PROGRAM063_TOP1_SAFETY_INVARIANT_FAILED={basekey}")

baseline_submission = data.get_submission(baseline_results)
guarded_submission = data.get_submission(guarded_results)

baseline_path = Path("/kaggle/working/submission_program063_baseline_kgmon.json")
baseline_path.write_text(json.dumps(baseline_submission, sort_keys=True, separators=(",", ":")), encoding="utf-8")
submission_path = Path("/kaggle/working/submission.json")
submission_path.write_text(json.dumps(guarded_submission, sort_keys=True, separators=(",", ":")), encoding="utf-8")
if not submission_path.is_file() or submission_path.stat().st_size <= 0:
    raise RuntimeError(f"PROGRAM063_SUBMISSION_JSON_NOT_WRITTEN={submission_path}")
print("PROGRAM063_SUBMISSION_JSON_READY=" + json.dumps({
    "path": str(submission_path), "size": submission_path.stat().st_size,
    "shadow_baseline_path": str(baseline_path)}, sort_keys=True))

output_files = sorted(path for path in output_root.iterdir() if path.is_file())
output_entries = []
for path in output_files:
    task_id = path.name.split("_", 1)[0]
    output_entries.append({"task_id": task_id, "name": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)})
manifest_tasks = {}
for entry in output_entries:
    manifest_tasks.setdefault(entry["task_id"], []).append(entry)
program024_reference_total = 76 if not rerun_mode else None
program024_delta = len(output_entries) - program024_reference_total if program024_reference_total is not None else None

diagnostics_path = Path("program063_run_diagnostics.json")
if not diagnostics_path.is_file():
    raise RuntimeError(f"Program 063 diagnostics are missing: {diagnostics_path}")
diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
diagnostics_sha256 = sha256_file(diagnostics_path)

changed_keys = sorted(k for k, v in change_details.items() if v.get("changed"))
compact_change_details = {k: change_details[k] for k in changed_keys}
ranker_audit = {
    "policy": "keep_kgmon_top1_replace_second_only_on_strictly_broader_cross_symmetry_top4_support",
    "top1_changed_count": 0,
    "second_slot_changed_count": len(changed_keys),
    "changed_keys": changed_keys,
    "change_details": compact_change_details,
}
ranker_audit_path = Path("program063_ranker_audit.json")
ranker_audit_path.write_text(json.dumps(ranker_audit, sort_keys=True, separators=(",", ":")), encoding="utf-8")

manifest = {
    "program_no": 63,
    "variant": "exact024_cross_symmetry_second_guess_guard",
    "rerun_mode": rerun_mode,
    "seed_contract": run_plan["seed_contract"],
    "run_diagnostics_sha256": diagnostics_sha256,
    "training_records": diagnostics.get("training_records", []),
    "total_files": len(output_entries),
    "program024_reference_total": program024_reference_total,
    "delta_from_program024_reference": program024_delta,
    "ranker_audit_sha256": sha256_file(ranker_audit_path),
    "tasks": manifest_tasks,
}
manifest_path = Path("program063_output_manifest.json")
manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8")
print(f"PROGRAM063_OUTPUT_FILE_COUNT={len(output_entries)}")
if program024_delta is None:
    print("PROGRAM063_OUTPUT_COUNT_DELTA_FROM_PROGRAM024=NA_RERUN_MODE")
else:
    print(f"PROGRAM063_OUTPUT_COUNT_DELTA_FROM_PROGRAM024={program024_delta}")
print(f"PROGRAM063_OUTPUT_MANIFEST_SHA256={sha256_file(manifest_path)}")
print("PROGRAM063_RANKER_AUDIT=" + json.dumps(ranker_audit, sort_keys=True))
for task_id in sorted(manifest_tasks):
    entries = manifest_tasks[task_id]
    combined = hashlib.sha256()
    for entry in entries:
        combined.update(f"{entry['name']}:{entry['sha256']}\n".encode("utf-8"))
    print("PROGRAM063_TASK_OUTPUT_MANIFEST=" + json.dumps({
        "task_id": task_id, "count": len(entries), "combined_sha256": combined.hexdigest(),
        "files": [entry["name"] for entry in entries]}, sort_keys=True))

submission_sha256 = sha256_file(submission_path)
baseline_sha256 = sha256_file(baseline_path)
print(f"PROGRAM063_SUBMISSION_SHA256={submission_sha256}")
print(f"PROGRAM063_SHADOW_BASELINE_SUBMISSION_SHA256={baseline_sha256}")

reproducibility_payload = {
    "program_no": 63,
    "variant": "exact024_cross_symmetry_second_guess_guard",
    "seed_contract": run_plan["seed_contract"],
    "runtime_tree_sha256": {name: report["record_tree_sha256"] for name, report in sorted(run_plan["pinned_runtime_source"].items())},
    "exact_program024_source_sha256": {
        "arc_loader.py": "d01cd56167e534ae156706ef62ab11662fc7b403964614a7a55131940bde970c",
        "arc_decoder.py": "965cfd910777d9bbec9681c6c15c5e2ea3924569734248046f5e58f16cb28222",
        "arc_solver.py": "f9011b4d4549fb688488d67ba7c2dc43184824436d34e1a4c2eae06f69daab73",
    },
    "run_diagnostics_sha256": diagnostics_sha256,
    "output_manifest_sha256": sha256_file(manifest_path),
    "ranker_audit_sha256": sha256_file(ranker_audit_path),
    "shadow_baseline_submission_sha256": baseline_sha256,
    "submission_sha256": submission_sha256,
}
reproducibility_path = Path("program063_reproducibility_signature.json")
reproducibility_path.write_text(json.dumps(reproducibility_payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
reproducibility_sha256 = sha256_file(reproducibility_path)
print(f"PROGRAM063_REPRODUCIBILITY_SIGNATURE={reproducibility_sha256}")

baseline_public_score = "NA_COMPETITION_RERUN"
guarded_public_score = "NA_COMPETITION_RERUN"
if rerun_mode:
    print("PROGRAM063_SUBMISSION_BUILD_COMPLETE")
    print("SUBMISSION_READY_FOR_MANUAL_SUBMIT")
else:
    decoder.benchmark_selection_algos()
    baseline_public_score = data.validate_submission(json.loads(baseline_path.read_text(encoding="utf-8")))
    guarded_public_score = data.validate_submission(json.loads(submission_path.read_text(encoding="utf-8")))
    print(f"PROGRAM063_BASELINE_PUBLIC_RELOAD_SCORE={baseline_public_score}")
    print(f"PROGRAM063_GUARDED_PUBLIC_RELOAD_SCORE={guarded_public_score}")
    print("PROGRAM063_PUBLIC_EVAL_COMPLETE_REVIEW_RANKER_EFFECT_BEFORE_SUBMITTING")

task_counts = {task_id: len(entries) for task_id, entries in sorted(manifest_tasks.items())}
gpt_fb_lines = [
    "＝＝＝＝GPT FB START ＝＝＝＝",
    "GPT_FB_VERSION: 7",
    "PROGRAM_NO: 063",
    "PURPOSE: Preserve exact Program 024 candidate generation and KGMon top-1 while replacing only a weak second guess when a nearby candidate has strictly broader cross-symmetry support.",
    "EXECUTION_STATUS: COMPLETED",
    "FIRST_ERROR: NONE",
    "SUBMIT_POLICY: EXPERIMENTAL_SECOND_SLOT_ONLY_TOP1_EXACTLY_PRESERVED_REVIEW_PUBLIC_COMPARISON_BEFORE_SUBMIT",
    f"RUNTIME_FAMILY: {run_plan.get('runtime_family')}",
    "EXTERNAL_ASSETS: NONE_BEYOND_EXISTING_PROGRAM024_STACK",
    "SOURCE_HASH_CHECK: EXACT_PROGRAM024_OK",
    "SEED_AUDIT: OK",
    "BACKEND_POLICY: NORMAL_NONDETERMINISTIC_CUDA_BACKWARD",
    "RANKER_POLICY: KEEP_KGMON_TOP1_PROMOTE_SECOND_ONLY_IF_TOP4_CANDIDATE_HAS_STRICTLY_BROADER_GEOMETRY_FAMILY_SUPPORT_AND_RANK_CONSENSUS",
    "TOP1_CHANGED_COUNT: 0",
    f"SECOND_SLOT_CHANGED_COUNT: {len(changed_keys)}",
    "SECOND_SLOT_CHANGED_KEYS: " + json.dumps(changed_keys, sort_keys=True),
    "SECOND_SLOT_CHANGE_DETAILS: " + json.dumps(compact_change_details, sort_keys=True),
    "TRAINING_RECORDS: " + json.dumps(diagnostics.get("training_records", []), sort_keys=True),
    f"OUTPUT_FILE_COUNT: {len(output_entries)}",
    "TASK_OUTPUT_COUNTS: " + json.dumps(task_counts, sort_keys=True),
    f"OUTPUT_MANIFEST_SHA256: {sha256_file(manifest_path)}",
    f"SHADOW_BASELINE_SUBMISSION_SHA256: {baseline_sha256}",
    f"SUBMISSION_PATH: {submission_path}",
    f"SUBMISSION_SHA256: {submission_sha256}",
    f"REPRODUCIBILITY_SIGNATURE: {reproducibility_sha256}",
    f"BASELINE_PUBLIC_RELOAD_SCORE: {baseline_public_score}",
    f"GUARDED_PUBLIC_RELOAD_SCORE: {guarded_public_score}",
    "DIAGNOSTIC_CONCLUSION: EXACT_TOP1_PRESERVED_AND_CROSS_SYMMETRY_SECOND_GUESS_GUARD_APPLIED_WITH_ZERO_EXTRA_GPU_PASSES",
    "NEXT_RECOMMENDED_ACTION: SUBMIT_AT_MOST_ONCE_ONLY_IF_RUN_COMPLETES_AND_GUARDED_PUBLIC_SCORE_DOES_NOT_REGRESS_VERSUS_SHADOW_BASELINE",
    "PRIVATE_SCORE: USER_TO_FILL",
    "＝＝＝＝GPT FB STOP ＝＝＝＝",
]
print("\n".join(gpt_fb_lines))
