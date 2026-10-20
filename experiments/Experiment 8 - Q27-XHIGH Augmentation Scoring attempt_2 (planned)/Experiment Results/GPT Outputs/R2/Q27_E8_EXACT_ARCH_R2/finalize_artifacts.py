from pathlib import Path
import json, hashlib, difflib, ast, zipfile, shutil, nbformat
R=Path('/mnt/data')
D=R/'Q27_E8_EXACT_ARCH_R2'
P=D/'Q27-XHIGH-v1_4_4_E8_EXACT_ARCH_FIXED_R2.ipynb'
BASE=R/'Q27-XHIGH-v1_4_4_E7_ORDERING_ONLY.ipynb'
PREV=R/'Q27_E8_EXACT_ARCH/Q27-XHIGH-v1_4_4_E8_EXACT_ARCH.ipynb'
DONOR=R/'arc-2026d-with-codex (24)(1).ipynb'
nb=nbformat.read(P,as_version=4)
base=nbformat.read(BASE,as_version=4)
prev=nbformat.read(PREV,as_version=4)
assert len(nb.cells)==12
assert all(nb.cells[i].source==base.cells[i].source for i in [0,1,3,4,5,6,7])
assert nb.cells[2].source!=base.cells[2].source
assert nb.cells[9].source==prev.cells[9].source

# Diff for configuration & report and E8 helper
for title,a,b in [
 ('BASE_E7_CELL1_TO_E8_CONFIG.patch',base.cells[2].source,nb.cells[2].source),
 ('E8_PREVIOUS_TO_FIXED_HELPERS.patch',prev.cells[10].source,nb.cells[10].source),
 ('BASE_E7_REPORT_TO_E8_FINAL.patch',base.cells[8].source,nb.cells[11].source),
]:
    diff=''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile=title+':original',tofile=title+':updated'))
    (D/title).write_text(diff)

d=json.load(open(DONOR))
donor_source=''.join(d['cells'][3]['source']).split('\n',1)[1]
loader_source=nb.cells[9].source
(D/'DONOR_LOADER_DIFF.patch').write_text(''.join(difflib.unified_diff(donor_source.splitlines(True),loader_source.splitlines(True),fromfile='Codex arc_loader.py',tofile='E8 donor loader')))

def dump_ast(tree,name):
    node=next(n for n in ast.walk(tree) if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
    return ast.dump(node, include_attributes=False)
donor_ast=ast.parse(donor_source);vendor_ast=ast.parse(loader_source)
for name in ['ArcDataset','augment','mod','mod_single','shuffle_ex','permute_mod','permute_rnd_all_']:
    assert dump_ast(donor_ast,name)==dump_ast(vendor_ast,name),name

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={
 'updated_notebook':P.name,
 'updated_sha256':sha(P),
 'base_sha256':sha(BASE),
 'donor_sha256':sha(DONOR),
 'base_primary_source_unchanged_cells':[0,1,3,4,5,6,7],
 'base_configuration_cell_changed':2,
 'base_report_cell_replaced':8,
 'donor_ArcDataset_exact_AST':True,
 'donor_loader_semantic_changes':['remove unused AutoTokenizer import and type annotation'],
 'E8_POLICY':'apply',
 'E8_RESERVED_SEC_default':1800,
 'E8_FINALIZE_SEC_default':120,
 'effective_scoring_window_sec':1680,
 'scoring_uses_vllm_prompt_logprobs':'GPU_UNVALIDATED',
 'CPU_test_log':'test_output.txt',
}
(D/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))

(D/'SOURCE_COMPARISON.md').write_text('''# E8 R2 — Source comparison and independent re-audit

## Updated vs frozen E7 Ordering Only

- **Identical source:** notebook cells 0, 1, 3, 4, 5, 6, and 7 (intro plus all E7 primary executable cells other than configuration).
- **Modified Cell 2 (configuration only):** primary deadline moves earlier to make an explicit 1,800-second E8 reserve. The 120-second finalisation margin leaves a 1,680-second post-primary scoring window by default. Primary model, prompt, ordering, sampling, 49,152 thinking, 8,192 answer, seeds, engine recipe, and primary scheduler logic are unchanged.
- **Original E7 report Cell 8 is replaced by Cell 11:** Stage 2 is executed before final validation, final scoring, reporting and ZIP. The EVAL diagnostic score is recalculated from the final selected pair; the baseline score is retained separately.
- **Added Cell 9:** Codex vendor loader. **Added Cell 10:** candidate builder, 8-view augmentation, teacher-forced Q27 scoring, selection and checkpoint/transaction logic.
- **Original base notebook is not modified**; the new notebook is a separate artifact.

## Updated vs Codex donor

- Donor `ArcDataset` class, `augment`, `mod`, `mod_single`, `shuffle_ex`, `permute_mod`, `permute_rnd_all_` have **identical AST** to the donor `arc_loader.py` code extracted from notebook Cell 3.
- Only loader changes: remove unused `AutoTokenizer` import and type annotation from `QwenFormatter.__init__`.
- Donor's Torch `calc_scores` is **not copied directly**: model logits/forward implementation is adapted to Q27's existing vLLM engine using `SamplingParams(prompt_logprobs=1)` and chosen-token NLL over precisely tracked answer token IDs.
- Donor per-puzzle LoRA/DFS candidate generation and Probmul/Kgmon ranking **are not part of this experiment**. E8 uses train-consistent deterministic candidates and independent normalized-NLL / raw-NLL ranking.

## F1–F7 fixes and validations

| ID | Fix | Verified CPU test |
|---|---|---|
| F1 | Proper `len(challenges[tid]['test'])` and strict task schema | 1,2,3-query valid/invalid submission |
| F2 | Retain all train-supported candidates before optional legacy fallback; no truncation of supported hypotheses | synthetic 90-degree rotation |
| F3 | Reserve E8 window by moving E7 primary-only deadlines earlier; Stage 2 uses `E8_STAGE_DEADLINE` | 3-hour and hidden 12-hour budget monotonicity |
| F4 | EVAL generated predictions and exact post-E8 score, baseline/net gain fields | controlled wrong/correct grid transition and integrated run |
| F5 | Explicit no-extra-special-token encoding, suffix/answer index parity, token-boundary diagnostics | chosen-token logprob and fake-engine batch |
| F6 | Cleanup includes failed request ID even after partial registration, abort failures escalate; checkpoint rollback | injected add/abort/submission-write failures |
| F7 | E8 diagnostic report write isolated from final submission | injected report I/O failure |

## Critical remaining evidence limitations

- The notebook has **not** been run on Kaggle 4×L4; Q27/vLLM 0.19.0/MTP3 `prompt_logprobs` compatibility, throughput and real GPU KV pressure remain unverified.
- The **1,800-second time allocation is an explicit experiment setting**, not a measured optimum, and changes primary coverage/deadlines in this E8 copy. This must be reported in comparisons with E7.
- Ranking may lose correct baseline fallback hits; the selection is experimentally active as requested. A positive net pass@2 benefit is **not** assumed.
- A huge number of augmented candidate views may exceed the allocated stage window; partial scoring does not invalidate the baseline checkpoint.
''')

(D/'README.md').write_text('''# Q27-XHIGH E8 — Exact Architecture FIXED R2

**Notebook:** `Q27-XHIGH-v1_4_4_E8_EXACT_ARCH_FIXED_R2.ipynb`

## Approved architecture
Stage 1 Frozen E7 Ordering Only → baseline attempts → **validated baseline submission checkpoint** → train-consistent candidate pool → exact Codex 8-view augmentation → Q27 vLLM teacher-forced NLL scorer → protected attempt_2 selector (**apply mode is ON**) → final validation → submission → report/bundle.

The original E7 and Codex donor files are intact. All E7 primary code cells remain identical except **Cell 1 time-deadline configuration**; the original report cell is replaced to run E8 **before** final output/report/bundle.

## Time allocation
`E8_RESERVED_SEC=1800` (30 min) before the original E7 hard stop; `E8_FINALIZE_SEC=120` leaves ~28 min of post-primary scoring time. Values are configurable using `Q27X_E8_RESERVED_SEC` and `Q27X_E8_FINALIZE_SEC` **before notebook execution**. This allocation is experimental and not GPU-calibrated; importantly the E7 primary scheduler ends earlier by design. No additional hidden opt-in is required: E8 applies when it finds valid scored candidates and retains the baseline on failure.

## Audit and tests
Run `python test_e8_r2.py` in an environment with NumPy/nbformat installed. The test script explicitly checks source-parity, F1-F7 regressions, scoring fake engine, submission atomic rollback, and deadline ordering. No Kaggle GPU execution has occurred in this audit.

## Files
- `SOURCE_COMPARISON.md` and `SOURCE_MANIFEST.json` — two-way source comparison.
- `BASE_E7_CELL1_TO_E8_CONFIG.patch`, `BASE_E7_REPORT_TO_E8_FINAL.patch`, `E8_PREVIOUS_TO_FIXED_HELPERS.patch`, `DONOR_LOADER_DIFF.patch` — exact diffs.
- `test_output.txt` — CPU/fake-engine results.
- `patch_e8_r2.py` — reproducible patch builder applied to previous E8 notebook.
''')

# Reproducibility and actual source files
(D/'patch_e8_r2.py').write_bytes((R/'Q27_E8_EXACT_ARCH/patch_e8_r2.py').read_bytes())
for source,out in [(BASE,'ORIGINAL_E7_ORDERING_ONLY.ipynb'),(DONOR,'ORIGINAL_CODEX_DONOR.ipynb')]:
    shutil.copy2(source,D/out)

# Keep pre-audit original E8 for reproduction of patch build
shutil.copy2(PREV,D/'ORIGINAL_E8_EXACT_ARCH_PRE_FIX.ipynb')

ZIP=R/'Q27_E8_EXACT_ARCH_FIXED_R2_CPU_AUDITED_2026-10-13.zip'
with zipfile.ZipFile(ZIP,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(D.iterdir()):
        if p.is_file() and p.name!='finalize_artifacts.py':
            z.write(p,arcname='Q27_E8_EXACT_ARCH_FIXED_R2/'+p.name)
with zipfile.ZipFile(ZIP) as z:
    assert z.testzip() is None
print('created',P, P.stat().st_size, 'bytes')
print('zip',ZIP, ZIP.stat().st_size, 'bytes')
print('sha256',manifest['updated_sha256'])
print('count_files',len(list(D.glob('*'))))
