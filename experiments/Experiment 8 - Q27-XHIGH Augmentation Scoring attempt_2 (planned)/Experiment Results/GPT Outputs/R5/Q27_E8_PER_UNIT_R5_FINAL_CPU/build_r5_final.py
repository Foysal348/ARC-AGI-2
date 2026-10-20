from pathlib import Path
import ast,json,nbformat,hashlib,difflib
D=Path(__file__).resolve().parent
r4=D/'Q27-XHIGH-v1_4_4_E8_PER_UNIT_R4.ipynb'
# R4 builder was run locally against the modified e8_per_unit_core.py;
# reconstruct from the original R4 instead to avoid accidental changes.
original=nbformat.read(D/'sources/Q27-XHIGH-v1_4_4_E8_PER_UNIT_R4.ipynb',as_version=4)
base=nbformat.read(D/'sources/Q27-XHIGH-v1_4_4_E7_ORDERING_ONLY.ipynb',as_version=4)
n=nbformat.v4.new_notebook(metadata=json.loads(json.dumps(original.metadata)))
n.cells=list(original.cells)
assert len(n.cells)==12
n.cells[7].source=n.cells[7].source.replace('## Experiment 8','## Experiment 8 R5 — CPU-hardened')
n.cells[7].source+='\n\n**R5 CPU updates:** context-aware chronological candidate parsing, deterministic fallback protection, primary-first admission, complete cancellation counts, fail-closed token alignment, and final EVAL report consistency. Original E7 deadlines unchanged. GPU risk remains untested.\n'
n.cells[9].source=(D/'e8_per_unit_core.py').read_text()
sched=original.cells[10].source
a='''        # Score one queued view at a time, using the SAME engine and the SAME
        # global concurrency cap. A score request temporarily occupies a slot.
        # Original E7 admission/drain/hard-stop timestamps are untouched.
        if E8_ENABLED and (E8_JOBS or E8_CURRENT is not None):
            e8_try_admit(len(active), now)
        while queue and len(active) + int(E8_ACTIVE is not None) < CONCURRENCY and now < ADMIT_DEADLINE and not drained:
            admit(queue.pop(0)); active = [s for s in STATE.values() if s["phase"] in (1, 2)]'''
b='''        # R5 primary-FIRST: admit waiting E7 requests before offering spare
        # capacity to the E8 scorer. Model cost remains shared.
        if E8_ENABLED and (E8_JOBS or E8_CURRENT is not None) and queue and now < ADMIT_DEADLINE and not drained:
            SCHED['e8_primary_waiting_steps'] = SCHED.get('e8_primary_waiting_steps', 0) + 1
        while queue and len(active) + int(E8_ACTIVE is not None) < CONCURRENCY and now < ADMIT_DEADLINE and not drained:
            admit(queue.pop(0)); active = [s for s in STATE.values() if s["phase"] in (1, 2)]
        if E8_ENABLED and (E8_JOBS or E8_CURRENT is not None):
            e8_try_admit(len(active), now)'''
assert sched.count(a)==1
sched=sched.replace(a,b)
n.cells[10].source=sched
report=original.cells[11].source
block='''    try: write_json_atomic(OUT/'e8_per_unit_report.json',E8_REPORT)
    except Exception as exc: print('E8 auxiliary report save failed:',repr(exc)[:300])'''
assert report.count(block)==1
report=report.replace(block,'    # E8 auxiliary report is written after final EVAL metrics are available.')
start=report.index('    if SOLUTIONS is not None:\n        # Validate actual final EVAL generated pairs')
end=report.index('    rows = RECORDS',start)
evalblock=report[start:end]
report=report[:start]+report[end:]
pos=report.index('    rows = RECORDS')
report=report[:pos]+evalblock+block+'\n'+report[pos:]
needle='    write_json_atomic(OUT / "run_report.json", report)'
assert report.count(needle)==1
report=report.replace(needle,"    report['e8'] = {k:v for k,v in E8_REPORT.items() if k!='rows'}\n"+needle)
n.cells[11].source=report
n.metadata.setdefault('e8',{}).update({'revision':'R5 CPU hardened','gpu_tested':False,'source':'E7 + Codex donor','primary_deadlines_unchanged':True})
nbformat.validate(n)
for c in n.cells:
 if c.cell_type=='code':ast.parse(c.source)
assert all(n.cells[i].source==base.cells[i].source for i in range(7))
out=D/'Q27-XHIGH-v1_4_4_E8_PER_UNIT_R5_FINAL_CPU.ipynb'
nbformat.write(n,out)
(D/'E7_SCHEDULER_TO_R5.patch').write_text(''.join(difflib.unified_diff(base.cells[7].source.splitlines(True),sched.splitlines(True),fromfile='E7/scheduler',tofile='E8_R5/scheduler')))
(D/'R4_TO_R5_HELPERS.patch').write_text(''.join(difflib.unified_diff(original.cells[9].source.splitlines(True),n.cells[9].source.splitlines(True),fromfile='E8_R4/helpers',tofile='E8_R5/helpers')))
(D/'R4_TO_R5_REPORT.patch').write_text(''.join(difflib.unified_diff(original.cells[11].source.splitlines(True),report.splitlines(True),fromfile='E8_R4/report',tofile='E8_R5/report')))
manifest={'notebook':out.name,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'base_cells_0_to_6_same':True,'donor_vendored_cell_same':n.cells[8].source==original.cells[8].source,'model_config_same':n.cells[6].source==base.cells[6].source,'gpu_run':False}
(D/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
print('BUILD OK',out, 'bytes',out.stat().st_size)
