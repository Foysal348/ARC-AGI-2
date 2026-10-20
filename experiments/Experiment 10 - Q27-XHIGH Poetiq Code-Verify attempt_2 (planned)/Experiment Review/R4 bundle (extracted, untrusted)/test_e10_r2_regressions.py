"""Independent E10 R2 regression tests for verified adapter bugs; no GPU needed."""
import runpy, os, json, hashlib, ast
from pathlib import Path
here=Path(__file__).resolve().parent
core=runpy.run_path(str(here/'poetiq_core.py'))
task={'train':[{'input':[[0]],'output':[[1]]}],'test':[{'input':[[0]]}]}
for tag,program in (
 ('lossy_float','import numpy as np\ndef transform(grid):\n return np.array([[1.99]])\n'),
 ('lossy_string','def transform(grid):\n return [["1"]]\n'),
 ('bool_as_int','def transform(grid):\n return [[True]]\n'),
):
 old=core['poetiq_verify_code'](program,task,3)
 assert not old['passed'],(tag,old)
 assert old['test_results'][0]['grid'] is None,(tag,old)
 print('PASS invalid-grid no false train-pass:',tag)
program='import numpy as np\nprint("module-debug")\ndef transform(grid):\n print("transform-debug")\n return np.array([[1]])\n'
sc=core['poetiq_verify_code'](program,task,3)
assert sc['passed'] and sc['test_results'][0]['grid']==[[1]],sc
print('PASS generated program may print debugging output')
os.environ['E10_PRIVATE_TEST_SECRET']='top_secret_value'
exfil='import os\ndef transform(grid):\n return [[1 if "E10_PRIVATE_TEST_SECRET" in os.environ else 2]]\n'
result=core['poetiq_sandbox_run'](exfil,[[0]],3)
assert result[0] and json.loads(result[1])==[[2]],result
print('PASS model-generated subprocess has no inherited private environment')
notebook=json.loads((here/'Q27-XHIGH-v1_4_4_E10_86K_POETIQ_PROGRAM_SOLVER_R3.ipynb').read_text())
base=json.loads((here/'Q27-XHIGH-v1_4_4_E10_BASE_86K.ipynb').read_text())
for i in (1,3,6):assert notebook['cells'][i]['source']==base['cells'][i]['source']
for idx,c in enumerate(notebook['cells']):
 if c['cell_type']=='code':ast.parse(''.join(c['source']))
assert 'E10_POETIQ_MAX_ITERATIONS' in ''.join(notebook['cells'][2]['source'])
assert 'E10_POETIQ_EXPERTS' in ''.join(notebook['cells'][2]['source'])
assert '77824' in ''.join(notebook['cells'][2]['source'])
assert 'poetiq_sandbox_run' in ''.join(notebook['cells'][5]['source'])
assert "marker='__E10_PROGRAM_RESULT__'" in ''.join(notebook['cells'][5]['source'])
assert 'if poetiq_valid_grid(raw):' in ''.join(notebook['cells'][5]['source'])
try:
 import nbformat
 nbformat.validate(nbformat.read(str(here/'Q27-XHIGH-v1_4_4_E10_86K_POETIQ_PROGRAM_SOLVER_R3.ipynb'),as_version=4))
except ImportError: pass
print('PASS notebook structure, base/source parity, donor capability configuration')
print('ALL E10 R2 REGRESSIONS PASS')
