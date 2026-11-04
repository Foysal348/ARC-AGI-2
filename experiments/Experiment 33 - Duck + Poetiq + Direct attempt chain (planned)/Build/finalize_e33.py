"""Reproducible local release from the selected, MTP-fixed ZIP base. Never pushes."""
import ast
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import build_e33 as B

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GPT = ROOT / 'GPT'
ARCHIVE = GPT / 'E33_CHAIN_SMOKE_AND_PRODUCTION_NOTEBOOKS.zip'
BASE = GPT / 'E33_CHAIN_SMOKE_AND_PRODUCTION_NOTEBOOKS.original.zip'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if not BASE.exists():
        shutil.copy2(ARCHIVE, BASE)
    with zipfile.ZipFile(BASE) as archive:
        nb = json.loads(archive.read('E33_CHAIN_PRODUCTION_COMPLETED.ipynb'))
    payload = {name: data for name, data in B.PAYLOAD.values()}
    assert all(isinstance(data, bytes) for data in payload.values())
    pins = {key: [name, digest(data)] for key, (name, data) in B.PAYLOAD.items()}
    cell = ''.join(nb['cells'][2]['source'])
    tree = ast.parse(cell)
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'E33D_FILES' for t in node.targets))
    lines = cell.splitlines(True)
    lines[assignment.lineno - 1:assignment.end_lineno] = ['E33D_FILES = ' + json.dumps(pins, sort_keys=True) + '\n']
    cell = ''.join(lines)
    needle = '    E33_CHAIN = E33L.check_chain(E33_CHAIN, os.environ.get("E33_CHAIN_JSON") if os.environ.get("E11_HARNESS") == "1" else None)\n'
    assert cell.count(needle) == 1
    cell = 'E33_A1_ORDER = "e28"  # fixed design: train pass, best train, Direct\n' + cell.replace(
        needle, needle + '    E33_CHAIN["a1_order"] = E33_A1_ORDER\n')
    nb['cells'][2]['source'] = cell.splitlines(True)
    facts = {'base_sha256': digest(BASE.read_bytes()), 'answer_order': 'e28', 'payload': pins,
             'gpu_run': False, 'artifacts': {}}
    outputs = {}
    for smoke, name in [(False, 'E33_CHAIN_PRODUCTION_2026-11-02.ipynb'),
                        (True, 'E33_CHAIN_150MIN_SMOKE_2026-11-02.ipynb')]:
        out = json.loads(json.dumps(nb))
        if smoke:
            source = ''.join(out['cells'][2]['source'])
            assert source.count('E29_PROFILE = "E28EVAL120"') == 1
            out['cells'][2]['source'] = source.replace('E29_PROFILE = "E28EVAL120"', 'E29_PROFILE = "E29SMOKE"').splitlines(True)
        for c in out['cells']:
            if c['cell_type'] == 'code':
                ast.parse(''.join(c['source']))
        data = json.dumps(out, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        assert len(data) < 400000, len(data)
        target = GPT / name
        backup = target.with_suffix('.original.ipynb')
        if target.exists() and not backup.exists():
            shutil.copy2(target, backup)
        target.write_bytes(data)
        outputs['E33_CHAIN_SMOKE_COMPLETED.ipynb' if smoke else 'E33_CHAIN_PRODUCTION_COMPLETED.ipynb'] = data
        B.write_push(B.SLUG + ('-smoke' if smoke else ''), B.TITLE + (' Smoke' if smoke else ''), data)
        if smoke:
            B.write_push(B.SLUG + '-submit', B.TITLE + ' Submit', data)
        facts['artifacts'][name] = {'sha256': digest(data), 'bytes': len(data)}
        canonical = ROOT / 'Experiment Notebooks' / ('Q27-XHIGH-E33_CHAIN_SMOKE.ipynb' if smoke else 'Q27-XHIGH-E33_CHAIN.ipynb')
        canonical.write_bytes(data)
    ds = B.DS_DIR
    ds.mkdir(parents=True, exist_ok=True)
    for name, data in payload.items():
        (ds / name).write_bytes(data)
    (HERE / 'e33_nbx.py').write_bytes(payload['e33_nbx.py'])
    (ds / 'SHA256SUMS').write_text(''.join(f'{digest(data)}  {name}\n' for name, data in sorted(payload.items())), encoding='utf-8')
    (ds / 'dataset-metadata.json').write_text(json.dumps({'title': B.DS_SLUG, 'id': B.DS_ID,
        'licenses': [{'name': 'CC0-1.0'}], 'isPrivate': True}, indent=2), encoding='utf-8')
    with zipfile.ZipFile(ARCHIVE, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in outputs.items():
            archive.writestr(name, data)
        for name, data in payload.items():
            archive.writestr('payload/' + name, data)
        archive.writestr('payload/dataset-metadata.json', (ds / 'dataset-metadata.json').read_bytes())
        archive.writestr('payload/SHA256SUMS', (ds / 'SHA256SUMS').read_bytes())
    (HERE / 'e33_release_facts.json').write_text(json.dumps(facts, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(facts, indent=2))


if __name__ == '__main__':
    main()
