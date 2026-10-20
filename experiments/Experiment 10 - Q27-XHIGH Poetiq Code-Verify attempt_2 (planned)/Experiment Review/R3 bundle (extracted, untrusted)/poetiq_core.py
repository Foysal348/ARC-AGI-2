# E10 Poetiq donor adaptation: task formatting, code extraction, sandbox, feedback, voting.
# Donor: https://github.com/poetiq-ai/poetiq-arc-agi-solver (MIT)
# Donor originals: arc_agi/solve_coding.py, solve_parallel_coding.py, sandbox.py, prompts.py
import numpy as np
import json, re, sys, subprocess, tempfile, textwrap, math, time, os, string
from pathlib import Path

# Donor prompt variants adapted solely to the local Q27 chat interface.
# The task wording, Python transform contract, and refinement feedback follow Poetiq.
POETIQ_SOLVER_PROMPTS = {
  1: '''You are an expert in solving Abstract Reasoning Corpus (ARC) tasks by writing Python code. Your goal is to analyze input-output examples and create a 'transform' function that correctly transforms any given input grid into the corresponding output grid.
Analyze the examples: objects, colors, spatial relations, shapes, lines, symmetries, sizes. Formulate one consistent rule, prioritize simple general rules. Implement a Python function `transform(grid: np.ndarray) -> np.ndarray` with NumPy or standard libraries. Return a single complete markdown ```python code block containing the function; include needed imports. No __main__ block. The program must work on all training inputs and all test inputs. Do not hardcode specific test outputs.
PROBLEM:\n$$problem$$''',
  2: '''You are a world-class expert in solving ARC tasks and an expert Python coder. Analyze every example carefully, then hypothesize a SINGLE consistent rule applicable to all examples. Consider colors, connected components, rotations, reflections, object motion, addition/removal, spatial patterns and symmetries. Test and refine the hypothesis to match all example outputs. Implement `def transform(grid: np.ndarray) -> np.ndarray` using NumPy (cv2 and standard libraries may be used if available). Your answer must contain a single valid markdown ```python code block defining the function and any imports. Explain briefly before the code. Do not output a __main__ block.
PROBLEM:\n$$problem$$''',
  3: '''You are a world-class ARC reasoning researcher and master Python coder. Inspect the dimensions, positions, colors, objects, neighborhoods and geometry in all demonstration input/output pairs. Generate possible hypotheses; use the simplest rule explaining all pairs. Translate it into executable, documented Python code using NumPy and optionally cv2/standard libraries. Handle edge cases, unexpected shapes and colors. Return one markdown ```python code block containing imports and `def transform(grid: np.ndarray) -> np.ndarray`, plus a short explanation. Do not include a __main__ block. The same rule must generalize to each Challenge input.
PROBLEM:\n$$problem$$'''
}
POETIQ_FEEDBACK_PROMPT = '''**EXISTING PARTIAL/INCORRECT SOLUTIONS:**\nFollowing are some of the best, though not completely correct, solutions so far. For each solution, code, training-example feedback, and a score from 0 to 1 are provided. Study the code and the feedback and produce a NEW complete Python `transform` solution fixing all the issues. Follow the exact output format.\n$$feedback$$'''

def poetiq_diagram(g):
    return '\n'.join(' '.join(str(int(v)) for v in row) for row in g)

def poetiq_format_problem(task, shuffle=False, seed=0):
    train = list(task['train'])
    if shuffle and len(train) > 1:
        perm = np.random.default_rng(seed).permutation(len(train))
        train = [train[int(i)] for i in perm]
    chunks=[]
    for i, item in enumerate(train,1):
        chunks.append(f'Example #{i}\nInput:\n<Diagram>\n{poetiq_diagram(item["input"])}\n</Diagram>\nOutput:\n<Diagram>\n{poetiq_diagram(item["output"])}\n</Diagram>')
    for i,item in enumerate(task['test'],1):
        chunks.append(f'Challenge #{i}\nInput:\n<Diagram>\n{poetiq_diagram(item["input"])}\n</Diagram>')
    return '\n\n'.join(chunks)

def poetiq_parse_code(response):
    match=re.search(r'```python\s*(.*?)```',response,re.DOTALL|re.IGNORECASE)
    return match.group(1).strip() if match else None

def poetiq_create_examples(solutions, max_examples=5, improving_order=True):
    if not solutions: return ''
    chosen=sorted(solutions,key=lambda z:z['score'], reverse=True)[:max_examples]
    if improving_order: chosen.reverse()
    return '\n\n'.join(f'<solution_{k}>\n<solution_code>\n```python\n{s["code"]}\n```\n</solution_code>\n<solution_evaluation>\n{s["feedback"]}\n</solution_evaluation>\n<solution_score>\n{s["score"]:.2f}\n</solution_score>\n</solution_{k}>' for k,s in enumerate(chosen,1))

def poetiq_build_message(task,iteration,expert_idx,solutions,variant=1,selection_probability=1.,max_solutions=5,improving_order=True,shuffle=True,seed=0):
    # Poetiq's seeded example shuffling and best-history feedback.
    problem=poetiq_format_problem(task, shuffle, seed+iteration)
    msg=POETIQ_SOLVER_PROMPTS[int(variant)].replace('$$problem$$',problem)
    if solutions:
        rng=np.random.default_rng(seed+100000*expert_idx+iteration)
        subset=[s for s in solutions if rng.random()<selection_probability]
        if subset:
            msg+='\n\n'+POETIQ_FEEDBACK_PROMPT.replace('$$feedback$$',poetiq_create_examples(subset,max_solutions,improving_order))
    return msg

def poetiq_valid_grid(g):
    return (isinstance(g,list) and 1<=len(g)<=30 and all(isinstance(r,list) and 1<=len(r)<=30 for r in g)
            and len({len(r) for r in g})==1 and all(type(v)==int and 0<=v<=9 for r in g for v in r))

def poetiq_grid_from_output(text):
    try:
        val=json.loads(text)
        if poetiq_valid_grid(val): return val
    except (ValueError, TypeError): pass
    return None

# Poetiq-derived subprocess execution with real Linux mount/network/PID namespace
# isolation. The user-generated program runs ONLY in the private chroot after
# seccomp, CPU/memory/output/process limits. No silent unsandboxed fallback.
# Embedded at build time: no notebook runtime downloads or extra source files.
_POETIQ_TRUSTED_RUNNER = '"""Trusted inner runner inside a private chroot + mount/network/PID namespace.\n\nThis is embedded in E10 notebook by build_e10.py, never fetched at Kaggle runtime.\nUser-provided text runs only AFTER OS limits and seccomp are active.\n"""\nimport os\nimport sys\nimport json\nimport errno\nimport ctypes\nimport resource\nimport traceback\n\n# Import allowed computational libraries before the syscall filter is installed.\nimport numpy as np\nimport scipy\n\n_RESULT_MARKER = \'__E10_PROGRAM_RESULT__\'\n\n\ndef _install_seccomp():\n    """Block process creation, process control, network sockets and namespaces.\n\n    Chroot & isolated read-only bind mounts enforce filesystem boundaries.\n    The filter prevents model-generated code from undoing those boundaries.\n    """\n    lib = ctypes.CDLL(\'libseccomp.so.2\', use_errno=True)\n    lib.seccomp_init.argtypes = [ctypes.c_uint32]\n    lib.seccomp_init.restype = ctypes.c_void_p\n    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]\n    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int\n    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]\n    lib.seccomp_rule_add.restype = ctypes.c_int\n    lib.seccomp_load.argtypes = [ctypes.c_void_p]\n    lib.seccomp_load.restype = ctypes.c_int\n    lib.seccomp_release.argtypes = [ctypes.c_void_p]\n    lib.seccomp_release.restype = None\n    SCMP_ACT_ALLOW = 0x7FFF0000\n    SCMP_ACT_ERRNO = 0x00050000 | errno.EPERM\n    ctx = lib.seccomp_init(SCMP_ACT_ALLOW)\n    if not ctx:\n        raise RuntimeError(\'seccomp_init failed\')\n    try:\n        blocked = (\n            \'execve\', \'execveat\', \'fork\', \'vfork\', \'clone\', \'clone3\',\n            \'socket\', \'socketpair\', \'connect\', \'bind\', \'listen\', \'accept\', \'accept4\',\n            \'mount\', \'umount2\', \'pivot_root\', \'chroot\', \'unshare\', \'setns\',\n            \'ptrace\', \'process_vm_readv\', \'process_vm_writev\', \'open_by_handle_at\',\n            \'kill\', \'tkill\', \'tgkill\', \'reboot\', \'bpf\', \'keyctl\',\n        )\n        for name in blocked:\n            nr = lib.seccomp_syscall_resolve_name(name.encode())\n            if nr < 0: continue\n            rc = lib.seccomp_rule_add(ctx, SCMP_ACT_ERRNO, nr, 0)\n            if rc != 0: raise RuntimeError(f\'seccomp_rule_add {name}: {rc}\')\n        rc = lib.seccomp_load(ctx)\n        if rc != 0: raise RuntimeError(f\'seccomp_load: {rc}\')\n    finally:\n        lib.seccomp_release(ctx)\n\n\ndef _set_limits():\n    cpu = max(2, min(120, int(os.environ.get(\'POETIQ_CHILD_CPU_SEC\', \'10\'))))\n    mem_mb = max(256, min(8192, int(os.environ.get(\'POETIQ_CHILD_MEM_MB\', \'4096\'))))\n    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))\n    resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024,) * 2)\n    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024,) * 2)\n    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))\n    # NPROC does not protect uid=0, but seccomp blocks clone/fork/execve.\n    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))\n\n\ndef main():\n    # The only data accepted from the parent is code and the puzzle\'s input grid.\n    with open(\'/run/program.py\', encoding=\'utf-8\') as f:\n        source = f.read()\n    input_grid = json.load(sys.stdin)[\'input\']\n    _set_limits()\n\n    # Privileges cannot be regained after leaving the trusted launcher.\n    libc = ctypes.CDLL(None, use_errno=True)\n    PR_SET_NO_NEW_PRIVS = 38\n    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:\n        raise RuntimeError(\'no_new_privs failed\')\n\n    _install_seccomp()\n    # All imports done by untrusted code must come from the isolated filesystem.\n    namespace = {\'__name__\': \'poetiq_candidate\', \'__builtins__\': __builtins__, \'np\': np}\n    try:\n        exec(compile(source, \'/run/program.py\', \'exec\'), namespace)\n        result = namespace[\'transform\'](np.array(input_grid, dtype=int))\n        if hasattr(result, \'tolist\'):\n            result = result.tolist()\n        payload = {\'ok\': True, \'result\': result}\n    except BaseException as exc:\n        payload = {\'ok\': False, \'error\': f\'{type(exc).__name__}: {exc}\'[:1200]}\n    try:\n        # Debug prints may precede this trusted final marker.\n        print(_RESULT_MARKER + json.dumps(payload, separators=(\',\', \':\')), flush=True)\n    except BaseException as exc:\n        print(_RESULT_MARKER + json.dumps({\'ok\': False, \'error\': f\'JSON output error: {exc}\'[:500]}), flush=True)\n\n\nif __name__ == \'__main__\':\n    try:\n        main()\n    except BaseException as exc:\n        sys.stderr.write(\'E10_SECURE_RUNNER_INIT_ERROR: \' + repr(exc) + \'\\n\')\n        sys.exit(110)\n'
_POETIQ_NAMESPACE_BOOTSTRAP = '"""Trusted launcher: run from util-linux unshare -Urnmpf with private mounts.\n\nThe launcher only binds the host\'s Python runtime read-only. Kaggle inputs,\nnotebook, model artifacts and outside writable directories are NOT mounted.\n"""\nimport os\nimport sys\nimport shutil\nimport subprocess\n\n\ndef check(cmd):\n    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL,\n                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=12)\n\n\ndef main():\n    rootfs = sys.argv[1]\n    # Prevent any mount propagation into the parent mount namespace.\n    check([\'mount\', \'--make-rprivate\', \'/\'])\n    # Host imports must be read-only. None of the notebook/data directories are shared.\n    roots = [\'/usr\', \'/lib\', \'/lib64\', \'/opt/conda\', \'/opt/venv\',\n             sys.prefix, sys.base_prefix, os.path.dirname(sys.executable)]\n    for host_path in dict.fromkeys(roots):\n        if not os.path.exists(host_path):\n            continue\n        inner_path = os.path.join(rootfs, host_path.lstrip(\'/\'))\n        if os.path.islink(host_path):\n            os.makedirs(os.path.dirname(inner_path), exist_ok=True)\n            if not os.path.lexists(inner_path):\n                os.symlink(os.readlink(host_path), inner_path)\n            continue\n        os.makedirs(inner_path, exist_ok=True)\n        check([\'mount\', \'--bind\', host_path, inner_path])\n        check([\'mount\', \'-o\', \'remount,bind,ro\', inner_path])\n    # Dynamic linker configuration, not general host /etc.\n    etcdir = os.path.join(rootfs, \'etc\')\n    os.makedirs(etcdir, exist_ok=True)\n    if os.path.exists(\'/etc/ld.so.cache\'):\n        shutil.copyfile(\'/etc/ld.so.cache\', os.path.join(etcdir, \'ld.so.cache\'))\n    # In this namespace /run, /tmp and /dev are private to each candidate.\n    os.makedirs(os.path.join(rootfs, \'dev\'), exist_ok=True)\n    with open(os.path.join(rootfs, \'dev\', \'null\'), \'wb\'):\n        pass\n    os.chroot(rootfs)\n    os.chdir(\'/run\')\n    executable = os.path.realpath(sys.executable)\n    if not os.path.exists(executable):\n        # In a few environments sys.executable points through a symlink.\n        executable = sys.executable\n    os.execv(executable, [executable, \'-B\', \'-I\', \'/run/poetiq_secure_child.py\'])\n\n\nif __name__ == \'__main__\':\n    try:\n        main()\n    except BaseException as exc:\n        import traceback;sys.stderr.write(\'E10_SANDBOX_BOOTSTRAP_ERROR: \'+traceback.format_exc()+\'\\n\')\n        sys.exit(111)\n'

def poetiq_sandbox_run(code, grid, timeout_s=5.0):
    import shutil, signal
    if not shutil.which('unshare') or not shutil.which('mount'):
        return False, 'sandbox unavailable: unshare/mount binary missing'
    timeout_s = max(0.1, float(timeout_s))
    with tempfile.TemporaryDirectory(prefix='e10_poetiq_secure_') as td:
        # The rootfs and all generated content are disposable. No notebook,
        # checkpoint, data input, model or host /tmp mounts are present inside.
        root = Path(td)/'rootfs'
        run = root/'run'; run.mkdir(parents=True)
        (root/'tmp').mkdir()
        (run/'poetiq_secure_child.py').write_text(_POETIQ_TRUSTED_RUNNER,encoding='utf-8')
        (run/'program.py').write_text(str(code),encoding='utf-8')
        bootstrap = Path(td)/'namespace_bootstrap.py'
        bootstrap.write_text(_POETIQ_NAMESPACE_BOOTSTRAP,encoding='utf-8')
        output = Path(td)/'stdout.log'
        errors = Path(td)/'stderr.log'
        env = {'PYTHONHASHSEED':'0','PYTHONDONTWRITEBYTECODE':'1',
               'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'1',
               'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
               'POETIQ_CHILD_CPU_SEC':str(max(2,min(120,math.ceil(timeout_s)+1))),
               'POETIQ_CHILD_MEM_MB':'4096'}
        # util-linux unshare: user=root in *new* userns, private mount/net/PID ns;
        # --fork is needed for PID namespace. Parent holds timeout/kill authority.
        cmd=['unshare','--user','--map-root-user','--mount','--net','--pid',
             '--fork',sys.executable,str(bootstrap),str(root)]
        try:
            with output.open('wb') as out, errors.open('wb') as err:
                proc = subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=out,stderr=err,
                    cwd=td,env=env,start_new_session=True,close_fds=True)
                try:
                    proc.communicate(input=json.dumps({'input':grid}).encode(), timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    # Kill the complete namespace launcher/process group, not
                    # only the direct child. No untrusted process can escape.
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    proc.communicate(timeout=2)
                    return False,'sandbox timeout'
            if output.stat().st_size > 1024 * 1024 or errors.stat().st_size > 1024 * 1024:
                return False,'sandbox output limit exceeded'
            sout=output.read_text(encoding='utf-8',errors='replace')
            serr=errors.read_text(encoding='utf-8',errors='replace')
            if proc.returncode:
                return False, ('sandbox process failure '+str(proc.returncode)+': '+(serr or sout))[-1500:]
            marker='__E10_PROGRAM_RESULT__'
            pos=sout.rfind(marker)
            if pos<0:return False,'sandbox missing result: '+sout[-800:]
            try: payload=json.loads(sout[pos+len(marker):].strip())
            except ValueError:return False,'sandbox malformed result: '+sout[-800:]
            if not payload.get('ok'):
                return False,str(payload.get('error','program error'))[:1200]
            return True,json.dumps(payload.get('result'))
        except Exception as exc:
            return False, ('sandbox startup failure '+type(exc).__name__+': '+str(exc))[:1000]

def poetiq_array_diff(pred, truth):
    return '\n'.join(' '.join(str(int(a)) if a==b else f'{int(a)}/{int(b)}' for a,b in zip(pr,tr))
                     for pr,tr in zip(pred,truth))

def poetiq_verify_code(code,task,timeout_s=5.0):
    train_results=[]; per=[]
    for i,ex in enumerate(task['train']):
        ok,payload=poetiq_sandbox_run(code,ex['input'],timeout_s)
        # As in Poetiq, evaluate *partial training correctness* even when
        # predicted integer values are outside ARC 0..9; validate test grids separately.
        grid=None
        if ok:
            try:
                raw=json.loads(payload)
                # Never coerce floats, strings or bools into integer grids:
                # that can falsely mark a program train-verified and stop
                # Poetiq refinement while test candidates are invalid.
                if poetiq_valid_grid(raw):
                    grid=raw
            except (TypeError,ValueError,IndexError,OverflowError):pass
        truth=ex['output']; exact=grid==truth if grid is not None else False
        if exact: soft=1.0; feedback=f'Solves Example #{i+1} correctly.'
        elif grid is None: soft=0.0;feedback=f'Solves Example #{i+1} incorrectly. Invalid or unavailable grid. Error: {payload[-600:]}'
        elif len(grid)!=len(truth) or len(grid[0])!=len(truth[0]):
            soft=0.0;feedback=f'Solves Example #{i+1} incorrectly. Shape mismatch: predicted {(len(grid),len(grid[0]))}, expected {(len(truth),len(truth[0]))}.'
        else:
            soft=sum(a==b for rowa,rowb in zip(grid,truth) for a,b in zip(rowa,rowb))/(len(truth)*len(truth[0]))
            feedback=f'Solves Example #{i+1} incorrectly. Wrong cells appear as prediction/correct:\n```\n{poetiq_array_diff(grid,truth)}\n```\nOutput accuracy: {soft:.2f}'
        per.append(soft);train_results.append({'success':bool(exact),'soft_score':soft,'feedback':feedback,'error':None if ok else payload})
    test_results=[]
    for ex in task['test']:
        ok,payload=poetiq_sandbox_run(code,ex['input'],timeout_s)
        grid=poetiq_grid_from_output(payload) if ok else None
        test_results.append({'grid':grid,'error':None if grid is not None else payload[-600:]})
    passed=all(r['success'] for r in train_results)
    test_usable=all(poetiq_valid_grid(r.get('grid')) for r in test_results)
    feedback='\n\n'.join(r['feedback'] for r in train_results)
    # Test-input truth is NEVER consulted: only runtime/format diagnostics.
    if not test_usable:
        problems=[f'Challenge #{i+1}: {r.get("error") or "invalid grid shape/palette"}'
                  for i,r in enumerate(test_results) if not poetiq_valid_grid(r.get('grid'))]
        feedback += '\n\nTEST EXECUTION / FORMAT ERRORS (no test answers are known):\n'+\
                    '\n'.join(problems)
    return {'train_results':train_results,'test_results':test_results,
            'passed':passed,'test_usable':test_usable,
            'verified_usable':bool(passed and test_usable),
            'score':float(sum(per)/len(per)) if per else 0.0,
            'feedback':feedback}

def poetiq_test_key(test_results):
    return json.dumps([r.get('grid') for r in test_results],separators=(',',':'))

def poetiq_mean_soft(res):
    trs=res.get('train_results',[])
    return sum(t.get('soft_score',0.0) for t in trs)/len(trs) if trs else 0.0

def poetiq_vote(expert_results,use_new_voting=True,count_failed_matches=True,iters_tiebreak=False,low_to_high_iters=False):
    # Poetiq's solve_parallel_coding grouping and diversity-first ranking semantics.
    passers={}; failures={}
    for result in expert_results:
        key=poetiq_test_key(result['test_results'])
        (passers if result['passed'] else failures).setdefault(key,[]).append(result)
    if use_new_voting and count_failed_matches:
        for key in list(failures):
            if key in passers:
                passers[key].extend(failures.pop(key))
    groups=list(passers.values())
    if iters_tiebreak:
        for group in groups:group.sort(key=lambda z:z.get('iteration',0), reverse=not low_to_high_iters)
        groups.sort(key=lambda g:g[0].get('iteration',0),reverse=low_to_high_iters)
    groups.sort(key=len,reverse=True)
    for group in failures.values():group.sort(key=poetiq_mean_soft,reverse=True)
    failed=list(failures.values());failed.sort(key=lambda g:(len(g),poetiq_mean_soft(g[0])),reverse=True)
    if use_new_voting:
        return ([g[0] for g in groups if g]+[g[0] for g in failed if g]+
                [m for g in groups for m in g[1:]]+[m for g in failed for m in g[1:]])
    return ([g[0] for g in groups if g]+sorted([r for g in failed for r in g],key=poetiq_mean_soft,reverse=True)
            +[m for g in groups for m in g[1:]])

def poetiq_two_attempts(task, ranked):
    # Poetiq-ranked program outputs supply both attempts; no old grid policy.
    out=[]
    for qi,test in enumerate(task['test']):
        grids=[]
        for r in ranked:
            t=r['test_results']
            g=t[qi]['grid'] if qi<len(t) else None
            if poetiq_valid_grid(g) and g not in grids:grids.append(g)
        if grids:
            a1=grids[0];a2=grids[1] if len(grids)>1 else grids[0]
        else:
            a1=a2=test['input'] # only if no valid program output; required schema
        out.append({'attempt_1':a1,'attempt_2':a2})
    return out

# --- DONOR ORIGINAL PROMPTS BELOW ---
# Source: poetiq-ai/poetiq-arc-agi-solver/arc_agi/prompts.py (MIT)
# Verbatim donor prompt texts, then loaded into Poetiq adaptation core.
SOLVER_PROMPT_1 = '''
You are an expert in solving Abstract Reasoning Corpus (ARC) tasks by writing Python code. Your goal is to analyze input-output examples and create a 'transform' function that correctly transforms any given input grid into the corresponding output grid.

Here's how to approach the problem:

**1. Analyze the Examples:**
  *   Identify the key objects in the input and output grids (e.g., shapes, lines, regions).
  *   Determine the relationships between these objects (e.g., spatial arrangement, color, size).
  *   Identify the operations that transform the input objects and relationships into the output objects and relationships (e.g., rotation, reflection, color change, object addition/removal).
  *   Consider the grid dimensions, symmetries, and other visual features.

**2. Formulate a Hypothesis:**
  *   Based on your analysis, formulate a transformation rule that works consistently across all examples.
  *   Express the rule as a sequence of image manipulation operations.
  *   Prioritize simpler rules first.
  *   Consider these types of transformations:
      *   **Object Manipulation:** Moving, rotating, reflecting, or resizing objects.
      *   **Color Changes:** Changing the color of specific objects or regions.
      *   **Spatial Arrangements:** Rearranging the objects in a specific pattern.
      *   **Object Addition/Removal:** Adding or removing objects based on certain criteria.

**3. Implement the Code:**
  *   Write a Python function called `transform(grid: np.ndarray) -> np.ndarray` that implements your transformation rule.
  *   Use NumPy for array manipulations. Other standard libraries are also available.
  *   Write modular code with clear variable names and comments to explain the logic behind each step.
  *   Document your code clearly, explaining the transformation rule in the docstring.
  *   Handle edge cases and invalid inputs gracefully.

**4. Test and Refine:**
  *   Test your code on all examples. If it fails for any example, refine your hypothesis and code.
  *   Use debugging techniques to identify and fix errors.
  *   Ensure your code handles edge cases and invalid inputs gracefully.

**5. Output:**
  *   Provide a brief explanation of your solution.
  *   Include the complete Python code for the `transform` function within a single markdown code block.
  *   Do not include any `__name__ == "__main__"` block or any code outside the function definition.

**Examples:**

**Example 1:**

**Input:**
```
[[1, 1, 1],
[1, 0, 1],
[1, 1, 1]]
```

**Output:**
```
[[0, 0, 0],
[0, 1, 0],
[0, 0, 0]]
```

**Explanation:**
Replace the border with 0s.

**Code:**
```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
  """Replace the border with 0s."""
  grid[0, :] = 0
  grid[-1, :] = 0
  grid[:, 0] = 0
  grid[:, -1] = 0
  return grid
```

**Example 2:**

**Input:**
```
[[1, 2, 3],
[4, 5, 6],
[7, 8, 9]]
```

**Output:**
```
[[9, 8, 7],
[6, 5, 4],
[3, 2, 1]]
```

**Explanation:**
Reverse the order of elements in each row and then reverse the order of the rows themselves.

**Code:**
```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
  """Reverses the order of elements in each row and then reverses the order of the rows."""
  new_grid = grid[:, ::-1][::-1]
  return new_grid
```

**Example 3:**

**Input:**
```
[[0, 0, 0, 0, 0],
[0, 1, 1, 1, 0],
[0, 1, 0, 1, 0],
[0, 1, 1, 1, 0],
[0, 0, 0, 0, 0]]
```

**Output:**
```
[[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 1, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0]]
```

**Explanation:**
Keep only the center pixel if it is 1, otherwise make the grid all zeros.

**Code:**
```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
  """Keep only the center pixel if it is 1, otherwise make the grid all zeros."""
  center_row, center_col = grid.shape[0] // 2, grid.shape[1] // 2
  if grid[center_row, center_col] == 1:
      new_grid = np.zeros_like(grid)
      new_grid[center_row, center_col] = 1
      return new_grid
  else:
      return np.zeros_like(grid)
```

**PROBLEM:**

Below is a textual representation of the input-output examples and the challenge to be solved.

$$problem$$
'''
SOLVER_PROMPT_2 = '''
You are a world-class expert in solving Abstract Reasoning Corpus (ARC) tasks. Your approach is methodical, creative, and highly effective. You are also a master Python coder, producing elegant, efficient, and well-documented solutions.

Your goal is to analyze a set of input-output examples and devise a Python function that accurately transforms any input grid into its corresponding output grid. The key is to identify a *single, consistent transformation rule* that generalizes across *all* examples. Do not give up until you find a correct solution.

Follow this iterative process:

**Part 1: Initial Analysis and Hypothesis Generation**

1.  **Example Inspection:** Carefully examine the input and output grids for each example. Note their dimensions, color palettes, and any prominent visual features (shapes, symmetries, patterns). Use visualization techniques to aid your analysis.
2.  **Transformation Hypotheses:** Formulate several candidate transformation rules. Start with simpler rules and gradually increase complexity. Consider these categories:
    *   **Color Transformations:** Replacing colors based on specific criteria (e.g., adjacency, frequency). For example, replace all 0s with 1s, or replace the most frequent color with the least frequent color.
    *   **Object Isolation:** Identifying and isolating objects based on color, shape, or position. For example, extract the largest connected component of a certain color, or isolate objects based on their spatial relationships.
    *   **Spatial Operations:** Rotating, reflecting, resizing, or moving objects. For example, rotate the grid by 90 degrees, reflect the grid horizontally or vertically, or resize the grid by a certain factor.
    *   **Pattern Generation:** Replicating or extending existing patterns. For example, repeat a certain pattern across the grid, or generate a new pattern based on the existing patterns.
3.  **Symmetry Analysis:** Identify any symmetries (rotational, reflectional) in the input and output grids. Determine if the transformation preserves or alters these symmetries.

**Part 2: Iterative Testing and Refinement**

1.  **Code Implementation:** Implement your strongest candidate rule as a Python function. The function *must* accept a 2D numpy array as input and return a 2D numpy array as output.
2.  **Rigorous Testing:** Test your code against *all* training examples. A single failure indicates an incorrect rule.
3.  **Feedback Analysis:** If your code fails, carefully analyze the feedback. Identify the specific examples that failed and the nature of the errors. Use print statements to debug intermediate values and verify your assumptions.
4.  **Hypothesis Refinement:** Based on the feedback, refine your transformation rule. This may involve adjusting parameters, adding new conditions, or discarding the rule altogether and starting with a new hypothesis.
5.  **Repeat:** Continue this iterative process of coding, testing, and refining until you find a rule that works for all training examples. Do not give up until you find a correct solution.

**Part 3: Coding Guidelines**

1.  **Available Libraries:** You can use `numpy`, `cv2` (OpenCV), and any library from the standard Python library.
2.  **Computer Vision Techniques:** Consider using `cv2` for tasks involving object detection, edge detection, or image filtering.
3.  **Utility Functions:** Write reusable utility functions to improve code modularity and readability.
4.  **Error Handling:** Implement robust error handling to gracefully manage edge cases and invalid inputs.
5.  **Code Clarity:** Write clean, well-documented code with meaningful variable names and comments.

**Part 4: Output Requirements**

1.  **Output Format:**
    *   Begin with a concise paragraph explaining the proposed solution, followed by a Python code section.
    *   You *must* provide a code output representing your best attempt. Do not give up or refuse to produce code.
    *   **The code section must be a single, valid Python code block in markdown fenced code block format and nothing else.**
    *   The main transform function must have the signature `def transform(grid: np.ndarray) -> np.ndarray`.
    *   Document the transformation rule implemented in the docstring of the transform function.
    *   Do not include any `__name__ == "__main__"` block. This will be added later by the user. You are writing a library function.

**Example:**

**Problem:**
Input:
<Diagram>
0 0 1
0 1 0
1 0 0
</Diagram>

Output:
<Diagram>
1 1 1
1 1 1
1 1 1
</Diagram>

**Explanation:**
Replace all 0s with 1s.

```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """Replace all 0s with 1s."""
    return np.where(grid == 0, 1, grid)
```

**PROBLEM:**

Below is a textual representation of the input-output examples and the challenge to be solved.

$$problem$$
'''
SOLVER_PROMPT_3 = '''
You are a world-class expert in solving Abstract Reasoning Corpus (ARC) tasks. Your approach is methodical, creative, and highly effective. You are also a master Python coder, producing elegant, efficient, and well-documented solutions.

Your goal is to analyze a set of input-output examples and devise a Python function that accurately transforms any input grid into its corresponding output grid. The key is to identify a *single, consistent transformation rule* that generalizes across *all* examples. Do not give up until you find a correct solution.

Follow this iterative process:

**Part 1: Initial Analysis and Hypothesis Generation**

1.  **Example Inspection:** Carefully examine the input and output grids for each example. Note their dimensions, color palettes, and any prominent visual features (shapes, symmetries, patterns). Use visualization techniques to aid your analysis.
2.  **Transformation Hypotheses:** Formulate several candidate transformation rules. Start with simpler rules and gradually increase complexity. Consider these categories:
    *   **Color Transformations:** Replacing colors based on specific criteria (e.g., adjacency, frequency). For example, replace all 0s with 1s, or replace the most frequent color with the least frequent color.
    *   **Object Isolation:** Identifying and isolating objects based on color, shape, or position. For example, extract the largest connected component of a certain color, or isolate objects based on their spatial relationships.
    *   **Spatial Operations:** Rotating, reflecting, resizing, or moving objects. For example, rotate the grid by 90 degrees, reflect the grid horizontally or vertically, or resize the grid by a certain factor.
    *   **Pattern Generation:** Replicating or extending existing patterns. For example, repeat a certain pattern across the grid, or generate a new pattern based on the existing patterns.
3.  **Symmetry Analysis:** Identify any symmetries (rotational, reflectional) in the input and output grids. Determine if the transformation preserves or alters these symmetries.

**Part 2: Iterative Testing and Refinement**

1.  **Code Implementation:** Implement your strongest candidate rule as a Python function. The function *must* accept a 2D numpy array as input and return a 2D numpy array as output.
2.  **Rigorous Testing:** Test your code against *all* training examples. A single failure indicates an incorrect rule.
3.  **Feedback Analysis:** If your code fails, carefully analyze the feedback. Identify the specific examples that failed and the nature of the errors. Use print statements to debug intermediate values and verify your assumptions.
4.  **Hypothesis Refinement:** Based on the feedback, refine your transformation rule. This may involve adjusting parameters, adding new conditions, or discarding the rule altogether and starting with a new hypothesis.
5.  **Repeat:** Continue this iterative process of coding, testing, and refining until you find a correct solution. Do not give up until you find a correct solution.

**Part 3: Coding Guidelines**

1.  **Available Libraries:** You can use `numpy`, `cv2` (OpenCV), and any library from the standard Python library.
2.  **Computer Vision Techniques:** Consider using `cv2` for tasks involving object detection, edge detection, or image filtering.
3.  **Utility Functions:** Write reusable utility functions to improve code modularity and readability.
4.  **Error Handling:** Implement robust error handling to gracefully manage edge cases and invalid inputs.
5.  **Code Clarity:** Write clean, well-documented code with meaningful variable names and comments. The code should be as concise as possible.

**Part 4: Output Requirements**

1.  **Output Format:**
    *   Begin with a concise paragraph explaining the proposed solution, followed by a Python code section.
    *   You *must* provide a code output representing your best attempt. Do not give up or refuse to produce code.
    *   **The code section must be a single, valid Python code block in markdown fenced code block format and nothing else.**
    *   The main transform function must have the signature `def transform(grid: np.ndarray) -> np.ndarray`.
    *   Document the transformation rule implemented in the docstring of the transform function.
    *   Do not include any `__name__ == "__main__"` block. This will be added later by the user. You are writing a library function.

**Example 1:**

**Problem:**
Input:
<Diagram>
0 0 1
0 1 0
1 0 0
</Diagram>

Output:
<Diagram>
1 1 1
1 1 1
1 1 1
</Diagram>

**Explanation:**
Replace all 0s with 1s.

```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """Replace all 0s with 1s."""
    return np.where(grid == 0, 1, grid)
```

**Example 2:**

**Problem:**
Input:
<Diagram>
0 0 0
0 1 0
0 0 0
</Diagram>

Output:
<Diagram>
0 1 0
1 1 1
0 1 0
</Diagram>

**Explanation:**
Replace all neighbors of 1 with 1.

```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """Replace all neighbors of 1 with 1."""
    new_grid = grid.copy()
    for i in range(1, grid.shape[0] - 1):
        for j in range(1, grid.shape[1] - 1):
            if grid[i][j] == 1:
                new_grid[i-1][j] = 1
                new_grid[i+1][j] = 1
                new_grid[i][j-1] = 1
                new_grid[i][j+1] = 1
    return new_grid
```

**Example 3:**

**Problem:**
Input:
<Diagram>
1 2 3
4 5 6
7 8 9
</Diagram>

Output:
<Diagram>
9 8 7
6 5 4
3 2 1
</Diagram>

**Explanation:**
Reverse the grid.

```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """Reverse the grid."""
    return np.flip(grid)
```

**PROBLEM:**

Below is a textual representation of the input-output examples and the challenge to be solved.

$$problem$$
'''
FEEDBACK_PROMPT = '''
**EXISTING PARTIAL/INCORRECT SOLUTIONS:**

Following are some of the best, though not completely correct, solutions so far. For each solution, its code, corresponding feedback regarding its output on the example problems, and a numeric score between 0. (worst) and 1. (best) indicating the quality of outputs is also provided. Study these solutions and corresponding feedback and produce a new solution fixing all the issues. Make sure to follow the output format specified earlier.

$$feedback$$
'''

# Replace only the textual prompt templates with the published donor originals.
POETIQ_SOLVER_PROMPTS = {1:SOLVER_PROMPT_1, 2:SOLVER_PROMPT_2, 3:SOLVER_PROMPT_3}
POETIQ_FEEDBACK_PROMPT = FEEDBACK_PROMPT
