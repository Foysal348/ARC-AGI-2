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
    if solutions and max_solutions > 0:
        rng=np.random.default_rng(seed+100000*expert_idx+iteration)
        subset=[s for s in solutions if rng.random()<selection_probability]
        if subset:
            msg+='\n\n'+POETIQ_FEEDBACK_PROMPT.replace('$$feedback$$',poetiq_create_examples(subset,max_solutions,improving_order))
    return msg

def poetiq_normalize_grid(g):
    """Lossless ARC palette normalization, rejecting booleans and fractions."""
    if not isinstance(g, list) or not 1 <= len(g) <= 30:
        return None
    if not all(isinstance(row,list) and 1 <= len(row) <= 30 for row in g):
        return None
    if len({len(row) for row in g}) != 1:
        return None
    normalized=[]
    for row in g:
        rr=[]
        for v in row:
            if isinstance(v, (bool, np.bool_)):
                return None
            if isinstance(v,(int,np.integer)):
                value=int(v)
            elif isinstance(v,(float,np.floating)) and math.isfinite(float(v)) and float(v).is_integer():
                value=int(v)
            else:
                return None
            if not 0 <= value <= 9:
                return None
            rr.append(value)
        normalized.append(rr)
    return normalized


def poetiq_valid_grid(g):
    return poetiq_normalize_grid(g) is not None


def poetiq_grid_from_output(text):
    try:
        return poetiq_normalize_grid(json.loads(text))
    except (TypeError,ValueError,OverflowError):
        return None


# Poetiq subprocess execution adapted for Kaggle Docker without unshare.
# UID separation + parent limits + no-new-privs + seccomp are enforced.
# These do not create a mount namespace. World-readable host files are NOT
# guaranteed inaccessible; see the README's explicit residual risk.
_POETIQ_TRUSTED_RUNNER = '"""Trusted Poetiq runner for Kaggle\'s namespace-restricted Docker containers.\n\nNot a mount namespace. It provides UID separation and syscall/resource limits;\nworld-readable host paths are NOT guaranteed unreadable without a mount or LSM.\n"""\nimport sys, os, json, ctypes, errno, resource, traceback\n# Import numerical/scientific libraries before syscall filtering.\nimport numpy as np\ntry:\n    import scipy\nexcept ImportError:\n    scipy = None\ntry:\n    import cv2\nexcept ImportError:\n    cv2 = None\n\n_RESULT_MARKER = \'__E10_PROGRAM_RESULT__\'\n\n\ndef drop_privileges_and_limits():\n    # Trusted code runs before compiling or executing model-generated Python.\n    if os.geteuid() != 0:\n        raise RuntimeError(\'Root is required for irreversible UID/GID drop\')\n    os.setgroups([])\n    os.setgid(65534)\n    os.setuid(65534)\n    cpu = max(2,min(120,int(os.environ.get(\'E10_CHILD_CPU\',\'10\'))))\n    mem = 4096*1024*1024\n    resource.setrlimit(resource.RLIMIT_CPU,(cpu,cpu+1))\n    resource.setrlimit(resource.RLIMIT_AS,(mem,mem))\n    resource.setrlimit(resource.RLIMIT_FSIZE,(1024*1024,1024*1024))\n    resource.setrlimit(resource.RLIMIT_CORE,(0,0))\n    resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))\n    resource.setrlimit(resource.RLIMIT_NPROC,(64,64))\n\n\ndef harden_process():\n    libc = ctypes.CDLL(None, use_errno=True)\n    if libc.prctl(38, 1, 0, 0, 0) != 0:\n        raise RuntimeError(\'PR_SET_NO_NEW_PRIVS failed\')\n    lib = ctypes.CDLL(\'libseccomp.so.2\', use_errno=True)\n    lib.seccomp_init.argtypes = [ctypes.c_uint32]\n    lib.seccomp_init.restype = ctypes.c_void_p\n    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]\n    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int\n    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]\n    lib.seccomp_rule_add.restype = ctypes.c_int\n    lib.seccomp_load.argtypes = [ctypes.c_void_p]\n    lib.seccomp_load.restype = ctypes.c_int\n    lib.seccomp_release.argtypes = [ctypes.c_void_p]\n    lib.seccomp_release.restype = None\n    ctx = lib.seccomp_init(0x7fff0000)\n    if not ctx:\n        raise RuntimeError(\'seccomp_init failed\')\n    try:\n        # Blocking exec/spawn/network/ptrace/mount remains compatible with\n        # numpy/scipy/cv2. Do not block clone(thread) indiscriminately.\n        blocked = (\n            \'execve\',\'execveat\',\'fork\',\'vfork\',\'clone3\',\n            \'socket\',\'socketpair\',\'connect\',\'bind\',\'listen\',\'accept\',\'accept4\',\n            \'mount\',\'umount2\',\'pivot_root\',\'chroot\',\'unshare\',\'setns\',\n            \'ptrace\',\'process_vm_readv\',\'process_vm_writev\',\'open_by_handle_at\',\n            \'reboot\',\'bpf\',\'keyctl\',\'setuid\',\'setgid\',\'setreuid\',\'setregid\',\n            \'setresuid\',\'setresgid\',\n        )\n        for name in blocked:\n            nr = lib.seccomp_syscall_resolve_name(name.encode())\n            if nr >= 0:\n                if lib.seccomp_rule_add(ctx, 0x00050000 | errno.EPERM, nr, 0):\n                    raise RuntimeError(\'seccomp rule failed: \' + name)\n        # clone() flags: require CLONE_THREAD when creating a thread.\n        # This blocks ordinary child processes without disabling numerical threads.\n        clone_nr = lib.seccomp_syscall_resolve_name(b\'clone\')\n        if clone_nr >= 0:\n            class ArgCmp(ctypes.Structure):\n                _fields_=[(\'arg\',ctypes.c_uint),(\'op\',ctypes.c_uint),\n                          (\'datum_a\',ctypes.c_uint64),(\'datum_b\',ctypes.c_uint64)]\n            # seccomp_rule_add_array(ctx, action, syscall, n, &cmp)\n            lib.seccomp_rule_add_array.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,\n                                                   ctypes.c_uint,ctypes.POINTER(ArgCmp)]\n            lib.seccomp_rule_add_array.restype=ctypes.c_int\n            # SCMP_CMP_MASKED_EQ=7; deny clone if CLONE_THREAD bit is zero.\n            cmp=ArgCmp(0,7,0x10000,0)\n            rc=lib.seccomp_rule_add_array(ctx,0x00050000|errno.EPERM,clone_nr,1,ctypes.byref(cmp))\n            if rc:\n                raise RuntimeError(\'seccomp clone restriction failed: \'+str(rc))\n        rc=lib.seccomp_load(ctx)\n        if rc:\n            raise RuntimeError(\'seccomp_load failed \'+str(rc))\n    finally:\n        lib.seccomp_release(ctx)\n\n\ndef main():\n    raw = json.load(sys.stdin)\n    program = raw[\'code\']\n    grid = raw[\'input\']\n    if not isinstance(program, str) or len(program)>250000:\n        raise ValueError(\'invalid code length\')\n    drop_privileges_and_limits()\n    harden_process()\n    namespace = {\'__name__\': \'poetiq_candidate\', \'__builtins__\': __builtins__, \'np\': np}\n    try:\n        exec(compile(program, \'<poetiq_candidate>\', \'exec\'), namespace)\n        result=namespace[\'transform\'](np.array(grid, dtype=int))\n        if hasattr(result,\'tolist\'):\n            result=result.tolist()\n        payload={\'ok\':True,\'result\':result}\n    except BaseException as ex:\n        payload={\'ok\':False,\'error\':(type(ex).__name__+\': \'+str(ex))[:1200]}\n    try:\n        print(_RESULT_MARKER + json.dumps(payload,separators=(\',\',\':\')),flush=True)\n    except BaseException as ex:\n        print(_RESULT_MARKER + json.dumps({\'ok\':False,\'error\':\'JSON output error: \'+str(ex)[:500]}),flush=True)\n\nif __name__==\'__main__\':\n    try:\n        main()\n    except BaseException as ex:\n        sys.stderr.write(\'E10_RUNNER_SETUP_ERROR: \'+repr(ex)+\'\\n\')\n        sys.exit(110)\n'



def poetiq_sandbox_run(code, grid, timeout_s=5.0):
    import signal
    timeout_s=max(.1,float(timeout_s))
    if os.geteuid()!=0:
        return False,'sandbox unavailable: cannot drop root privileges to UID 65534'
    try:
        import ctypes
        ctypes.CDLL('libseccomp.so.2')
    except Exception as e:
        return False,'sandbox unavailable: libseccomp: '+str(e)[:200]
    with tempfile.TemporaryDirectory(prefix='e10_poetiq_r4_') as td:
        # Child cwd and TMPDIR are an isolated per-program scratch folder.
        # It is NOT a mount-based read sandbox; UID/seccomp prevent spawning,
        # outbound sockets and writing to root-owned Kaggle checkpoints.
        os.chown(td,65534,65534)
        os.chmod(td,0o700)
        output=Path(td)/'stdout.log'; errors=Path(td)/'stderr.log'
        env={'PYTHONHASHSEED':'0','PYTHONDONTWRITEBYTECODE':'1',
             'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1',
             'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
             'NUMEXPR_NUM_THREADS':'1','TMPDIR':td,
             'HOME':td, 'PATH':'/usr/bin:/bin'}
        env['E10_CHILD_CPU']=str(max(2,min(120,math.ceil(timeout_s)+1)))
        cmd=[sys.executable,'-I','-B','-c',_POETIQ_TRUSTED_RUNNER]
        try:
            with output.open('wb') as outf,errors.open('wb') as errf:
                proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=outf,stderr=errf,
                      cwd=td,env=env,start_new_session=True,close_fds=True)
                try:
                    proc.communicate(input=json.dumps({'code':code,'input':grid}).encode(), timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    proc.communicate(timeout=2)
                    return False,'sandbox timeout'
            if output.stat().st_size>1024*1024 or errors.stat().st_size>1024*1024:
                return False,'sandbox output limit exceeded'
            sout=output.read_text(encoding='utf-8',errors='replace')
            serr=errors.read_text(encoding='utf-8',errors='replace')
            if proc.returncode:
                return False,('sandbox process failure '+str(proc.returncode)+': '+(serr or sout))[-1500:]
            marker='__E10_PROGRAM_RESULT__'
            pos=sout.rfind(marker)
            if pos<0:
                return False,'sandbox missing result: '+sout[-800:]
            try: payload=json.loads(sout[pos+len(marker):].strip())
            except ValueError:
                return False,'sandbox malformed result: '+sout[-800:]
            if not payload.get('ok'):
                return False,str(payload.get('error','program error'))[:1200]
            return True,json.dumps(payload.get('result'))
        except Exception as exc:
            return False,('sandbox startup failure '+type(exc).__name__+': '+str(exc))[:1000]

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
                grid=poetiq_normalize_grid(raw)
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

# A positive runtime test executes before vLLM engine loading (notebook Cell 6).
# Never claim readiness based on a seccomp library import alone.
def poetiq_sandbox_selftest():
    import tempfile
    test_grid=[[2]]
    ok, payload=poetiq_sandbox_run(
        'import numpy as np\ndef transform(grid):\n return np.asarray(grid)', test_grid, 8)
    if not ok or poetiq_grid_from_output(payload)!=test_grid:
        raise RuntimeError('Poetiq sandbox positive test FAILED: '+str(payload)[:500])
    with tempfile.TemporaryDirectory(prefix='e10_root_protected_') as td:
        protected=Path(td)/'no_write'
        protected.write_text('original',encoding='utf-8')
        protected.chmod(0o600)
        attack=f"def transform(grid):\n open({str(protected)!r}, 'w').write('MODIFIED')\n return grid"
        ok,err=poetiq_sandbox_run(attack,test_grid,8)
        if ok or protected.read_text(encoding='utf-8')!='original':
            raise RuntimeError('Poetiq sandbox protected-file write test FAILED')
    for label,attack in (
        ('spawn',"import subprocess\ndef transform(grid):\n subprocess.check_output(['/usr/bin/id'])\n return grid"),
        ('network',"import socket\ndef transform(grid):\n socket.socket()\n return grid"),
    ):
        ok,err=poetiq_sandbox_run(attack,test_grid,8)
        if ok or 'Operation not permitted' not in str(err):
            raise RuntimeError('Poetiq sandbox '+label+' restriction FAILED: '+str(err)[:300])
    return {'ok':True,'mode':'uid_drop+seccomp+rlimits+timeout',
            'filesystem_isolation':'NOT a private mount; world-readable host paths may be readable',
            'verified':['numpy_transform','protected_file_write_blocked','spawn_blocked','network_blocked']}

E10_SANDBOX_SELFTEST=poetiq_sandbox_selftest()
print('[E10] Sandbox self-test PASS:',E10_SANDBOX_SELFTEST,flush=True)
