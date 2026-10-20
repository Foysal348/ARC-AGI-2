"""Q27 v1.9: strict, no-tools xhigh program synthesis and deterministic verification.

No solution labels are loaded or used in this module. Phase-3 repair is explicitly
DISABLED in the first real-model microprobe. No automatic thinking downgrade.
"""
import ast
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from rescue import render_grid, valid_grid
from repl_agent import verify, _nonportable

THINK = 49152
ANSWER = 8192
MAX_CONTEXT = 81920
MARGIN = 512
BLOCKS = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.I | re.S)


DENIED_MODULE_ROOTS = {
    'os', 'sys', 'subprocess', 'socket', 'ctypes', 'cffi', 'pathlib',
    'shutil', 'importlib', 'builtins', 'pickle', 'glob', 'inspect',
    'tempfile', 'urllib', 'requests', 'http', 'sqlite3'
}
DENIED_CALLS = {'open', 'eval', 'exec', 'compile', '__import__', 'input'}
DENIED_ARRAY_IO = {'load', 'save', 'loadtxt', 'savetxt', 'genfromtxt',
                   'memmap', 'fromfile', 'tofile'}


def source_guard_reason(tree):
    """Best-effort source guard; NOT a substitute for OS sandboxing."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] in DENIED_MODULE_ROOTS:
                    return 'disallowed import: ' + alias.name
        if isinstance(node, ast.ImportFrom):
            if (node.module or '').split('.')[0] in DENIED_MODULE_ROOTS:
                return 'disallowed from-import: ' + str(node.module)
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in DENIED_CALLS:
                return 'disallowed call: ' + f.id
            if isinstance(f, ast.Attribute) and f.attr in DENIED_ARRAY_IO:
                return 'disallowed file-like API: ' + f.attr
    return None


def build_synthesis_prompt(task):
    parts = ["You are solving an abstract reasoning puzzle. Digits 0-9 represent colors; 0 is usually background. "
             "Infer a single transformation that maps each example INPUT to OUTPUT.\n"]
    for i, p in enumerate(task['train'], 1):
        parts.append(f"Example {i} INPUT:\n{render_grid(p['input'])}\nExample {i} OUTPUT:\n{render_grid(p['output'])}\n")
    for j, p in enumerate(task['test'], 1):
        parts.append(f"TEST {j} INPUT:\n{render_grid(p['input'])}\n")
    parts.append('''Think through a rule that reproduces ALL training examples. Then output EXACTLY ONE fenced Python block
of standalone executable code defining def transform(grid): ...
The function takes ONE input grid (list of lists of ints) and returns a new grid (list of lists of ints 0-9).
You may define helpers and imports in the same code block. The program must compute from grid alone;
it must not read files, evaluate other code, use train/train_inputs/train_outputs/test_inputs/puzzle, print,
or look up hardcoded training output examples. Do not return exploratory snippets. Do not add text after the code block.
Output format: ```python followed by source code, followed by ```.
''')
    return '\n'.join(parts)


def extract_program(content):
    """Strict extraction, no free-text or unfinished-fence fallback."""
    for block in reversed(BLOCKS.findall(content or '')):
        src = block.strip()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name == 'transform']
        if len(functions) != 1 or isinstance(functions[0], ast.AsyncFunctionDef):
            continue
        # Call requires an input parameter. No @decorator/wrapping transformation accepted.
        fn = functions[0]
        if len(fn.args.posonlyargs) + len(fn.args.args) < 1 or fn.decorator_list:
            continue
        bad = sorted(_nonportable(src))
        if bad:
            return {'status': 'nonportable', 'code': src, 'bad': bad}
        unsafe = source_guard_reason(tree)
        if unsafe:
            return {'status': 'unsafe_program', 'code': src, 'bad': [unsafe]}
        return {'status': 'program', 'code': src}
    return {'status': 'no_program', 'code': None}


def _post(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8')), None
    except urllib.error.HTTPError as e:
        return None, {'code': e.code, 'message': e.read().decode(errors='replace')[:900]}
    except Exception as e:
        return None, {'code': None, 'message': '%s: %s' % (type(e).__name__, str(e)[:400])}


def count_prompt_tokens(base_url, model, messages, template_kwargs, timeout=12):
    """Try exact chat-tokenize; conservative UTF-8 byte count if unavailable.

    /tokenize is separate from /v1/tokenize on vLLM. If the endpoint or request
    shape is unsupported, byte count is a conservative upper bound for text.
    """
    root = base_url.removesuffix('/v1').rstrip('/')
    payload = {'model': model, 'messages': messages, 'chat_template_kwargs': template_kwargs}
    result, err = _post(root + '/tokenize', payload, timeout)
    if not err and isinstance(result, dict):
        n = result.get('count', result.get('num_tokens'))
        if n is None and isinstance(result.get('tokens'), list):
            n = len(result['tokens'])
        if isinstance(n, int) and n > 0:
            return n, 'server_chat_tokenize'
    # Reserve 256 tokens for chat-template wrappers (usually far fewer).
    return sum(len(m.get('content', '').encode('utf-8')) for m in messages) + 256, 'byte_upper_bound'


def count_text_tokens(base_url, model, text, timeout=12):
    """v1.9.1: exact token count of a plain text (no chat template) via /tokenize, or None."""
    if not text:
        return 0
    root = base_url.removesuffix('/v1').rstrip('/')
    result, err = _post(root + '/tokenize', {'model': model, 'prompt': text, 'add_special_tokens': False}, timeout)
    if err or not isinstance(result, dict):
        return None
    n = result.get('count', result.get('num_tokens'))
    if n is None and isinstance(result.get('tokens'), list):
        n = len(result['tokens'])
    return n if isinstance(n, int) else None


def think_accounting(response, base_url, model, count_text=None):
    """v1.9.1: thinking tokens per request. vLLM 0.19 usually leaves usage.completion_tokens_details empty, so the
    count is derived as completion_tokens - tokens(content). hit_budget = thinking ran into the 49,152 cap (the
    server closed it) rather than ending naturally."""
    think = response.get('reasoning_tokens')
    source = 'usage'
    if think is None:
        n_content = (count_text or count_text_tokens)(base_url, model, response.get('content') or '')
        comp = int(response.get('completion_tokens') or 0)
        think = max(0, comp - n_content) if n_content is not None and comp else None
        source = 'derived' if think is not None else 'unknown'
    return {'think_tokens': think, 'think_source': source,
            'think_hit_budget': (think >= THINK - 64) if think is not None else None}


def strict_payload(model, messages, cfg, thinking=True, max_tokens=None):
    budget = int(cfg.get('synth_think', THINK))
    answer = int(cfg.get('synth_answer', ANSWER))
    if budget != THINK:
        raise ValueError('strict synth thinking budget must equal 49152')
    if answer != ANSWER:
        raise ValueError('strict synth answer cap must equal 8192')
    if cfg.get('agent_mode', 'synth') != 'synth' or int(cfg.get('repair_turns', 0)) != 0:
        raise ValueError('microprobe must be synthesis only, repair=0')
    use_thinking = bool(thinking)
    return {'model': model, 'messages': messages,
            'max_tokens': (budget + answer if use_thinking else int(max_tokens or answer)),
            'temperature': cfg.get('temperature', 0.6), 'top_p': cfg.get('top_p', 0.95),
            'top_k': cfg.get('top_k', 20),
            'chat_template_kwargs': {'enable_thinking': use_thinking,
                                     'reasoning_effort': 'xhigh' if use_thinking else 'low'},
            **({'thinking_token_budget': budget} if use_thinking else {})}


def _is_timeout(error):
    msg = (error.get('message') or '').lower()
    return error.get('code') is None and ('timed out' in msg or 'timeout' in msg)


def _request(base_url, payload, timeout, *, retry_transient=False, deadline=None, margin_s=10.0):
    """v1.9.1: a timeout is never retried (the first request already used the whole remaining time), and a retry
    gets only the time left before `deadline`, so one task can never run past the deadline."""
    url = base_url.rstrip('/') + '/chat/completions'
    for attempt in range(2 if retry_transient else 1):
        body, error = _post(url, payload, timeout)
        if error:
            desc = (error['message'] or '').lower()
            if 'thinking_token_budget' in desc or 'thinking budget' in desc:
                return {'status': 'budget_rejected', 'error': error, 'usage_known': False}
            if error['code'] in (400, 401, 403, 404, 422):
                return {'status': 'request_error', 'error': error, 'usage_known': False}
            if _is_timeout(error):
                return {'status': 'request_timeout', 'error': error, 'usage_known': False}
            if attempt == 0 and retry_transient and (error['code'] is None or error['code'] in (429, 500, 502, 503, 504)):
                time.sleep(min(5.0, timeout / 10))
                if deadline is not None:
                    timeout = min(timeout, deadline - time.time() - margin_s)
                    if timeout < 60:
                        return {'status': 'request_error', 'error': error, 'usage_known': False,
                                'retry_skipped': 'deadline'}
                continue
            return {'status': 'request_error', 'error': error, 'usage_known': False}
        try:
            msg = body['choices'][0]['message'] or {}
            # Plain synth must never have a tool call.
            if msg.get('tool_calls'):
                return {'status': 'unexpected_tool_call', 'completion_tokens': body.get('usage', {}).get('completion_tokens', 0)}
            reasoning = msg.get('reasoning_content') or msg.get('reasoning') or ''
            return {'status': 'ok', 'content': msg.get('content') or '',
                    'reasoning_tail': reasoning[-8000:], 'reasoning_chars': len(reasoning),
                    'finish_reason': body['choices'][0].get('finish_reason'),
                    'completion_tokens': int(body.get('usage', {}).get('completion_tokens', 0)),
                    'prompt_tokens': int(body.get('usage', {}).get('prompt_tokens', 0)),
                    'usage_known': bool(body.get('usage')),
                    'reasoning_tokens': (body.get('usage') or {}).get('completion_tokens_details', {}).get('reasoning_tokens') if isinstance((body.get('usage') or {}).get('completion_tokens_details'), dict) else None}
        except (KeyError, IndexError, TypeError, ValueError) as e:
            return {'status': 'malformed_response', 'error': repr(e)[:300], 'usage_known': False}
    raise AssertionError('unreachable')


def synthesize_one(task_id, task, cfg, *, count_fn=None, now=None):
    start = time.time()
    result = {'task_id': task_id, 'status': 'not_started', 'code': None, 'n_ok': 0,
              'n_train': len(task.get('train') or []), 'test': [None] * len(task.get('test') or []),
              'gen_tokens': 0, 'think_tokens': None, 'program_source': None, 'feedback': ''}
    prompt = build_synthesis_prompt(task)
    messages = [{'role': 'user', 'content': prompt}]  # no system prompt, no tools
    token_counter = count_fn or count_prompt_tokens
    try:
        n_prompt, method = token_counter(cfg['base_url'], cfg.get('model','qwen'), messages,
                                        {'enable_thinking': True, 'reasoning_effort': 'xhigh'})
    except Exception:
        n_prompt, method = len(prompt.encode('utf-8')) + 256, 'byte_upper_bound_error'
    result.update(prompt_tokens_est=n_prompt, prompt_count_method=method)
    if n_prompt + THINK + ANSWER + MARGIN > int(cfg.get('max_model_len', MAX_CONTEXT)):
        result.update(status='context_too_large', feedback='Refused to lower 49152 thinking budget.')
        result['wall_s'] = round(time.time()-start, 3)
        return result
    deadline = float(cfg.get('deadline_epoch', float('inf')))
    remaining = deadline - time.time() - float(cfg.get('request_margin_s', 10))
    if remaining <= 1:
        result.update(status='deadline')
        result['wall_s'] = round(time.time()-start, 3)
        return result
    payload = strict_payload(cfg.get('model','qwen'), messages, cfg)
    response = _request(cfg['base_url'], payload, min(float(cfg.get('request_timeout_s', 7200)), remaining),
                        retry_transient=True, deadline=deadline,
                        margin_s=float(cfg.get('request_margin_s', 10)))
    result.update(gen_tokens=int(response.get('completion_tokens', 0)),
                  finish=response.get('finish_reason'), usage_known=response.get('usage_known', False),
                  think_tokens=response.get('reasoning_tokens'),
                  reasoning_chars=response.get('reasoning_chars'), request_wall_s=round(time.time() - start, 1))
    if response['status'] == 'ok':
        try:
            result.update(think_accounting(response, cfg['base_url'], cfg.get('model', 'qwen'),
                                           cfg.get('_count_text')))
        except Exception as e:                  # accounting must never cost the answer
            result.update(think_source='error: %s' % type(e).__name__)
    if response['status'] != 'ok':
        result.update(status=response['status'], feedback=str(response.get('error',''))[:700])
        result['wall_s'] = round(time.time()-start, 3)
        return result
    parsed = extract_program(response['content'])
    result['program_source'] = 'primary' if parsed['status'] == 'program' else None
    if parsed['status'] == 'no_program' and cfg.get('synth_commit', True):
        # Only one explicit no-thinking commit; no tools, no forced submit, no inherited budget.
        left = deadline - time.time() - float(cfg.get('request_margin_s', 10))
        if left > max(30, float(cfg.get('commit_min_remaining_s', 120))):
            commit_tail = (response.get('content') or response.get('reasoning_tail') or '')[-7000:]
            commit_messages = [messages[0], {'role':'assistant', 'content': commit_tail},
                {'role':'user','content': 'Return your final complete standalone Python program now. Exactly one ```python fenced block containing def transform(grid). No discussion.'}]
            # A commit request can fit after long reasoning; if not, omit middle assistant tail.
            commit_n, _ = token_counter(cfg['base_url'], cfg.get('model','qwen'), commit_messages,
                                          {'enable_thinking': False, 'reasoning_effort': 'low'})
            if commit_n + ANSWER + MARGIN > int(cfg.get('max_model_len', MAX_CONTEXT)):
                commit_messages = [messages[0], commit_messages[-1]]
            cp = strict_payload(cfg.get('model','qwen'), commit_messages, cfg, thinking=False, max_tokens=ANSWER)
            c = _request(cfg['base_url'], cp, min(left, 720), retry_transient=False)
            result['commit_status'] = c['status']
            result['gen_tokens'] += int(c.get('completion_tokens', 0))
            if c['status']=='ok':
                parsed = extract_program(c.get('content',''))
                if parsed['status']=='program':
                    result['program_source'] = 'commit'
    result['status'] = parsed['status']
    if parsed['status'] != 'program':
        result['feedback'] = ('No standalone Python program produced.' if parsed['status']=='no_program'
                              else str(parsed['status']) + ': ' + str(parsed.get('bad')))
        result['wall_s'] = round(time.time()-start, 3)
        return result
    code = parsed['code']
    result['code'] = code
    try:
        n_ok, tests, feedback = verify(code, task, timeout=float(cfg.get('sandbox_timeout_s', 10)))
        result.update(n_ok=n_ok, test=tests, feedback=feedback)
        train_ok = bool(task['train']) and n_ok == len(task['train'])
        usable = len(tests) == len(task['test']) and all(valid_grid(g) for g in tests)
        result.update(train_verified=train_ok, test_usable=usable)
        result['status'] = ('verified_usable' if train_ok and usable else
                            'verified_test_error' if train_ok else 'train_mismatch')
    except Exception as e:
        result.update(status='verification_error', feedback='%s: %s' % (type(e).__name__,str(e)[:300]))
    result['wall_s'] = round(time.time()-start, 3)
    return result


def solve_synthesis(tasks, cfg, on_done=None):
    # First pass: all requested tasks receive an attempt, bounded by global deadline.
    records = []
    def do(tid):
        try:
            return synthesize_one(tid, tasks[tid], cfg)
        except BaseException as e:
            return {'task_id': tid, 'status':'crashed', 'error': repr(e)[:500], 'code':None,
                    'n_ok':0, 'test':[None] * len(tasks[tid].get('test',[])), 'gen_tokens':0}
    with ThreadPoolExecutor(max_workers=min(int(cfg.get('concurrency',16)),max(1,len(tasks)))) as pool:
        futures = {pool.submit(do, tid):tid for tid in tasks}
        for fut in as_completed(futures):
            rec = fut.result()
            records.append(rec)
            if on_done:
                on_done(rec)
    return records
