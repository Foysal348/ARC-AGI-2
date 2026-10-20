"""Q27 v1.9 bootstrap. Reuses the v1.8.3 measured vLLM startup and wheelhouse.
No forced tool choice, no tool smoke, no effort ladder, no budget-dropping fallback.
"""
import json
import os
import shutil
import signal
import sys
import time
import traceback

import submit_main as legacy  # startup/install only; NEVER calls legacy.main
from synth import _request, strict_payload

REPORT_PATH=legacy.REPORT_PATH


def budget_contract_smoke(cfg, seconds=45):
    # Cheap request-shape probe. Not an accuracy or full 49k-generation test.
    payload=strict_payload(cfg.get('model','qwen'),
                           [{'role':'user','content':'In your final answer, write the integer 7.'}],cfg)
    assert payload['thinking_token_budget']==49152
    assert payload['max_tokens']==57344
    assert payload['chat_template_kwargs']['reasoning_effort']=='xhigh'
    assert 'tools' not in payload and 'tool_choice' not in payload
    # Keep actual thinking budget constant but cap the smoke generation.
    payload['max_tokens']=64
    r=_request(cfg['base_url'],payload,max(2.0,seconds),retry_transient=False)
    return {'ok':r.get('status')=='ok', 'status':r.get('status'),
            'finish':r.get('finish_reason'), 'error':str(r.get('error',''))[:300],
            'synth_think_retained':r.get('status')=='ok',
            'request_contract':{'thinking_token_budget':49152,'max_tokens_smoke':64,
                'max_tokens_full':57344,'effort':'xhigh','tool_choice':None}}


def _atomic_report(obj):
    p=REPORT_PATH; p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.tmp')
    q.write_text(json.dumps(obj,indent=2,default=str))
    os.replace(q,p)


def main(cfg):
    cfg = dict(cfg, base_url=cfg.get('base_url', 'http://127.0.0.1:1234/v1'),
               model=cfg.get('model', 'qwen'))
    proc=None
    report={'version':'v1.9.1','architecture':'strict_no_tools_synthesis',
            'rungs':[],'budget_fallback_allowed':False}
    try:
        from check_config import validate
        validate(cfg)
        signal.signal(signal.SIGTERM,legacy._on_sigterm)
        t0=time.time()
        legacy.install(cfg)
        report['install_s']=round(time.time()-t0,2)
        report['libcuda']=('dryrun' if cfg.get('dryrun') else legacy.ensure_libcuda())
        # Only rungs with server-level --reasoning-config are allowed.
        all_rungs=[[]] if cfg.get('dryrun') else legacy.server_ladder(cfg)
        rungs=[r for r in all_rungs if cfg.get('dryrun') or '--reasoning-config' in r]
        if not rungs:
            report['error']='NO_REASONING_CONFIG_RUNG';_atomic_report(report);return
        for i,extra in enumerate(rungs):
            remaining=cfg['deadline_epoch']-time.time()-float(cfg.get('min_agent_s',900))
            if remaining<90:
                report['error']='INSUFFICIENT_TIME_FOR_SERVER';break
            t=time.time()
            proc=legacy.start_server(cfg,extra,timeout=min(1500.0,remaining))
            entry={'index':i,'extra':extra,'started':proc is not None,
                   'start_s':round(time.time()-t,1)}
            if proc:
                smoke=budget_contract_smoke(cfg,min(90.0,max(5.0,remaining-20)))
                entry['budget_smoke']=smoke
                report['rungs'].append(entry)
                if smoke['ok']:
                    report['chosen']={'index':i,'extra':extra}
                    report['budget_contract_pass']=True
                    break
                legacy.kill(proc);proc=None
            else:
                entry['startup_error']=legacy.classify_startup_failure()
                report['rungs'].append(entry)
        _atomic_report(report)
        if not proc or not report.get('budget_contract_pass'):
            print('v1.9 NO-GO: no healthy server with strict 49,152 budget',flush=True)
            return
        if cfg.get('dryrun'):
            print('v1.9 fake-server rehearsal; no model accuracy claims',flush=True)
        try:
            log_text=legacy.LOG.read_text(errors='replace')
            import re
            kv=re.findall(r'GPU KV cache size: ([0-9,]+) tokens',log_text)
            report['kv_cache_tokens']=int(kv[-1].replace(',','')) if kv else None
        except Exception as e:
            report['kv_error']=repr(e)[:250]
        _atomic_report(report)
        if cfg.get('eval'):
            from run_eval_v19 import evaluate
            report['eval_report_summary']=evaluate(cfg)
        else:
            # The hidden-run decision is DIRECT-only until microprobe/EVAL evidence.
            # Preserve previous durable submission/fallback path; no experimental AGENT.
            from run_submit import main as direct_hidden
            direct_hidden(dict(cfg,mode='repl',agent_disabled=True,rescue=True))
        _atomic_report(report)
    except BaseException as e:
        report['error']=repr(e)
        report['traceback']=traceback.format_exc()[-5000:]
        _atomic_report(report)
        print('v1.9 caught exception (fallback retained):',report['traceback'],flush=True)
    finally:
        if proc is not None:
            legacy.kill(proc)
        shutil.rmtree(legacy.SITE,ignore_errors=True)

if __name__=='__main__':
    main(json.loads(sys.argv[1]))
