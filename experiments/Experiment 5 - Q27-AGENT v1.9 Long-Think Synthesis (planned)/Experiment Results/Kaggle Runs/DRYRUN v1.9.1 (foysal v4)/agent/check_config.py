"""Runtime/launch invariants, before any GPU or hidden evaluation."""

def validate(cfg):
    if cfg.get('agent_mode') != 'synth':
        raise ValueError('agent_mode must be synth')
    if int(cfg.get('synth_think',-1)) != 49152:
        raise ValueError('synth_think must be exactly 49152')
    if int(cfg.get('synth_answer',-1)) != 8192:
        raise ValueError('synth_answer must be exactly 8192')
    if int(cfg.get('repair_turns',-1)) != 0:
        raise ValueError('v1.9 microprobe repair_turns must be zero')
    if cfg.get('force_submit_after') or cfg.get('effort_ladder'):
        raise ValueError('forced submit or effort ladder prohibited')
    if cfg.get('template_kwargs',{}).get('reasoning_effort')!='xhigh':
        raise ValueError('xhigh required')
    if cfg.get('thinking') is False:
        raise ValueError('thinking may not be disabled')
    if int(cfg.get('max_model_len',0))<57344+512:
        raise ValueError('context too small')
    # v1.9.1: the hidden-test DIRECT path (inherited v1.8.3 rescue) must never shrink the 49,152 thinking budget
    # (user rule: no budget below the LB base anywhere); with rescue_min_think = 49,152 time_budget() gives either the
    # full budget or None (= not started), never a smaller one
    if int(cfg.get('rescue_think',-1)) != 49152 or int(cfg.get('rescue_answer',-1)) != 8192:
        raise ValueError('rescue_think / rescue_answer must be exactly 49152 / 8192')
    if int(cfg.get('rescue_min_think',-1)) < 49152:
        raise ValueError('rescue_min_think below 49152 would cut the DIRECT thinking budget')
    if cfg.get('rescue_effort','xhigh') != 'xhigh':
        raise ValueError('rescue_effort must be xhigh')
    if cfg.get('eval') and cfg.get('eval_direct') not in ('ref',None):
        raise ValueError('v1.9 comparison must use recorded DIRECT, not live DIRECT')
    if not cfg.get('dryrun') and not any('--reasoning-config' in r for r in cfg.get('server_ladder',[])):
        raise ValueError('server ladder lacks reasoning-config')
    return True
