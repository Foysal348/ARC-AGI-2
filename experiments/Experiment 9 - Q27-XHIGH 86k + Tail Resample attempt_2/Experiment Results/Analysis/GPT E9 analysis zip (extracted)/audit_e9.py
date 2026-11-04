import json,re,ast,hashlib
from pathlib import Path
from collections import Counter
import argparse
ap=argparse.ArgumentParser(description='CPU-only forensic audit of an extracted E9 bundle; executes the inspected parser definitions, never model inference.')
ap.add_argument('bundle_dir',type=Path);ap.add_argument('--out',type=Path,default=Path.cwd())
args=ap.parse_args();R=args.bundle_dir;O=args.out;O.mkdir(parents=True,exist_ok=True)
C=json.loads((R/'data/eval12_challenges.json').read_text(encoding='utf-8'));S=json.loads((R/'data/eval12_solutions.json').read_text(encoding='utf-8'))
nb=json.loads((R/'notebook/Q27-XHIGH-v1_4_4_E9_86K_RESAMPLE.ipynb').read_text(encoding='utf-8'))
ns={'re':re,'json':json}
src=''.join(nb['cells'][5]['source']).split('# ---- self-tests')[0]
exec(compile(src,'parser_from_notebook','exec'),ns)
records=[json.loads(x) for x in (R/'outputs/q27x/records.jsonl').read_text(encoding='utf-8').splitlines()]
def shape(g):return [len(g),len(g[0])] if g else None
def diffs(a,b):
 if shape(a)!=shape(b) or any(len(r)!=len(b[0]) for r in a):return None
 return [[i,j,x,b[i][j]] for i,row in enumerate(a) for j,x in enumerate(row) if x!=b[i][j]]
def align(a,b):
 # Oracle diagnostic only: minimum edit cost with row gaps at width cost.
 if not a or not b or len({len(r) for r in a+b})!=1:return None
 n,m=len(a),len(b);w=len(b[0]);d=[[0]*(m+1) for _ in range(n+1)]
 for i in range(n+1):d[i][0]=i*w
 for j in range(m+1):d[0][j]=j*w
 for i in range(1,n+1):
  for j in range(1,m+1):d[i][j]=min(d[i-1][j-1]+sum(x!=y for x,y in zip(a[i-1],b[j-1])),d[i-1][j]+w,d[i][j-1]+w)
 i,j=n,m;missing=[];extra=[];res=[]
 while i or j:
  if i and j and d[i][j]==d[i-1][j-1]+sum(x!=y for x,y in zip(a[i-1],b[j-1])):
   res.extend([[i-1,j-1,k,a[i-1][k],b[j-1][k]] for k in range(w) if a[i-1][k]!=b[j-1][k]]);i-=1;j-=1
  elif j and d[i][j]==d[i][j-1]+w:missing.append(j-1);j-=1
  else:extra.append(i-1);i-=1
 return {'cost':d[n][m],'missing_solution_rows':sorted(missing),'extra_answer_rows':sorted(extra),'residual_cells':res}
rows=[]
for rec in records:
 tid,q=rec['task_id'],rec['query'];suffix='_r1' if 'resample_of' in rec else ''
 p=R/f'outputs/q27x/answers/{tid}_q{q}{suffix}.txt';text=p.read_text(encoding='utf-8');ans=text.split('### ANSWER',1)[1];g=ns['parse_grid'](ans)
 blocks=re.findall(r'```grid\s*\n(.*?)```',ans,re.S)
 raw=[[int(x) for x in z.split()] for z in blocks[-1].strip().splitlines()] if blocks else None
 inp=C[tid]['test'][q]['input'];gt=S[tid][q];dif=diffs(g,gt) if g else None
 tr=C[tid]['train'];same=all(shape(x['input'])==shape(x['output']) for x in tr);const=len({tuple(shape(x['output'])) for x in tr})==1
 row={'unit':rec['unit'],'mode':rec['mode'],'p1_finish':rec['p1_finish'],'input_shape':shape(inp),'truth_shape':shape(gt),'parsed_shape':shape(g),'raw_rows':len(raw) if raw else None,'raw_widths':dict(Counter(map(len,raw))) if raw else None,'same_size_train':same,'constant_train_output':const,'train_shapes':[[shape(x['input']),shape(x['output'])] for x in tr],'wrong_cells':len(dif) if dif is not None else None,'diffs':dif,'copy_wrong_cells':len(diffs(inp,gt)) if diffs(inp,gt) is not None else None,'hit':g==gt,'alignment':align(raw,gt),'answer_tokens':rec['p2_tokens'],'grid':g}
 rows.append(row)
 if 'resample_of' not in rec:
  print(json.dumps({k:v for k,v in row.items() if k not in ('grid','diffs','alignment','train_shapes')}))
  if dif is not None and len(dif)<=10:print('DIFF',dif)
  if shape(raw)!=shape(gt) or not g:print('ALIGN',row['alignment'])
print('RESAMPLES')
for row in rows:
 if '~r1' in row['unit']:print(row['unit'],row['parsed_shape'],row['wrong_cells'],row['p1_finish'])
p=[r for r in records if 'resample_of' not in r];rs=[r for r in records if 'resample_of' in r]
print('TOKENS',{'primary':sum(r['p1_tokens']+r['p2_tokens'] for r in p),'resample':sum(r['p1_tokens']+r['p2_tokens'] for r in rs),'drained_resample':sum(r['p1_tokens']+r['p2_tokens'] for r in rs if r['p1_finish']=='drained')})
(O/'e9_independent_audit.json').write_text(json.dumps(rows,indent=2))
print('SHA',hashlib.sha256((R/'notebook/Q27-XHIGH-v1_4_4_E9_86K_RESAMPLE.ipynb').read_bytes()).hexdigest())
