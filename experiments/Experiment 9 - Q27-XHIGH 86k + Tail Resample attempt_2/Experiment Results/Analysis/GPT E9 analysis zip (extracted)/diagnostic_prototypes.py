"""Post-hoc diagnostic prototypes for two named ARC tasks, NOT a general solver.
Developed after inspecting this evaluation bundle; matches are not prospective model gains.
Only read the supplied archive; no model, GPU, network, or generated-code execution.
Run: python diagnostic_prototypes.py path/to/E9_GPU_EVAL12_RUN_FOR_GPT.zip
"""
import json,sys,zipfile
def push_mask(g):
 out=[r[:] for r in g];h=len(g);w=len(g[0]);offset=w//2;wall=5
 for col in range(offset+1,w-1):
  mask=[r for r in range(1,h-1) if g[r][col-offset]==wall]
  colored=[r for r in range(1,h-1) if g[r][col] not in [0,wall]]
  if mask and colored:
   shift=min(max(0,max(colored)-min(mask)+1),min(colored)-1)
   for r in colored:out[r][col]=0
   for r in colored:out[r-shift][col]=g[r][col]
 for r in range(h):
  for c in range(offset-1):
   if g[r][c]==wall:out[r][c]=0;out[r][c+offset]=wall
 return out

def stamp_sources(g,reverse=True):
 h,w=len(g),len(g[0]); sep=next(c for c in range(1,w-2) if len({g[r][c] for r in range(h)})==1 and g[0][c]!=0)
 frame=sep+1; out=[r[:] for r in g];colors=set(v for r in g for v in r[:sep])-{0};objs=[]
 for color in colors:
  cells=[(r,c) for r in range(h) for c in range(sep) if g[r][c]==color]
  ys=sorted({r for r in range(1,h-1) if g[r][frame]==color or g[r][w-1]==color})
  xs=sorted({c for c in range(frame+1,w-1) if g[0][c]==color or g[h-1][c]==color})
  if not ys or not xs:continue
  cr=(min(r for r,c in cells)+max(r for r,c in cells))//2;cc=(min(c for r,c in cells)+max(c for r,c in cells))//2
  for r,c in cells:out[r][c]=0
  objs.append((min(r for r,c in cells),color,[(r-cr,c-cc) for r,c in cells],ys,xs))
 for _,color,offsets,ys,xs in sorted(objs,reverse=reverse):
  for y in ys:
   for x in xs:
    for dr,dc in offsets:
     r,c=y+dr,x+dc
     if 0<r<h-1 and frame<c<w-1:out[r][c]=color
 return out
if __name__ == '__main__':
 with zipfile.ZipFile(sys.argv[1]) as z:
  tasks=json.loads(z.read('data/eval12_challenges.json'))
  solutions=json.loads(z.read('data/eval12_solutions.json'))
 for tid,fn in [('247ef758',stamp_sources),('4c3d4a41',push_mask)]:
  t=tasks[tid]
  train=[fn(x['input'])==x['output'] for x in t['train']]
  test=[fn(x['input'])==y for x,y in zip(t['test'],solutions[tid])]
  print(tid,json.dumps({'train':train,'test':test}))
  assert all(train+test)
 # A wrong overlap ordering must fail a training example.
 assert stamp_sources(tasks['247ef758']['train'][0]['input'],False)!=tasks['247ef758']['train'][0]['output']
 print('PASS: 5 training examples + 4 test inputs; adverse overlap control rejected.')
