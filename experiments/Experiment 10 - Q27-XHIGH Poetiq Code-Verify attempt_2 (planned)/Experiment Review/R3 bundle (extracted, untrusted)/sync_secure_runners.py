from pathlib import Path
p=Path(__file__).resolve().parent
f=p/'poetiq_core.py'
s=f.read_text()
for key,filename in [('_POETIQ_TRUSTED_RUNNER','poetiq_secure_child.py'),('_POETIQ_NAMESPACE_BOOTSTRAP','poetiq_namespace_bootstrap.py')]:
 start=s.index(key+' = ')
 end=s.index('\n',start)
 s=s[:start]+key+' = '+repr((p/filename).read_text())+s[end:]
f.write_text(s)
