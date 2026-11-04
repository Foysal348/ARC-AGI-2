import os, json, hashlib, requests, time
from pathlib import Path
os.chdir("D:/ARC AGI 2/_scratch")
lines=open("foysal_kit_log.py",encoding="utf-8").read().split("\n"); cut=next(i for i,l in enumerate(lines) if "api.authenticate()" in l)
g={"__name__":"pre"}; exec("\n".join(lines[:cut+1]),g); api=g["api"]
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
BASE=Path("D:/ARC AGI 2/papers/1 - NVARC Solution 2025 Limitation/kaggle versions")
slugs=[x["slug"] for x in json.load(open("D:/ARC AGI 2/resources/NVARC family/index.json",encoding="utf-8"))]
manifest={}
for slug in slugs:
    manifest[slug]={}; miss=0
    for n in range(1,41):
        vl=f"v{n}"
        try:
            with api.build_kaggle_client() as k:
                r=ApiListKernelSessionOutputRequest(); r.user_name="foysalemonshanto"; r.kernel_slug=slug; r.page_size=500; r.version_label=vl
                resp=k.kernels.kernels_api_client.list_kernel_session_output(r)
        except Exception as e:
            miss+=1; manifest[slug][vl]="404" if "404" in str(e) else str(e)[:60]
            if miss>=3: break
            continue
        miss=0; d=BASE/slug/vl/"output"; d.mkdir(parents=True,exist_ok=True)
        files={}
        for f in resp.files or []:
            fn=f.file_name
            if "unsloth_compiled" in fn or "__pycache__" in fn or fn.startswith("program063_runtime_profiles"): continue
            c=requests.get(f.url).content; p=d/fn; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(c)
            files[fn]=[len(c),hashlib.sha256(c).hexdigest()[:8]]
        (d/f"{slug}_{vl}.log").write_text(resp.log or "",encoding="utf-8")
        files["_log_chars"]=len(resp.log or "")
        manifest[slug][vl]=files
        print(slug,vl,len(files)-1,"files log",files["_log_chars"],flush=True)
    json.dump(manifest,open(BASE/"manifest.json","w",encoding="utf-8"),indent=1)
print("DONE")
