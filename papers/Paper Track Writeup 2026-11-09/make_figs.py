import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from datetime import datetime
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12})
INK="#1f2430"; MUT="#6b7280"; A="#2563eb"; B="#f59e0b"; C="#10b981"; D="#ef4444"; G="#9ca3af"

# ---- LB history (public LB, our team's scored submissions, 0 / error-like runs < 5 dropped)
sub = [  # date, score, family
("2026-04-26",28.61,"T"),("2026-05-04",29.31,"T"),("2026-05-05",28.06,"T"),("2026-05-06",28.06,"T"),("2026-05-07",27.64,"T"),("2026-05-08",26.94,"T"),
("2026-05-12",28.33,"T"),("2026-05-15",27.22,"T"),("2026-05-17",28.06,"T"),("2026-05-19",28.61,"T"),("2026-05-22",30.14,"T"),("2026-05-25",26.81,"T"),
("2026-05-26",28.89,"T"),("2026-05-28",23.75,"T"),("2026-05-29",29.17,"T"),("2026-06-01",26.25,"T"),("2026-06-02",24.58,"T"),("2026-06-03",26.11,"T"),
("2026-06-05",29.72,"T"),("2026-06-06",32.22,"T"),("2026-06-07",28.06,"T"),("2026-06-10",28.33,"T"),("2026-06-11",28.61,"T"),("2026-06-12",29.72,"T"),
("2026-06-13",29.31,"T"),("2026-06-14",30.97,"T"),("2026-06-15",30.14,"T"),("2026-06-17",27.64,"T"),("2026-06-18",29.31,"T"),("2026-06-19",27.22,"T"),
("2026-06-22",29.72,"T"),("2026-06-23",28.61,"T"),("2026-06-24",28.06,"T"),("2026-06-25",28.89,"T"),("2026-06-26",28.06,"T"),("2026-06-27",26.11,"T"),
("2026-06-28",32.64,"T"),("2026-06-29",29.72,"T"),("2026-06-30",30.14,"T"),("2026-07-01",27.36,"T"),("2026-07-02",30.14,"T"),("2026-07-03",28.89,"T"),
("2026-07-04",28.61,"T"),("2026-07-05",27.22,"T"),("2026-07-08",25.97,"T"),("2026-07-10",30.69,"T"),("2026-07-11",27.64,"T"),("2026-07-12",28.89,"T"),
("2026-07-14",28.89,"T"),("2026-07-15",27.64,"T"),("2026-07-16",28.47,"T"),("2026-07-16",28.06,"T"),("2026-07-22",29.31,"T"),("2026-07-23",29.31,"T"),
("2026-07-24",29.03,"T"),("2026-07-25",30.14,"T"),("2026-07-26",29.72,"T"),("2026-07-31",28.89,"T"),("2026-08-02",27.64,"T"),("2026-08-03",29.86,"T"),
("2026-08-05",27.64,"T"),("2026-08-06",29.31,"T"),("2026-08-07",29.31,"T"),("2026-08-07",30.14,"T"),("2026-08-11",30.97,"T"),("2026-08-19",26.81,"T"),
("2026-08-30",27.22,"T"),("2026-09-07",28.06,"T"),("2026-09-10",30.56,"T"),("2026-09-13",30.28,"T"),
("2026-10-06",18.75,"D27"),("2026-10-10",26.94,"D27"),("2026-10-12",27.92,"D27"),("2026-10-14",22.92,"D27"),("2026-10-15",21.39,"D27"),("2026-10-16",34.58,"D27"),
("2026-10-17",32.08,"D27"),("2026-10-18",29.17,"D27"),("2026-10-19",31.81,"D27"),("2026-10-22",31.94,"D27"),("2026-10-25",26.94,"D27"),("2026-10-27",31.53,"D27"),("2026-10-29",31.53,"D27"),
("2026-10-20",17.50,"A27"),("2026-10-23",22.78,"A27"),("2026-10-24",22.92,"A27"),
("2026-11-02",49.58,"AFN"),("2026-10-31",54.17,"AFNW4"),
]
fam={"T":("Apr-Sep: derivatives of public notebooks (mostly NVARC 2025)",G),"D27":("Qwen3.8-27B, code-verifying / direct solvers",B),"A27":("Qwen3.8-27B, tool agent",C),
     "AFN":("Flash-Next 3-bit, tool agent",A),"AFNW4":("Flash-Next W4A16, tool agent (this paper)",D)}
fig,ax=plt.subplots(figsize=(11,5.6),dpi=150)
for k,(lab,col) in fam.items():
    xs=[datetime.fromisoformat(d) for d,s,f in sub if f==k]; ys=[s for d,s,f in sub if f==k]
    ax.scatter(xs,ys,s=110 if k=="AFNW4" else 42,color=col,label=lab,zorder=3,edgecolor="white" if k!="AFNW4" else INK,linewidth=0.8)
ax.annotate("54.17\nsubmission 57851995",(datetime(2026,10,31),54.17),xytext=(datetime(2026,9,16),50.5),
            arrowprops=dict(arrowstyle="->",color=INK),fontsize=11,color=INK)
ax.set_ylabel("Public leaderboard score (%)"); ax.set_ylim(15,58)
ax.set_title("Our scored ARC-AGI-2 submissions, by method family",loc="left",color=INK,fontsize=14)
ax.grid(axis="y",color="#e5e7eb"); [ax.spines[s].set_visible(False) for s in ("top","right")]
ax.legend(loc="upper left",frameon=False,fontsize=10)
fig.tight_layout(); fig.savefig("figures/fig2_lb_history.png"); plt.close(fig)

# ---- pipeline
fig,ax=plt.subplots(figsize=(12,5.4),dpi=150); ax.set_xlim(0,12); ax.set_ylim(0,5.4); ax.axis("off")
def box(x,y,w,h,t,c,fs=10.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04,rounding_size=0.12",fc=c,ec=INK,lw=1.2))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,color=INK)
def arr(x1,y1,x2,y2): ax.annotate("",(x2,y2),(x1,y1),arrowprops=dict(arrowstyle="-|>",color=INK,lw=1.4))
box(0.2,3.5,2.4,1.5,"240 hidden tasks\npriority order\nattempt 1 for all,\nthen independent\nattempt 2","#f3f4f6")
box(3.2,3.5,2.8,1.5,"Tool agent\n(one session per attempt)\nthinks, writes Python,\ncalls check()","#dbeafe")
box(6.6,3.5,2.4,1.5,"Python sandbox\ntransform(grid)\nrun on every\ntrain pair","#d1fae5")
box(9.6,3.5,2.2,1.5,"submit()\nhost re-verifies\nin a clean\nsandbox","#fee2e2")
arr(2.6,4.25,3.2,4.25); arr(6.0,4.45,6.6,4.45); arr(6.6,3.95,6.0,3.95); arr(9.0,4.25,9.6,4.25)
ax.text(7.8,3.2,"check(): exact pairs + wrong cells fed back",ha="center",fontsize=9,color=MUT)
box(3.2,0.4,5.8,2.1,"Merlin v3.0.0 on 4 x L4 (offline), TP2 x PP2\nQwen3.8-Flash-Next: W4A16 routed experts, FP8 n-gram table\nFP8 KV cache (738k tokens), MTP speculation (3 tokens)\nup to 9 sessions, 104k tokens per attempt, 717-min budget","#fef3c7")
arr(4.4,3.5,4.4,2.5); arr(5.4,2.5,5.4,3.5)
box(9.4,0.4,2.5,2.1,"submission.json\nper test input:\nverified > partial\nprogram > rule library\n> identity\n(rewritten each attempt)","#f3f4f6",fs=9.5)
arr(10.7,3.5,10.7,2.5)
fig.tight_layout(); fig.savefig("figures/fig1_pipeline.png"); plt.close(fig)

# ---- verification precision
fig,ax=plt.subplots(figsize=(10,4.8),dpi=150)
lab=["15 matched tasks\nW4A16 (submitted engine)","15 matched tasks\n3-bit klee100","92 shared tasks, 6 h, attempt 1\nW4A16 build","92 shared tasks, 6 h, attempt 1\n3-bit klee100"]
right=[24,22,61,50]; wrong=[2,6,18,29]
y=[3.3,2.5,1.0,0.2]
ax.barh(y,right,color=C,height=0.6,label="verified and correct on every test output")
ax.barh(y,wrong,left=right,color=D,height=0.6,label="verified but wrong")
for yy,r,w in zip(y,right,wrong): ax.text(r+w+1,yy,f"{r}/{r+w} = {100*r/(r+w):.1f} %",va="center",fontsize=11,color=INK)
ax.set_yticks(y); ax.set_yticklabels(lab,fontsize=10)
ax.text(52,2.9,"Fisher p = 0.25",fontsize=10,color=MUT); ax.text(100,0.6,"Fisher p = 0.08",fontsize=10,color=MUT)
ax.set_xlim(0,120); ax.set_xlabel("train-verified programs")
ax.set_title("Share of train-verified programs that are right on the test",loc="left",fontsize=13,color=INK)
[ax.spines[k].set_visible(False) for k in ("top","right")]; ax.legend(frameon=False,loc="upper right",fontsize=9)
fig.tight_layout(); fig.savefig("figures/fig3_verification.png"); plt.close(fig)

# ---- cover 1200x675
fig=plt.figure(figsize=(12,6.75),dpi=100); fig.patch.set_facecolor("#0f172a"); ax=fig.add_axes([0,0,1,1]); ax.axis("off")
ax.text(0.06,0.70,"A verifying tool agent on\nQwen3.8-Flash-Next, 4 x L4",color="white",fontsize=34,weight="bold",va="center")
ax.text(0.06,0.42,"ARC-AGI-2 public leaderboard 54.17  |  submission 57851995",color="#93c5fd",fontsize=22)
ax.text(0.06,0.32,"W4A16 experts  +  FP8 KV  +  MTP speculation  +  train-checked Python programs",color="#e5e7eb",fontsize=17)
ax.text(0.06,0.12,"Team Hack2Publish  ·  ARC Prize 2026 Paper Track",color="#9ca3af",fontsize=15)
import random; random.seed(3)
cols=["#000000","#1e93ff","#f93c31","#4fcc30","#ffdc00","#999999","#e53aa3","#ff851b","#87d8f1","#921231"]
for i in range(6):
    for j in range(6):
        ax.add_patch(plt.Rectangle((0.83+i*0.025,0.60+j*0.045),0.023,0.041,color=random.choice(cols),transform=ax.transAxes))
fig.savefig("figures/cover.png",facecolor=fig.get_facecolor()); plt.close(fig)
print("ok")
