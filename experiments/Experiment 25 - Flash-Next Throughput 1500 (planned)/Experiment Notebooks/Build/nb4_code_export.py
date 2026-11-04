# ==================================================================== CELL 1 — config, inputs, lab module (RC2 verbatim + E25 NB4 marked edits)
import os, sys, json, math, time, re, shutil, subprocess, hashlib, tarfile, threading, traceback
from pathlib import Path
T_START = time.time()
MODE = "FULL"
RERUN = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true", "yes"}
FAKE = os.environ.get("E23_FAKE_SERVER")                          # local CPU harness only
BUDGET = int(os.environ.get("NB4_BUDGET_SEC") or 150 * 60)    # E25 NB4: hard stop 150 min (install + T0 / G / C / P / T0b)
HARD_STOP = T_START + BUDGET
MAX_LEN = 131072                                                  # user 2026-10-29: thinking up to 120k
OUT = Path(os.environ.get("E23_OUT", "/kaggle/working/nb4"))   # E25 NB4
OUT.mkdir(parents=True, exist_ok=True)
LOGS = OUT / "logs"
LOGS.mkdir(exist_ok=True)
TRACES = OUT / "traces"                                           # memory rule: the full text of every long generation
TRACES.mkdir(exist_ok=True)
TMP = Path(os.environ.get("E23_TMP", "/tmp"))
INP = Path(os.environ.get("E23_INPUT", "/kaggle/input"))
R = {"what": "E25 NB4 TICK: stage-0 gap timeline + FULL graph / commit graph / P5 A/B",   # E25 NB4
      "budget_min": BUDGET // 60, "max_len": MAX_LEN,
     "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "setup": {}, "sections": {}, "errors": [], "order": []}
_SAVE_LOCK = threading.Lock()
NB1_EXTRA, NB1_ENV, NB1_KV = {}, {}, {}   # E25 NB4 (NB1 hooks): per-launch trailing vllm flags, env and KV pin
NB4_PREBUILD = {}                         # E25 NB4: per-launch env applied BEFORE Merlin build() (DRAFT_HEAD)


def save():
    with _SAVE_LOCK:
        for _i in range(5):                                      # E25 NB4 (klee100 R1-13): another thread may mutate R mid-dump
            try:
                _txt = json.dumps(R, indent=1, default=str)
                break
            except RuntimeError:
                time.sleep(0.05)
        else:
            return
        tmp = OUT / "nb4_results.json.tmp"                       # E25 NB4
        tmp.write_text(_txt)
        tmp.replace(OUT / "nb4_results.json")                    # E25 NB4


def left():
    return HARD_STOP - time.time()


def log(*a):
    print(f"[{(time.time() - T_START) / 60:6.1f} min]", *a, flush=True)


COMP = INP / "competitions" / "arc-prize-2026-arc-agi-2"


def write_fallback_submission():
    tp = COMP / "arc-agi_test_challenges.json"
    if tp.is_file():
        tasks = json.load(open(tp))
        sub = {k: [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in v["test"]] for k, v in tasks.items()}
        Path(os.environ.get("RC2_SUBMISSION", "/kaggle/working/submission.json")).write_text(json.dumps(sub))


if RERUN:
    write_fallback_submission()
    print("E25 NB4 tick diagnostic: competition rerun -> fallback submission.json only")
else:
    LAB_SRC = (lambda _t: _t if __import__("hashlib").sha256(_t.encode()).hexdigest() == "ef393fbb2ff7669ee4c2c99ee841b60da66514a3b86b7c0ce81fd5f64689ab5e" else (_ for _ in ()).throw(RuntimeError("e23_lab: embedded source sha256 mismatch")))(__import__("lzma").decompress(__import__("base64").b85decode("{Wp48S^xk9=GL@E0stWa8~^|S5YJf5;K_bFNnHRDXbna~F$u0Pgj)JG27#5$Z?y(-4y`r~GNitw^S8-9YLUa|c(I<zlpY<#1Ng|!{;sE_(y@QAm{+XUC0$9Wo_B4PIC3RK!O+U5`<kC926s2nq?Cuejv$rZzjOYWPz;7sm6LeHL>>ku(i)C6zVWq&QZ3++F01hkKCjG3Y8Qr>u+$I?;|RzAz_a-*#*=T%cj*&N(f=+mO5{RzcHUwwXtH=6@r*#gO@*DKa(MoOAUs<xf9vRT?dRmlvaz0K`rv?kh@g!*yL6F-Ja$;&(Z%oqX~JE!VX^ld5QUSE*gCLD$b04(iit-UTcBL{IQ3l5aRaC7<-RR)cBN>$vI06t4@J!7PON?87C+1OQEa`Xe4@|HS->F*k)^on>Zx_ByK{buu+>$saj~O2fg2Cm@PI8w1Tt>}7#1=WTte>?bewTyEcssU4oWzrpT6sUZ4y)-C89dPUMvk|JGq`oKlPm<l@LJ9NAsnNpgVWO8#i)*p1?{i(%zFqP>h@%XCf$ebF}%~wUtM4HLJ|zZ)dWj*NnDzG3*%HqwrqwX2oc2H|{xcmmt<pNgzTLs_nfh$Aw_{Gfv8jKUw7!Qy!SsFMJzG)iL2c*j<*Bj4G>-cATvvR0yl8VOJVrH`d+0vgRzT#8QA)P#(L9iy7;1oyNXvk@7=lgXPq(fiSeKw|~thn)lr|R(I)q33SEMq4P~Xj*$r&Nf>>cJEX8YU+pYagqtbSWUHpm%GPQx;}$c=U)heFNl`MSsqaU284+ZuNRQZwXBFAVdjimY=EYI1<iuv5xep-WNIE-(qd%$>|89+iAX<uUY(?7wm-;*lr`ZbJe_q!It}{QvQlbsNsx3SJ@~g1+sjutCRNM`qzeV_D?|pLC8AcPxf>7}cZSRx~DdZI*Z%NO!`;o<3Y=d9XguJ&C_q#6yMwirL2L@L}QW&*oAPU+$d`SF#f=F?nJ;KcnhR&nrFEV9L*0^hS_oambm3rZq*pA4#-PV&+I{^DUh{mT)G}r@bbd%o-MnUTv5w;pDnAFPN5Ofn&gaL7GSZqxm@GP==7f}r3bXRK9rpQ`YB)t<cK+&S&oBFy{B4i5`grOo>yKT@PP>itj%Sb(@kO4gx-J+}k&9;@J#`R;7R&zO`T&)#AjW1Y3``zKf0wfnIv01x#+f{rMa)<$W&BJ1f*L_VkUmfJej9xZw{aewieJ=$bh!kG8iP${~jOVJs`wCsd^*w4O{iiDol)#s(OtQi4P8@Z_p{o5{_<wS&z4f2Rk4>3qlv_z)Xg5x$LOAK7{}tO>;|8f!AWi){?+9lxwd3Ua%GQPuE?wUN;?5;oHXG?;-V<CVwFJPpiv~<NRYDKEDtD}$Y?*axa;gB0#%%+_O?j8y<;@5Rde7G<x__0}oE?sb`qfXH4VoJ4C(lKTmh{mK4U1s@>#S<sW9k4=--gP(<3A+Bn`JD}>ay!~8rW*FNyoqfIWM<&2Hq)4Py|Eir4uqr;|j_vFW@2e#OLmD%$oOL5R5!Z1uwd`f)%4xNoDZ#vpezDD)Uup=HAJg>_;7<qW=18W-F1*%Of8OO(e;ZoTF<%**})bMDQF%AMp`)Rzx*6jaPg*1X1F#Gy%^$(^K9mA+=GG8|$t8I$R7?TQ%`ijcll=9wI4C?zavWu=h9}GB2kVXqm$rN=5*IlRL#L=lwz*vSB5)8S+uu_SO2goBZ4Wr6xze)iBC`Y|5JM4)yN=!Cx<m>CUSRh%y0`9<~e+5q^SEP{7vb-IHc*5P?kWkzkFBD}vt613H)T2YSpFIcxZrJ`<H+Jl^;LVkTb)bK$&OFKL<`m)knW0e*$5IX~$vjiFUkySY?dzscY~z5R-x>fAoe9b)^xK#3sQ2eH<F57nrrlRaLFG$N71k|rdNp~5Ie-%0|WI;@>1JisN%3r^kLR*jjkD7h=uhi=Xs*R^lr5PDq{wb6v^vKnz4gMUlkTQrTr=S2Lr5Qo~t|7{#tMh`9lqXYgn*k&!y?wDiKX!tSaaT{OCfEa`<GJd?ArEw`*vyQT~lxAR&n3pi${8z!j%I$urV8#q<)%)aU-|(IKShVmZvznf3#wRWkiif`|M>zb}WS%X%q*38Y^0<pf`wO6!3#Zmf3JLjN!4`oD^aY>9l*uK%7nDA7;F%s91M)pMp#-8KKFwTa6#4!HQ`h8z)<1j0r&DsxYX@>%2G+=eYnoj17<0CjDF`vjGMa?MQVdKGA<H&Uj_-a|<mdHx2@{UUa5EU1gbAGLUP2(hB3WoHQk+lXu!AWybN?|An+x^nBP*-gKqTi!-n)sBh~zH)6oeU?9Qzt~fxP|K&V?pDi$3EYFA|`2yZ3mn9h<!_-Es!%4)jpU#3Vg1E?-z)WRzTa#X~@mlidY=^EGW3X0HgE7U{1V)9hx`a+kw1`1!Mu>Y`o_M9MwKp~zOaNoHD@jNpj|fH*ScDCke>;7A+jo-sR5iFSqjuXWV#V!qs0<<*KJ3iMz_j|cu~+0TE$`Kv4f-q-0Hk$D0ux5;(>JFK!tCg<!1MA`FUXbA$A0O`vh5js1?hkfI?Y(q|!6ZMg-d^yb4J4JFwtfAI9@R3~goZ5nMVMC0cBYbp&RTA#pjm>HhQ+Y+p(*V+Qv`V^}nO=2N47>nIe;l$CQ&w-a2J;$|IVAvG4GJa@sh=}*EWw8;wwP!bP6gI@4hCLqJn++;(9yVL0);C-REaGVW>*S=&6$fQzIj+XeyUi4wLA%1tG(|JVONJT%(l_j723*r5TR%0%tc=1X2*0HBP8u5a8#S%iq&0hMizvp8V>t#h%OW|S-*4@R9tjx?>_2OtPAMBjDMD}=y|w<vkv$w$IuURamGvBXLOnImP2}!&T#Y)10er_NX-|^RS_6cvac+sD&P1@HS^!Vp6_>tWBwz|7MhffTCK|48Hz!ObuU8KcW2maa`XRd3Lc}#{af^kJk^4^m6TfFFA_Fd_QmQcDP>LnmocihCX?x4$^;re6SY}^LKdoy>q()ubWDk~`yd15wpOn0)h1q?mqa-=34aU6g5`-gun>1huM8EiHZgxdKlV1eGfY=c95(G94Zs7~X3n(^EmnJgD0vSNQ}CSBM{xeV+uG)IcH0AxmB)S2^VT||ELDo%s;x|`aW_g;9;Bi6p~&;^ij99OXj9I6?8ZFVcnJEFc&km~H3OOf{RR%Tpyo~h>y&%!P_%5fS~45&p-2Oa^B?1=#kpU*<BB;_FsWniNv4tzOEyaCXRBXxH&3(G<>9C6LbQ>kmQ*?K<t5X4H8)2=i^^CuR=}>h&k=1pVnUd|b?~cJj$%W%LH1{Aii8wyjIxQ>FaNhNpe&ytZSZ!h)S>#HXWaT5tu2RigG_V#XO2CSP2b)pmVz@0Ik|{7?VHl_IAVdEY(I_{GhC-J=%jmC+bZt*dRj{8`N;2AjhY+i+(svJ8;le8$#Hf$8xLOU9K6&c;<)LeoA#W?PDO+_dzkagRki<jEn9bEUrGa1Xqi%eG3oa7dU9>Z$NJdT)@(qq;o(}#J&GwwYm`R^r$$Dw_h)_uJJHgv^UwNGpkP6Y%e$gutnNcH;I8N03ZU(OHZ5>J_tHj>H;}||>Y@c!D+dA~O#BZSMg<1>@GyvL6WXck%=y!yS~#?*kYj{_FO=N-a7KwTB%#w;G^$zZ_-QSLlgtp86Cipp>OI9MC}J=(M3lsW6|($~2e#0jK*T7~5KAdDGkNNB>8vR6D?U3k5rBSwjx5rgJ2@zluB~I79AulhZdc_bw_FSuQC!;Na|oX7@S_E%mBpohG4GbL?3LPgUo(ke6IkOk4GUNGO}BKYmk~O#<>479Fyh1S5>E#^7@o(gVNwOQm9hnFL%X-WWKjgm^<PJWV(b5KM1(sMfr0xVnI99ct$xh0wS6F7ajbN;Enx`#{6j;wGRa1}^z+gW|8jIIK|2*inANVw@t(1EeJ(CmR5R-V!b?-dl%}2+)tQ$chy*W0&%y|2&5?j;pw==6+ATb*>CkZHWSyRjAjH*qr`I=$xG{!)3iOKMyxh9^YEEaBQJg}K`hD>+>ILHTD(;*(<VH5<LMO=x`mfr~5wQK*e)ji-jx{A6B+0&)P3Qua8F=)ADff>0dn6T@VMSF0Iznz3iQnLGa$#^m6tez)-pFVYx^AeZx*<l?`}xt7jGE%_(Ih5`FuqL1ci|6DpsjV0^#XdRn;#-mjj+AesK5*cwx45MW?Lr~r)aS&@b8zsV3!-tj27CekGv)?ss@b)(ZB^kF0yj!h~UHJyl}z>bOMbi&=y+}Q(Avj%c%i8Z4|sb5CX4@^JEf=1fd^0Wd<7Fv#(XT3U<lmlQ>t<CM3AGISf=roM|_x*$mwXDEf8OPzB*QaqH=Hnt0_rjVR<1uCD{5M%lkAIGP3tVn+UU-V|s7f07kwJ_4s6JuCU)#kp={>jr*{bP2tzl7-1f;(cB8ixjvm2bY23&Rk$GV`VvMgipxAAMZtTdz9H8*$@8@x>_!_H}~h*I~*xJILj|g5!`fUG*Bf^-=qc!4%*8BPd87}@$idnxAYe-jk+zmgFHPBGFn`7z6WhbwSUK0dpdl}5O?j5b&;J3?#~k{pA&8)|0K9f7Cmofs3@8e-})9iJfE6mQ?n_I#4t@mFI0vF8VBj7=QY8I?0`XXTXoGV_!;RX2>d<pf5$K?;4A_ZbUn)1DvoV02D8l_3@=K~j6(7O|B#A>J3l^YWYU{<xfT`ScJVwq@h@!pP<M=+y?08hWlU!e+fL2@tio)R)!CDKnReI&A9CyW|I}9O-NsKZOoth{M<c`u3seaOA-G>s7Ihx~bYcHEv+CQP6G<1O<<(Z1p)UWhPOEg4jbCp+)*7x2E#ieD`XfgkGBRzFra8fD)fe(~m8_T7{B&iBK!?c@2&?65qw#(SRVF%3R3~$scAxA@M6OlMLp;oS|Lv(X<-m{qoTS3n_S}3(DPdLL;DYs*DXq-dxP^{(h*88gVPafyW6kE~N03()BXO6wEygYMB7OW$jT$FD7h3`^Xm$qA5;<pP*;`6ieC`^D#p~euNgOXZ)uF_^eKIEy1dkS00rmR!^o9{F)b{R3r`#2cWj1|e3N@{g7?NFjxwhZ!{76cHa%M1_L;!U%eyn?`PKSqYOiit+mna+k${g)Ukwmx4;}{hZQpMQlNk3aj|J$L#<d_4nCMG*DS~oqJFh276ITDX&?Kp{<S>!EGiwDK?6_|Hvg)@FF)#BB90$%Gk<Jdha3sPsDN43Ds=~MG&JFnSpw(xg?@9x=HNW0TroYIGGxP(dn0U#S3VD!v5_q0WZ@hlAo?I~Wu-~o&7LG$KVjV+P;e?w;vEn-bNyqf*B<p<mJz5j-BHnylp#$G(eiST`3j;XjJpm+}e0&<XBR|Mb+*DtRGp^`=}D@iqlc_<hIXOEI#Ja5iS%?VbK%OvRIof+L$=nw&JB~ADxC^4<|*XJB+|6~ZidkJvqDL#woWN2f^aYaiesokhI!~Uqy;K!cE$kqytBI;R{av8T)R4A6xduK;GTyuFOob~8Ip(a);b9iZ>4yCG~nK)06PVNzw#9x=Q1UH2<YNf>He0No*07PlCyT$`EQ&!utV^FUK_+ig$2cPWa;XF38m<hVWHK+rI7-+10cE0Xq7Bvw>M&!hd&o@aCv`eqw$pZ5~OaOIQvEq^!;)R1>N3giNO5pRa%*vP`L_BkUa(j(OxF^p(B~I}~;x^~*m|ZjlO)rRtM7TH}!Uc^Az(u&}+!s~6qY``(5U-;xw_n=GnTQOlTwwhr*M|w&!XEPyt)<7)znrK3q~FA!&GP68uyQdA17M{%r+c+JbvV|r7mKsz;S+=+9MoABuFoVqy;X8$*TY?rE>rL#22f15pls#uwqJ6qDR;Z^JJitI3?^taXZTT3?}hfh6CUvwW@U<`gn<U&#hs)N;h(Mw?gc%5s`#Z>H7kL`8SGZsivXLr`$DIr-RCULbd{OMat1s%Ae5jWiOL(>`tEp4cS=EFrs6CiZ-E=9#gM6g{x;8mRy|$-H5^Iq6(eOl{Q8(b6`L-H$hciA*rj!DJ1!zD*!&U|tzk4rO8(pY#HG5(Af=X9uaXDmBaMC?Oxk3PMVk!$$-`yAf_j267T4nGoK8GWL$TncCq*JQ*c{VOpS#6|rTj|S;cbYq*9_~|5fsS8=m1OMfnln}>$9N%-{vVPgH!pK>w-o<l_2`zj$F_r2ZQHO{RHII?v1|I5D3JSr;<?ra;^L9-LFI(nJA;fI}m*`$J0m}D%|coZA-p+f1Jg)ioBFU4k$4#1@-p}%Bta4-?DQQ?=GMSnH|d}V*#w0e90^WA2)^c+0JnU@l-{s^33VZgqzkN#JBRQD5z?v-$@YA?z?JBus0h}gHz9)qFW3uw2`~15(RppFuCj9XZIQ<E`@4;8nk6rY+CA8lL)*paOp)&V=Hv~FZ^WXW?A8D0N?{M54e;M%?7oNXGJ1r0|iI)QuZ>h%rx8&Hrg`G46Zm!Q~{ROElO*P^A86UtFSWr=kIz2&iU<u;y^uJ`bnBhILN4(U}d}m7RpKFz={fL`jxKNS+)a5xjthG*Qh9NzB#+J+cH4x*#r1U!TK%)1_tO5P!oSeg1~mU6)>VVpdsmMazLD9WcHiD#b2hmvQ;`|XQS+#CHUDDnTTo_6agX11NExo*AMN=SfT63uq``cA;boP)NT(0c^nsUNAWY%)I-QkOq2T68f~<tF_;*gThh{8k#BLEhe6_FR${E*(zHMSUN|YmLEt7hugyI0Zp)*g+K}ZnjK+ljmYEH)p-rfZU7p4k4KTk`;u(iLGTjI57oy!(`%|tZ*^v{?V{uG6NzDXuqja!8DdIe)_U}1{n-h7(Q{aIf!sTwoY^|V<{MPi*KkgE`5srPrQ7$o7aCZ0Y-t^S3aSWnhS1Y!nk|Y~ay`n+tSr|*5V7w}D{j{sCtJq;^g!{#QM||N%mgt|MT@O8w4ruN%^iqoD&W}nRJNxL&RVY*mI-<v9!lP$_4U=ylK2>BR{cnwW&T!|mL*URKcq+c&o*DpC?i&+_@;Ng_lx;-L3vfY&6W5oWCm5k>d!_v`%Uf>}A-vPM?7xkZE<e69>?UCgeF*KN^r=F?d3UbGVhW{yeZfKBhQ;rz#EDM+w2&5!-m)4Coybx}uyH?!1vT(-2`He-n38Z&2Fr!sX8mRx8@UeHt!OS0X(j&|ME*>&O6VYTo7$sGtXW2vsA@!3m?&OZ(2og9?I+YW0)*j7qw~AE^=^q~z;tBf_GzzvO0o}IX7o?G@ro?JG82H4UeARHx`_;f(R&(F?73s+>*c)N&BJiWD=OPK4d`tY^K!CBL#k8ApNALF3Gy2gpuScpT!sYIE}Z845gl+X>l+C-Sg`t>;0cUDlEC`D=qZ~>Y?=w>{V63F2b{jLJQ{qc4Dz2Q(INfQA?z!({jC0d;KyaB9piCe$+)QCgY4v7GfpR~0=pjk*DZEvOS!SaSrqhf-1i!HzQ>AA6)U2r8&87qOZp_H-h(>33D7S*SIYQpQm+i9=Fd3-l8oFH!IB=!P4A__4aviR#nKl~u+7cXI@>ZGEK7X*r*W=sUUySU%#<EtZLxn8`nYWMg4z*Jv8-H)Ty_I_!(HIIA*^^)e{#_#kl?Dl@+DKc9Q3*oDU*Tl1>W>k!0;6Kh=7S%v8Q*_Ha7S$R|#?)W1he%%hy$Ff@vD4C%gRFpiL_QWGabH<>4#$0{TXZecGhguava>bb_U~gN^6Itq3Kc`;Vz!A7gUd#Y>ZlwBz2JX0!L4?b|*fOSx)4Gv+rCn=F>%0ZYDlOl=}}w)Qovlx^y&lRB{6BdG5fu+Pjng$`9Qy2_Z#T61Hsf0w?K1s}ZWkhm<e%ASyc@Q)XHr1JIBAm$*)A+Q3fkqD<_z^J0~+yfAcnoB-M`)a>X@LsgSA0uGnN+R3usA#Z7=1e;!@9>KA;u300I!5OfRhx+N!BkT!ky7Z|DNVG_2OOf7iU^ZJ@j!bdLx+=JXw4G51<u`hOcOw_8mLiVjzmL8obM&4*Kpqh3J;_bp6}G11)_-kqMUgQ`$W5n7@3R}jl$13kq>Cuh2c==9yGU6U#c^d&nJIe4SkoR>;du{07dOgLQVNU8OQ6bn7ON<vK9&Zeyb*(KrE=jg+wel?v~VSIM@?qaCj`ejQIMuw=42GC-i>|{WK<^tk|ESK`>J>O0jpN0TLRMx92yg4hx43>1|?6F{J|4f84nd@53h<w@D%ye^L$k!*GjAGss39o{C1Cenuj7-A`?9Jc~9rG4`b6HhYbT1Cl08gddUG6;If$68o%(__NsW-k0o=1dcorJDYDh#$FY7ysq7iswU>u#v#Lvron8RhUW}5UOYp@!BbPD18-_rwy1s+gXYl``}JG9;>T+fd1o24CBmR1PW@E$N6~&{``5t}IM9!^(Wr*W8t-l@9L7Z`pCwuv`2`~pqeE{*JB9L<*l8Wu;0e-Z#S_q;=+|$NU}kP(T&2ICcTgQ91PTNf7LM1=RuC{mEzoWUg^RwVXvhoy!Fb)264VJA8N_d5Vwz1{eH)bFjz}7PRqOR#iLRe#O>@$%ei*A!Y~Hb7_1VxV4wre5zKM&FGiV7l*}{A_y=~Z^s;|ZJPXYEKx(2usR92kb6-{mYRPsbPu|B<)KmK+dfzz7a1Oi9B0!b=DIpi5kx24rDyE4x?&_AUI!l&2F0ch&R4K+=~A5l0b-lz1kSUgR#YqAP3!0s6&E;{QDd+)M!M)zR(*EG27GcSSQNhVKq$y^l0xJ(v~GuMnQIvuMhG5&B+GruxKs`#f$UsuxLqrea#9D<Bz+(Wz@O6iqferE;^nKIx7ud*9`V*|pz^{4ix6AyJ4){4O=t_L4paV3Ienw(=}GuZA&sm@bvoPbarGy<Y)Ri?fG%x|?_8k&aVo#8k+3WR~m5$YL&Wo1K11IEKG>Cb)SJIQ|;ZjxBmze(5J7vdfU+!d4}J5vquXHQ*O47oO>rFQX7PXxpOY=RRpR5V7CR0tXKfnPzSmX&IC#4dE!^kgT>`M$bFZ#<iHkDMhql(tohQzllI%6e_+*Imdb$_pv%@~mD;M<j3YF=0=R^|D}W`Gl_4fL<cwk@M2<;MBMoTFCsuHpzg5>EHZ8LScU1c#+*|20f5OUD%a$Jq(H&hkpmA#$kul3|{2b*HiFwQ$XvdlYDzL&hy^n=~!s)*l1sI5`YVFHb|~A?-_orp1;wq<Qv*M+v`q`5Y|tRb=-lQJ6F<?k>3@>9fu7nPv5=RUt9YxZ&Vzt^lcYn@e*4>%Wzr7ddyz+PW*);MW09QH6F8W`Po1DiPn?~K3%QKD`6mxN*!PTHkD52wO=74zxvf0v01i+5w4NPP*I`4F#puHY;KVk!wIVD<t|V&KrYFQ_fiAnftY5X?Di(5ed3gl=w;mTYQZk(Kt0NSx!5#~EaApY2);R0;0<p;uNY9a2OI1|PRBP4*7oRQ%|~vwq=5$E4yn}!0bw4xF|^3B$#f8N`pOw25f$3p#}H7K<Z%2?!?(gs4m2>@Xtl%UL#&)3FHC7V{?~!p_`uu`*^0ZIz1Z$U;&x-Bl~9=JKc#xdA9qWM%r)g0iBGv??j*)uG)zEq95B&@4fawLYHrrh=!{)vJutH{JmpWmV&5he{9!8jnc^!3(;R*K7n^a4-Elw0GbW7?c>RmTkLkETu3)!T60E|Sj@|aA7bEncEp5nMC2{DAi?Ceoc%I7VPZ_Ntq;jJ4pxQ9g#dKxt$(H&<njy@5y9v3uk@eP{ZXzadBFH$LjP)Ge>%vb}Dwvi+p`n|5g7<@V;7HAIt-@g9gyz3Ha^a%93R0lQWXb?`@&%O)8vC6btJ|S`L`S4el~>>68E0GfYd4U=#drmXO`}L}9X@fX2=8_&Tptph$dV0S(cpeiPWB8?N_2(%0$xCP={2s^52X1nc6<H4>)^DCcfF8kYG;C7_aqNx0z<;(wHf0)>;i2T)fq0Ly>LYB8hQy2QB{5*Z@Qn4S#o5pgQ?)xu^HV^j${-YcCHvJGm&x9R!7xN7-P(`8ufHztj<}&y-$sd9!Zy@w|Qobx_5kae1yXf0)<e_J<2{}OXoujSn_2YeL(KsbSbyGDBjZB!f7stz0EN@YPf{4t~V2vy#nB)-^9vtPP{o-cum$u<uWu#l?#!|QEwgX^veX6C1?6F_;5%>e*5O{z2nHm-{#hrkvl31<LOi1Bq^@&PUf=iBKke(xlmbZG6ufAJIOT(`y>vUoauvzRqN<%>t`=@r`g178dag0eC&bRQ~l7_Eb*XR$@xx_oo?kSJ5Aoh2sqkx_%KGY0$0{jlp-R`XI#{5M1|i4l>#bQj!^N5#pucdh7Q{{yOkoPtu#XeCpuhy+DmXBvoc=JC1RRnpixzN4Ohv#pD4hFyw~hmfmnj`-gw7g3s<>cpyWy)f*-4~$QkiUqs_^58z1r@FqX&bf>LRRM1Z>t@<ta1L7@s^{4obnqU|sZg7p5}h^{p%Znu9>F@UR=K#O&Y-O&0wH?F%U+5*TcWmZ1rjX$^p_`(5!rkwm$Dncmt>iENT?Sb!a$uG6!3^Us3<6~s@6#4sR8h90fHyfu+FMjpFikO6vrea7TpL#}m#K`Nbp?cm_SMCBa#ysm5)T`sazg6>KXHF3Y^wF?i>3#`D4Ozqj0henm+TI!P^GBU28EXXlsDS0Y<X`q^09|btkrrnWgX#yJtE!3JvjQUi=|<LKmwK`D|JFaeePpRIb07Ww5^sW$7w@F^&uXp#a)v1<dP^}(DK3aSLYLf-ILI4L`C5z@@<;^5@tt>ObXF0Wv!~#^XA{};ahvP5?*=?+uGxSY1KGI|8*J?PZw3tDB)42A5ScD~s^jmG+82wO<dNkiMC5Qzjyr4sX<q-dLEUL*W!m}OE_h4adm}{#?^zg{p3bbZZsimvU6Od@@qO|lJFiX|W3>b3kYXHg&`#;4=66FSDB~+s`+GI@D|cIZ=HnP9J*SUG>VoY_M!ZaT_(6M@)m29OTq0H6npN%VC8{eCmA#3KD0-m2SKyPhVGvOH(WRTt%kyUWit4wFN*;#v(SZZXQ&_Hxzl|rg)&8U2xlw0kO4w2_b?A;K26WR>Bv=juoDwNkmVGUN(O?W72mJ)z#XqMhCc`=wf9pMt%7oO5Mh+R8n(QK2%|BHz)ZL$hhx$w$Pl-X})F6%fq_KN1Drc*}Lq~jqBN$wqO4_N~x}^7o%4Kb^&isEd=Gwb4yON(zvmKNvgVaEWVwI8-b0F_we7(n!f|N`MWiuZ}`I4L&{=0Y#k|kYmINiqlj(MYu#-GAs_RGBef9aC)lmpA-L$ab|pIdP?>PAlF)?#E~?bi!sfR2o?26X}fHt2UB1gujnN#VM#b_kpwG9+(7l2~kto`4`Hk&dsuuw^X4u#vX@1TZe1C;PrtDa?~w=Y$n!?>>7Hl;<CedCV78^Ky53JpapYroR}!r#v-d*4LQ}T1S9`lK=>_G_{o&oK}M4fO$s<oA=k29NUIK={h$4*XI$9NS<Z8!G}W3yRW@YXY}vN{_ykSj&zwJf2Bwr%lM7CNn2dgigNt>e)zN?-P%47gV(V?tsgzdR3Q0Pk>T1yUJm^NzR|$&WF8P(CzK2P{E^=g@tgz|@%Ps0apXf;P-vLOt{}_3S#Ad4lBrP%rRNHhV7G#}-F9p}`~pyp66GTrY1g3Dhm%1P8lI{$CJGJZO|5W{xmmxMN$2x*!ca!Jpa3$WB|Y1mIu^mwFO(()SBUebGl6>~NC=%AZdF<nfX+t;vtNUqCRC)eHqoCb0mQr;Hi8gos{x*C#AbY3pz#6|bP0p`8Eb9j4aX_9JW9hyaQs1z%|UB=5n6d0-Unt5un(+3G-6h`breyCIb<tf1mCBJ61?vBL^=^Ejo;#K{H@PjmZ8v=%Y0jI2(p-9RmGZI2=|L<J3>T^pIN|}_A3`7Bt}v=$D2Wst*GxShKSg~s;byA_Ax8x=!`m?hW|c+Z&kt_N<xfyf|>CAaDRSE0%ZiuRagBP{Y`oYP(E;5P@?KtOh-f<y2Y7J(x&+JvL!h$CexVfzKJG1&C?}R=~D*Ds{*u1%V#@A82Pym;ZNG?rXc%U_TO^snkC4`x(+YEkt>Y=4UY%S(p)EexRa}U_2<-S*YUgh=`KqW!`dHsI_7MDpl;7_$RgZN772Qp@VgkZx$_?ir!b){n=Xl{j#SNb&;W2CVttGIAZe3nGv(S{M4m2q;R)=C{ZU$<Vdg?Zk<1Nva8-Agwu;aZT@iPG+wM08h&$u<`*>e-rmM3Z!++#QL+O*}?ScwtIVA?H*Zg7HV-(^jC3m*a<+mP#ch5J=tsG!jj{IXq1NGtp3WvDnscr$QuEa1^iWNt>#z;w5$VpK@Rnx;6IPyOwC_fW+=OzB<asj&TR_k;1L-CkCVKx(Ns1r^&jiT8aTvIVOs_<VtKmVvvbIuo0w9I!ObBVlt-Az@9{4>?ew*uVS<Yq|tKfCo1jt%p>ytt6NcM8wdLG@>rVRb1l!xm7HUu3}9)`R(!YWT2!fg?hGS8(Z6jjdmZ3>jkacE9ZJ@@6Uj%cegpd-&Yhf5u{GArgt=shJ2le=?P&?u^Eqf~G9_+wRx64@i9A(V2Ie;#i@I`v**53r11()2fKE22Mh}xq1dAlbOI-xdNuNF7ZToAHfEmphb;YiA9$m(~;9WGyF*ImtxnsygEw4)CUk>rT0?qxM!G_Jl78*(|v7gyWj0hl@9&Y2@Zr`-Ku?Ck=hMq9<wg{m)AFAOQ_{mi<t(T2=|9f;4N861d*H>wrJw1MK!~)B?ZJMGUVcOYUTk5@Fr7cpDYklM>p&Ex7D+_SQaC^(pgLfr)sIF0?fH01knm%%Tub`S+VHr2zD<ktcD?7K?1#7=53*1IV_R%Q)ZB1^Q%`}$7XSci1HI0ObzMI@Mku-fQn#e`zRW`l*{kABxRCbB^9Ire<b<WVNKHx9FL#TaI<fia1W%O^=CIR=a4W}RmS46M+UXJ%of4<qb-f*fbsXD<{>e1)A#jq_5^5##g}6i*&5e+mmbzP)TKM_A#K$yqBpyn^)FAMxy*@7N*fVjZD_Yb6l-SdHETH|t2rjA@l_@krb|~87Ptx)y>T<u5fgxohLm)sCE4a<80Khgpm=JHK@Y+M{30reJ&yi~rt+?jB2e+BzbQ27K2nt%PJyfSd`9XIBU%L|_>5rDlQX2>$(dmMJ^+{J0rLX4Gbj0jNvKo_=d9S;NAGH#PdJv1{MlI#g1oZ^KdHx`i;0G?V4IZE?`8Q>Iaz$G>?&e&IbDmG45@x9JDvV)D*_hIz0V5F<)^#lkvuSQjSpB$$b};~rCM`Ln*4+|Y?S8X!x1wEyhU7H|HB;E#Mu)w*&vg8`xgO>B6T^3dLesy0%8;fw2li&I$|KAeb@$$q>>SZ(SVoBY2{K&ywxOyC8`*p$@kKqiw$}R3%8+s{S|3E3(;j1k~D0dayki80;j|*E#S4=W#iGyf|9r%C&AawTqn`Ki_VSK&p?5_k1j5Xx7zz?Gl}*jVpT5oXOmq=>XZb9E47ALkrN`C95c1sq9W$jF@1j@%rujuKFTw$qtj*S7IDa^O1X3VIl;eg!32)k{eD9<Yv7ad7gonS=+*p$HX#f}B-a*`jlm^x<9+Si80>IZc6P_rf6{XM2Dj$wclWtx<QE%|{$CzYfmg6rOg}NCrF&eAvSFHerI>5w7wmaeNv|7uYsUX<>2d8SF?(ofFlr<<>hktcqw{qEl9#Svjd0~AS?##BI`e#b=lwDB(8qAc?snCG!jTO_<gHfE7ZGPJmDofCc{H#gSTBngKC4{h*N$xvz{88Nl2z7~+&b-FvW`n+CX#?>NIEUJql7|htB3fR<pRH%l0Pyw4103)*m9KnI5_slLVQeO>_8FhfIiY7|7d11(Nc<P<PPn2gbfua$0`a8(vWNb=5%9vZ!5Jfn76D+03^JdY3#YF-hpo){690n7M@0Z3NDDaCB7%sF@}8B2*I1@2t1soGR%b>)28PH({EXmVre`2B3$zG<M&<=-IOU*O8q-KV|$BH?Vw8AYNXRe8eV67Xyn^Cc=R!Q7nUP@$1S}URrMK6f^D5&?jD@`+xsYm+d}zWl}9%Sb2YJ}FTYG6liM6c?W#R|w@ixESkp^7^bS-(%HAxqmn&z=HzGfky8|V;^&mlWmSa-zk!5dM&K^i+sEtZp*y=CLoFRsHRyLsZlSRtUDHg#iSm#6^Usp&T#z`1kxxfpjycH7V3{ykk;Q^n3Y6jIfDVn?`{m(<E;7Q{L!rfTJ@PQwYJYqg4hxDwuMnAGcn!ff!V*h@@SDFieQxTBDS;0FE7sa;9A%Eq-W_(u7ET%SYb&?(Xtami)$0+yteYa|V%)pX(l^OYL2#VmYY`I*FEM|!!BaUFsXfhtX)(KQGE&H+F5^Dhyy306bt&;#ltw>LgjfYld95Nsa*u3fp5OmKFj394kCFP5b`3UO*(nmofld`x3;MzoAnbxL@dGf=zpEJe#$i$-6DD@AZrr3N>dfEDbFCYzj4`Cz}b5^1Z(+n&3p0%Q()W?fO+b%)AKbyhZbmg>(58JO3Ql^|aQ_bQ$IrqA*1O}6B9!n=%88&EUMjf{l4Xh!qTkDYmAL6l#2$ic+;)RsWc-r^I)f<#J7jukWvoQtkH5UGC$SQd+JkBV9faGv_ekV<#TKB#?NH1Rl9CQ4u3zOs}3Rie(EmV4DM*AleM?_ztcIS45zk>T7?}@HRFby_a=~T<h46TaHCO_EWI57OgUMA!!LgDw@(-{r{1ai5ewccbnzZ$<ReGc{D1OUP9${*n^1|2M;PUvYJrX2wz`-aQ~p++0Rwo>(z8qyDx8za`a=^mVunyFE!mn-_p5(*|YN0M@ek2MC|5u;GP7u$$yYx5vsCDkH!P9W{ml)~o{Ho2)0#mbcQ-Q6o-KvXL)JO<to&Z>4mEZ<s#5mh8bvI92t+m+wmy8154_MM0G*9^TFAB1!vtOFoMHF9riK$-B)Z=7-7O-HEA1mz~3gLR3|{Mj_jCv6M}Z5;8O2qxDM2&(VU0G%qo^CDNcKcY7?>PZ^sjP1CoY2)GZ+me(1&o;75T&FXJqr}sk>)CIm2y`-tf!&TTOVBC6wZYg4o*@R)yG)?3LTKcqPD|<MEyfLm74x2;@N0$DB#+4um2+99a6%DY9B<ru$`KPf=-VKaLv=~|Q2TJwH1r*(#rC@WGrjeZ*h|6(=Ovi}D^PVuCfperrr3QC4dIRW`IYGX{m@Jhn+mgp_y<*6_jHh-32bh51Mi9UCMz1$6QfZAtKi_GiSAxAZbW?>0~;TVmw}_-UX8Wp=z11(KODk=GeK|42r2*udxZu}F*>^xi$z+^?s8xK`AA_@w@R0&Fj<JfJ+VPwM!Wu4#F^?xL`BZ-9KJm+vhG<7hQwC2qxuTYg0xS*2z=zRjWuG#M+l#7F3w~A`E=>e=B6s%m^Vm6BrQgDH$<4H+(yYseep9E8re1`r!%)hU*2csUz792R>d<13MSF8kp|#PnYYj|8hD}*YS<2%?BerlA|bLzRyGyv?e@fq(l^|+n};6=PqYIP*==&^nJc$cB*d$z0+`tVvis@I){nI-;5x>hLL`t(>F6M11aBoBb}(_IbQoMiN!{g@cJ?ThcyN~ik2IH7=fm(V+fV=h_ata&g3`PG84;-~na36)x5gPoOcG*LxBs8+XY8OrGPHdrj#lj8`|*N46Da|`y-}h7W|wPor|meAIT7jwyyoC-8HB*5V4a6&_ug_0vBGUN+5MK374*=Q(n6!$y+S?ZNP9ickry|y%+8DN8c|gBV}sr!B0T(zIgf3sj_;tgI49<!(3*l}Hmo-U={Cl=GedB^Uef26&7@IYThx*Az*6zgflhDH_!we2Oy{BnEtKk|30<dw0?v5$z*$n;O1L#z{d+3hzMqJ+s3mPAC|HHK+@d(r9kLVVDe=tLlro%YPhjQpd^z)f1I~|R^%P{lG>gJABH7m(R8(vcs~Xmw-Cb81?IJ%hRU6MI$r75fLLz804oOUT?H(8kkyx{Ot=Z_|U%#Ylx1Tcd%Ba_*oNCu0cyBUrE9oVo$97Gw1K>sOLt}70U@CaiRAPbNV^d%P3DM<R_y4lVO0J{Vpg-8>g0p)bT#aK}OLW<w@6dPEt-Jlh2bLb1UD2pAwQEz&Bx^ws7oZ$XFFBM{zTjnG3nf=rVR(qa4=aF7pIhU(YMTpLZ2p+A^=!HxB*>%6xB<a9Iq;ndHO30S=__^JHv&V3FV*z*Ie;ka=7Ev=nUG9WXK5E$RkKICwKXS_4e|65Jq+R?L>9H1AE_dI+s0xgr6DhMqZ<RFB1oB7tp_QppA6EI_M>VsZw(~DX9g>S9&_p`e!<bzpXx7jFMq?VC&MHeYbsNK<V}q_?xS-v+QQ7A;t!A<7vnDkRXO!KKRYYq63`T8_y!<hxF(#pJ;m`Qn#-LF?!3%Ae;GJP15A#%z+j8bXwpu-+6Y$mI0<@=j|v`OS5#h6Ms_qVBM7~d$fO(RSWHA!W@`8mmsm;zl?1<jgS{|1#MRS8ej*rv*-9#skIgB|RRD97(00&~Op8T7cChl~npkE}h`E)lx-$R|P6ipP;&SX^4Y1wq{XNVnBdoO&^eZe4x}v;{$(8yYQHsTEDY_%jxYwl_2K-MlFP9zP<m!KS*~pE1W-5ET10|sAV)MMn=}=c(d>zD0T6klxDL^jcwn%lrypMhau4&>LqZC<q@kC8zKkw}wMZe1Zr8#9%NA5y3Sa6{3C&axA5P>KmvU^Wm2E66&TP92sux8CXL;zR5qO^~l!&=3nF*sZJlvYk<)~!2kc)j@8woPj%%6{Y|v^arz2z`I9FEK3{2Iv8ZqVxJ*8g38Eu&i_CrACBKCt|wY8x%jS+1caI00yn5fEgF>EhqXi$pP-&;V363(+-4wOCX9tllH#Yzaj)9O<gRld=NBTk)WOiZviNB0K|K`D+oAO%0GQq^j8~!6w^@DatzK6Kj9CTF5{!{3i#6Q1~&_`xMfuxGGM}0qu}tOzJ0gZnzlOmBQ9u(=K2V+hVn_r9ixkADB<vPCQ1;O;I{s|s)OOxzZ$ef8Q<j6v8ka7Pcc;p`$1(vr6GM8vamrJ;659&`l#5;9b$gjN~mRU%dz^?{rTDBW_rC+S23Rzc*%~3{mN(B!?2`B8!Q0w^Ywi09t=O+r{Pq-ZX%Wo8+~Y+l2={}w)bVe|3(a?$&ON_vc;p6-5uy`Y62n057i8sisx3YD<de(55!lTpy`xxr+E{6bPArhwFAPyY4`NvZjFMno`-6S74z%?K!lMReR=6B(lv9etVJgkubuj{55k9>xoS5A604ju4-OqbmAdMipipB!!>J4qV7QXdw_m#tjB*%9x{qp-)@_pU;Bj~#F=<bz+!(eiweuw&a4_s5;E~;CjI5QroesdB;<D5}<=!7fPT=qc!qeaMnt31rbfYFsSdN+<w%?d)Y|T!Q&5h#Uf(bkeo!jAdFqtvyhp(`=TY@n;?!|<k*gIYO-Cs7rwI=+jLS%k{(6_P`dO^;T1P1>4T9`(j267|m6!Tu@J0(I)t$RC!MEl0U&n%dwi{;24SRfOxvE$z_XzN^-ai?fbaBap`!?`_4k+{dTLhrbqePe-iT|YxH>w6*MLnKyg0qJr#%D9np9BQa>nCfohgGaonxV)~@Z}z}W-C)3OZb$u<mT-G`=3IX_1-^s)dnQTr0*)D-5S(7j#~xYdb*;JPfLl3O?#%K9r{c;kLIg>S8Uh3wzdcxKXK)!!|0C0M7ta(nBNTfYhe$5_=`AqVOsAvhftp?)emzjAPN^QLyH0>R5<*Kid_K}V=%vrtRw4QzNn=G~k3(%K69kXuttdbo&2egi9M6@T@I-l5qUHqaH#Bd4v0aA;%qR8Rex1cK_KyB>W#Kt@G0%LsNKGjRZATErTU%;+!02Eguk7$5BEPBbDQ7a=*WiGJ{gX#Z=P52454UL)SE$(P^t_F{c~YYeZN-e+S(DVCQGK_7g-L;rwZRvR#@fK|8n0WG%1J&eA)h^f&0vjfh42skY3a;fnG#BDTsUoJ!1-YLrB}qQN*AIMkp)w(f#*N1VFG<*Xq<6t8&`72bD>io+WGV{{M^`Bka{bza)(-|wsySW-OByPJqEI&RY6rlc~ap)X;aS$Lcz!~rk{q7#rW=ddL3X~l4P^NdR&U!rWVKm2?M?6P}F-ucS0w3fxEIC;mHaehqHwbXQ-~OCajKz_H{eXFhS=HrC;gFBB~jtD#iJjiE50YWw%59T0;W-PiY_qTN&P6PgJm72(d^mFe$0PbL6!lSJzhvB{A4KKM*hZyg@4r&aCvE{)C>0HiJ5N@D5(=ZUwNpLbYoXbTo1jbsIaG=#%2_Nr46NNe$x>dR}vGeN1G*s2$N>dCYOms<LW&4_6C^+t7>Rgpci}Td}g+BtJr;yn;Vyl?g>7WUtuxOYqa)!OT9IneoK01@kNxKnD7L_=+WZ$;ZmYsZkkwV4>Zq$-a6(kIRszu*yz=fO6}UqTe!)Fzdr=IV1By>m5@|A!KEA^oP=|h6-2N8gc1hL#2sk#>e-Wh>OSyv-lEY=_|DB8!F!i(wVFZv*?20Az>`ERQIYS@pV~gC!i`0<Z(emSf;lkfO;kA`afo?JR|=hn^_#X00UAvRg4Ls9TO<qt(?41#<}{;=xm+}Md<YCG}9!kf*IloK`C87-$)_}vCQmDteQI-*OV&68D5{dvGuLdNX&~Y$tg1&6`DX^ZwC)w&jhC%b^C;A7|Pa8P+PXqU>4mEx)f$(!t>>!Y4q<hqj?50;*g|Itx^enpuUMJGzZ3H6>ZLdmeHt>$8mYCXEU@L%v>?3^)*Ri{<nqKB|XTgOBg+3&l(o<Zw^U7U6~L~V04HXM5)OA^;GUxRGpzB!jdG6Y6pn&pV&&n1|7g10gGQrpt98&Bk17Dmtx+-DDr`h)G$4y3?14HPmK!xIgHn31u}o6AhM|S>D_cOU}8U9BdNJtf0~8I4gw%_8E~0Z5wn8%VXL3)>IVW2Gqgj-@;Au-{VnZjKA~ko+M%jXR+C#F*Djij5^vw+;D!YD{^WG)Ii`m_*#7XgK@$V`;OaW_bvb1fqJxAEC2Zlcz@5UUxvm^jB2fOf2cPjDN>K2}&EGasMUww^u)IgKG4)fScy$^{jp{nIoa9{a%@g<|Q8R3B>&s{RaWxqMc-AYL9@dZ`v9h)~tYXw5u;f#{>9br&T-6$Dvj`}xmP_6wqE_L#)d*&ZJ8ylw90Wqin<3S1=>|ldwxBjx_)eH%@<^XSU7e5s_qT<FX#f5jJQ3)!yKt)#LnefcAT1%l4~8e@3?iuiZThR}A2!kM*l-<-L@>RkiZXaKsI^>ER}Mq<d4?qhN2(Y#ohI^PgmZd<(a!*eQ+C;^>)rPce+g^~J+{_(fC@mfX2W!Lq*;Ky*HPj#xPZ1UDN&HLAIE3Q{*|66w{HYK*$SgKxrGAiKAqmRQ|>`mC;*?ffOn&R!;lwiD^eh_g*#nbk4<fPvr<%k000004Y*qbx|kS100HH8|B?d$FLLqtvBYQl0ssI200dcD")).decode("utf-8"))   # E25 NB4: the E23 lab verbatim (lzma, sha ef393fbb2ff7669e checked)
    (TMP / "e23_lab.py").write_text(LAB_SRC)
    sys.path.insert(0, str(TMP))
    import e23_lab as L
    REQ_LOG = TRACES / "requests.jsonl"                           # memory rule: the full text of EVERY request
    _stream_run = L.Stream.run

    def _traced_run(self):
        try:
            _stream_run(self)
        finally:
            if not self.err and any(m is not None and not (isinstance(m, (int, float)) and math.isfinite(m))
                                    for m in self.margins):
                self.err = "NONFINITE_LOGPROB"           # final audit H2: NaN / Inf logprobs are never data
            try:
                r = self.row()
                pr = self.body.get("prompt") or []
                r.update(port=self.port, sampling={k: v for k, v in self.body.items() if k != "prompt"}, prompt_len=len(pr),
                         prompt_sha=hashlib.sha256(json.dumps(pr).encode()).hexdigest()[:16],
                         ttft=round(self.t_first - self.t_start, 2) if self.t_first and self.t_start else None,
                         sec=round(self.t_end - self.t_start, 2) if self.t_end and self.t_start else None, t=time.time())
                with _SAVE_LOCK, open(REQ_LOG, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
            except Exception:
                pass
    L.Stream.run = _traced_run
    PLE_EXPECTED = json.loads(r"""{"model-plefp8-00000.safetensors": [5200027199, "dc2e845b7edd35bda92834fba3626bf7d199e28d6aceac99fee654aade390cfd"], "model-plefp8-00001.safetensors": [5200027202, "899eaa0716e28594468a1389ee58cb065c23907f1270de3831f5ecb0a4f82d56"], "model-plefp8-00002.safetensors": [5200027198, "06fd8a11abf0419a669f89397b8d70dd6ff42d401e6b2a037c65e49704faaf71"], "model-plefp8-00003.safetensors": [5200027189, "c6aaa1fc08e84eced3c8151ac8679ed943888eba0fcef2556963693430f95bd9"], "model-plefp8-00004.safetensors": [5200027190, "d94e97c96d3ea09208614da016960f3f4b429f47a044c898a51b493f42f74ba2"], "model-plefp8-00005.safetensors": [5200027190, "03d5d4792e14a4ab55bae50bb459624b01786f818bbcfcaed1a5d0235af484c6"], "model-plefp8-00006.safetensors": [5200027190, "586cccbc12383021bc9bc02f206d9b19fbd1672373ed9a5d91e3b0ce34c2418f"], "model-plefp8-00007.safetensors": [5200027190, "c4a23bcc10f3cde6b633e82c282a9a518e3a64d433628283e1bc592c94cf3d6c"], "model-plefp8-00008.safetensors": [5200027190, "2dc8098c0d020bff277c9cf499a6b908e17836b70a5f949dfa24793371c9a87e"], "model-plefp8-00009.safetensors": [4400023163, "61de98b89bb79f386795787d7a76827a26f1e292c26edbb0a1b613da455f5a9c"]}""")                     # 10 FP8 PLE shards: [size, sha256]
    # exact Kaggle mount paths (foysalemonshanto Models, checked through the API 2026-10-31; no recursive search)
    MD = INP / "models" / "foysalemonshanto"
    PART_DIRS = {1: MD / "qwen3-8-flash-next-w4a16-merlin-p1/transformers/default/1",
                 2: MD / "qwen3-8-flash-next-w4a16-merlin-p2/transformers/default/1",
                 3: MD / "qwen3-8-flash-next-w4a16-merlin-p3/transformers/default/1",
                 4: MD / "qwen3-8-flash-next-w4a16-merlin-p4/transformers/default/1",
                 5: MD / "qwen3-8-flash-next-w4a16-merlin-p5/transformers/default/1"}
    PARTS = {}
    for k, d in PART_DIRS.items():                                  # the five checkpoint parts (Models from the kernel outputs)
        mf = d / f"q27_fn_merlin_part{k}_manifest.json"
        PARTS[k] = (d, json.loads(mf.read_text())) if mf.is_file() else None
    PLE_DIR = MD / "qwen38-flash-next-fp8ple-overlay/pytorch/albucino-fp8ple/1"      # our FP8 PLE overlay = albucino's shards
    PLE_DIR = PLE_DIR if (PLE_DIR / "model-plefp8-00000.safetensors").is_file() else None
    MERLIN_PAY = MD / "e23-merlin-env/other/v1/1/merlin_offline_tar.bin"
    MERLIN_PAY = MERLIN_PAY if MERLIN_PAY.is_file() else None
    INTEL = INP / "models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1"
    INTEL = INTEL if (INTEL / "config.json").is_file() else None
    WHEELHOUSE = INP / "datasets/foysalemonshanto/vllm-0-32-0-cu13-wheelhouse/install_wheelhouse.py"
    WHEELHOUSE = WHEELHOUSE if WHEELHOUSE.is_file() else None
    ARC = COMP / "arc-agi_evaluation_challenges.json"
    if not ARC.is_file():
        raise FileNotFoundError(f"{ARC}: attach the competition data")
    MERLIN_OK = all(PARTS.values()) and PLE_DIR is not None and MERLIN_PAY is not None
    R["inputs"] = {"parts": {k: (str(v[0]) if v else None) for k, v in PARTS.items()}, "ple_dir": str(PLE_DIR),
                   "merlin_pay": str(MERLIN_PAY), "intel": str(INTEL), "wheelhouse": str(WHEELHOUSE), "arc": str(ARC),
                   "merlin_inputs_ok": MERLIN_OK}
    ABORT = {"why": None}
    NVSMI_OK = bool(L.gpu_mem()) and not FAKE
    R["setup"]["gpus_at_start"] = L.gpu_mem()
    if not FAKE and not os.environ.get("RC2_CPU_KIT") and len(R["setup"]["gpus_at_start"] or []) != 4:   # CPU kit: no GPUs
        R["errors"].append(f"expected 4 GPUs, nvidia-smi shows {R['setup']['gpus_at_start']}")
        save()
        raise RuntimeError("GPU_PREFLIGHT_FAILED: " + repr(R["setup"]["gpus_at_start"]))
    L.start_watchdog(HARD_STOP, lambda: ABORT.update(why="HARD_STOP reached: watchdog killed live servers"))
    log("NB4 | budget", BUDGET // 60, "min |", json.dumps(R["inputs"])[:900])
    save()


if not RERUN:
    # ==================================================================== CELL 2 — host facts, checkpoint merge, engines
    def run(name, cmd, timeout=3600, env=None):
        t = time.time()
        try:
            with open(LOGS / f"{name}.log", "w") as f:
                r = subprocess.run(["bash", "-c", "set -o pipefail; " + cmd], stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                                   env={**os.environ, **(env or {})})
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = -9
        txt = (LOGS / f"{name}.log").read_text(errors="replace")
        R["setup"][name] = {"rc": rc, "sec": round(time.time() - t, 1), "tail": txt[-1500:]}
        log(name, "rc", rc, f"{time.time() - t:.0f}s")
        save()
        return rc, txt


    SHIM = TMP / "e23_libcuda"          # FlashInfer JIT links -lcuda; Kaggle has only libcuda.so.1
    SHIM.mkdir(exist_ok=True)
    for d in ("/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/local/cuda/compat", "/usr/lib64"):
        c = sorted(Path(d).glob("libcuda.so.1")) if Path(d).is_dir() else []
        if c and not (SHIM / "libcuda.so").exists():
            (SHIM / "libcuda.so").symlink_to(c[0])


    def host_facts():
        """A0: what the Kaggle box gives us (glibc, driver, GPUs, PCIe, peer access, cgroup memory, /dev/shm, disk)."""
        f = {}
        for k, cmd in (("glibc", "ldd --version | head -1"), ("uname", "uname -a"),
                       ("nvidia_smi", "nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader"),
                       ("cuda_version", "nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"), ("shm", "df -h /dev/shm | tail -1"),
                       ("tmp_disk", f"df -h {TMP} | tail -1"), ("nproc", "nproc"),
                       ("cgroup_max", "cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes"),
                       ("cgroup_cur", "cat /sys/fs/cgroup/memory.current 2>/dev/null || cat /sys/fs/cgroup/memory/memory.usage_in_bytes"),
                       ("meminfo", "grep -E 'MemTotal|MemAvailable' /proc/meminfo"), ("ulimit_l", "ulimit -l"),
                       ("omp", "echo OMP_NUM_THREADS=$OMP_NUM_THREADS"), ("topo", "nvidia-smi topo -m; nvidia-smi topo -p2p r"),
                       ("pcie_now", "nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current,power.limit --format=csv,noheader")):
            try:
                f[k] = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=60).stdout.strip()[:600]
            except Exception as e:
                f[k] = repr(e)[:200]
        try:
            import torch
            n = torch.cuda.device_count()
            f["peer_access"] = [[torch.cuda.can_device_access_peer(i, j) if i != j else None for j in range(n)] for i in range(n)]
        except Exception as e:
            f["peer_access"] = repr(e)[:200]
        return f


    def host_mem():
        """MiB. avail_mib = max - (current - inactive_file), Merlin's own rule (RC1's guard counted page cache as used and
        skipped arm 3 with MemAvailable 55 GB); /dev/shm counts against the cgroup."""
        out = {}
        try:
            cur = int(Path("/sys/fs/cgroup/memory.current").read_text())
            mx = Path("/sys/fs/cgroup/memory.max").read_text().strip()
            st = dict(l.split() for l in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines() if len(l.split()) == 2)
            out.update(cgroup_cur_mib=cur >> 20, cgroup_max_mib=None if mx == "max" else int(mx) >> 20,
                       **{k + "_mib": int(st[k]) >> 20 for k in ("anon", "file", "shmem", "inactive_file") if k in st})
            if mx != "max":
                out["avail_mib"] = (int(mx) - (cur - int(st.get("inactive_file", 0)))) >> 20
        except Exception:
            pass
        try:
            for l in open("/proc/meminfo"):
                if l.startswith(("MemAvailable", "MemTotal")):
                    out[l.split(":")[0] + "_mib"] = int(l.split()[1]) >> 10
        except Exception:
            pass
        try:
            out["dev_shm_used_mib"] = shutil.disk_usage("/dev/shm").used >> 20
        except Exception:
            pass
        return out


    TELEMETRY = OUT / "telemetry.jsonl"


    def _telemetry():
        """Host RAM + GPU memory every 10 s for the whole run (Merlin's floor ~69 GiB inside a ~173 GiB cgroup)."""
        while True:
            try:
                with open(TELEMETRY, "a") as fh:
                    fh.write(json.dumps({"t": round(time.time() - T_START, 1), "host": host_mem(), "gpu": L.gpu_mem()}) + "\n")
            except Exception:
                pass
            time.sleep(10)


    threading.Thread(target=_telemetry, daemon=True).start()


    SHM = Path(os.environ.get("RC2_SHM", "/dev/shm"))


    def shm_names():
        try:
            return {p.name for p in SHM.iterdir()}
        except Exception:
            return set()


    def shm_cleanup(before):
        """After a boot's server is dead: only /dev/shm entries that appeared during THIS boot (absent from `before`) and
        match the engine's prefixes are unlinked; everything else is recorded, never touched."""
        out = {"host_before": host_mem(), "removed": [], "kept": []}
        try:
            for p in sorted(SHM.iterdir()):
                try:
                    sz = p.stat().st_size >> 20
                except Exception:
                    sz = None
                if p.name not in before and re.match(r"(torch_|psm_|vllm|ple|nccl|sem\.mp-|cuda\.shm)", p.name, re.I):
                    try:
                        p.unlink() if not p.is_dir() else shutil.rmtree(p)
                        out["removed"].append([p.name, sz])
                    except Exception as e:
                        out["kept"].append([p.name, sz, repr(e)[:80]])
                else:
                    out["kept"].append([p.name, sz])
        except Exception as e:
            out["error"] = repr(e)[:200]
        out["host_after"] = host_mem()
        out["removed"], out["kept"] = out["removed"][:60], out["kept"][:60]
        return out


    class MServer(L.Server):
        """Merlin passes --shutdown-timeout 60 and the PLE transport unlinks its /dev/shm storages only on a graceful
        teardown: SIGTERM, wait 75 s, then E23's kill (SIGTERM + 40 s + SIGKILL of the whole session)."""
        def kill(self):
            if self.proc is not None and self.proc.poll() is None:
                try:
                    os.killpg(self.proc.pid, 15) if hasattr(os, "killpg") else self.proc.terminate()
                except Exception:
                    pass
                try:
                    self.proc.wait(timeout=75)
                except Exception:
                    pass
            super().kill()


    def boot_gate():
        """Merlin's pin leaves ~0.9 GiB free per card: every boot needs near-empty GPUs (< 150 MiB, not E23's 1024)."""
        g = L.wait_gpu_release(300, fake=bool(FAKE), gpus="0,1,2,3", free_mib=150)
        return {"ok": g, "gpus": L.gpu_mem(), "host": host_mem()}       # callers launch only on ok is True


    CKPT = TMP / "fn_merlin"
    SHA = {"thread": None, "ok": None, "bad": [], "sec": None}


    def merge_ckpt():
        """Symlink the 5 parts + 10 PLE shards into one directory; names and sizes now, sha256 in a background thread."""
        SHA.update(thread=None, ok=None, bad=[], sec=None)        # a stale result never gates a new merge
        shutil.rmtree(CKPT, ignore_errors=True)
        CKPT.mkdir(parents=True)
        expected = {}
        for k, (d, man) in PARTS.items():
            for n, (size, sha) in man["files"].items():
                expected[n] = (d / man.get("model_subdir", ".") / n, size, sha)
        for n, (size, sha) in PLE_EXPECTED.items():
            expected[n] = (PLE_DIR / n, size, sha)
        bad = []
        for n, (src, size, sha) in sorted(expected.items()):
            if not src.is_file():
                bad.append(f"missing {n}")
                continue
            if not FAKE and src.stat().st_size != size:
                bad.append(f"{n}: {src.stat().st_size} bytes, expected {size}")
            try:
                (CKPT / n).symlink_to(src)
            except OSError:
                if not FAKE:
                    raise
                shutil.copyfile(src, CKPT / n)               # local Windows harness only (no symlink privilege)
        idx = json.loads((CKPT / "model.safetensors.index.json").read_text())
        need = set(idx["weight_map"].values())
        bad += [f"index needs {n}, not merged" for n in sorted(need - set(expected))]
        cfg = json.loads((CKPT / "config.json").read_text())
        if (cfg.get("text_config") or {}).get("ple_embedding_dtype") != "float8_e4m3fn":
            bad.append("config.json: text_config.ple_embedding_dtype is not float8_e4m3fn")
        rec = {"files": len(expected), "index_files": len(need), "bad": bad[:20], "ok": not bad}

        def hash_all():
            t = time.time()
            import concurrent.futures as CF

            def one(item):
                n, (src, size, sha) = item
                if FAKE:
                    return n, True
                with open(src, "rb") as fh:
                    return n, hashlib.file_digest(fh, "sha256").hexdigest() == sha
            with CF.ThreadPoolExecutor(6) as ex:
                res = list(ex.map(one, sorted(expected.items())))
            with _SAVE_LOCK:                   # E25 NB4 (klee100 R1-13): never mutate R while save() dumps it
                SHA.update(ok=all(ok for _, ok in res), bad=[n for n, ok in res if not ok], sec=round(time.time() - t, 1))
                R["setup"]["sha256"] = {k: SHA[k] for k in ("ok", "bad", "sec")}
            save()
        if rec["ok"]:
            SHA["thread"] = threading.Thread(target=hash_all, daemon=True)
            SHA["thread"].start()
        return rec


    def sha_gate(timeout):
        """The first launch waits for the full sha256 check (Merlin itself never hashes shards)."""
        th = SHA["thread"]
        if th is not None:
            th.join(max(1, timeout))
        if th is None or th.is_alive():
            return False, "sha256 check not finished"
        return bool(SHA["ok"]), ("ok" if SHA["ok"] else f"sha256 mismatch: {SHA['bad'][:5]}")


    def prefetch_bg(paths, tag):
        """rev 3: warm the page cache of a cold 17-30 GB /kaggle/input checkpoint in the background (many readers)."""
        def go():
            R["setup"]["prefetch_" + tag] = L.prefetch([p for p in paths if p]) if not FAKE else {"fake": True}
            save()
        th = threading.Thread(target=go, daemon=True)
        th.start()
        return th


    MERLIN_ENG = None


    def merlin_install():
        """P2 payload (.bin = a tar Kaggle does not unpack) -> install_merlin_offline.sh (UV_OFFLINE); payload deleted after;
        a failure is cached so later Merlin steps fail fast (rev 3)."""
        global MERLIN_ENG
        if MERLIN_ENG is not None:
            if MERLIN_ENG.get("failed"):
                raise RuntimeError("Merlin install failed earlier: " + MERLIN_ENG["failed"])
            return MERLIN_ENG
        if not MERLIN_OK or MERLIN_PAY is None:
            MERLIN_ENG = {"failed": "Merlin checkpoint / env not attached or incomplete"}
            raise RuntimeError(MERLIN_ENG["failed"])
        if FAKE:
            MERLIN_ENG = {"fake": True}
            return MERLIN_ENG
        try:
            pay_root = TMP / "merlin_pay"
            shutil.rmtree(pay_root, ignore_errors=True)
            pay_root.mkdir()
            t = time.time()
            with tarfile.open(MERLIN_PAY) as tf:
                tf.extractall(pay_root, filter="data")
            R["setup"]["merlin_extract_sec"] = round(time.time() - t, 1)
            pay = pay_root / "merlin_offline"
            shutil.rmtree(TMP / "merlin_off", ignore_errors=True)
            rc, txt = run("install_merlin", f"bash {pay}/install_merlin_offline.sh {pay} {TMP}/merlin_off", timeout=int(max(600, left() - 100 * 60)))   # E25 NB4 (klee100 R1-7): never outlives the arms
            kv = dict(re.findall(r"^(MERLIN_[A-Z]+)=(.*)$", txt, re.M))
            shutil.rmtree(pay_root, ignore_errors=True)
            if rc or "INSTALL OK" not in txt:
                raise RuntimeError("Merlin offline install failed (logs/install_merlin.log)")
            sys.path.insert(0, kv["MERLIN_TREE"] + "/tools/merlin")
            import launch as ML
            site = subprocess.run([kv["MERLIN_VENV"] + "/bin/python", "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
                                  capture_output=True, text=True).stdout.strip()
            MERLIN_ENG = {"kv": kv, "ML": ML, "cuda_home": site + "/nvidia/cu13"}
            return MERLIN_ENG
        except Exception as e:
            MERLIN_ENG = {"failed": f"{type(e).__name__}: {e}"[:300]}
            raise


    MERLIN_CC = {"mode": "gcc13"}       # rev 3c: Merlin README:229 / install-env.sh: FlashInfer JIT at first start needs GCC >= 13



    KV_FP8, KV_BF16 = 2479838700, 2464415744      # Merlin's own L4 pins (23,034 MiB cards; preflight.kv_bytes_for_vram, 150 / 79 blocks)
    PROFILE = "fn-quad-24g-sm89"
    XHIGH = json.dumps({"enable_thinking": True, "reasoning_effort": "xhigh"})
    HOST_PHASE = {}


    def merlin_fn_server(name, kv, port, kv_bytes):
        """Merlin's launch.build() for Flash-Next on the L4 profile: KV pin lowered for 22.49 GiB cards, the checkpoint's own
        chat template at xhigh (Merlin passes medium otherwise), Merlin's host_phase checks recorded, E23's env overrides."""
        eng = merlin_install()
        extra = ["--served-model-name", "q27", "--max-model-len", str(MAX_LEN), "--port", str(port),
                 "--default-chat-template-kwargs", XHIGH] + list(NB1_EXTRA.get(name) or [])   # E25 NB1: per-boot flags (last wins)
        if FAKE:
            return MServer(name, [sys.executable, FAKE, str(CKPT) + "-w4a16", "--tensor-parallel-size", "2", "--max-num-seqs",
                                  "8" if kv == "fp8" else "3", "--port", str(port), "--max-model-len", str(MAX_LEN),
                                  "--kv-cache-memory-bytes", str(kv_bytes)] + list(NB1_EXTRA.get(name) or []),   # E25 NB1 (fake only)
                           {**os.environ, **(NB1_ENV.get(name) or {})}, port, LOGS)
        kvv = eng["kv"]
        gcc_bin = str(Path(kvv["MERLIN_CC"]).parent) if kvv.get("MERLIN_CC") else ""
        base = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CUDA_PATH", "CUDA_HOME", "LD_PRELOAD", "CC", "CXX")}
        base.update(GPUS="0,1,2,3", PORT=str(port), KV_BYTES=str(kv_bytes), P2P="off",
                    VENV=kvv["MERLIN_VENV"], TREE=kvv["MERLIN_TREE"], CHAT_TEMPLATE="", WARM="0", CACHE_ROOT=f"{TMP}/merlin-cache",
                    PATH=f"{kvv['MERLIN_VENV']}/bin:{gcc_bin}:{eng['cuda_home']}/bin:{os.environ.get('PATH', '')}",
                    CUDA_HOME=eng["cuda_home"])
        base.update(NB4_PREBUILD.get(name) or {})          # E25 NB4: build() knobs (DRAFT_HEAD) of this launch
        if kv == "fp8":
            base["SCALES"] = str(CKPT / "qsa_kv_scales_262k.json")
        plan = eng["ML"].build("flash-next", PROFILE, str(CKPT), base, extra=extra, kv=kv, checkpoint="flash-next", status="untested")
        try:                                               # audits: tree products, venv, checkpoint marker, PLE dtype, toolchain,
            eng["ML"].host_phase(plan)                     # scales file: a failure stops this boot before any server starts
            HOST_PHASE[name] = "ok"
        except BaseException as e:
            HOST_PHASE[name] = f"{type(e).__name__}: {e}"[:600]
            R["setup"].setdefault("host_phase_error", {})[name] = HOST_PHASE[name]
            save()
            raise RuntimeError("MERLIN_HOST_PHASE_FAILED: " + HOST_PHASE[name])
        env = dict(plan.env)
        env.update(LD_PRELOAD=kvv["MERLIN_SHIM"], CUDA_HOME=eng["cuda_home"], VLLM_USE_RUST_FRONTEND="0", HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", LIBRARY_PATH=f"{SHIM}:{os.environ.get('LIBRARY_PATH', '')}",
                   PATH=base["PATH"], VLLM_CACHE_ROOT=f"{TMP}/merlin-cache", TRITON_CACHE_DIR=f"{TMP}/merlin-triton",
                   FLASHINFER_WORKSPACE_BASE=f"{TMP}/merlin-fi", CUDA_DEVICE_ORDER="PCI_BUS_ID", VLLM_PLE_OFFLOAD_READY_TIMEOUT="3600")
        for k in ("CC", "CXX", "CUDAHOSTCXX"):
            env.pop(k, None)
        if MERLIN_CC["mode"] == "gcc13" and kvv.get("MERLIN_CC") and kvv.get("MERLIN_CXX"):
            env.update(CC=kvv["MERLIN_CC"], CXX=kvv["MERLIN_CXX"], CUDAHOSTCXX=kvv["MERLIN_CXX"])
            lib13 = Path(kvv["MERLIN_CC"]).parents[1] / "lib"
            if (lib13 / "libstdc++.so.6").exists():
                env["LD_LIBRARY_PATH"] = f"{lib13}:{env.get('LD_LIBRARY_PATH', '')}"
        try:                                               # Merlin's own scheme: PORT 8000..8355 -> 29500+d / 30000+100d
            mp, vp = eng["ML"].distributed_ports(str(port))
            env.update(MASTER_PORT=str(mp), VLLM_PORT=str(vp))
        except Exception:
            env.update(MASTER_PORT=str(29500 + port - 8000), VLLM_PORT=str(30000 + 100 * (port - 8000)))
        env.update(NB1_ENV.get(name) or {})              # E25 NB1: after build(), which strips every MERLIN_* variable
        argv = plan.argv()
        HELP[name] = L.help_check("merlin", argv[:argv.index("serve") + 1], env, argv)
        R["setup"].setdefault("merlin_plans", {})[name] = {"argv_tail": argv[-60:], "status": plan.status, "host_phase": HOST_PHASE[name],
                                                           "env_keys": {k: env[k] for k in sorted(env) if k.startswith(("VLLM_", "NCCL_", "KV_", "P2P", "SCALES"))}}
        save()
        return MServer(name, argv, env, port, LOGS)


    STOCK = None


    def stock_install():
        """Fallback only: stock vLLM 0.32 venv (E18 / E23 installer) with the -lcuda shim on LIBRARY_PATH (RC1's sampler crash)."""
        global STOCK
        if STOCK is not None:
            return STOCK
        if FAKE:
            STOCK = {"vllm": [sys.executable, FAKE], "env": dict(os.environ)}
            return STOCK
        rc, txt = run("install_stock", f"{sys.executable} {WHEELHOUSE} {TMP}/vllm-venv", timeout=2400)
        kv = dict(l.split("=", 1) for l in txt.splitlines() if l.startswith(("VENV_PYTHON=", "CUDA_HOME=")))
        if rc or "CUDA_HOME" not in kv:
            raise RuntimeError("stock vLLM 0.32 install failed (logs/install_stock.log)")
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CUDA_PATH", "LD_PRELOAD")}
        env.update(CUDA_HOME=kv["CUDA_HOME"], PATH=f"{TMP}/vllm-venv/bin:{kv['CUDA_HOME']}/bin:{env.get('PATH', '')}",
                   LIBRARY_PATH=f"{SHIM}:{env.get('LIBRARY_PATH', '')}", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   VLLM_NO_USAGE_STATS="1", VLLM_CACHE_ROOT=f"{TMP}/vllm-cache", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        STOCK = {"vllm": [f"{TMP}/vllm-venv/bin/vllm", "serve"], "env": env}
        return STOCK


    def stock_fn_server(name, port):
        eng = stock_install()
        flags = [str(INTEL), "--served-model-name", "q27", "--host", "127.0.0.1", "--port", str(port),
                 "--tensor-parallel-size", "4", "--enable-expert-parallel", "--max-model-len", "32768", "--max-num-seqs", "4",
                 "--max-num-batched-tokens", "4096", "--gpu-memory-utilization", "0.88", "--disable-custom-all-reduce",
                 "--language-model-only", "--engram-config", json.dumps({"cpu_offload": True}), "--load-format", "safetensors",
                 "--dtype", "bfloat16", "--kv-cache-dtype", "fp8_e4m3", "--default-chat-template-kwargs", XHIGH]
        # diagnostic only: RC1 had a 30,208-token BF16 pool at TP4; no MTP here, so prefix caching (default ON) is unaffected
        if FAKE:
            flags = [str(INTEL) + "-w4a16", "--tensor-parallel-size", "4", "--max-num-seqs", "4", "--port", str(port), "--max-model-len", "32768"]
        return L.Server(name, list(eng["vllm"]) + flags, eng["env"], port, LOGS)
    save()


if not RERUN:
    # ==================================================================== CELL 3 — prompts (Flash-Next tokenizer + its own template at xhigh)
    R["sections"]["A0"] = {}
    _t = time.time()
    R["sections"]["A0"]["host"] = host_facts()
    R["sections"]["A0"]["merge"] = merge_ckpt() if MERLIN_OK else {"ok": False, "why": "Merlin inputs missing", "inputs": R["inputs"]}
    R["sections"]["A0"]["host_mem"] = host_mem()
    log("A0 host", json.dumps(R["sections"]["A0"]["host"])[:1200])
    log("A0 merge", json.dumps(R["sections"]["A0"]["merge"])[:800])
    save()
    TOKDIR = TMP / "fn_tok"
    shutil.rmtree(TOKDIR, ignore_errors=True)
    TOKDIR.mkdir()
    # the Merlin checkpoint's tokenizer when the merge is OK; otherwise the Intel (stock fallback) checkpoint's own
    MERLIN_PROMPTS = bool(R["sections"]["A0"]["merge"].get("ok")) and (CKPT / "tokenizer.json").exists()
    _src = CKPT if MERLIN_PROMPTS else INTEL
    if _src is None or not (_src / "tokenizer.json").exists():
        raise FileNotFoundError("no tokenizer: the Merlin merge failed and the Intel fallback checkpoint is not attached")
    for _n in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (_src / _n).exists():
            shutil.copyfile(_src / _n, TOKDIR / _n)
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    TOK = Tokenizer.from_file(str(TOKDIR / "tokenizer.json"))
    _HF = AutoTokenizer.from_pretrained(str(TOKDIR))
    _tpl = ((TOKDIR / "chat_template.jinja").read_text() if (TOKDIR / "chat_template.jinja").exists()
            else json.loads((TOKDIR / "tokenizer_config.json").read_text())["chat_template"])
    _tasks = json.load(open(ARC))
    _ids = sorted(_tasks)


    def _grid(g):
        return "\n".join("".join(str(v) for v in row) for row in g)


    def _task_text(k):
        t = _tasks[k]
        parts = [f"## Task {k}"]
        for i, ex in enumerate(t["train"]):
            parts.append(f"Example {i + 1} input:\n{_grid(ex['input'])}\nExample {i + 1} output:\n{_grid(ex['output'])}")
        for ex in t["test"]:
            parts.append(f"Test input:\n{_grid(ex['input'])}")
        return "\n\n".join(parts)


    _INTRO = "Find the transformation rule of each ARC task below and solve the test inputs.\n\n"
    _PH = "E13_BODY"


    def _render(effort):
        return _HF.apply_chat_template([{"role": "user", "content": _INTRO + _PH}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True, reasoning_effort=effort, chat_template=_tpl)


    _RX, _RM = _render("xhigh"), _render("medium")
    assert _RX.count(_PH) == 1
    _head_txt, _tail_txt = _RX.split(_PH)
    _HEAD = TOK.encode(_head_txt, add_special_tokens=False).ids
    _TAIL = TOK.encode(_tail_txt, add_special_tokens=False).ids
    TPL_MSG = [{"role": "user", "content": _INTRO + _task_text(_ids[0])}]
    TPL_IDS = TOK.encode(_HF.apply_chat_template(TPL_MSG, tokenize=False, add_generation_prompt=True, enable_thinking=True,
                                                 reasoning_effort="xhigh", chat_template=_tpl), add_special_tokens=False).ids
    R["prompt_render"] = {"head_text": _head_txt[-600:], "tail_text": _tail_txt, "head_tokens": len(_HEAD), "tail_tokens": len(_TAIL),
                          "xhigh_differs_from_medium": _RX != _RM, "xhigh_sentence": "xhigh" in _head_txt,
                          "template_sha256": hashlib.sha256(_tpl.encode()).hexdigest(),
                          "tokenizer_json_sha256": hashlib.sha256((TOKDIR / "tokenizer.json").read_bytes()).hexdigest()}
    _TASK_IDS = {k: TOK.encode(_task_text(k) + "\n\n", add_special_tokens=False).ids for k in _ids}


    def prompt(band, variant):
        body, i = [], (variant * 7) % len(_ids)
        need = band - len(_HEAD) - len(_TAIL)
        while len(body) < need:
            body += _TASK_IDS[_ids[i % len(_ids)]]
            i += 1
        return _HEAD + body[:need] + _TAIL


    TOK_SHA = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"   # = the 27B Merlin tokenizer.json
    TPL_SHA = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"   # = the 27B FP8 chat_template.jinja
    R["prompt_render"]["source"] = "merlin" if MERLIN_PROMPTS else "intel_fallback"
    R["prompt_render"]["hashes_ok"] = (R["prompt_render"]["tokenizer_json_sha256"] == TOK_SHA and
                                       R["prompt_render"]["template_sha256"] == TPL_SHA)
    if MERLIN_PROMPTS:                    # the pinned hashes belong to the Merlin checkpoint only (Intel's are recorded)
        assert R["prompt_render"]["hashes_ok"], "Flash-Next tokenizer / template differ from the pinned Merlin checkpoint"
    assert R["prompt_render"]["xhigh_sentence"] and R["prompt_render"]["xhigh_differs_from_medium"], "xhigh render missing"
    BANDS = {"2k": 2048, "8k": 8192, "40k": 40000, "90k": 90000, "116k": 116000}
    NSTREAM = 8
    PROMPTS = {b: [prompt(n, v) for v in range(NSTREAM)] for b, n in BANDS.items()}
    MAX_NEW = {b: MAX_LEN - n - 512 for b, n in BANDS.items()}
    ORACLE_P = [PROMPTS["2k"][0], PROMPTS["2k"][1], PROMPTS["8k"][2]]
    LONG_P = [PROMPTS["40k"][3]]
    TF_P = {"40k": PROMPTS["40k"][5]}
    R["prompts"] = {b: {"n": len(p), "len": len(p[0])} for b, p in PROMPTS.items()}
    log("prompts", json.dumps(R["prompts"]), "| render", json.dumps({k: v for k, v in R["prompt_render"].items() if "text" not in k}))
    save()


if not RERUN:
    # ==================================================================== CELL 4 — the step runner (E23, verbatim)
    HELP = HELP if "HELP" in globals() else {}
    WIN = float(os.environ.get("E23_WIN_SEC") or 90.0)   # 120 s FULL windows
    EST_SCALE = float(os.environ.get("E23_EST_SCALE") or 1.0) if FAKE else 1.0   # local harness only: walk the whole order
    SM_TF = MODE == "SMOKE"                                            # SMOKE teacher-forced: the 40k row only
    ROWS = {}            # oracle greedy rows by key (same weights)
    TF = {}              # teacher-forced rows by key
    NTOK = 128 if MODE == "FULL" else 64
    LIVE = {"rec": None}                 # rev 3c: windows land in the step record before every save (a hard kill keeps them)
    COMPILE_ERR = re.compile(r"ninja: build stopped|Error building extension|error: command '.*(nvcc|g\+\+|gcc|c\+\+)' failed|"
                             r"GLIBCXX_[0-9.]+' not found|unsupported GNU version|fatal error: ")
    OOM_ERR = re.compile(r"OutOfMemoryError|CUDA out of memory|out of memory")
    XID = re.compile(r"NVRM: Xid|\bXid \d+|CUDA error: an illegal memory access|illegal memory access was encountered|"
                     r"unspecified launch failure|CUDA error: misaligned address")       # rev 3e: device faults, not any CUDA error


    def keep_after_error(rec, e):
        """E25 NB1 R2: an exception after the measurements (post-processing) keeps the whole record + postprocess_error /
        postprocess_tb. A clean boot (status None / NOT_RUN / DONE / TF_OOM, correctness CLEAN, quality PASS) keeps its status
        (None / NOT_RUN -> DONE), so its dependent boots still run; any other record -> POSTPROCESS_ERROR (status_before kept).
        R2 audit N1-3: postprocess_ok False; whether the dependents run is decided by core_measurements_complete (core_check)."""
        rec["postprocess_ok"] = False
        rec["postprocess_error"] = f"{type(e).__name__}: {e}"[:800]
        rec["postprocess_tb"] = traceback.format_exc()[-2500:]
        if (rec.get("status") in (None, "NOT_RUN", "DONE", "TF_OOM") and rec.get("correctness") == "CLEAN"
                and rec.get("quality_gate") == "PASS"):
            if rec.get("status") in (None, "NOT_RUN"):
                rec["status"] = "DONE"
        else:
            rec["status_before_postprocess"] = rec.get("status")
            rec["status"] = "POSTPROCESS_ERROR"
        return rec


    def step(sec, name, est_min, fn):
        """One measured step: skipped when the budget cannot hold it, recorded on any exception, never PASS by default."""
        rec = R["sections"].setdefault(sec, {}).setdefault(name, {"status": "NOT_RUN"})
        R["order"].append(f"{sec}/{name}")
        if ABORT["why"]:
            rec.update(status="SKIPPED_ABORT", why=ABORT["why"])
            log("---", sec, name, "SKIPPED_ABORT")
            save()
            return rec
        if left() < est_min * EST_SCALE * 60 + 300:
            rec.update(status="SKIPPED_BUDGET", left_min=round(left() / 60, 1), est_min=est_min)
            log("---", sec, name, "SKIPPED_BUDGET", f"(est {est_min} min, left {left() / 60:.0f} min)")
            save()
            return rec
        LIVE["rec"] = rec
        LIVE["out"] = None                       # E25 NB1 R2: the body's record once it returned (set by watched() / run_boot())
        t = time.time()
        log(">>>", sec, name, f"(est {est_min} min, left {left() / 60:.0f} min)")
        out = None                               # E25 NB1 R2
        try:
            out = fn() or {}
            rec.update(out)
            if rec.get("status") in (None, "NOT_RUN"):
                rec["status"] = "DONE"
        except Exception as e:
            part = out if isinstance(out, dict) and out else LIVE.get("out")   # E25 NB1 R2: never discard what the body produced
            if isinstance(part, dict) and part:                               # E25 NB1 R2
                rec.update(part)                                              # E25 NB1 R2
                keep_after_error(rec, e)                                      # E25 NB1 R2: clean -> DONE + postprocess_error
            else:                                                             # E25 NB1 R2
                rec.update(status="ERROR", error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2500:])
        rec["sec"] = round(time.time() - t, 1)
        LIVE["rec"] = None
        LIVE["out"] = None                       # E25 NB1 R2
        if ABORT["why"] and rec["status"] in ("DONE", "SERVER_DIED", "LAUNCH_FAILED", "BODY_ERROR"):
            rec["status"] = "PARTIAL_HARD_STOP"
        log("<<<", sec, name, rec.get("status"), f"{rec['sec']:.0f}s")
        save()
        return rec


    def with_servers(servers, timeout, body):
        """Start, wait ready, record facts + memory, run body(servers), always kill; then the GPU-release gate on these
        servers' GPUs only; abort the lab if they stay held."""
        out = {"servers": {}}
        timeout = max(60, min(timeout, left() - 120))
        try:
            for s in servers:
                s.start()
            for s in servers:
                ok, why = s.wait_ready(timeout)
                out["servers"][s.name] = {"ready": ok, "why": why, **s.facts(), "argv_tail": s.argv[-44:], "gpus": s.gpus,
                                          "help_check": HELP.get(s.name)}
                if not ok:
                    out["status"] = "LAUNCH_FAILED"
                    out["servers"][s.name]["log_tail"] = s.log()[-4000:]
                    _xl = [l for l in (s.log() or "").splitlines() if XID.search(l)][:8]   # rev 3e/3f: start-up / capture faults
                    out["servers"][s.name].update(xid=bool(_xl), xid_lines=_xl)
                    return out
            mem0 = L.gpu_mem()
            for s in servers:
                out["servers"][s.name].update(gpu_mem_after_ready=mem0, host_rss_mb=s.rss_mb())
            try:                             # rev 3: an exception in the body keeps everything measured so far
                out.update(body(servers) or {})
            except Exception as e:
                out.update(status="BODY_ERROR", body_error=f"{type(e).__name__}: {e}"[:800], body_tb=traceback.format_exc()[-2000:])
            for s in servers:
                out["servers"][s.name].update(s.facts(), gpu_mem_end=L.gpu_mem(), host_rss_mb_end=s.rss_mb(),
                                              alive_after_body=s.alive(), rc=None if s.alive() else s.proc.returncode,
                                              xid_lines=[l for l in (s.log() or "").splitlines() if XID.search(l)][:8])
                out["servers"][s.name]["xid"] = bool(out["servers"][s.name]["xid_lines"])
                if not s.alive():
                    out["server_status"] = "SERVER_DIED"         # E25 NB4 (klee100 R2-3): a specific body status is kept
                    if out.get("status") in (None, "DONE", "NOT_RUN"):
                        out["status"] = "SERVER_DIED"
                    out["servers"][s.name]["log_tail"] = s.log()[-4000:]
            return out
        finally:
            for s in servers:
                s.kill()
            gpus = ",".join(sorted({g for s in servers for g in s.gpus.split(",")}))
            out["engine_procs_before_release"] = clear_engine_procs()      # E25 NB4 (klee100 R1-6)
            rel = L.wait_gpu_release(180, fake=bool(FAKE), gpus=gpus)
            if rel is False:                 # rev 3c: one more kill + 300 s before the lab aborts
                for s in servers:
                    s.kill()
                out["engine_procs_before_fail"] = clear_engine_procs()     # E25 NB4 (klee100 R1-6): before FAIL_GPU_RELEASE
                rel = L.wait_gpu_release(300, fake=bool(FAKE), gpus=gpus)
            if rel is None and NVSMI_OK:     # nvidia-smi worked at start: a failure now is not "released"
                rel = False
            out["gpu_released"] = rel
            if rel is False:
                ABORT["why"] = f"GPUs {gpus} still held after {[s.name for s in servers]}"
                out["status"] = "FAIL_GPU_RELEASE"


    def oracle(port, key=None, refs=()):
        """Greedy rows (2k, 2k, 8k) + one 40k row; saved under `key` (a target-only oracle) or compared with ROWS[ref]."""
        rows = L.greedy(port, ORACLE_P + LONG_P, n_tok=NTOK, timeout=max(120, min(900, left() / 6)))
        clean = all(not r.get("error") and r.get("tokens") and not L.degenerate(r.get("text")) for r in rows)
        if key and clean:            # rev 3d: an errored / empty / garbage reference is never stored (consumers: NO_ORACLE)
            ROWS[key] = rows
        elif key:
            R["errors"].append(f"oracle {key}: reference not stored (error / empty / garbage row)")
        res = {"garbage": [L.degenerate(r["text"]) for r in rows], "errors": [r["error"] for r in rows],
               "text_head": [r["text"][:120] for r in rows]}
        if not FAKE:                 # rev 3h: chat-path render vs the production xhigh render (recorded per arm)
            res["template_probe"] = L.template_probe(port, TPL_MSG, TPL_IDS)
        for ref in refs:
            res["vs_" + ref] = L.compare(ROWS.get(ref), rows) if ROWS.get(ref) else {"status": "INCONCLUSIVE_NO_ORACLE"}
        return res


    def tforced(port, key=None, ref=None):
        """Teacher-forced 40k / 90k prompt logprobs (last 256 positions); saved under `key` or compared with TF[ref].
        rev 3c: on stock vLLM only on its own short launch (prompt logprobs = chunk x 248k-vocab logits: OOM at 0.90)."""
        rows = {b: L.teacher_forced(port, ids) for b, ids in TF_P.items() if not (SM_TF and b != "40k")}
        if key and L.tf_complete(rows):                                 # rev 3e: complete references only
            TF[key] = rows
        elif key:
            R["errors"].append(f"teacher-forced {key}: reference not stored (incomplete band)")
        if ref:
            return {b: L.compare_tf((TF.get(ref) or {}).get(b), rows[b]) for b in rows}
        return {b: {"n": (rows[b] or {}).get("n"), "error": (rows[b] or {}).get("error")} for b in rows}


    def levels(port, band, lv, server=None, **kw):
        """rev 3c: the guard includes admission (prefill) time; a band stops once KV-bound (the fit is recorded); a dead
        server stops the band; every window is written into the live step record before save()."""
        out, fit = {}, None
        for n in lv:
            k = f"{band}@{n}"
            need = WIN + 90 + min(3600, n * BANDS[band] / 1500)
            if ABORT["why"] or left() < need + 120:
                out[k] = {"status": "SKIPPED_BUDGET", "need_sec": round(need)}
            elif not due(need + 120):                         # E25 NB4 (klee100 R1-1): the boot body cap, not only the run budget
                out[k] = {"status": "SKIPPED_BOOT_BUDGET", "need_sec": round(need), "boot_left_sec": round(boot_left())}
            elif fit is not None:
                out[k] = {"status": "SKIPPED_KV_BOUND", "streams_fit": fit}
            elif server is not None and not server.alive():
                out[k] = {"status": "SKIPPED_SERVER_DEAD"}
            else:
                try:                         # rev 3: one failing level never loses the others
                    out[k] = L.steady(port, PROMPTS[band], n, secs=WIN, max_new=MAX_NEW[band], **kw)
                except Exception as e:
                    out[k] = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"[:300]}
                rm = out[k].get("running_max")
                if out[k].get("status") == "CAPACITY_LIMITED" or (rm is not None and 0 < rm < n and
                                                                  str(out[k].get("status")).startswith("INCONCLUSIVE")):
                    fit = rm
            if LIVE["rec"] is not None:
                LIVE["rec"].setdefault("live_windows", {})[k] = out[k]
            save()
        return out
    save()


if not RERUN:
    # ==================================================================== CELL 5 — the arms
    LADDER = [1, 2, 4, 5]                    # 40k: 8 streams would need 8 x (10 + 7) blocks of 4096 > 149 FP8 blocks (preempts)
    TEMP = 0.6                               # production sampling (Merlin's --override-generation-config is T 1.0)
    E23_TF_SCALE = {"tp4_vs_tp1_layout": 0.0772, "nvfp4_vs_bf16_kv": 0.3787}   # E23 SMOKE rev3g mean |d logprob|, for scale
    PREFILL = {"tok_s": 700.0, "measured": False}
    OOM_STEP = 268435456                     # Merlin's note: lower KV_BYTES in 0.25 GiB steps if the boot OOMs
    # Pre-run capacity (Merlin's rule: blocks per request = ceil(L / 4096) + 7; FP8 149 usable blocks, BF16 ~78)
    R["capacity_plan"] = {
        "fp8": {"pool_blocks": 149, "40k@5": 5 * (math.ceil(130560 / 4096) + 7), "8k@8": 8 * (math.ceil(32000 / 4096) + 7),
                "90k@4": 4 * (math.ceil(131072 / 4096) + 7), "116k@3": 3 * (math.ceil(131072 / 4096) + 7),
                "116k@4": 4 * (math.ceil(131072 / 4096) + 7),
                "note": "blocks if every stream grew to max length; the ladder windows are 90 s, so real use is far lower; "
                        "90k@4 and 116k@4 are capacity-limit probes"},
        "bf16": {"pool_blocks": 78, "40k@3": 3 * (math.ceil(130560 / 4096) + 7)},
        "long_rows": "1 natural xhigh ARC row (max 120,000 new) + 1 forced 60,000-token row, temperature 0.6, run CONCURRENTLY "
                     "(concurrency 2): not a single-stream latency number",
    }
    R["correctness_meaning"] = ("Both Merlin boots run MTP K3: CLEAN = BF16_MTP3 vs FP8_MTP3 parity plus probes / sampled / greedy "
                                "evidence; it is NOT target-only (MTP off) correctness. A production GO needs an MTP-off oracle and "
                                "a forced-rejection test.")
    CLEAN = ("EXACT", "NEAR_TIE")


    def _windows(o):
        """Every steady window of a step record (levels dicts + steady_multi replicas)."""
        out = []
        for v in (o.get("steady") or {}).values():
            if isinstance(v, dict):
                out.append(v)
        for k in ("40k", "90k"):
            for r in ((o.get(k) or {}).get("replicas") or []) if isinstance(o.get(k), dict) else []:
                if isinstance(r, dict):
                    out.append(r)
        return out


    def oracle_gate(o, need=(), extra=()):
        """None when clean; otherwise the NO_GO / INCONCLUSIVE status. `need` = refs that must compare clean; `extra` =
        further comparison dicts (replica greedy, 90k greedy) that must compare clean too. rev 3e (audits B-O1, C-F2):
        garbage under load in any window and failed arithmetic probes (real servers; the fake has no chat) are NO_GO too."""
        orc = o.get("oracle") or {}
        if any(orc.get("errors") or []):
            return "NO_GO_ORACLE_REQUEST_ERROR"
        if any(orc.get("garbage") or []):
            return "NO_GO_ORACLE_GARBAGE"
        if any(w.get("status") == "FAIL_GARBAGE_UNDER_LOAD" for w in _windows(o)):
            return "NO_GO_GARBAGE_UNDER_LOAD"
        if not FAKE and isinstance(o.get("probes"), dict) and o["probes"].get("ok") is False:
            return "NO_GO_SHORT_PROBE"
        sts = [(orc.get("vs_" + k) or {}).get("status") for k in need] + [(x or {}).get("status") for x in extra]
        if any(s == "FAIL_HIGH_MARGIN" for s in sts):
            return "NO_GO_ORACLE"
        if any(s not in CLEAN for s in sts):
            return "INCONCLUSIVE_NO_ORACLE" if any(s in (None, "INCONCLUSIVE_NO_ORACLE") for s in sts) else "INCONCLUSIVE_ORACLE"
        return None


    def gated(o, need=(), extra=()):
        g = oracle_gate(o, need, extra)
        o["correctness"] = g or "CLEAN"
        if g and o.get("status") in (None, "DONE"):
            o["status"] = g
        return o




    def _finite(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


    _compare_e23, _compare_tf_e23 = L.compare, L.compare_tf


    def _compare_rc2(ref, cand):
        """Final audit H2: a non-finite margin on either side makes the comparison INCONCLUSIVE before any token logic
        (E23's compare read a NaN reference margin as NEAR_TIE)."""
        for rows in (ref or [], cand or []):
            for r in rows:
                if any(m is not None and not _finite(m) for m in (r.get("margins") or [])):
                    return {"status": "INCONCLUSIVE_NONFINITE"}
        return _compare_e23(ref, cand)


    def _compare_tf_rc2(ref, cand):
        """Final audit H2: a non-finite teacher-forced logprob is never MEASURED."""
        for side in (ref, cand):
            for p in ((side or {}).get("positions") or []):
                if p and p[0] is not None and not _finite(p[0]):
                    return {"status": "INCONCLUSIVE_NONFINITE"}
        return _compare_tf_e23(ref, cand)


    L.compare, L.compare_tf = _compare_rc2, _compare_tf_rc2


    def short_gate_failed(o):
        """Final audit H1: a failed probe (real servers; the fake has no chat) or a failed / garbage sampled request stops
        the boot's body at once; gated() then gives the NO_GO verdict."""
        p, s = o.get("probes") or {}, o.get("sampled") or {}
        return (p.get("ok") is not True and not FAKE) or s.get("ok") is not True or bool(s.get("garbage"))


    def oracle_ok(orc):
        return bool(orc) and orc.get("complete") is True and not any(orc.get("errors") or []) and not any(orc.get("garbage") or [])


    def _windows(o):
        """Every load window of a record: the ladder AND the long census (audit: census_long was not gated)."""
        out = []
        for key in ("steady", "census_long"):
            for v in (o.get(key) or {}).values():
                if isinstance(v, dict):
                    out.append(v)
        return out


    def evidence_gate(o, expected_rows, require_sampled=True):
        """Fail-closed evidence (three pre-GPU audits): probes, the sampled path and a COMPLETE oracle are mandatory;
        a missing piece is never CLEAN. expected_rows None = no oracle in this arm (diagnostic arms)."""
        pr = o.get("probes")
        if not isinstance(pr, dict):
            return "INCONCLUSIVE_PROBES_MISSING"
        if pr.get("ok") is not True and not FAKE:          # the fake server has no chat endpoint (E23 rule)
            return "NO_GO_SHORT_PROBE"
        if require_sampled:
            sm = o.get("sampled")
            if not isinstance(sm, dict):
                return "INCONCLUSIVE_NO_SAMPLED_TEST"
            if sm.get("ok") is not True or sm.get("garbage"):
                return "NO_GO_SAMPLED_PATH"
        if expected_rows is None:
            return None
        orc = o.get("oracle")
        if not isinstance(orc, dict):
            return "INCONCLUSIVE_NO_ORACLE"
        if any(orc.get("errors") or []):
            return "NO_GO_ORACLE_REQUEST_ERROR"
        if any(orc.get("garbage") or []):
            return "NO_GO_ORACLE_GARBAGE"
        if orc.get("row_count") != expected_rows or orc.get("complete") is not True:
            return "INCONCLUSIVE_ORACLE_INCOMPLETE"
        return None


    _gated_e23 = gated


    def gated(o, need=(), extra=(), expected_rows=4, require_sampled=True):
        g = evidence_gate(o, expected_rows, require_sampled)
        if g:
            o["correctness"] = g
            if o.get("status") in (None, "DONE"):
                o["status"] = g
            return o
        return _gated_e23(o, need, extra)


    def oracle(port, key=None, refs=(), prompts=None):
        """E23's oracle plus completeness: every expected row present, no error, exactly NTOK tokens (greedy rows run with
        ignore_eos, so a short row is a defect); a reference is stored only when complete and garbage-free."""
        prompts = list(prompts) if prompts is not None else ORACLE_P + LONG_P
        rows = L.greedy(port, prompts, n_tok=NTOK, timeout=max(120, min(900, left() / 6)))
        garbage = [L.degenerate(r.get("text")) for r in rows]
        complete = len(rows) == len(prompts) and all(not r.get("error") and len(r.get("tokens") or []) == NTOK for r in rows)
        if key and complete and not any(garbage):
            ROWS[key] = rows
        elif key:
            R["errors"].append(f"oracle {key}: reference not stored (incomplete / error / garbage)")
        res = {"garbage": garbage, "errors": [r.get("error") for r in rows], "text_head": [(r.get("text") or "")[:120] for r in rows],
               "row_count": len(rows), "expected_rows": len(prompts), "tokens": [len(r.get("tokens") or []) for r in rows],
               "complete": complete}
        if not FAKE:
            res["template_probe"] = L.template_probe(port, TPL_MSG, TPL_IDS)
        for ref in refs:
            res["vs_" + ref] = L.compare(ROWS.get(ref), rows) if ROWS.get(ref) else {"status": "INCONCLUSIVE_NO_ORACLE"}
        return res


    def kv_census(s):
        log_t = s.log() or ""
        lines = log_t.splitlines()
        kv = re.findall(r"GPU KV cache size: ([\d,]+) tokens", log_t)
        conc = re.findall(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", log_t)
        pick = lambda rx, n=24: [l[:240] for l in lines if re.search(rx, l, re.I)][:n]
        return {"kv_tokens": [int(x.replace(",", "")) for x in kv][:2], "max_concurrency": conc[:2],
                "scale_lines": pick(r"K/V scale|kv scale|scaling factor|DEFAULT 1\.0|static K/V"),
                "engine_lines": pick(r"WNA16 MoE backend|MarlinExperts|PLE offload|engram|cudagraph_mode|CUDAGraphMode|PIECEWISE"),
                "startup_lines": pick(r"Loading weights took|Model loading took|torch\.compile takes|torch\.compile took|"
                                      r"Graph capturing finished|init engine|PLE.*ready|Available KV cache memory"),
                "oom_lines": [l[:240] for l in lines if OOM_ERR.search(l)][:6], "host_mem": host_mem()}


    def sampled_check(s):
        """RC1 died at its first sampled request (FlashInfer sampling JIT: ld cannot find -lcuda): one short sampled request
        right after ready, then the engine must still be alive."""
        g = L.Stream(s.port, PROMPTS["2k"][1], 64, temperature=TEMP, seed=5)
        g.start()
        g.join(900)
        if g.is_alive():
            g.stop()
            g.join(10)
        r = g.row()
        return {"tokens": len(r["tokens"]), "error": r["error"], "finish": r["finish_reason"], "alive_after": s.alive(),
                "garbage": L.degenerate(r["text"]), "ok": len(r["tokens"]) == 64 and not r["error"] and s.alive()}


    def prefill_probe(port):
        """Time to first token of one cold 40k prompt (1 new token): the measured prefill rate the ladder admission uses."""
        ids = PROMPTS["40k"][7]
        g = L.Stream(port, ids, 1, temperature=0.0)
        g.start()
        g.join(1800)
        if g.is_alive():
            g.stop()
            g.join(10)
        sec = (g.t_first or g.t_end or time.perf_counter()) - (g.t_start or time.perf_counter())
        ok = not g.err and g.t_first is not None
        if ok:
            PREFILL.update(tok_s=round(len(ids) / max(1e-3, sec), 1), measured=True)
        return {"prompt": len(ids), "ttft_sec": round(sec, 1), "tok_s": round(len(ids) / max(1e-3, sec), 1), "error": g.err, "ok": ok}


    def window_garbage(ids, win=2000, min_len=200):
        """Final audit H3: the first garbage 2,000-token window of a row (None if every window is clean). A whole-text check
        would read 100k tokens of healthy reasoning as LOW_VARIETY (unique words < 8 %), so windows are checked one by one."""
        for i in range(0, len(ids), win):
            seg = ids[i:i + win]
            if len(seg) < min_len:
                break
            v = L.degenerate(TOK.decode(seg))
            if v:
                return {"start_token": i, "verdict": v}
        return None


    def long_quality(name, r):
        if not isinstance(r, dict):
            return "MISSING_LONG"
        if r.get("error"):
            return "FAIL_LONG_REQUEST"
        if r.get("window_garbage"):
            return "FAIL_LONG_GARBAGE"
        if name == "forced60k" and not r.get("reached_60k"):
            return "INCOMPLETE_LONG"
        if name == "natural" and r.get("cut_by_cap"):
            return "INCOMPLETE_LONG"
        return "PASS_LONG"


    def long_gate(o):
        """None when the record has no long rows or both rows PASS_LONG; otherwise the verdict for the top-level status.
        Priority: request error > natural garbage > forced garbage (synthetic past its first EOS) > incomplete."""
        lo = o.get("long")
        if not isinstance(lo, dict):
            return None
        q = {n: long_quality(n, lo.get(n)) for n in ("natural", "forced60k")}
        lo["long_quality"] = q
        if "MISSING_LONG" in q.values():
            return "INCONCLUSIVE_LONG_MISSING"
        if "FAIL_LONG_REQUEST" in q.values():
            return "NO_GO_LONG_REQUEST_ERROR"
        if q["natural"] == "FAIL_LONG_GARBAGE":
            return "NO_GO_LONG_GARBAGE"
        if q["forced60k"] == "FAIL_LONG_GARBAGE":
            return "NO_GO_FORCED60K_GARBAGE"
        if "INCOMPLETE_LONG" in q.values():
            return "INCONCLUSIVE_LONG_INCOMPLETE"
        return None


    def long_rows(port, secs_cap=3000):
        """One natural xhigh ARC request (its own stop, up to 120,000 new tokens) and one forced 60,000-token row
        (ignore_eos: 'synthetic' past its first EOS), CONCURRENTLY (concurrency 2). Full text in traces/ + requests.jsonl.
        A row cut by the time cap is incomplete, not a correctness failure."""
        cap = max(60, min(secs_cap, left() - 900))
        jobs = {"natural": L.Stream(port, TPL_IDS, min(120000, MAX_LEN - len(TPL_IDS) - 512), temperature=TEMP, seed=11,
                                    ignore_eos=False),
                "forced60k": L.Stream(port, PROMPTS["8k"][0], 60000, temperature=TEMP, seed=12, ignore_eos=True)}
        t = time.time()
        for g in jobs.values():
            g.start()
        for g in jobs.values():
            g.join(max(1, cap - (time.time() - t)))
        for g in jobs.values():
            if g.is_alive():
                g.stop()
                g.join(10)
        out = {"concurrency": 2, "cap_sec": round(cap)}
        for name, g in jobs.items():
            row = g.row()
            (TRACES / f"long_{port}_{name}.txt").write_text(row["text"])
            ids = row["tokens"]
            n_tok = len(ids)
            sec = (g.t_end or time.perf_counter()) - (g.t_start or time.perf_counter())
            sl = {}
            for k, end in (("to_8k", 8000), ("to_32k", 32000), ("to_60k", 60000), ("to_100k", 100000)):
                sl[k] = L.degenerate(TOK.decode(ids[end - 2000:end])) if n_tok >= end else "NOT_REACHED"
            out[name] = {"tokens": n_tok, "sec": round(sec, 1), "tok_s": round(n_tok / max(1e-6, sec), 1), "error": row["error"],
                         "finish": row["finish_reason"], "garbage": sl, "window_garbage": window_garbage(ids),
                         "think_closed": "</think>" in row["text"], "cut_by_cap": row["finish_reason"] is None,
                         "tail": row["text"][-400:]}
        out["forced60k"]["reached_60k"] = out["forced60k"]["tokens"] >= 60000
        out["forced60k"]["label"] = "synthetic past the first EOS (ignore_eos)"
        try:
            m = {k: v for k, v in L.prom(port).items() if k.startswith(("vllm:spec_decode", "vllm:num_preemptions", "vllm:kv_cache_usage"))}
        except Exception as e:
            m = {"error": repr(e)[:200]}
        acc, drf = m.get(L.ACC), m.get(L.DRF)
        out["metrics_end"] = m
        out["mtp_accept_per_draft"] = round(acc / drf, 3) if acc and drf else None
        return out


    def tf_last(s, key=None, ref=None, aa=False):
        """Teacher-forced 40k LAST in a boot (chunk x 248k-vocab prompt logits can OOM the last PP rank); free memory first.
        How a server death here is classified is decided from its log in merlin_boot (TF_OOM vs TF_SERVER_DIED)."""
        o = {"gpu_before": L.gpu_mem()}
        if key:
            o["ref"] = tforced(s.port, key=key)
            if aa and s.alive():
                o["aa"] = tforced(s.port, ref=key)              # A/A: the same boot, the same prompt (the noise floor)
                TF["aa"] = o["aa"]
        if ref:
            o["vs_" + ref] = tforced(s.port, ref=ref)
            o["aa_" + ref] = TF.get("aa")
            o["e23_scale"] = E23_TF_SCALE
        o["alive_after"] = s.alive()
        return o


    def merlin_boot(name, kv, port, timeout, body, need=(), expected_rows=4, require_sampled=True):
        """Boot gate (ok must be True), one Merlin boot, cleanup of the /dev/shm entries this boot created; a boot OOM
        retries once 0.25 GiB lower. A server death during TF is TF_OOM only with an OOM line and no device fault."""
        kvb = NB1_KV.get(name) or (KV_FP8 if kv == "fp8" else KV_BF16)   # E25 NB1: the B3 pin step
        tries = []
        while True:
            gate = boot_gate()
            if gate["ok"] is not True:
                return {"status": "GPU_NOT_EMPTY", "boot_gate": gate, "tries": tries}
            shm0 = shm_names()
            o = with_servers([merlin_fn_server(name, kv, port, kvb)], timeout, body)
            o.update(kv_bytes=kvb, boot_gate=gate, shm_cleanup=shm_cleanup(shm0))
            srv = (o.get("servers") or {}).get(name) or {}
            oom = o.get("status") == "LAUNCH_FAILED" and bool(OOM_ERR.search(srv.get("log_tail") or ""))
            tries.append({"kv_bytes": kvb, "status": o.get("status"), "oom": oom})
            if not oom or len(tries) > 1 or left() < timeout + 900:
                break
            kvb -= OOM_STEP
            log("boot OOM:", name, "retry with KV_BYTES", kvb)
        o["tries"] = tries
        if o.get("tf_died") and o.get("status") == "SERVER_DIED":
            srv = (o.get("servers") or {}).get(name) or {}
            if OOM_ERR.search(srv.get("log_tail") or "") and not srv.get("xid"):
                o["status"] = "TF_OOM"
                o["status_note"] = "the server died of OOM in the teacher-forced call (run last); everything before it is valid"
            else:
                o["status"] = "TF_SERVER_DIED"
                o["status_note"] = "the server died in the teacher-forced call without an OOM line (or with a device fault)"
        g = gated(o, need=need, expected_rows=expected_rows, require_sampled=require_sampled)
        if g.get("status") == "TF_OOM" and g.get("correctness") != "CLEAN":
            g["status"] = g["correctness"]
        lg = long_gate(g)                                  # final audit H3: the long rows have their own verdict
        g["long_verdict"] = lg or ("PASS_LONG" if isinstance(g.get("long"), dict) else None)
        if lg and g.get("status") in ("DONE", "TF_OOM"):
            g["status"] = lg
        return g


    def arm_bf16():
        def body(ss):
            s = ss[0]
            p = s.port
            o = {"census": kv_census(s), "probes": L.chat_probes(p), "sampled": sampled_check(s)}
            if short_gate_failed(o):
                o.update(early_stop="SHORT_CORRECTNESS_GATE_FAILED", census_end=kv_census(s))
                return o
            o["prefill"] = prefill_probe(p)
            o["oracle"] = oracle(p, key="bf16")
            o["census_end"] = kv_census(s)
            if not oracle_ok(o["oracle"]):
                o["early_stop"] = "ORACLE_INCOMPLETE"          # no reference: the TF A/A would measure nothing usable
                return o
            o["tf"] = tf_last(s, key="bf16", aa=True)
            o["tf_died"] = not o["tf"]["alive_after"]
            return o
        ok, why = sha_gate(min(1500, left() - 900))
        R["setup"]["sha_gate"] = {"ok": ok, "why": why}
        if not ok:
            return {"status": "SKIPPED_SHA_GATE", "why": why}
        return merlin_boot("merlin_bf16", "bf16", 8000, 45 * 60, body)


    def fp8_gate(o):
        vs = (o["oracle"].get("vs_bf16") or {}).get("status")
        if (o.get("sampled") or {}).get("ok") is not True:
            return "FAIL_SAMPLED"
        if (o.get("probes") or {}).get("ok") is not True and not FAKE:
            return "FAIL_PROBES"
        if any(o["oracle"].get("errors") or []) or any(o["oracle"].get("garbage") or []) or vs == "FAIL_HIGH_MARGIN":
            return "FAIL"
        if o["oracle"].get("complete") is True and vs in CLEAN:
            return "PASS"
        return "INCONCLUSIVE"


    def arm_fp8():
        def body(ss):
            s = ss[0]
            p = s.port
            o = {"census": kv_census(s), "probes": L.chat_probes(p), "sampled": sampled_check(s)}
            if short_gate_failed(o):
                o.update(early_stop="SHORT_CORRECTNESS_GATE_FAILED", census_end=kv_census(s),
                         fp8_gate="FAIL_SAMPLED" if (o["sampled"] or {}).get("ok") is not True else "FAIL_PROBES")
                return o
            if not PREFILL["measured"]:
                o["prefill"] = prefill_probe(p)
            o["oracle"] = oracle(p, refs=("bf16",))
            o["fp8_gate"] = fp8_gate(o)
            if o["fp8_gate"] == "PASS":                      # load tests only on a PASS (audits: INCONCLUSIVE never runs them)
                kw = {"server": s, "temperature": TEMP, "prefill_tok_s": PREFILL["tok_s"]}
                o["steady"] = levels(p, "40k", LADDER, **kw)
                o["steady"].update(levels(p, "8k", [8], **kw))
                o["census_long"] = {**levels(p, "90k", [3, 4], **kw), **levels(p, "116k", [3, 4], **kw)}
                o["long"] = long_rows(p)
            else:
                o["load_tests_skipped"] = f"FP8 comparison gate: {o['fp8_gate']}"
                o["status"] = "NO_GO_FP8_KV" if o["fp8_gate"].startswith("FAIL") else "INCONCLUSIVE_FP8_KV"
            o["census_end"] = kv_census(s)
            if o["fp8_gate"] == "PASS" or os.environ.get("RC2_TF_ON_FP8_FAIL") == "1":   # default: fail fast
                o["tf"] = tf_last(s, ref="bf16")
                o["tf_died"] = not o["tf"]["alive_after"]
            return o
        return merlin_boot("merlin_fp8", "fp8", 8001, 40 * 60, body, need=("bf16",))


    def arm_bf16_load():
        """Only when FP8 fails: the ladder, the census and the long rows on BF16 KV (seqs 3). DIAGNOSTIC_ONLY: this arm
        has no oracle of its own (the A1 boot is the BF16 reference)."""
        def body(ss):
            s = ss[0]
            p = s.port
            o = {"census": kv_census(s), "probes": L.chat_probes(p), "sampled": sampled_check(s)}
            if short_gate_failed(o):
                o["early_stop"] = "SHORT_CORRECTNESS_GATE_FAILED"
            else:
                kw = {"server": s, "temperature": TEMP, "prefill_tok_s": PREFILL["tok_s"]}
                o["steady"] = levels(p, "40k", [1, 2, 3], **kw)
                o["census_long"] = levels(p, "116k", [1, 2], **kw)
                o["long"] = long_rows(p)
            o["census_end"] = kv_census(s)
            o["certification"] = "DIAGNOSTIC_ONLY"
            return o
        return merlin_boot("merlin_bf16_load", "bf16", 8002, 30 * 60, body, expected_rows=None)


    def arm_stock():
        """Fallback when Merlin cannot serve: stock vLLM 0.32 + Intel W4A16 (RC1's correct engine), -lcuda shim, FP8 KV."""
        if INTEL is None or WHEELHOUSE is None:
            return {"status": "FALLBACK_UNAVAILABLE", "why": f"intel={INTEL} wheelhouse={WHEELHOUSE}"}

        def body(ss):
            s = ss[0]
            p = s.port
            o = {"census": kv_census(s), "probes": L.chat_probes(p), "sampled": sampled_check(s)}
            if short_gate_failed(o):
                o["early_stop"] = "SHORT_CORRECTNESS_GATE_FAILED"
                return o
            o["oracle"] = oracle(p, key="stock", prompts=ORACLE_P)          # 2k / 2k / 8k (32k max length: no 40k row)
            if oracle_ok(o["oracle"]):
                o["steady"] = levels(p, "8k", [1, 2, 3], server=s, temperature=TEMP)
            return o
        gate = boot_gate()
        if gate["ok"] is not True:
            return {"status": "GPU_NOT_EMPTY", "boot_gate": gate}
        shm0 = shm_names()
        o = gated(with_servers([stock_fn_server("stock_fn", 8003)], 40 * 60, body), expected_rows=len(ORACLE_P))
        o.update(boot_gate=gate, shm_cleanup=shm_cleanup(shm0))
        return o


    save()


if not RERUN:
    # ==================================================================== CELL 6 — reused tools, verbatim (NB0 CELL 6, NB1 CELL 6) + marked edits
    import statistics
    import numpy as np
    import urllib.request as UR
    import urllib.error as UE
    from collections import Counter
    R.pop("capacity_plan", None)          # RC2-only records written by RC2's CELL 5 definitions
    R.pop("correctness_meaning", None)
    RC2_REF = json.loads("{\"fp8\": {\"argv_tail\": [\"127.0.0.1\", \"--port\", \"8001\", \"--tensor-parallel-size\", \"2\", \"--pipeline-parallel-size\", \"2\", \"--enable-expert-parallel\", \"--disable-custom-all-reduce\", \"--speculative-config\", \"{\\\"method\\\":\\\"mtp\\\",\\\"num_speculative_tokens\\\":3,\\\"draft_sample_method\\\":\\\"probabilistic\\\"}\", \"--max-model-len\", \"262144\", \"--max-num-seqs\", \"8\", \"--max-num-batched-tokens\", \"1024\", \"--gpu-memory-utilization\", \"0.96\", \"--kv-cache-memory-bytes\", \"2479838700\", \"--kv-cache-dtype\", \"fp8_e4m3\", \"--limit-mm-per-prompt\", \"{\\\"image\\\":42,\\\"video\\\":0}\", \"--mm-processor-cache-gb\", \"0\", \"--mm-processor-kwargs\", \"{\\\"size\\\":{\\\"longest_edge\\\":8388608,\\\"shortest_edge\\\":65536}}\", \"--compilation-config\", \"{\\\"cudagraph_mode\\\":\\\"FULL_AND_PIECEWISE\\\",\\\"cudagraph_capture_sizes\\\":[1,2,3,4,6,8,12,16,24,32]}\", \"--enable-prefix-caching\", \"--mamba-cache-mode\", \"align\", \"--enable-prompt-tokens-details\", \"--no-enable-flashinfer-autotune\", \"--enable-auto-tool-choice\", \"--tool-call-parser\", \"qwen3_coder\", \"--reasoning-parser\", \"qwen3\", \"--override-generation-config\", \"{\\\"temperature\\\":1.0,\\\"top_p\\\":0.95,\\\"top_k\\\":20,\\\"min_p\\\":0.0,\\\"presence_penalty\\\":0.0,\\\"repetition_penalty\\\":1.0}\", \"--default-chat-template-kwargs\", \"{\\\"enable_thinking\\\": true, \\\"reasoning_effort\\\": \\\"medium\\\"}\", \"--shutdown-timeout\", \"60\", \"--block-size\", \"4096\", \"--use-replayssm\", \"--prefix-match-unit\", \"64\", \"--served-model-name\", \"q27\", \"--max-model-len\", \"131072\", \"--port\", \"8001\", \"--default-chat-template-kwargs\", \"{\\\"enable_thinking\\\": true, \\\"reasoning_effort\\\": \\\"xhigh\\\"}\"], \"env\": {\"KV_BYTES\": \"2479838700\", \"NCCL_P2P_DISABLE\": \"1\", \"NCCL_VERSION\": \"2.25.1-1\", \"P2P\": \"off\", \"SCALES\": \"/tmp/fn_merlin/qsa_kv_scales_262k.json\", \"VLLM_CACHE_ROOT\": \"/tmp/merlin-cache\", \"VLLM_E2_GUARD_MODE\": \"warn\", \"VLLM_E2_PLE_EARLY_REQUEST\": \"1\", \"VLLM_E2_PLE_FAST_IDS\": \"1\", \"VLLM_E2_QSA_LOGITS_GRANULE_MB\": \"256\", \"VLLM_HOST_EMBED_TABLE\": \"1\", \"VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS\": \"0\", \"VLLM_PLE_OFFLOAD_HOME_DEVICE\": \"0\", \"VLLM_PLE_OFFLOAD_READY_TIMEOUT\": \"3600\", \"VLLM_PORT\": \"30100\", \"VLLM_PP_LAYER_PARTITION\": \"25,23\", \"VLLM_QSA_KV_SCALES\": \"/tmp/fn_merlin/qsa_kv_scales_262k.json\", \"VLLM_SKIP_P2P_CHECK\": \"1\", \"VLLM_SPARSE_INDEXER_MAX_LOGITS_MB\": \"256\", \"VLLM_USE_BREAKABLE_CUDAGRAPH\": \"0\", \"VLLM_USE_RUST_FRONTEND\": \"0\", \"VLLM_USE_V2_MODEL_RUNNER\": \"1\"}, \"kv_bytes\": 2479838700, \"kv_tokens\": 513940, \"seqs\": 8}, \"bf16\": {\"argv_tail\": [\"127.0.0.1\", \"--port\", \"8000\", \"--tensor-parallel-size\", \"2\", \"--pipeline-parallel-size\", \"2\", \"--enable-expert-parallel\", \"--disable-custom-all-reduce\", \"--speculative-config\", \"{\\\"method\\\":\\\"mtp\\\",\\\"num_speculative_tokens\\\":3,\\\"draft_sample_method\\\":\\\"probabilistic\\\"}\", \"--max-model-len\", \"262144\", \"--max-num-seqs\", \"3\", \"--max-num-batched-tokens\", \"1024\", \"--gpu-memory-utilization\", \"0.96\", \"--kv-cache-memory-bytes\", \"2464415744\", \"--kv-cache-dtype\", \"bfloat16\", \"--limit-mm-per-prompt\", \"{\\\"image\\\":42,\\\"video\\\":0}\", \"--mm-processor-cache-gb\", \"0\", \"--mm-processor-kwargs\", \"{\\\"size\\\":{\\\"longest_edge\\\":8388608,\\\"shortest_edge\\\":65536}}\", \"--compilation-config\", \"{\\\"cudagraph_mode\\\":\\\"FULL_AND_PIECEWISE\\\",\\\"cudagraph_capture_sizes\\\":[1,2,3,4,6,8,12]}\", \"--enable-prefix-caching\", \"--mamba-cache-mode\", \"align\", \"--enable-prompt-tokens-details\", \"--no-enable-flashinfer-autotune\", \"--enable-auto-tool-choice\", \"--tool-call-parser\", \"qwen3_coder\", \"--reasoning-parser\", \"qwen3\", \"--override-generation-config\", \"{\\\"temperature\\\":1.0,\\\"top_p\\\":0.95,\\\"top_k\\\":20,\\\"min_p\\\":0.0,\\\"presence_penalty\\\":0.0,\\\"repetition_penalty\\\":1.0}\", \"--default-chat-template-kwargs\", \"{\\\"enable_thinking\\\": true, \\\"reasoning_effort\\\": \\\"medium\\\"}\", \"--shutdown-timeout\", \"60\", \"--block-size\", \"4096\", \"--use-replayssm\", \"--prefix-match-unit\", \"64\", \"--served-model-name\", \"q27\", \"--max-model-len\", \"131072\", \"--port\", \"8000\", \"--default-chat-template-kwargs\", \"{\\\"enable_thinking\\\": true, \\\"reasoning_effort\\\": \\\"xhigh\\\"}\"], \"env\": {\"KV_BYTES\": \"2464415744\", \"NCCL_P2P_DISABLE\": \"1\", \"NCCL_VERSION\": \"2.25.1-1\", \"P2P\": \"off\", \"VLLM_CACHE_ROOT\": \"/tmp/merlin-cache\", \"VLLM_E2_GUARD_MODE\": \"warn\", \"VLLM_E2_PLE_EARLY_REQUEST\": \"1\", \"VLLM_E2_PLE_FAST_IDS\": \"1\", \"VLLM_E2_QSA_LOGITS_GRANULE_MB\": \"256\", \"VLLM_HOST_EMBED_TABLE\": \"1\", \"VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS\": \"0\", \"VLLM_PLE_OFFLOAD_HOME_DEVICE\": \"0\", \"VLLM_PLE_OFFLOAD_READY_TIMEOUT\": \"3600\", \"VLLM_PORT\": \"30000\", \"VLLM_PP_LAYER_PARTITION\": \"25,23\", \"VLLM_SKIP_P2P_CHECK\": \"1\", \"VLLM_SPARSE_INDEXER_MAX_LOGITS_MB\": \"256\", \"VLLM_USE_BREAKABLE_CUDAGRAPH\": \"0\", \"VLLM_USE_RUST_FRONTEND\": \"0\", \"VLLM_USE_V2_MODEL_RUNNER\": \"1\"}, \"kv_bytes\": 2464415744, \"kv_tokens\": 269042, \"seqs\": 3}, \"tf40k\": {\"fp8_vs_bf16\": {\"status\": \"MEASURED\", \"n\": 256, \"mean_abs_dlogprob\": 1.8764, \"p99_abs_dlogprob\": 5.464, \"max_abs_dlogprob\": 7.15, \"rank1_agreement\": 0.3672, \"forced_token_rank1_both_or_neither\": 0.457}, \"bf16_aa\": {\"status\": \"MEASURED\", \"n\": 256, \"mean_abs_dlogprob\": 0.0, \"p99_abs_dlogprob\": 0.0, \"max_abs_dlogprob\": 0.0, \"rank1_agreement\": 1.0, \"forced_token_rank1_both_or_neither\": 1.0}}, \"oracle_rows\": {\"rc2_bf16\": [{\"text\": \"We need solve ARC task. Need infer transformation from examples. User gives Task 0934a4d8 with Example 1 input/output, Example 2 input/output, Example 3 input only? Need solve test inputs? They provided Example 3 input: 194499276 maybe incomplete? Need output? Need find transformation rule and solve test inputs. We need analyze.\\n\\nWe have ARC tasks represented as strings of digits? Example 1 input is 31 rows? Let's count. Example 1 input lines:\\n1 3533665414\", \"tokens\": [1596, 1144, 11290, 67964, 3274, 13, 14235, 22846, 17454, 494, 9989, 13, 2570, 6486, 5261, 220, 15, 24, 18, 19, 64, 19, 67, 23, 440, 12996, 220, 16, 1879, 45447, 11, 12996, 220, 17, 1879, 45447, 11, 12996, 220, 18, 1879, 1132, 30, 14235, 11290, 1228, 10804, 30, 2302, 3766, 12996, 220, 18, 1879, 25, 220, 16, 24, 19, 19, 24, 24, 17, 22, 21, 6970, 31084, 30, 14235, 2468, 30, 14235, 1423, 17454, 5728, 321, 11290, 1228, 10804, 13, 1165, 1144, 22903, 13, 271, 1596, 599, 67964, 8806, 14807, 430, 8796, 314, 17955, 30, 12996, 220, 16, 1879, 369, 220, 18, 16, 6761, 30, 6558, 579, 1698, 13, 12996, 220, 16, 1879, 4965, 25, 198, 16, 220, 18, 20, 18, 18, 21, 21, 20, 19, 16, 19], \"margins\": [11.375, 10.75, 0.25, 3.5, 0.125, 1.75, 2.375, 0.625, 4.375, 0.125, 4.125, 1.375, 0.125, 1.0, 1.25, 4.875, 8.6875, 10.375, 10.375, 9.6875, 8.25, 10.1875, 14.0, 10.4375, 0.125, 0.875, 1.25, 8.375, 3.875, 1.0, 3.75, 7.375, 3.75, 12.0, 2.625, 8.375, 6.75, 2.5, 8.75, 10.5, 6.625, 0.125, 0.375, 0.5, 0.75, 8.25, 4.125, 1.0, 1.0, 0.125, 0.25, 4.625, 6.375, 9.25, 2.875, 0.375, 4.125, 4.25, 10.9375, 11.5625, 10.625, 10.5625, 9.5625, 11.5, 10.6875, 0.5, 2.25, 4.875, 2.125, 1.25, 0.125, 0.5, 0.625, 0.625, 4.25, 1.125, 2.875, 3.125, 3.125, 4.125, 0.75, 1.375, 1.75, 0.75, 1.0, 4.25, 1.5, 0.5, 1.125, 0.625, 4.625, 1.625, 0.25, 4.5, 1.0, 0.0, 1.25, 8.5, 5.0, 1.125, 2.75, 3.375, 0.625, 0.125, 1.125, 1.875, 7.25, 1.875, 2.125, 0.25, 0.875, 6.25, 1.875, 1.625, 2.625, 2.875, 2.125, 2.375, 8.125, 10.9375, 10.0625, 9.0625, 9.875, 9.375, 11.25, 11.375, 11.875, 12.9375], \"error\": null, \"prompt_sha\": \"8df9ed2c640ef5bd\", \"prompt_len\": 2048, \"finish_reason\": \"length\"}, {\"text\": \"We need solve ARC tasks. Need infer transformation rules from examples and apply to test inputs. We have two tasks: 1818057f and 195c6913. Need output transformed test grids. Need analyze carefully.\\n\\nWe need produce final answer likely grids. Need infer rules.\\n\\nTask 1818057f:\\nExamples: inputs/outputs with colors 2,4,8. Output changes some 4s to 8? Let's inspect.\\n\\nExample1 input 10x12? rows:\\n1 22224\", \"tokens\": [1596, 1144, 11290, 67964, 8806, 13, 14235, 22846, 17454, 5426, 494, 9989, 321, 3668, 310, 1228, 10804, 13, 1165, 599, 1330, 8806, 25, 220, 16, 23, 16, 23, 15, 20, 22, 69, 321, 220, 16, 24, 20, 66, 21, 24, 16, 18, 13, 14235, 2468, 22773, 1228, 54455, 13, 14235, 22903, 15060, 13, 271, 1596, 1144, 7936, 1534, 4087, 4222, 54455, 13, 14235, 22846, 5426, 13, 271, 6065, 220, 16, 23, 16, 23, 15, 20, 22, 69, 25, 198, 39044, 25, 10804, 14, 39652, 440, 7736, 220, 17, 11, 19, 11, 23, 13, 8984, 4203, 1010, 220, 19, 82, 310, 220, 23, 30, 6558, 579, 23331, 13, 271, 12929, 16, 1879, 220, 16, 15, 87, 16, 17, 30, 6761, 25, 198, 16, 220, 17, 17, 17, 17, 19], \"margins\": [11.875, 11.375, 0.75, 3.375, 2.625, 3.0, 3.25, 0.5, 3.125, 0.25, 0.625, 4.75, 1.125, 3.25, 2.0, 5.5, 0.875, 3.75, 0.375, 0.5, 1.375, 6.5, 1.125, 2.25, 12.625, 12.0625, 11.3125, 12.9375, 12.4375, 8.875, 10.8125, 10.75, 1.75, 7.375, 13.5625, 13.875, 11.6875, 11.25, 10.75, 12.375, 12.375, 10.75, 2.25, 2.25, 1.625, 0.875, 1.0, 0.375, 0.875, 1.375, 0.125, 1.875, 6.5, 1.5, 1.0, 1.375, 1.0, 4.75, 4.625, 1.75, 1.0, 1.625, 0.875, 0.625, 0.25, 3.5, 1.0, 1.25, 4.625, 11.625, 5.75, 11.125, 12.3125, 10.8125, 10.9375, 11.0, 11.8125, 0.125, 0.0, 2.625, 0.375, 0.25, 0.875, 7.875, 0.75, 0.875, 1.875, 8.125, 0.75, 7.375, 2.875, 3.875, 2.125, 0.25, 2.0, 3.25, 2.625, 0.625, 2.625, 2.125, 10.75, 10.625, 0.75, 1.75, 7.5, 2.75, 6.375, 0.25, 1.75, 1.375, 0.75, 3.25, 8.375, 4.625, 3.75, 4.375, 2.625, 1.5, 0.25, 0.5, 5.375, 1.375, 0.5, 9.75, 4.625, 12.625, 11.4375, 8.375], \"error\": null, \"prompt_sha\": \"8750743e9dcadca9\", \"prompt_len\": 2048, \"finish_reason\": \"length\"}, {\"text\": \"We need solve ARC tasks. Need find transformation rule for each task and solve test inputs. We have multiple tasks: 247ef758, 269e22fb, 271d71e2, 28a6681f, 291dc1e1. Need output transformed grids for test inputs. Need infer rules from examples. Need be careful. We need produce final answer likely grids. Need analyze each task.\\n\\nWe need parse tasks. There are two test inputs for 247ef758? It says Test input: ...\", \"tokens\": [1596, 1144, 11290, 67964, 8806, 13, 14235, 1423, 17454, 5728, 364, 1754, 3274, 321, 11290, 1228, 10804, 13, 1165, 599, 5081, 8806, 25, 220, 17, 19, 22, 797, 22, 20, 23, 11, 220, 17, 21, 24, 68, 17, 17, 10481, 11, 220, 17, 22, 16, 67, 22, 16, 68, 17, 11, 220, 17, 23, 64, 21, 21, 23, 16, 69, 11, 220, 17, 24, 16, 7385, 16, 68, 16, 13, 14235, 2468, 22773, 54455, 364, 1228, 10804, 13, 14235, 22846, 5426, 494, 9989, 13, 14235, 381, 16097, 13, 1165, 1144, 7936, 1534, 4087, 4222, 54455, 13, 14235, 22903, 1754, 3274, 13, 271, 1596, 1144, 4563, 8806, 13, 2532, 513, 1330, 1228, 10804, 364, 220, 17, 19, 22, 797, 22, 20, 23, 30, 1049, 2640, 3284, 1879, 25, 2423], \"margins\": [11.75, 9.875, 1.125, 1.25, 3.875, 2.0, 2.875, 0.375, 5.25, 0.875, 1.5, 5.625, 0.875, 3.0, 2.875, 3.125, 7.75, 6.75, 0.5, 0.5, 1.5, 5.625, 1.375, 3.0, 11.75, 13.375, 12.625, 11.3125, 12.0, 13.5, 12.625, 0.625, 8.4375, 11.625, 11.0, 12.5625, 10.5625, 11.0, 10.375, 8.375, 7.375, 9.75, 11.5625, 9.125, 10.0, 9.75, 9.4375, 10.3125, 10.4375, 9.0, 8.1875, 9.4375, 12.625, 9.75, 11.0, 11.3125, 10.9375, 9.625, 9.5625, 10.0, 8.1875, 7.625, 12.375, 11.125, 11.375, 10.625, 10.25, 10.375, 10.0, 2.5, 3.0, 2.125, 0.125, 0.125, 1.875, 0.125, 4.125, 3.125, 2.0, 0.75, 3.0, 0.75, 8.75, 4.125, 0.125, 1.25, 3.875, 2.75, 0.125, 0.25, 0.625, 3.875, 1.625, 1.875, 0.75, 1.0, 0.375, 2.375, 2.125, 0.5, 0.5, 2.5, 1.875, 0.125, 0.75, 0.375, 3.625, 0.125, 1.375, 0.0, 4.125, 7.375, 2.5, 0.375, 12.75, 6.75, 12.375, 4.625, 8.75, 12.1875, 10.25, 2.0, 0.625, 0.25, 1.125, 7.625, 1.75, 2.125], \"error\": null, \"prompt_sha\": \"fc63e297d382cd8f\", \"prompt_len\": 8192, \"finish_reason\": \"length\"}, {\"text\": \"We need solve ARC tasks. Need find transformation rule for each task and solve test inputs. We need output likely transformed grids. Need analyze many tasks. Need be careful. We have multiple tasks with examples and test inputs. Need infer rules.\\n\\nWe need produce final answer with transformed test inputs for each task. There are tasks: 2c181942, 2d0172a1 (two test inputs?), 31f7f899, 332f06d7, 35ab12c3, 36a08778\", \"tokens\": [1596, 1144, 11290, 67964, 8806, 13, 14235, 1423, 17454, 5728, 364, 1754, 3274, 321, 11290, 1228, 10804, 13, 1165, 1144, 2468, 4222, 22773, 54455, 13, 14235, 22903, 1599, 8806, 13, 14235, 381, 16097, 13, 1165, 599, 5081, 8806, 440, 9989, 321, 1228, 10804, 13, 14235, 22846, 5426, 13, 271, 1596, 1144, 7936, 1534, 4087, 440, 22773, 1228, 10804, 364, 1754, 3274, 13, 2532, 513, 8806, 25, 220, 17, 66, 16, 23, 16, 24, 19, 17, 11, 220, 17, 67, 15, 16, 22, 17, 64, 16, 318, 19186, 1228, 10804, 66926, 220, 18, 16, 69, 22, 69, 23, 24, 24, 11, 220, 18, 18, 17, 69, 15, 21, 67, 22, 11, 220, 18, 20, 365, 16, 17, 66, 18, 11, 220, 18, 21, 64, 15, 23, 22, 22, 23], \"margins\": [10.125, 11.5, 1.125, 0.25, 3.5, 2.125, 2.625, 0.5, 5.25, 1.0, 1.75, 5.75, 0.375, 2.375, 3.625, 4.25, 6.625, 7.375, 0.25, 0.125, 1.25, 0.5, 0.25, 2.0, 0.5, 1.625, 1.0, 1.0, 4.375, 2.875, 0.75, 2.125, 4.125, 4.0, 0.375, 0.5, 1.625, 3.875, 0.375, 2.125, 2.125, 2.125, 2.875, 1.625, 2.875, 1.125, 0.75, 1.375, 0.125, 1.625, 2.25, 1.0, 2.875, 2.5, 3.0, 0.125, 1.125, 0.75, 1.25, 2.375, 1.75, 1.625, 0.625, 3.0, 0.125, 4.5, 0.5, 11.625, 10.625, 12.0625, 13.0625, 9.5625, 10.9375, 10.375, 10.875, 1.0, 6.5, 9.8125, 10.25, 11.9375, 13.5, 12.3125, 11.625, 11.25, 11.25, 0.375, 2.5, 4.125, 7.25, 0.625, 5.375, 9.75, 9.625, 12.4375, 11.25, 10.4375, 9.0, 11.4375, 10.8125, 3.25, 8.375, 11.875, 7.375, 11.5, 11.0, 10.5625, 8.8125, 10.875, 10.375, 5.625, 10.125, 13.125, 8.625, 7.375, 11.1875, 10.5, 10.25, 10.9375, 5.25, 9.75, 10.1875, 7.25, 11.4375, 12.0, 8.9375, 11.0625, 9.5, 10.3125], \"error\": null, \"prompt_sha\": \"49b3958bb234d4fd\", \"prompt_len\": 40000, \"finish_reason\": \"length\"}], \"rc2_fp8\": [{\"text\": \"We need solve ARC task. Need infer transformation from examples. User gives Task 0934a4d8. Example 1 input 30x30? output 9x4? Example 2 input 30x30? output 4x5? Example 3 input: \\\"194499276\\\" only one row? Maybe truncated? Need find transformation rule and solve test inputs. But only one test input? It says Example 3 input: 194499276. Maybe that's the test input? Need output. Need infer from\", \"tokens\": [1596, 1144, 11290, 67964, 3274, 13, 14235, 22846, 17454, 494, 9989, 13, 2570, 6486, 5261, 220, 15, 24, 18, 19, 64, 19, 67, 23, 13, 12996, 220, 16, 1879, 220, 18, 15, 87, 18, 15, 30, 2468, 220, 24, 87, 19, 30, 12996, 220, 17, 1879, 220, 18, 15, 87, 18, 15, 30, 2468, 220, 19, 87, 20, 30, 12996, 220, 18, 1879, 25, 328, 16, 24, 19, 19, 24, 24, 17, 22, 21, 1, 1132, 799, 2713, 30, 10380, 57739, 30, 14235, 1423, 17454, 5728, 321, 11290, 1228, 10804, 13, 1921, 1132, 799, 1228, 1879, 30, 1049, 2640, 12996, 220, 18, 1879, 25, 220, 16, 24, 19, 19, 24, 24, 17, 22, 21, 13, 10380, 421, 579, 279, 1228, 1879, 30, 14235, 2468, 13, 14235, 22846, 494], \"margins\": [11.5, 11.25, 0.25, 3.625, 0.375, 2.25, 2.25, 0.875, 4.125, 0.25, 4.25, 1.25, 0.125, 1.0, 1.75, 4.625, 9.0, 9.9375, 10.375, 9.5, 8.875, 10.6875, 14.0, 10.625, 0.125, 0.75, 0.875, 7.5, 4.5, 1.25, 3.5, 0.75, 0.0, 1.75, 4.0, 2.625, 1.25, 9.375, 5.5, 4.25, 2.0, 0.25, 2.625, 0.875, 8.875, 3.75, 4.375, 5.75, 5.5, 4.25, 3.5, 5.5, 0.0, 8.0, 7.25, 7.375, 4.625, 5.875, 1.5, 3.75, 4.875, 9.125, 6.75, 0.25, 0.625, 8.25, 8.25, 12.3125, 11.5, 10.3125, 11.0, 10.1875, 11.625, 9.875, 4.375, 2.25, 0.375, 1.0, 4.875, 0.0, 0.25, 5.75, 2.125, 0.0, 0.25, 2.375, 2.875, 6.75, 9.125, 5.125, 1.875, 0.75, 2.375, 0.0, 0.875, 1.875, 2.125, 0.75, 2.25, 1.625, 9.1875, 7.125, 9.125, 5.625, 1.125, 7.9375, 7.0625, 10.375, 11.375, 10.8125, 10.5625, 11.1875, 11.5, 11.625, 1.0, 1.625, 1.375, 2.625, 0.5, 2.375, 1.75, 2.625, 2.0, 2.625, 0.125, 1.5, 0.25, 1.5], \"error\": null, \"prompt_sha\": \"8df9ed2c640ef5bd\", \"prompt_len\": 2048, \"finish_reason\": \"length\"}, {\"text\": \"We need solve ARC tasks. Need infer transformation rules from examples and apply to test inputs. We have two tasks: 1818057f and 195c6913. Need output transformed test grids. Need analyze carefully.\\n\\nWe need produce final answer likely grids. Need infer rules.\\n\\nTask 1818057f examples:\\nInput/output grids with colors 2,4,8. Output changes some 4s to 8? Let's inspect.\\n\\nExample1 10x12? Input rows:\\n1 222244\", \"tokens\": [1596, 1144, 11290, 67964, 8806, 13, 14235, 22846, 17454, 5426, 494, 9989, 321, 3668, 310, 1228, 10804, 13, 1165, 599, 1330, 8806, 25, 220, 16, 23, 16, 23, 15, 20, 22, 69, 321, 220, 16, 24, 20, 66, 21, 24, 16, 18, 13, 14235, 2468, 22773, 1228, 54455, 13, 14235, 22903, 15060, 13, 271, 1596, 1144, 7936, 1534, 4087, 4222, 54455, 13, 14235, 22846, 5426, 13, 271, 6065, 220, 16, 23, 16, 23, 15, 20, 22, 69, 9989, 25, 198, 2425, 45447, 54455, 440, 7736, 220, 17, 11, 19, 11, 23, 13, 8984, 4203, 1010, 220, 19, 82, 310, 220, 23, 30, 6558, 579, 23331, 13, 271, 12929, 16, 220, 16, 15, 87, 16, 17, 30, 5396, 6761, 25, 198, 16, 220, 17, 17, 17, 17, 19, 19], \"margins\": [11.875, 11.75, 0.5, 3.125, 2.625, 3.0, 3.125, 0.5, 3.0, 0.5, 0.625, 4.875, 1.0, 3.25, 2.0, 5.625, 0.625, 3.875, 0.375, 0.75, 1.375, 6.75, 1.375, 2.125, 12.8125, 11.9375, 11.6875, 13.1875, 12.0625, 8.4375, 10.3125, 10.5, 1.625, 7.25, 13.5625, 12.625, 10.5625, 9.375, 12.0, 12.6875, 11.3125, 11.0625, 2.0, 2.375, 1.625, 0.875, 1.0, 0.125, 0.5, 1.25, 0.125, 1.875, 6.625, 1.625, 1.5, 1.5, 1.0, 4.75, 4.625, 2.0, 0.875, 1.5, 1.0, 0.625, 0.5, 3.375, 1.0, 1.125, 4.5, 12.375, 5.625, 11.125, 12.625, 10.0625, 9.875, 10.125, 11.625, 0.125, 1.375, 0.5, 1.0, 0.75, 0.375, 1.25, 1.0, 1.75, 7.75, 0.75, 7.75, 2.375, 3.625, 1.625, 0.25, 2.0, 3.5, 2.75, 1.125, 2.25, 2.375, 10.8125, 10.75, 0.75, 1.875, 7.5, 2.75, 5.25, 0.25, 2.125, 1.25, 0.0, 7.125, 3.625, 5.5, 4.5, 2.0, 1.5, 0.75, 0.875, 0.0, 6.375, 1.0, 0.75, 10.0, 4.75, 12.625, 11.5625, 8.375, 11.4375], \"error\": null, \"prompt_sha\": \"8750743e9dcadca9\", \"prompt_len\": 2048, \"finish_reason\": \"length\"}, {\"text\": \"We need solve ARC tasks. Need find transformation rule for each task and solve test inputs. We have multiple tasks: 247ef758, 269e22fb, 271d71e2, 28a6681f, 291dc1e1. Need output transformed test inputs. Need infer rules from examples. Need be careful. We can analyze each.\\n\\nWe need produce final answer likely grids for each test input. Need determine exact transformations.\\n\\nLet's parse tasks.\\n\\nTask 247ef758: examples show\", \"tokens\": [1596, 1144, 11290, 67964, 8806, 13, 14235, 1423, 17454, 5728, 364, 1754, 3274, 321, 11290, 1228, 10804, 13, 1165, 599, 5081, 8806, 25, 220, 17, 19, 22, 797, 22, 20, 23, 11, 220, 17, 21, 24, 68, 17, 17, 10481, 11, 220, 17, 22, 16, 67, 22, 16, 68, 17, 11, 220, 17, 23, 64, 21, 21, 23, 16, 69, 11, 220, 17, 24, 16, 7385, 16, 68, 16, 13, 14235, 2468, 22773, 1228, 10804, 13, 14235, 22846, 5426, 494, 9989, 13, 14235, 381, 16097, 13, 1165, 628, 22903, 1754, 13, 271, 1596, 1144, 7936, 1534, 4087, 4222, 54455, 364, 1754, 1228, 1879, 13, 14235, 7995, 4581, 51192, 13, 271, 9764, 579, 4563, 8806, 13, 271, 6065, 220, 17, 19, 22, 797, 22, 20, 23, 25, 9989, 1420], \"margins\": [11.875, 9.75, 1.25, 1.5, 3.625, 2.125, 3.0, 0.125, 5.25, 0.875, 1.625, 5.5, 0.875, 3.125, 2.875, 2.875, 8.5, 7.125, 0.375, 0.375, 1.875, 5.375, 1.625, 3.0, 12.125, 13.3125, 12.9375, 11.8125, 11.8125, 13.5, 12.5, 0.375, 8.875, 11.6875, 10.9375, 12.25, 11.0625, 10.6875, 10.375, 8.3125, 7.375, 10.4375, 11.1875, 9.0, 9.8125, 10.125, 10.25, 10.8125, 10.125, 9.3125, 8.0625, 9.5, 12.4375, 10.4375, 11.0, 11.1875, 11.1875, 9.6875, 10.375, 9.0, 7.75, 7.625, 12.875, 10.8125, 11.375, 9.625, 10.625, 11.0625, 9.875, 2.375, 2.875, 2.125, 0.125, 0.0, 0.875, 0.75, 2.0, 0.625, 3.125, 0.875, 8.375, 3.875, 0.125, 1.25, 4.125, 2.875, 0.375, 0.125, 0.5, 0.125, 0.875, 1.375, 2.5, 2.25, 0.75, 4.0, 1.75, 2.0, 0.875, 0.625, 1.125, 3.375, 4.25, 3.75, 0.875, 0.125, 0.5, 0.875, 6.25, 1.5, 3.125, 10.25, 2.625, 2.25, 3.25, 3.625, 3.625, 6.75, 10.375, 7.625, 12.5, 11.75, 11.25, 11.875, 13.4375, 1.5, 0.0, 0.25], \"error\": null, \"prompt_sha\": \"fc63e297d382cd8f\", \"prompt_len\": 8192, \"finish_reason\": \"length\"}, {\"text\": \"We need solve ARC tasks. Need find transformation rule for each task and solve test inputs. We have multiple tasks with examples and test inputs. Need output transformed grids likely. Need infer rules. This is complex. Need analyze each task carefully. Need produce final answer with outputs for each test input. Need be accurate.\\n\\nWe have tasks:\\n2c181942 (one test input)\\n2d0172a1 (two test inputs? It lists Test input twice)\\n31f7f899 (one test)\\n332f06d7 (one test)\", \"tokens\": [1596, 1144, 11290, 67964, 8806, 13, 14235, 1423, 17454, 5728, 364, 1754, 3274, 321, 11290, 1228, 10804, 13, 1165, 599, 5081, 8806, 440, 9989, 321, 1228, 10804, 13, 14235, 2468, 22773, 54455, 4222, 13, 14235, 22846, 5426, 13, 1061, 369, 6150, 13, 14235, 22903, 1754, 3274, 15060, 13, 14235, 7936, 1534, 4087, 440, 15796, 364, 1754, 1228, 1879, 13, 14235, 381, 12995, 13, 271, 1596, 599, 8806, 25, 198, 17, 66, 16, 23, 16, 24, 19, 17, 318, 588, 1228, 1879, 8, 198, 17, 67, 15, 16, 22, 17, 64, 16, 318, 19186, 1228, 10804, 30, 1049, 11140, 3284, 1879, 10598, 8, 198, 18, 16, 69, 22, 69, 23, 24, 24, 318, 588, 1228, 8, 198, 18, 18, 17, 69, 15, 21, 67, 22, 318, 588, 1228, 8], \"margins\": [10.0, 11.25, 1.0, 0.625, 3.375, 2.5, 2.875, 0.375, 5.125, 0.875, 1.75, 5.875, 0.375, 2.875, 3.625, 4.125, 6.375, 7.5, 0.25, 0.0, 0.25, 6.25, 0.25, 3.75, 2.0, 2.625, 3.125, 2.875, 4.125, 2.75, 0.5, 1.375, 0.625, 0.5, 1.875, 0.125, 1.375, 0.25, 0.0, 9.625, 0.125, 1.375, 1.0, 0.625, 3.125, 4.125, 0.25, 1.875, 0.125, 0.875, 3.875, 1.5, 2.75, 0.0, 2.5, 0.875, 3.0, 4.75, 1.25, 1.0, 0.125, 1.75, 10.625, 1.875, 2.625, 0.125, 3.0, 3.5, 1.125, 0.25, 10.5, 13.375, 13.3125, 12.0625, 10.375, 9.625, 11.0, 0.125, 1.625, 7.5, 1.0, 3.5, 7.75, 7.0, 10.375, 13.0625, 13.1875, 12.0, 11.1875, 11.875, 11.1875, 7.125, 5.75, 5.25, 8.5, 0.625, 0.75, 0.75, 1.375, 7.375, 0.75, 0.0, 4.375, 8.9375, 9.875, 12.8125, 11.8125, 10.6875, 10.1875, 10.5625, 12.125, 4.625, 5.25, 1.125, 0.0, 10.875, 11.6875, 6.875, 13.0, 11.125, 12.0, 10.0625, 11.375, 10.5, 9.25, 9.0, 2.625, 5.25], \"error\": null, \"prompt_sha\": \"49b3958bb234d4fd\", \"prompt_len\": 40000, \"finish_reason\": \"length\"}]}, \"prompt_shas\": {\"8750743e9dcadca9\": 2048, \"cfc35f3fcb1e4c3c\": 40000, \"8df9ed2c640ef5bd\": 2048, \"fc63e297d382cd8f\": 8192, \"49b3958bb234d4fd\": 40000, \"23e0dd6cf518ffd6\": 40000, \"860a315bad6a77cd\": 40000, \"d9b32f9b8dccc17e\": 40000, \"cde642a1f2a6c19b\": 40000, \"f4212f34e7a2cc3a\": 8192, \"452f4eeeffdffcba\": 8192, \"cabba475611f1ca9\": 8192, \"b3074e57ef93918d\": 8192, \"e5f0513c6123c379\": 8192, \"5c55ca1649204598\": 8192, \"2395ccdb274f065a\": 8192, \"29ccfbd3ae41464d\": 90000, \"6d6f401b8828e42d\": 90000, \"0fc6beedaad76444\": 90000, \"44546bcef138a34a\": 90000, \"057ef8fde4806648\": 116000, \"6e42532a0cbaf769\": 116000, \"eced77cf1dce0ac9\": 116000, \"f581c9630300ae10\": 116000, \"d445d5b5d3d7c21c\": 4894}}")      # NB0's build-time RC2 facts: argv tails / env / pins / oracle rows / prompt shas
    # the globals the picked NB0 / NB1 functions read (NB0 CELL 6 values; NB1 CELL 6 values)
    NB0 = {"topk": int(os.environ.get("NB0_TF_TOPK") or 5), "sc_tok": 256, "tf_variant": 5, "sc_variant": {"2k": 6, "8k": 7, "40k": 6},
           "tf_bands": ["2k", "8k", "40k"], "sc_bands": [], "last": 256, "boot_watchdog_sec": 1800.0, "ready_sec": 1500.0,
           "host_floor_mib": (69 + 4) * 1024}
    SCREEN = {"rank1_min": 0.90, "mean_max": 0.15}       # NB0 (external audit 2): conservative screening, not a published limit
    TRIAGE = {"mean_max": 1.0, "rank1_min": 0.5}         # NB0 (external audit 3): severe-discrepancy triage
    RAW_DIR = OUT / "tf_raw"
    RAW_DIR.mkdir(exist_ok=True)
    NB0_REQ = TRACES / "nb4_requests.jsonl"        # every non-streaming request (teacher-forced) with its meta + generated text
    TFR, SCR, LOADMON, SIDS = {}, {}, {}, set()
    PROC = Path(os.environ.get("NB0_PROC", "/proc"))
    CG = Path(os.environ.get("NB0_CGROUP", "/sys/fs/cgroup"))
    HOSTMEM = OUT / "hostmem"
    HOSTMEM.mkdir(exist_ok=True)
    BLOCK, PP, GEN_RATE = 4096, 2, 45.0           # NB1: block size, pipeline stages (one engine step = one micro-batch), window growth
    RAM_NEED_MIB = (69 + 4) * 1024                 # Merlin's ~69 GiB resident floor + 4 GiB margin (NB1 host_ram_gate)
    EXT_FLAG = ["--worker-extension-cls", "nb1_worker_ext.NB1WorkerExt"]
    DIAG_ENV = {"VLLM_SERVER_DEV_MODE": "1"}      # /collective_rpc (read-only extension) + /reset_prefix_cache; 127.0.0.1 only


    def flags(argv):
        """--flag -> value, the LAST occurrence wins (argparse); a flag followed by another flag is a switch (True)."""
        out = {}
        for i, a in enumerate(argv):
            if isinstance(a, str) and a.startswith("--"):
                nxt = argv[i + 1] if i + 1 < len(argv) else None
                out[a] = nxt if (nxt is not None and not str(nxt).startswith("--")) else True
        return out


    ENV_KEYS = ("VLLM_", "NCCL_", "KV_", "P2P", "SCALES", "NB0_")


    def _set_value(a, flag, value):
        idx = [i for i, x in enumerate(a) if x == flag]
        if len(idx) != 1:
            raise ValueError(f"{flag} occurs {len(idx)} times in the Merlin argv (expected 1)")
        a[idx[0] + 1] = value


    def nb0_argv(argv, boot):
        """Merlin's argv with this boot's edits. MTP off = the same command without --speculative-config (Merlin
        models/flash-next/CHANGELOG.md: "the no-MTP line is the same command without --speculative-config"; launch.py:441/452
        always emits it, R.MTP_K = 3 has no env knob). seqs / captures: the value of the existing flag (no new flags)."""
        a = [str(x) for x in argv]
        if not boot["mtp"]:
            idx = [i for i, x in enumerate(a) if x == "--speculative-config"]
            if len(idx) != 1:
                raise ValueError(f"--speculative-config occurs {len(idx)} times (expected 1)")
            del a[idx[0]:idx[0] + 2]
        if boot.get("seqs"):
            _set_value(a, "--max-num-seqs", str(boot["seqs"]))
        if boot.get("captures"):
            _set_value(a, "--compilation-config", boot["captures"])
        if "--enable-prefix-caching" not in a:
            raise ValueError("prefix caching must stay on in every NB0 boot")
        return a


    def config_diff(argv, env, kv_bytes, ref="fp8"):
        """Flag / env / KV-pin diff of a boot vs RC2's recorded plan (last occurrence wins on both sides)."""
        rf, cf = flags(RC2_REF[ref]["argv_tail"]), flags(argv)      # RC2's tail starts at --host's value (127.0.0.1)
        head = set(flags(argv[:-len(RC2_REF[ref]["argv_tail"]) + 2])) - set(rf)   # Merlin flags before RC2's recorded window
        fl = {k: [rf.get(k), cf.get(k)] for k in sorted(set(rf) | set(cf))
              if k not in ("--port", "--host") and k not in head and rf.get(k) != cf.get(k)}
        renv = RC2_REF[ref]["env"]
        cenv = {k: v for k, v in env.items() if k.startswith(ENV_KEYS)}
        skip = ("VLLM_PORT", "MASTER_PORT", "NCCL_VERSION")
        ev = {k: [renv.get(k), cenv.get(k)] for k in sorted(set(renv) | set(cenv)) if k not in skip and renv.get(k) != cenv.get(k)}
        out = {"flags": fl, "env": ev}
        if kv_bytes != RC2_REF[ref]["kv_bytes"]:
            out["kv_bytes"] = [RC2_REF[ref]["kv_bytes"], kv_bytes]
        return out


    def cg_snapshot(full=True):
        """cgroup v2 memory files (MiB for sizes). avail_mib = max - (current - inactive_file), Merlin's own rule."""
        o, kv = {}, {}
        for n in ("memory.current", "memory.max", "memory.peak"):
            try:
                o[n] = (CG / n).read_text().strip()
            except Exception:
                pass
        try:
            st = (CG / "memory.stat").read_text()
            kv = dict(l.split() for l in st.splitlines() if len(l.split()) == 2)
            o["stat_mib"] = {k: int(kv[k]) >> 20 for k in ("anon", "file", "active_file", "inactive_file", "shmem", "file_mapped",
                                                           "file_dirty", "file_writeback") if k in kv}
            o["stat_count"] = {k: int(kv[k]) for k in ("pgmajfault", "workingset_refault_file", "pgscan", "pgsteal") if k in kv}
            if full:
                o["memory.stat"] = st[:6000]
        except Exception:
            pass
        for n in ("memory.events", "memory.pressure"):
            try:
                o[n] = (CG / n).read_text().strip()[:800]
            except Exception:
                pass
        try:
            mx = o.get("memory.max")
            if mx and mx != "max":
                o["avail_mib"] = (int(mx) - (int(o["memory.current"]) - int(kv.get("inactive_file", 0)))) >> 20
        except Exception:
            pass
        return o


    def _pids():
        try:
            return [int(p.name) for p in PROC.iterdir() if p.name.isdigit()]
        except Exception:
            return None


    def _stat(pid):
        try:
            t = (PROC / str(pid) / "stat").read_text()
        except Exception:
            return None
        i = t.rfind(")")
        f = t[i + 2:].split()
        try:
            return {"pid": pid, "comm": t[t.find("(") + 1:i], "state": f[0], "ppid": int(f[1]), "session": int(f[3]),
                    "minflt": int(f[7]), "majflt": int(f[9]), "utime": int(f[11]), "stime": int(f[12]), "rss_pages": int(f[21])}
        except Exception:
            return None


    def _io(pid):
        try:
            return {k: int(v) for k, v in (l.split(": ") for l in (PROC / str(pid) / "io").read_text().splitlines())
                    if k in ("rchar", "read_bytes", "syscr")}
        except Exception:
            return {}


    def session_procs(sids):
        out = []
        for pid in _pids() or []:
            st = _stat(pid)
            if st and st["session"] in sids and st["state"] != "Z":
                st.update(_io(pid))
                out.append(st)
        return out


    def _roles(text):
        return {int(p): n for n, p in re.findall(r"\((\w+) pid=(\d+)\)", text or "")}


    def load_monitor(s, key, deadline):
        """Every 5 s from launch to ready: cgroup current / avail / stat / pressure and every process of the server session
        (majflt, minflt, rchar, read_bytes, rss, cpu) -> logs/load_monitor_<boot>.jsonl + a per-process summary."""
        path = LOGS / f"load_monitor_{key}.jsonl"
        while s.proc is None and time.time() < deadline:
            time.sleep(0.5)
        if s.proc is None:
            return
        sid = s.proc.pid
        SIDS.add(sid)
        tck = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        t0, first, last, n = time.time(), {}, {}, 0
        peak = {"cg_current_mib_max": None, "avail_mib_min": None, "pressure_full_avg10_max": None}
        while s.alive() and s.ready_sec is None and time.time() < deadline:
            cg = cg_snapshot(False)
            ps = session_procs({sid})
            try:
                with open(path, "a") as fh:
                    fh.write(json.dumps({"t": round(time.time() - t0, 1), "current": cg.get("memory.current"),
                                         "avail_mib": cg.get("avail_mib"), "stat_mib": cg.get("stat_mib"),
                                         "stat_count": cg.get("stat_count"), "pressure": cg.get("memory.pressure"),
                                         "procs": ps}) + "\n")
            except Exception:
                pass
            for p in ps:
                first.setdefault(p["pid"], p)
                last[p["pid"]] = p
            try:
                c = int(cg["memory.current"]) >> 20
                peak["cg_current_mib_max"] = max(peak["cg_current_mib_max"] or 0, c)
            except Exception:
                pass
            if cg.get("avail_mib") is not None:
                peak["avail_mib_min"] = min(x for x in (peak["avail_mib_min"], cg["avail_mib"]) if x is not None)
            m = re.search(r"full avg10=([\d.]+)", cg.get("memory.pressure") or "")
            if m:
                peak["pressure_full_avg10_max"] = max(peak["pressure_full_avg10_max"] or 0.0, float(m.group(1)))
            n += 1
            time.sleep(0.5 if FAKE else 5)
        roles = _roles(s.log())
        per = {}
        for pid, a in first.items():
            z = last[pid]
            per[str(pid)] = {"role": roles.get(pid, a["comm"]), "majflt": z["majflt"] - a["majflt"],
                             "minflt": z["minflt"] - a["minflt"],
                             "read_gib": round((z.get("read_bytes", 0) - a.get("read_bytes", 0)) / 2 ** 30, 2),
                             "rchar_gib": round((z.get("rchar", 0) - a.get("rchar", 0)) / 2 ** 30, 2),
                             "rss_gib_end": round(z["rss_pages"] * 4096 / 2 ** 30, 2),
                             "cpu_s": round((z["utime"] + z["stime"] - a["utime"] - a["stime"]) / tck, 1)}
        loads = re.findall(r"\((\w+) pid=(\d+)\)[^\n]*Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds", s.log() or "")
        LOADMON[key] = {"samples": n, "ready_sec": s.ready_sec, "alive": s.alive(), "per_process": per, "peak": peak,
                        "model_loading": [[w, int(p), float(g), float(sec)] for w, p, g, sec in loads], "file": path.name}


    def _post(port, body, timeout=1800):
        t = time.perf_counter()
        try:
            r = json.loads(UR.urlopen(UR.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                                 headers={"Content-Type": "application/json"}), timeout=timeout).read())
            return r, None, round(time.perf_counter() - t, 2)
        except UE.HTTPError as e:
            try:
                msg = e.read().decode("utf-8", "replace")[:400]
            except Exception:
                msg = ""
            return None, f"HTTP {e.code}: {msg}", round(time.perf_counter() - t, 2)
        except Exception as e:
            return None, repr(e)[:300], round(time.perf_counter() - t, 2)


    def _req_log(rec):
        try:
            with _SAVE_LOCK, open(NB0_REQ, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            pass


    def ids_sha(ids):
        return hashlib.sha256(json.dumps([int(x) for x in ids]).encode()).hexdigest()


    def _cached(r):
        return (((r or {}).get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")


    def pl_arrays(pl, ids, k):
        """Every prompt position p >= 1 of an OpenAI prompt_logprobs list: the actual token's logprob and rank, the unique
        rank-1 id (E23's rule: an ambiguous rank 1 is -1), the best logprob, the top-k ids / logprobs by rank. NaN is kept
        (a non-finite value is never data); a missing position is valid=False."""
        n = len(ids)
        a = {"ids": np.asarray(ids, np.int32), "lp": np.full(n, np.nan, np.float32), "rank": np.full(n, -1, np.int32),
             "top1": np.full(n, -1, np.int32), "top1_lp": np.full(n, np.nan, np.float32),
             "topk_ids": np.full((n, k), -1, np.int32), "topk_lp": np.full((n, k), np.nan, np.float32), "valid": np.zeros(n, bool)}
        amb = 0
        for pos in range(1, n):
            d = pl[pos] if pos < len(pl) else None
            if not isinstance(d, dict):
                continue
            e = d.get(str(ids[pos])) or d.get(f"token_id:{ids[pos]}")
            if not isinstance(e, dict):
                continue
            a["valid"][pos] = True
            v = e.get("logprob")
            a["lp"][pos] = float(v) if isinstance(v, (int, float)) else np.nan
            a["rank"][pos] = int(e.get("rank") or -1)
            ent = []
            for kk, vv in d.items():
                if isinstance(vv, dict) and isinstance(vv.get("logprob"), (int, float)):
                    t = L._tok_id(kk)
                    if isinstance(t, int):
                        ent.append((vv.get("rank") or 10 ** 9, -vv["logprob"], t, vv["logprob"]))
            ent.sort()
            r1 = [x for x in ent if x[0] == 1]
            if len(r1) == 1:
                a["top1"][pos] = r1[0][2]
            else:
                amb += 1
            if ent:
                a["top1_lp"][pos] = max(x[3] for x in ent)
            for j, x in enumerate([x for x in ent if x[0] <= k][:k]):
                a["topk_ids"][pos, j], a["topk_lp"][pos, j] = x[2], x[3]
        a["top1_ambiguous"] = amb
        return a


    def e23_rows(a, last=256):
        """The same per-position rows E23's teacher_forced() keeps ([logprob, rank, rank-1 id]) for the last `last` positions,
        so RC2's compare_tf (with its non-finite guard) computes the RC2-comparable metric."""
        n, out = len(a["ids"]), []
        for pos in range(max(1, n - last), n):
            if not a["valid"][pos]:
                out.append(None)
                continue
            out.append([float(a["lp"][pos]), int(a["rank"][pos]), int(a["top1"][pos]) if a["top1"][pos] >= 0 else None])
        return {"positions": out, "n": len(out)}


    def tf_request(key, port, ids, band, run, k=None):
        """One teacher-forced request (max_tokens 1, T 0, prompt_logprobs k, ids as ids). k > 1 refused by Merlin's prompt-
        scoring budget (prompt_logprobs.py validate) -> retried with k 1 = RC2's value, recorded. Raw arrays -> tf_raw/*.npz."""
        k = k or NB0["topk"]
        body = {"model": "q27", "prompt": [int(x) for x in ids], "max_tokens": 1, "temperature": 0, "prompt_logprobs": k,
                "return_tokens_as_token_ids": True}
        r, err, sec = _post(port, body)
        note = None
        if err and k > 1 and "prompt-scoring tensor bound" in err:
            note = f"prompt_logprobs {k} refused ({err[:160]}); retried with 1 (RC2's value)"
            k = body["prompt_logprobs"] = 1
            r, err, sec = _post(port, body)
        meta = {"boot": key, "band": band, "run": run, "k": k, "note": note, "sec": sec, "error": err, "prompt_len": len(ids),
                "prompt_sha256": ids_sha(ids), "cached_tokens": _cached(r), "t": round(time.time() - T_START, 1)}
        arr = None
        if r is not None:
            try:
                ch = r["choices"][0]
                pl = ch.get("prompt_logprobs") or []
                meta.update(prompt_logprobs_len=len(pl), gen_token_ids=ch.get("token_ids"), gen_text=ch.get("text"))
                arr = pl_arrays(pl, ids, k)
                vv = arr["valid"]
                meta.update(valid=int(vv.sum()), expected=len(ids) - 1, nonfinite=int((~np.isfinite(arr["lp"][vv])).sum()),
                            top1_ambiguous=arr["top1_ambiguous"])
                f = RAW_DIR / f"{key}_{band}_{run}.npz"
                np.savez_compressed(f, **{x: arr[x] for x in ("ids", "lp", "rank", "top1", "top1_lp", "topk_ids", "topk_lp", "valid")})
                meta["raw"] = f.name
            except Exception as e:
                meta["error"] = meta["error"] or f"PARSE {type(e).__name__}: {e}"[:300]
                arr = None
        _req_log({**meta, "kind": "teacher_forced", "body": {kk: v for kk, v in body.items() if kk != "prompt"}})
        return {"meta": meta, "arr": arr}


    def cmp_last256(ra, rb):
        if not ra or not rb or ra.get("arr") is None or rb.get("arr") is None:
            return {"status": "INCONCLUSIVE_MISSING_TF"}
        if ra["meta"]["prompt_sha256"] != rb["meta"]["prompt_sha256"]:
            return {"status": "INCONCLUSIVE_PROMPT_MISMATCH"}
        return L.compare_tf(e23_rows(ra["arr"], NB0["last"]), e23_rows(rb["arr"], NB0["last"]))


    def chunk_check(pos, d, edge=8):
        """|dlogprob| at the first / last `edge` positions of each 1024-token scheduler chunk (--max-num-batched-tokens) and
        each 256-row prompt-logprob chunk (VLLM_MERLIN_PROMPT_LOGPROBS_CHUNK_SIZE) vs mid-chunk."""
        out = {}
        for unit in (1024, 256):
            off = pos % unit
            e = (off < edge) | (off >= unit - edge)
            if e.any() and (~e).any():
                me, mm = float(d[e].mean()), float(d[~e].mean())
                out[str(unit)] = {"edge_mean": round(me, 4), "mid_mean": round(mm, 4), "edge_n": int(e.sum()),
                                  "localised": bool(me > 3 * mm + 0.05)}
        return out


    def cmp_full(a, b, bins=True):
        """Every prompt position: mean / p99 / max |dlogprob|, rank-1 agreement, 4096-token bins (does it grow with context?),
        the first 256-position window over the screening mean, chunk-edge localisation."""
        if a["ids"].shape != b["ids"].shape or not np.array_equal(a["ids"], b["ids"]):
            return {"status": "INCONCLUSIVE_PROMPT_MISMATCH"}
        va, vb = a["valid"] & np.isfinite(a["lp"]), b["valid"] & np.isfinite(b["lp"])
        if (a["valid"] & ~np.isfinite(a["lp"])).any() or (b["valid"] & ~np.isfinite(b["lp"])).any():
            return {"status": "INCONCLUSIVE_NONFINITE"}
        pos = np.nonzero(va & vb)[0]
        n = len(a["ids"])
        if not len(pos):
            return {"status": "INCONCLUSIVE_NO_VALID"}
        d = np.abs(a["lp"][pos].astype(np.float64) - b["lp"][pos].astype(np.float64))
        t1 = (a["top1"][pos] == b["top1"][pos]) & (a["top1"][pos] >= 0)
        ds = np.sort(d)
        out = {"status": "MEASURED" if len(pos) == n - 1 else "PARTIAL", "n": int(len(pos)), "expected": n - 1,
               "mean_abs_dlogprob": round(float(d.mean()), 4), "p99_abs_dlogprob": round(float(ds[min(len(ds) - 1, int(round(0.99 * (len(ds) - 1))))]), 3),
               "max_abs_dlogprob": round(float(d.max()), 3), "rank1_agreement": round(float(t1.mean()), 4)}
        if bins:
            bl = []
            for lo in range(0, n, 4096):
                m = (pos >= lo) & (pos < lo + 4096)
                if m.any():
                    bl.append([lo, int(m.sum()), round(float(d[m].mean()), 4), round(float(t1[m].mean()), 4)])
            out["bins_4096"] = bl                        # [start, n, mean |dlogprob|, rank-1 agreement]
            if len(d) >= 256:
                cs = np.convolve(d, np.ones(256) / 256, "valid")
                idx = np.nonzero(cs > SCREEN["mean_max"])[0]
                out["first_window_over_screen_mean"] = int(pos[idx[0]]) if len(idx) else None
            out["chunk"] = chunk_check(pos, d)
        return out


    def shift_check(a, b, window=2048, shifts=(-2, -1, 0, 1, 2)):
        """Alignment test on the last `window` positions: reference rank-1 id at p vs candidate rank-1 id at p + s. A shifted
        agreement >= 0.9 that beats s = 0 by > 0.2 = a position-alignment artefact."""
        n = min(len(a["top1"]), len(b["top1"]))
        i0 = np.arange(max(1, n - window), n)
        res = {}
        for s in shifts:
            j = i0 + s
            m = (j >= 1) & (j < n)
            i, j = i0[m], j[m]
            v = a["valid"][i] & b["valid"][j] & (a["top1"][i] >= 0) & (b["top1"][j] >= 0)
            if not v.any():
                res[str(s)] = None
                continue
            dl = np.abs(a["top1_lp"][i][v].astype(np.float64) - b["top1_lp"][j][v].astype(np.float64))
            res[str(s)] = {"n": int(v.sum()), "top1_agree": round(float((a["top1"][i][v] == b["top1"][j][v]).mean()), 4),
                           "top1_lp_mean_abs_d": round(float(np.nanmean(dl)), 4) if np.isfinite(dl).any() else None}
        ok = {k: x for k, x in res.items() if x}
        best = max(ok, key=lambda k: (ok[k]["top1_agree"], k == "0")) if ok else None
        a0 = (ok.get("0") or {}).get("top1_agree")
        art = bool(best and best != "0" and ok[best]["top1_agree"] >= 0.9 and (a0 is None or ok[best]["top1_agree"] > a0 + 0.2))
        return {"shifts": res, "best_shift": int(best) if best is not None else None, "alignment_artefact": art}


    def screen(c):
        if not isinstance(c, dict) or c.get("status") != "MEASURED":
            return "INCONCLUSIVE"
        m, r1 = c.get("mean_abs_dlogprob"), c.get("rank1_agreement")
        if not (_finite(m) and _finite(r1)):
            return "INCONCLUSIVE_NONFINITE"
        return "DIVERGENT" if (r1 < SCREEN["rank1_min"] or m > SCREEN["mean_max"]) else "CLEAN"


    TF_EST = {"2k": 15, "8k": 25, "40k": 60, "90k": 150}


    def tf_suite(key, s, bands, runs=("fresh", "repeat")):    # E25 NB4 (klee100 R2-4)
        out = {}
        for band in bands:
            ids = PROMPTS[band][NB0["tf_variant"]]
            rec = TFR.setdefault(key, {}).setdefault(band, {})
            o = {"prompt_len": len(ids), "prompt_sha256": ids_sha(ids), "variant": NB0["tf_variant"]}
            for run in runs:                            # E25 NB4 (klee100 R2-4): non-base arms run the fresh requests only
                if not s.alive():
                    o[run] = {"status": "SKIPPED_SERVER_DEAD"}
                elif ABORT["why"] or left() < TF_EST[band] + 240:        # E25 NB4 (klee100 R1-15): WD (never set) removed
                    o[run] = {"status": "SKIPPED_BUDGET"}
                elif not due(TF_EST[band] + 300):          # E25 NB4 (klee100 K-1): the boot body cap, not only the run budget
                    o[run] = {"status": "SKIPPED_BOOT_BUDGET"}
                else:
                    rec[run] = tf_request(key, s.port, ids, band, run)
                    o[run] = rec[run]["meta"]
            o["aa"] = cmp_last256(rec.get("fresh"), rec.get("repeat"))
            out[band] = o
            save()
        return out


    def scales_facts(t):
        """The static scales sidecar's values, the per-stage 'applied ... to N of 12' lines, every 'E4M3 KV active on <layer>'
        line matched to the file (logged %.6g), and the MTP layer's DEFAULT 1.0 line."""
        path = CKPT / "qsa_kv_scales_262k.json"
        try:
            raw = path.read_bytes()
            file, sha = json.loads(raw), hashlib.sha256(raw).hexdigest()
        except Exception as e:
            file, sha = {}, f"unreadable: {e!r}"[:120]
        applied = re.findall(r"\((\w+) pid=(\d+)\)[^\n]*applied static K/V scales from (.+?) to (\d+) of (\d+) layer\(s\) "
                             r"\(pipeline layers (\d+)-(\d+)\)", t)
        act = {}
        for name, k, v, dflt in re.findall(r"QSA E4M3 KV active on (\S+): k_scale=([-\d.eE+]+) v_scale=([-\d.eE+]+)(\s+\(DEFAULT 1\.0)?", t):
            act.setdefault(name, set()).add((float(k), float(v), bool(dflt)))
        tol = lambda a, b: abs(a - b) <= 1e-5 * max(1e-12, abs(b))
        matched, mismatched = [], []
        for name, sc in sorted(file.items()):
            got = act.get(name)
            if got:
                (matched if all(tol(k, sc["k_scale"]) and tol(v, sc["v_scale"]) and not d for k, v, d in got) else mismatched).append(name)
        stages = sorted({f"{a}-{b}" for *_, a, b in applied})
        return {"file": str(path), "sha256": sha, "values": file, "applied_lines": len(applied), "applied_stages": stages,
                "applied_counts": sorted({f"{n} of {m}" for _, _, _, n, m, _, _ in applied}),
                "applied_paths": sorted({p for _, _, p, *_ in applied}), "active_layers": sorted(act), "matched": len(matched),
                "mismatched": mismatched, "missing": sorted(set(file) - set(act)), "not_in_file": sorted(set(act) - set(file)),
                "mtp_default_scale": sorted(n for n, g in act.items() if any(d for _, _, d in g)),
                "ok": bool(file) and not mismatched and not (set(file) - set(act)) and stages == ["0-25", "25-48"]}


    def psha(ids):
        return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


    P60490 = "# vLLM PR #60490 (issue #60489) rebased onto Merlin v3.0.0 (commit ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5,\n# tree 07c16a79d077f1e06c19d0eea84efd64f0b63540 = the tree the Kaggle payload installs, install_merlin.log).\n# Same code as upstream; only the import context differs (Merlin's import block). One marker comment added.\n# pre-patch sha256  a23d5f90507ad5185f8b8e8d6ba13ae0cb53b2506264fdfbbf6a60b1171dc653\n# post-patch sha256 88fe0a8f4dd2ef90fc6aefb5771b5b48e40c0b1d575039283c52aeca19b8bd6b\n--- a/vllm/models/qwen4_exp/nvidia/qsa.py\n+++ b/vllm/models/qwen4_exp/nvidia/qsa.py\n@@ -18,8 +18,9 @@\n from torch import nn\n from transformers import Qwen4ExpTextConfig\n \n+import vllm.envs as envs\n from vllm.compilation.breakable_cudagraph import eager_break_during_capture\n-from vllm.config import VllmConfig\n+from vllm.config import VllmConfig, get_current_vllm_config_or_none\n from vllm.config.cache import CacheConfig, CacheDType\n from vllm.distributed import get_tensor_model_parallel_world_size\n from vllm.forward_context import get_forward_context\n@@ -45,6 +46,7 @@\n )\n from vllm.platforms import current_platform\n from vllm.platforms.interface import DeviceCapability\n+from vllm.utils.math_utils import cdiv\n from vllm.utils.torch_utils import (\n     LayerNameType,\n     _encode_layer_name,\n@@ -75,6 +77,7 @@\n from . import model\n from .indexer_qsa import QSAIndexer\n from .ops.qsa import qsa_e4m3_uses_native_cast\n+from .ops.qsa_indexer import _TOPK_WORKSPACE_BYTES\n \n logger = init_logger(__name__)\n \n@@ -782,6 +785,31 @@\n         if isinstance(metadata, list):\n             metadata = metadata[0]\n         if not isinstance(metadata, dict):\n+            # E25 NB1: backport of vLLM PR #60490 (issue #60489).\n+            # Profiling/dummy run: the indexer allocates its selection scratch\n+            # on demand, so touch its peak here for the memory profiler (as\n+            # sparse_attn_indexer does): the prefill logits budget or the\n+            # worst-case decode logits [decode tokens, compressed columns],\n+            # plus the top-k workspace.\n+            logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024\n+            cfg = get_current_vllm_config_or_none()\n+            if cfg is not None:\n+                spec = cfg.speculative_config\n+                num_spec = spec.num_speculative_tokens if spec is not None else 0\n+                decode_tokens = min(\n+                    cfg.scheduler_config.max_num_seqs * (num_spec + 1),\n+                    cfg.scheduler_config.max_num_batched_tokens,\n+                )\n+                block_size = cfg.cache_config.block_size\n+                columns = cdiv(cfg.model_config.max_model_len, block_size) * cdiv(\n+                    block_size, self.indexer.compress_ratio\n+                )\n+                logits_bytes = max(logits_bytes, decode_tokens * columns * 4)\n+            _ = torch.empty(\n+                logits_bytes + _TOPK_WORKSPACE_BYTES,\n+                dtype=torch.uint8,\n+                device=output.device,\n+            )\n             output.zero_()\n             return\n         main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])\n"


    P60490_REL = "vllm/models/qwen4_exp/nvidia/qsa.py"


    P60490_PRE, P60490_POST = "a23d5f90507ad5185f8b8e8d6ba13ae0cb53b2506264fdfbbf6a60b1171dc653", "88fe0a8f4dd2ef90fc6aefb5771b5b48e40c0b1d575039283c52aeca19b8bd6b"


    P60490_MARKER = "# E25 NB1: backport of vLLM PR #60490 (issue #60489)."


    NB1_EXT_SRC = "\"\"\"E25 NB1 worker extension (read-only): reached through POST /collective_rpc {\"method\": \"nb1_cache_info\"} with\nVLLM_SERVER_DEV_MODE=1 and --worker-extension-cls nb1_worker_ext.NB1WorkerExt. Reports the ACTUAL cache tensors bound\nto every attention / Mamba layer (dtype, shape, bytes), the KV cache specs (padded vs unpadded Mamba page bytes), the\nphysical block count and this rank's free memory. It never raises: every part returns its error text instead.\"\"\"\nimport json\nimport math\n\n\ndef _t(x):\n    try:\n        import torch\n        if isinstance(x, torch.Tensor):\n            return {\"dtype\": str(x.dtype).replace(\"torch.\", \"\"), \"shape\": list(x.shape), \"bytes\": int(x.numel() * x.element_size())}\n    except Exception:\n        pass\n    return None\n\n\ndef _flat(kc):\n    if _t(kc):\n        return [kc]\n    if isinstance(kc, (list, tuple)):\n        if kc and isinstance(kc[0], (list, tuple)) and not _t(kc[0]):\n            kc = kc[0]                              # one entry per virtual engine\n        return [x for x in kc if _t(x)]\n    return []\n\n\n_RT = {\"on\": False, \"installed\": False, \"rows\": [], \"passes\": 0, \"calls\": 0, \"max_steps\": 200, \"bytes\": 0, \"errors\": [],\n       \"first\": None, \"skipped_capturing\": 0, \"skipped_prefill\": 0}\n_RT_MAX_BYTES = 16 << 20\n_RT_DECODE_MAX_ROWS = 128          # a decode / verify micro-batch: requests x (1 + K) tokens; prefill chunks are larger\n\n\ndef _rt_layer(name):\n    import re as _r\n    m = _r.search(r\"layers\\.(\\d+)\", str(name))\n    return int(m.group(1)) if m else -1\n\n\ndef _rt_wrap(orig):\n    def select_experts(self, hidden_states, router_logits, *a, **k):\n        out = orig(self, hidden_states, router_logits, *a, **k)\n        if _RT[\"on\"] and _RT[\"passes\"] < 2 * _RT[\"max_steps\"] and _RT[\"bytes\"] < _RT_MAX_BYTES:\n            try:\n                import torch\n                if torch.cuda.is_current_stream_capturing():\n                    _RT[\"skipped_capturing\"] += 1\n                else:\n                    ids = out[1]\n                    lay = _rt_layer(getattr(self, \"nb1_layer\", None))\n                    if _RT[\"first\"] is None:\n                        _RT[\"first\"] = lay\n                    n = int(ids.shape[0])\n                    if n > _RT_DECODE_MAX_ROWS:\n                        _RT[\"skipped_prefill\"] += 1\n                    else:\n                        if lay == _RT[\"first\"]:\n                            _RT[\"passes\"] += 1     # one pass of a micro-batch through this stage's first MoE layer\n                        keep = ids.to(torch.int16).cpu().numpy().reshape(n, -1)\n                        _RT[\"rows\"].append((lay, max(0, _RT[\"passes\"] - 1), n, keep))\n                        _RT[\"bytes\"] += keep.nbytes\n                        _RT[\"calls\"] += 1\n            except Exception as e:\n                if len(_RT[\"errors\"]) < 5:\n                    _RT[\"errors\"].append(repr(e)[:160])\n        return out\n    return select_experts\n\n\nclass NB1WorkerExt:\n    def nb1_routing(self, action, arg=\"\"):\n        \"\"\"Router top-k expert ids, ONLY between 'start' and 'stop' (the profiling window): FusedMoERouter.select_experts is\n        wrapped once at class level; every decode / verify-sized call (<= 128 token rows) that runs in Python (not during\n        a CUDA-graph capture) keeps its ids (int16) with the layer index and the pass index (counted at this stage's first\n        MoE layer; NB1 maps pass -> step = pass // 2, mb_index = pass % 2 under PP2); <= max_steps steps, <= 16 MiB per\n        rank. FULL-graph replays run no Python, so graph-replayed decode steps are NOT recorded (vLLM's\n        --enable-return-routed-experts writes the ids from inside the graph; it changes the forward graph and memory, so it\n        is not used on a measured boot). 'stop' writes the rank's part file <arg>_rank<r>.npz.\"\"\"\n        out = {\"action\": action}\n        try:\n            import torch.distributed as dist\n            rank = dist.get_rank() if dist.is_initialized() else 0\n            out[\"rank\"] = rank\n            if action == \"start\":\n                from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter\n                if not _RT[\"installed\"]:\n                    FusedMoERouter.select_experts = _rt_wrap(FusedMoERouter.select_experts)\n                    _RT[\"installed\"] = True\n                    model = getattr(getattr(self, \"model_runner\", None), \"model\", None)\n                    named = 0\n                    if model is not None:\n                        for name, mod in model.named_modules():\n                            for attr in (\"router\", \"_router\"):\n                                r = getattr(mod, attr, None)\n                                if isinstance(r, FusedMoERouter):\n                                    r.nb1_layer = name\n                                    named += 1\n                    out[\"routers_named\"] = named\n                _RT.update(on=True, rows=[], passes=0, calls=0, bytes=0, errors=[], first=None, skipped_capturing=0,\n                           skipped_prefill=0, max_steps=int(arg or 200))\n                return out\n            _RT[\"on\"] = False\n            out.update(passes=_RT[\"passes\"], calls=_RT[\"calls\"], bytes=_RT[\"bytes\"], errors=_RT[\"errors\"],\n                       skipped_capturing=_RT[\"skipped_capturing\"], skipped_prefill=_RT[\"skipped_prefill\"])\n            if action == \"stop\" and arg and _RT[\"rows\"]:\n                import numpy as np\n                rows = _RT[\"rows\"]\n                offs, o_ = [], 0\n                for r in rows:\n                    offs.append(o_)\n                    o_ += r[2]\n                path = f\"{arg}_rank{rank}.npz\"\n                np.savez_compressed(path, ids=np.concatenate([r[3] for r in rows], axis=0), layer=np.array([r[0] for r in rows]),\n                                    pass_index=np.array([r[1] for r in rows]), mb_size=np.array([r[2] for r in rows]),\n                                    offset=np.array(offs), rank=np.array([rank]))\n                out[\"path\"] = path\n            _RT[\"rows\"] = []\n        except Exception as e:\n            out[\"error\"] = repr(e)[:300]\n        return out\n\n    def nb1_cache_info(self):\n        out = {}\n        try:\n            import torch.distributed as dist\n            out[\"rank\"] = dist.get_rank() if dist.is_initialized() else None\n        except Exception as e:\n            out[\"rank_error\"] = repr(e)[:120]\n        try:\n            ctx = self.vllm_config.compilation_config.static_forward_context\n            sig = {}\n            for name, mod in ctx.items():\n                ts = [_t(x) for x in _flat(getattr(mod, \"kv_cache\", None))]\n                if not ts:\n                    continue\n                key = type(mod).__name__ + \"|\" + json.dumps([(x[\"dtype\"], x[\"shape\"]) for x in ts])\n                g = sig.setdefault(key, {\"module\": type(mod).__name__, \"tensors\": ts, \"layers\": 0, \"example\": name})\n                g[\"layers\"] += 1\n            out[\"cache_tensors\"] = list(sig.values())[:40]\n        except Exception as e:\n            out[\"cache_tensors_error\"] = repr(e)[:300]\n        try:\n            agg = {}\n            for name, sp in self.get_kv_cache_spec().items():\n                d = {\"type\": type(sp).__name__}\n                for k in (\"block_size\", \"page_size_bytes\", \"page_size_padded\", \"dtype\", \"num_kv_heads\", \"head_size\", \"mamba_type\"):\n                    try:\n                        v = getattr(sp, k, None)\n                        d[k] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)\n                    except Exception as e:\n                        d[k] = \"ERR \" + repr(e)[:60]\n                shapes, dtypes = getattr(sp, \"shapes\", None), getattr(sp, \"dtypes\", None)\n                if shapes and dtypes:\n                    try:\n                        d[\"state_dtypes\"] = [str(x).replace(\"torch.\", \"\") for x in dtypes]\n                        d[\"state_shapes\"] = [list(s) for s in shapes]\n                        d[\"unpadded_page_bytes\"] = int(sum(math.prod(s) * x.itemsize for s, x in zip(shapes, dtypes)))\n                    except Exception as e:\n                        d[\"unpadded_error\"] = repr(e)[:80]\n                key = json.dumps(d, sort_keys=True, default=str)\n                a = agg.setdefault(key, dict(d, layers=0, example=name))\n                a[\"layers\"] += 1\n            out[\"kv_cache_specs\"] = list(agg.values())[:40]\n        except Exception as e:\n            out[\"kv_cache_specs_error\"] = repr(e)[:300]\n        try:\n            out[\"num_gpu_blocks\"] = getattr(self.cache_config, \"num_gpu_blocks\", None)\n            out[\"mamba_page_size_padded\"] = getattr(self.cache_config, \"mamba_page_size_padded\", None)\n            out[\"mamba_ssm_cache_dtype\"] = str(getattr(self.cache_config, \"mamba_ssm_cache_dtype\", None))\n            out[\"cache_block_size\"] = getattr(self.cache_config, \"block_size\", None)\n        except Exception as e:\n            out[\"cache_config_error\"] = repr(e)[:200]\n        try:\n            for obj in (getattr(self, \"model_runner\", None), self):\n                cfg = getattr(obj, \"kv_cache_config\", None)\n                if cfg is not None:\n                    out[\"kv_cache_config\"] = {\n                        \"num_blocks\": getattr(cfg, \"num_blocks\", None),\n                        \"groups\": [{\"spec\": type(g.kv_cache_spec).__name__, \"layers\": len(g.layer_names),\n                                    \"block_size\": getattr(g.kv_cache_spec, \"block_size\", None),\n                                    \"page_size_bytes\": getattr(g.kv_cache_spec, \"page_size_bytes\", None)}\n                                   for g in getattr(cfg, \"kv_cache_groups\", [])][:16],\n                        \"tensors\": [{\"size\": getattr(t, \"size\", None), \"shared_by\": len(getattr(t, \"shared_by\", None) or [])}\n                                    for t in getattr(cfg, \"kv_cache_tensors\", [])][:24]}\n                    break\n        except Exception as e:\n            out[\"kv_cache_config_error\"] = repr(e)[:300]\n        try:\n            import torch\n            fr, tot = torch.cuda.mem_get_info()\n            out[\"mem\"] = {\"free_mib\": fr >> 20, \"total_mib\": tot >> 20, \"allocated_mib\": torch.cuda.memory_allocated() >> 20,\n                          \"reserved_mib\": torch.cuda.memory_reserved() >> 20, \"max_allocated_mib\": torch.cuda.max_memory_allocated() >> 20}\n        except Exception as e:\n            out[\"mem_error\"] = repr(e)[:200]\n        return out\n\n\n\n# ==================================================================== E25 NB4: read-only additions\ndef nb4_env(self, keys=\"\"):\n    \"\"\"os.environ of this worker for the given comma-separated keys (CUDA reads CUDA_DEVICE_MAX_CONNECTIONS at context\n    creation: arm S's change must be visible in every worker) + this worker's pid.\"\"\"\n    import os as _os\n    out = {\"pid\": _os.getpid()}\n    try:\n        import torch.distributed as dist\n        out[\"rank\"] = dist.get_rank() if dist.is_initialized() else None\n    except Exception as e:\n        out[\"rank_error\"] = repr(e)[:120]\n    ks = [k for k in str(keys).split(\",\") if k] or [k for k in _os.environ if k.startswith((\"NCCL_\", \"CUDA_\", \"NB4_\", \"OMP_\"))]\n    out[\"env\"] = {k: _os.environ.get(k) for k in ks}\n    return out\n\n\ndef nb4_p5_stats(self, read_device=\"1\"):\n    \"\"\"Agent B's P5 engaged counters (nb4_p5_backend.stats_json); call between windows only (a synchronous device read).\"\"\"\n    import json as _json\n    try:\n        import nb4_p5_backend as b\n        return b.stats_json(read_device=str(read_device) == \"1\")\n    except Exception as e:\n        return _json.dumps({\"error\": f\"{type(e).__name__}: {e}\"})\n\n\nNB1WorkerExt.nb4_env = nb4_env\nNB1WorkerExt.nb4_p5_stats = nb4_p5_stats\n"


    def apply_unified(text, patch):
        """Strict unified-diff applier (no fuzz, no offset): every context / removed line must match at its stated line,
        else ValueError. Lines before the first hunk (comments, ---/+++ headers) are ignored."""
        src = text.splitlines(True)
        hunks, cur = [], None
        for ln in patch.splitlines(True):
            if ln.startswith("@@"):
                m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", ln)
                if not m:
                    raise ValueError("bad hunk header: " + ln[:80])
                cur = [int(m.group(1)), int(m.group(2) or 1), []]
                hunks.append(cur)
            elif cur is not None and ln[:1] in (" ", "-", "+"):
                cur[2].append(ln)
            elif cur is not None and ln.startswith("\\"):
                raise ValueError("'No newline at end of file' hunks are not supported")
        if not hunks:
            raise ValueError("patch has no hunks")
        out, pos = [], 0
        for start, n_old, body in hunks:
            i = start - 1
            if i < pos:
                raise ValueError(f"overlapping hunk at line {start}")
            out += src[pos:i]
            old_seen = 0
            for ln in body:
                tag, s = ln[0], ln[1:]
                if tag in " -":
                    if i >= len(src) or src[i] != s:
                        raise ValueError(f"context mismatch at line {i + 1}: {s[:60]!r}")
                    i += 1
                    old_seen += 1
                if tag in " +":
                    out.append(s)
            if old_seen != n_old:
                raise ValueError(f"hunk at line {start}: {old_seen} old lines, header says {n_old}")
            pos = i
        return "".join(out + src[pos:])


    def apply_60490():
        """After merlin_install, before B1: patch the installed tree's qsa.py (pre-sha must be Merlin ad42e37's), verify the
        post-sha, the marker, a py_compile with the Merlin venv and that `vllm` resolves to the tree; write the read-only
        worker extension next to the tree's vllm/ (on the server's PYTHONPATH). Any miss raises: no boot is started."""
        if FAKE:
            tree = os.environ.get("NB1_FAKE_TREE")
            if not tree:
                raise RuntimeError("PATCH_60490_NO_TREE: NB1_FAKE_TREE is not set (local harness)")
            tree, py = Path(tree), sys.executable
        else:
            eng = merlin_install()
            tree, py = Path(eng["kv"]["MERLIN_TREE"]), eng["kv"]["MERLIN_VENV"] + "/bin/python"
        f = tree / P60490_REL
        raw = f.read_bytes()
        h0 = hashlib.sha256(raw).hexdigest()
        rec = {"file": str(f), "sha_before": h0, "pre_expected": P60490_PRE, "post_expected": P60490_POST}
        if h0 == P60490_POST:
            rec["applied"] = "already_applied"
        elif h0 != P60490_PRE:
            raise RuntimeError(f"PATCH_60490_PRE_SHA_MISMATCH: {h0} (expected Merlin ad42e37 {P60490_PRE})")
        else:
            new = apply_unified(raw.decode("utf-8"), P60490).encode("utf-8")
            if hashlib.sha256(new).hexdigest() != P60490_POST:
                raise RuntimeError("PATCH_60490_POST_SHA_MISMATCH: " + hashlib.sha256(new).hexdigest())
            tmp = f.with_name(f.name + ".nb1tmp")
            tmp.write_bytes(new)
            os.replace(tmp, f)
            rec["applied"] = "applied"
        for pyc in (f.parent / "__pycache__").glob("qsa.*.pyc"):
            pyc.unlink()
        data = f.read_bytes()
        rec.update(sha_after=hashlib.sha256(data).hexdigest(), marker=P60490_MARKER.encode() in data)
        ext = tree / "nb1_worker_ext.py"
        ext.write_text(NB1_EXT_SRC)
        rec["worker_ext"] = {"file": str(ext), "sha256": hashlib.sha256(ext.read_bytes()).hexdigest()}
        cp = subprocess.run([py, "-c", "import py_compile, sys\nfor f in sys.argv[1:]: py_compile.compile(f, doraise=True)", str(f), str(ext)],
                            capture_output=True, text=True, timeout=300)
        rec["py_compile"] = {"rc": cp.returncode, "err": cp.stderr[-400:]}
        origin_ok = True
        if not FAKE:
            envp = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            envp["PYTHONPATH"] = str(tree)            # what Merlin's build() exports for the server
            cp2 = subprocess.run([py, "-c", "import importlib.util as u; print(u.find_spec('vllm').origin); print(u.find_spec('nb1_worker_ext').origin)"],
                                 capture_output=True, text=True, timeout=300, env=envp)
            rec["vllm_origin"] = cp2.stdout.strip()[-400:]
            origin_ok = all(l.startswith(str(tree)) for l in cp2.stdout.strip().splitlines()[-2:]) and len(cp2.stdout.strip().splitlines()) >= 2
        if rec["sha_after"] != P60490_POST or not rec["marker"] or cp.returncode or not origin_ok:
            raise RuntimeError("PATCH_60490_VERIFY_FAILED: " + json.dumps({k: rec.get(k) for k in ("sha_after", "marker", "py_compile", "vllm_origin")})[:600])
        rec["ok"] = True
        rec["forward_test"] = "B1's greedy oracle vs RC2's recorded FP8 rows (unpatched RC2 FP8; B1 adds text-only): B1 oracle.vs_rc2_fp8"
        return rec


    def cgroup_snapshot(tag):
        """memory.current / stat / events / pressure (first 12 kB each) to hostmem/<tag>.txt; key numbers returned."""
        rec, txt = {"tag": tag}, []
        for n in ("memory.current", "memory.stat", "memory.events", "memory.pressure"):
            try:
                s = (Path("/sys/fs/cgroup") / n).read_text()[:12000]
            except Exception:
                continue
            txt.append(f"--- {n} ---\n{s}")
            if n == "memory.current":
                rec["current_mib"] = int(s.strip()) >> 20
            elif n == "memory.events":
                rec["events"] = {a: int(b) for a, b in (l.split() for l in s.splitlines() if len(l.split()) == 2)}
            elif n == "memory.pressure":
                rec["pressure"] = s.strip().splitlines()[:2]
        try:
            (HOSTMEM / f"{tag}.txt").write_text("\n".join(txt))
        except Exception:
            pass
        rec["host_mem"] = host_mem()
        return rec


    ENGINE_RX = re.compile(r"vllm|PleOffload|ple_offload|EngineCore|merlin_off/venv", re.I)


    def engine_procs():
        """Processes of an engine (vLLM server, workers, PLE offload) other than this notebook; None off Linux."""
        if not Path("/proc/self/stat").exists():
            return None
        me, out = os.getpid(), []
        for d in Path("/proc").glob("[0-9]*"):
            try:
                pid = int(d.name)
                if pid == me:
                    continue
                cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                if ENGINE_RX.search(cmd) and "py_compile" not in cmd:
                    out.append([pid, cmd[:160]])
            except Exception:
                pass
        return out


    def clear_engine_procs():
        left0 = engine_procs()
        rec = {"found": left0}
        if left0:
            for pid, _ in left0:
                try:
                    os.kill(pid, 9)
                except Exception:
                    pass
            time.sleep(10)
            rec["after_kill"] = engine_procs()
        rec["ok"] = not (rec.get("after_kill") if left0 else left0)
        return rec


    def drop_page_cache():
        """posix_fadvise(DONTNEED) on every checkpoint file (merged dir symlinks resolved + the Model mounts); only after the
        previous server is dead (never on the pinned PLE pages of a live server). A measured attempt, not an assumed fix."""
        if not hasattr(os, "posix_fadvise"):
            return {"available": False}
        t, seen, n, b, err = time.time(), set(), 0, 0, 0
        roots = [CKPT] + [v[0] for v in PARTS.values() if v] + ([PLE_DIR] if PLE_DIR else [])
        for root in roots:
            for dp, _, fns in os.walk(str(root)):
                for x in fns:
                    p = os.path.realpath(os.path.join(dp, x))
                    if p in seen:
                        continue
                    seen.add(p)
                    try:
                        fd = os.open(p, os.O_RDONLY)
                        try:
                            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                            b += os.fstat(fd).st_size
                            n += 1
                        finally:
                            os.close(fd)
                    except OSError:
                        err += 1
        return {"available": True, "files": n, "gib": round(b / 2 ** 30, 1), "errors": err, "sec": round(time.time() - t, 1)}


    def host_ram_gate(tag):
        before = cgroup_snapshot(tag + "_before_drop")
        drops, tries = [drop_page_cache()], 0
        after = host_mem()
        while True:
            av = after.get("avail_mib", after.get("MemAvailable_mib"))
            if av is None:
                ok = bool(FAKE)               # local harness only; on Kaggle an unreadable host is never "enough"
                why = "no cgroup / meminfo reading" + (" (local harness: skipped)" if FAKE else ": HOST_RAM_GATE fails closed")
                break
            if av >= RAM_NEED_MIB:
                ok, why = True, f"avail {av} MiB >= {RAM_NEED_MIB}"
                break
            if FAKE:                          # local / CPU-kit harness only (a 30 GiB CPU box): recorded, never enforced
                ok, why = True, f"avail {av} MiB < {RAM_NEED_MIB} MiB (local harness: recorded, not enforced)"
                break
            tries += 1
            if tries > 3:
                ok, why = False, f"HOST_RAM_GATE: avail {av} MiB < {RAM_NEED_MIB} MiB after {tries} page-cache drops"
                break
            time.sleep(20)
            drops.append(drop_page_cache())
            after = host_mem()
        return {"ok": ok, "why": why, "need_mib": RAM_NEED_MIB, "before": before, "after_drop": cgroup_snapshot(tag + "_after_drop"),
                "drops": drops}


    _boot_gate_rc2, _shm_cleanup_rc2 = boot_gate, shm_cleanup


    BOOT = {"deadline": None, "fired": None, "tag": "boot"}


    def boot_gate():
        """RC2's GPU release gate (< 150 MiB per card), the previous engine's processes gone, then the host-RAM gate."""
        pre = clear_engine_procs()                     # E25 NB4 (klee100 R1-6): before the release wait, not only after it
        g = _boot_gate_rc2()
        g["engine_procs_before_wait"] = pre
        if g.get("ok") is not True:
            return g
        g["engine_procs"] = clear_engine_procs()
        if not g["engine_procs"]["ok"]:
            g["ok"], g["why"] = False, "ENGINE_PROCS_LEFT: " + json.dumps(g["engine_procs"].get("after_kill"))[:300]
            return g
        h = host_ram_gate(BOOT["tag"])
        g["host_ram"] = h
        if not h["ok"]:
            g["ok"], g["why"] = False, h["why"]
        return g


    def shm_cleanup(before):
        """RC2's /dev/shm cleanup (this boot's torch_/psm_/vllm/ple/nccl/sem.mp-/cuda.shm entries) + the loky semaphores the
        RC2 log reported leaked at shutdown (sem.loky-*), only when created during this boot."""
        out = _shm_cleanup_rc2(before)
        out["removed_loky"] = []
        try:
            for p in sorted(SHM.iterdir()):
                if p.name not in before and re.match(r"sem\.loky-", p.name):
                    try:
                        p.unlink()
                        out["removed_loky"].append(p.name)
                    except Exception:
                        pass
        except Exception:
            pass
        return out


    def boot_left():
        d = BOOT["deadline"]
        return min(left(), (d - time.time()) if d else float("inf"))


    def due(sec):
        return boot_left() > sec and not ABORT["why"]


    def _pick(t, rx, n=6):
        return [l.strip()[:300] for l in t.splitlines() if re.search(rx, l)][:n]


    def server_caps(t, argv=()):
        na = next((l for l in t.splitlines() if "non-default args:" in l), "")
        caps = {"source": "log" if na else "argv"}
        for k in ("max_num_seqs", "max_num_batched_tokens", "block_size", "max_model_len", "kv_cache_memory_bytes", "num_speculative_tokens"):
            m = re.search(rf"'{k}': (\d+)", na)
            caps[k] = int(m.group(1)) if m else None
        for k in ("kv_cache_dtype", "mamba_ssm_cache_dtype", "mamba_cache_mode", "worker_extension_cls"):
            m = re.search(rf"'{k}': '([^']+)'", na)
            caps[k] = m.group(1) if m else None
        m = re.search(r"'language_model_only': (True|False)", na)
        caps["language_model_only"] = m.group(1) == "True" if m else False
        m = re.search(r"'cudagraph_capture_sizes': \[([\d, ]+)\]", na)
        caps["capture_sizes"] = [int(x) for x in m.group(1).split(",")] if m else None
        m = re.findall(r"Setting attention block size to (\d+) tokens", t)
        if m:
            caps["block_size"] = int(m[-1])
        if not na:                                     # fallback (the E23 fake has no args line): the argv, last wins
            a = [str(x) for x in argv]
            for flag, key in (("--max-num-seqs", "max_num_seqs"), ("--block-size", "block_size")):
                idx = [i for i, x in enumerate(a) if x == flag]
                if idx and idx[-1] + 1 < len(a):
                    caps[key] = int(a[idx[-1] + 1])
        caps["block_size"] = caps.get("block_size") or BLOCK
        sizes = caps.get("capture_sizes") or []
        k = caps.get("num_speculative_tokens") or 3
        caps["captures_cover_seqs_x_1pk"] = bool(sizes) and caps.get("max_num_seqs") is not None and max(sizes) >= caps["max_num_seqs"] * (1 + k)
        return caps


    def pool_facts(t):
        f = {}
        m = re.findall(r"GPU KV cache size: ([\d,]+) tokens", t)
        f["kv_tokens"] = int(m[-1].replace(",", "")) if m else None
        m = re.findall(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", t)
        f["max_concurrency"] = float(m[-1][1]) if m else None
        f["max_concurrency_tokens"] = int(m[-1][0].replace(",", "")) if m else None     # R2: M of the line (F_log)
        m = re.findall(r"Padding mamba page size by ([\d.]+)%", t)
        f["mamba_pad_pct"] = float(m[-1]) if m else None
        f["min_block_for_mamba"] = math.ceil(BLOCK / (1 + f["mamba_pad_pct"] / 100)) if m else None
        f["kv_group_sizes"] = (re.findall(r"kv cache group sizes \(([^)]*)\)", t) or [None])[-1]
        f["side_cache_layout"] = _pick(t, r"Side-cache layout", 2)
        f["initial_free_reserved_gib"] = [[float(a), float(b)] for a, b in re.findall(r"Initial free memory ([\d.]+) GiB, reserved ([\d.]+) GiB", t)][:4]
        f["model_load_gib_sec"] = [[float(a), float(b)] for a, b in re.findall(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds", t)][:4]
        f["graph_capture_sec_gib"] = [[int(a), float(b)] for a, b in re.findall(r"Graph capturing finished in (\d+) secs, took ([\d.]+) GiB", t)][:4]
        f["rank_load_gib"] = {int(p) * 2 + int(q): float(g) for p, q, g in re.findall(
            r"\(Worker_PP(\d)_TP(\d)\S*(?: pid=\d+)?\) INFO [^\n]*?Model loading took ([\d.]+) GiB memory", t)}
        f["rank_load_sec"] = {int(p) * 2 + int(q): float(g) for p, q, g in re.findall(
            r"\(Worker_PP(\d)_TP(\d)\S*(?: pid=\d+)?\) INFO [^\n]*?Model loading took [\d.]+ GiB memory and ([\d.]+) seconds", t)}
        f["rank_graph_gib"] = {int(p) * 2 + int(q): float(g) for p, q, g in re.findall(
            r"\(Worker_PP(\d)_TP(\d)\S*(?: pid=\d+)?\) INFO [^\n]*?Graph capturing finished in \d+ secs, took ([\d.]+) GiB", t)}
        f["ssm_dtype_lines"] = _pick(t, r"mamba_ssm_dtype|mamba-ssm-cache-dtype")
        f["gdn_fallback_lines"] = _pick(t, r"Falling back to the Triton GDN spec decode path")
        f["share_workspace_lines"] = _pick(t, r"MERLIN_SHARE_DRAFT_WORKSPACE")
        f["mm_warmup_lines"] = _pick(t, r"Multi-modal warmup completed")
        f["host_ram_lines"] = _pick(t, r"host memory available now|Cgroup memory preflight|Skipping auto-prefetch", 6)
        f["sched_warn_lines"] = _pick(t, r"max_num_scheduled_tokens", 2)
        f["oom_lines"] = _pick(t, r"OutOfMemoryError|CUDA out of memory", 4)
        f["static_scale_lines"] = _pick(t, r"applied static K/V scales", 8)
        f["default_scale_lines"] = _pick(t, r"DEFAULT 1\.0|no static calibration", 4)    # audit 2: the MTP layer's QSA FP8 KV
        f["ext_lines"] = _pick(t, r"Injected .*NB1WorkerExt|nb1_worker_ext", 2)
        f["shm_leak_lines"] = _pick(t, r"leaked (semaphore|shared_memory) objects", 4)
        return f


    def graph_facts(t):
        caps = sorted({(a, int(c)) for a, b, c in re.findall(r"Capturing CUDA graphs \((\w+)\):\s*100%\|[^|]*\|\s*(\d+)/(\d+)", t)})
        return {"captured": caps, "resolved_modes": _pick(t, r"resolved cudagraph modes before capture", 8),
                "draft_identity": _pick(t, r"e2-identity\[.*role=draft", 4), "recoverssm_commit": _pick(t, r"\[recoverssm-commit\]"),
                "recoverssm_path": _pick(t, r"RecoverSSM path taken", 4), "jit": jit_kernels(t)}


    def jit_kernels(t):
        return re.findall(r"JIT compilation during inference: (\w+)", t)


    def nccl_transport(t):
        via = Counter(re.findall(r"NCCL INFO Channel \d+/\d+ ?: ?\d+\[\d+\] -> \d+\[\d+\] (?:\[\w+\] )?via (\S+)", t))
        return {"channel_via": dict(via), "net": sorted(set(re.findall(r"NCCL INFO (Using network \w+|NET/\w+ : Using [^\n]{0,60})", t)))[:6],
                "comm_inits": len(re.findall(r"NCCL INFO comm 0x[0-9a-f]+ rank \d+ nranks \d+", t)),
                "pynccl_groups": _pick(t, r"all-reduce backends .* for group", 6),
                "debug_on": "NCCL INFO" in t}


    def rpc_cache_info(port):
        """The read-only worker extension over /collective_rpc (dev mode): actual cache tensor dtypes, specs, free memory."""
        body = {"method": "nb1_cache_info", "timeout": 120}
        try:
            r = json.loads(L.U.urlopen(L.U.Request(f"http://127.0.0.1:{port}/collective_rpc", data=json.dumps(body).encode(),
                                                   headers={"Content-Type": "application/json"}), timeout=180).read())
            res = r.get("results") or []
        except Exception as e:
            return {"error": repr(e)[:300]}
        out = {"ranks": res}
        dts = sorted({x["dtype"] for w in res if isinstance(w, dict) for g in (w.get("cache_tensors") or [])
                      if "mamba" in g.get("module", "").lower() or "gdn" in g.get("module", "").lower() or "linear" in g.get("module", "").lower()
                      for x in g.get("tensors") or []})
        out["recurrent_tensor_dtypes"] = dts
        out["num_gpu_blocks"] = sorted({w.get("num_gpu_blocks") for w in res if isinstance(w, dict)} - {None})
        out["free_mib_per_rank"] = [((w or {}).get("mem") or {}).get("free_mib") for w in res if isinstance(w, dict)]
        pads = []
        for w in res:
            for s in ((w or {}).get("kv_cache_specs") or []) if isinstance(w, dict) else []:
                if s.get("unpadded_page_bytes") and s.get("page_size_bytes"):
                    pads.append({"rank": w.get("rank"), "type": s.get("type"), "layers": s.get("layers"), "state_dtypes": s.get("state_dtypes"),
                                 "padded": s["page_size_bytes"], "unpadded": s["unpadded_page_bytes"],
                                 "pad_pct": round(100 * (s["page_size_bytes"] - s["unpadded_page_bytes"]) / s["unpadded_page_bytes"], 2)})
        out["mamba_padding"] = pads[:16]
        return out


    def rpc(port, method, *args, timeout=120):
        """POST /collective_rpc (dev mode): a worker method by name with string args; the per-rank results or the error."""
        body = {"method": method, "args": [str(a) for a in args], "timeout": timeout}
        try:
            r = json.loads(L.U.urlopen(L.U.Request(f"http://127.0.0.1:{port}/collective_rpc", data=json.dumps(body).encode(),
                                                   headers={"Content-Type": "application/json"}), timeout=timeout + 60).read())
            return {"results": r.get("results")}
        except Exception as e:
            return {"error": repr(e)[:300]}


    def commit_mode(port, mode):
        """Merlin's runtime RecoverSSM commit switch (gpu_worker.set_recoverssm_commit_mode: eager | staged | graph); returns
        the previous arm's counters (eager_steps, staged_steps, replays, captures, fallbacks) per rank."""
        return rpc(port, "set_recoverssm_commit_mode", mode)


    def commit_counter_delta(before, after):
        """Per arm, summed over ranks: commits graph-replayed vs run eager / staged, and the fallbacks by reason."""
        try:
            b = {k: sum((x or {}).get(k, 0) for x in before) for k in ("eager_steps", "staged_steps", "replays")}
            a = {k: sum((x or {}).get(k, 0) for x in after) for k in ("eager_steps", "staged_steps", "replays")}
        except Exception as e:
            return {"error": repr(e)[:200]}
        d = {k: a[k] - b[k] for k in a}
        tot = sum(d.values())
        fb = {}
        for x in after or []:
            for k, v in ((x or {}).get("fallbacks") or {}).items():
                fb[k] = fb.get(k, 0) + v
        return dict(d, replay_share=round(d["replays"] / tot, 3) if tot else None, fallbacks_total_after=fb)


    def kvu(port):
        return L.prom(port).get(L.KVU)


    def profiler_tools():
        """ncu / nsys availability (DRAM bytes per stage need a server LAUNCHED under one of them: a separate boot)."""
        out = {}
        cands = ["ncu", "nsys", "/usr/local/cuda/bin/ncu", "/usr/local/cuda/bin/nsys"]
        try:
            eng = MERLIN_ENG if isinstance(MERLIN_ENG, dict) else {}
            if eng.get("cuda_home"):
                cands += [eng["cuda_home"] + "/bin/ncu", eng["cuda_home"] + "/bin/nsys"]
        except Exception:
            pass
        for c in cands:
            path = shutil.which(c) if "/" not in c else (c if os.path.exists(c) else None)
            if path:
                try:
                    v = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=60).stdout.strip()[:200]
                except Exception as e:
                    v = repr(e)[:100]
                out[c] = {"path": path, "version": v}
        return {"found": out, "dram_bytes": "not measured in NB1: ncu / nsys cannot attach to a running vLLM server; DRAM bytes per "
                "stage need a boot launched under ncu (dram__bytes_read.sum on the Marlin MoE + dense kernels) or nsys: NB3 / NB4 item",
                "torch_profiler_window": "the PP2 trace window runs after the ladder, on its own streams; no benchmark window overlaps it"}


    def nb1_complete(port, ids, max_new, temperature=0.0, seed=None, logprobs=0, timeout=1800):
        """Non-streaming /v1/completions with token ids: ids, usage, cached tokens (--enable-prompt-tokens-details) and, with
        logprobs, the chosen-token logprobs + top-1/top-2 margins. Every request is saved in traces/requests.jsonl."""
        body = {"model": "q27", "prompt": list(ids), "max_tokens": int(max_new), "temperature": temperature,
                "top_p": 0.95 if temperature else 1.0, "ignore_eos": True, "return_token_ids": True}
        if temperature:
            body["top_k"] = 20
        if seed is not None:
            body["seed"] = int(seed)
        if logprobs:
            body["logprobs"] = int(logprobs)
            body["return_tokens_as_token_ids"] = True
        t = time.time()
        try:
            r = json.loads(L.U.urlopen(L.U.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                                   headers={"Content-Type": "application/json"}), timeout=timeout).read())
            ch = r["choices"][0]
            u = r.get("usage") or {}
            out = {"ids": list(ch.get("token_ids") or []), "text": ch.get("text") or "", "usage": u, "error": None,
                   "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens")}
            lp = ch.get("logprobs") or {}
            if logprobs:
                out["lps"] = list(lp.get("token_logprobs") or [])
                out["margins"] = []
                for top in lp.get("top_logprobs") or []:
                    v = sorted((x for x in (top or {}).values() if isinstance(x, (int, float))), reverse=True)
                    out["margins"].append(round(v[0] - v[1], 4) if len(v) >= 2 else None)
        except Exception as e:
            out = {"ids": [], "text": "", "usage": None, "cached": None, "error": repr(e)[:300]}
        out["sec"] = round(time.time() - t, 2)
        try:
            with _SAVE_LOCK, open(REQ_LOG, "a") as fh:
                fh.write(json.dumps({"kind": "nb1_complete", "port": port, "prompt_len": len(ids), "max_new": max_new,
                                     "prompt_sha": psha(ids)[:16], "text": out["text"], "tokens": out["ids"], "lps": out.get("lps"),
                                     "usage": out["usage"], "error": out["error"], "sec": out["sec"], "t": time.time()}) + "\n")
        except Exception:
            pass
        return out


    def lp_compare(a, b):
        """Greedy rows a (reference) vs b: first differing token, the reference's top-1 - top-2 margin there (L.FAIL_MARGIN),
        mean / max |delta chosen-token logprob| over the identical prefix."""
        if not a or not b or a.get("error") or b.get("error") or not a.get("ids") or not b.get("ids"):
            return {"status": "INCONCLUSIVE"}
        x, y = a["ids"], b["ids"]
        d = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), None)
        la, lb = a.get("lps") or [], b.get("lps") or []
        k = d if d is not None else min(len(x), len(y))
        dl = [abs(p - q) for p, q in zip(la[:k], lb[:k]) if isinstance(p, (int, float)) and isinstance(q, (int, float))]
        if d is None:
            st, m = ("EXACT" if len(x) == len(y) else "INCONCLUSIVE"), None
        else:
            mm = a.get("margins") or []
            m = mm[d] if d < len(mm) else None
            st = "INCONCLUSIVE" if m is None else ("FAIL_HIGH_MARGIN" if m >= L.FAIL_MARGIN else "NEAR_TIE")
        return {"status": st, "first_diff": d, "ref_margin": m, "mean_abs_dlogprob_prefix": round(statistics.mean(dl), 5) if dl else None,
                "max_abs_dlogprob_prefix": round(max(dl), 5) if dl else None, "tokens": [len(x), len(y)],
                "cached": [a.get("cached"), b.get("cached")]}


    def warmup(port, n, secs=None):
        """Multi-stream 2k decode, ramped 1 -> n streams, excluded from every metric: compiles the kernels RC2 saw compiled
        during inference (_fill_num_accepted_kernel, _causal_conv1d_update_kernel, _gdn_recoverssm_verify_kernel,
        _ple_conv_kernel, _ple_conv_writeback_kernel) before any measured window."""
        secs = secs or (45.0 if not FAKE else 4.0)
        srv = next((x for x in list(L.LIVE) if x.port == port), None)
        j0 = jit_kernels(srv.log()) if srv else []
        ss, t0 = [], time.time()
        try:
            for i in range(n):
                g = L.Stream(port, PROMPTS["2k"][i % len(PROMPTS["2k"])], 4096, temperature=TEMP, seed=900 + i, ignore_eos=True)
                g.start()
                ss.append(g)
                time.sleep(min(3.0, secs / max(1, 2 * n)))
            while time.time() - t0 < secs:
                time.sleep(1)
        finally:
            for g in ss:
                g.stop()
            for g in ss:
                g.join(10)
        L.wait_idle(port, 60)
        j1 = jit_kernels(srv.log()) if srv else []
        return {"streams": n, "sec": round(time.time() - t0, 1), "jit_before": j0, "jit_compiled_in_warmup": j1[len(j0):],
                "errors": sorted({g.err for g in ss if g.err})[:3]}


    def accept_pos(port):
        try:
            txt = L.U.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=15).read().decode("utf-8", "replace")
        except Exception:
            return None
        pos = {}
        for m in re.finditer(r"^vllm:spec_decode_num_accepted_tokens_per_pos_total\{([^}]*)\}\s+(\S+)", txt, re.M):
            p = re.search(r'position="(\d+)"', m.group(1))
            if p:
                pos[int(p.group(1))] = pos.get(int(p.group(1)), 0.0) + float(m.group(2))
        return {"pos": pos, "drafts": L.prom(port).get(L.DRF, 0.0)}


    def _prefill_done(r, n):
        tail = ((r.get("admission") or {}).get("tail") or [])
        t = tail[-1] if tail else None
        return bool(t) and t[1] == n and t[2] == 0 and t[3] >= n


    def _req_lines():
        try:
            with open(REQ_LOG) as fh:
                return sum(1 for _ in fh)
        except Exception:
            return 0


    def _ttfts(port, start_line):
        out = []
        try:
            with open(REQ_LOG) as fh:
                for i, l in enumerate(fh):
                    if i >= start_line:
                        r = json.loads(l)
                        if r.get("port") == port and r.get("ttft") is not None:
                            out.append(r["ttft"])
        except Exception:
            pass
        return out


    def _counters(port):
        try:
            m = L.prom(port)
            return m.get(L.PROMPT, 0.0), m.get("vllm:prefix_cache_hits_total", 0.0)
        except Exception:
            return None, None


    def window(port, s, band, n, max_new=None, secs=None):
        """One steady window, timed only after every stream finished admission + prefill (the lab's settle needs all running
        streams to have emitted and be gaining, then an explicit check of the last admission sample; Merlin's cost-aware
        admission may hold a cold miss up to 120 s, inside the lab's >= 420 s admission limit); a window with prefill inside is
        retried once. Recorded: tick_ms (one engine step = one micro-batch of C/2 requests under PP2) AND cycle_ms (= PP x
        tick, the per-request advance time), per-request TTFT, uncached prefill tok/s (prompt minus prefix-hit tokens),
        retained (idle) blocks before the window, per-position acceptance."""
        retained_before = None
        if L.wait_idle(port, 60):
            retained_before = kvu(port)
        def run():
            if max_new is None:                 # RC2 / E23 levels(): MAX_NEW[band], WIN, the budget / KV-bound guards
                return levels(port, band, [n], server=s, temperature=TEMP, prefill_tok_s=PREFILL["tok_s"]).get(f"{band}@{n}") or {}
            if not due((secs or WIN) + 210 + n * BANDS[band] / 1500):        # E25 NB4 (klee100 R1-1)
                return {"status": "SKIPPED_BOOT_BUDGET", "boot_left_sec": round(boot_left())}
            return L.steady(port, PROMPTS[band], n, secs=secs or WIN, max_new=max_new, temperature=TEMP, prefill_tok_s=PREFILL["tok_s"])
        a0, c0, l0 = accept_pos(port), _counters(port), _req_lines()
        r = run()
        if r.get("status") == "INCONCLUSIVE_PREFILL_IN_WINDOW" and due(WIN + 120 + n * BANDS[band] / max(500.0, PREFILL["tok_s"])):
            c0, l0 = _counters(port), _req_lines()
            r = dict(run(), retried_after="INCONCLUSIVE_PREFILL_IN_WINDOW")
        a1, c1 = accept_pos(port), _counters(port)
        r["prefill_done_before_window"] = _prefill_done(r, n)
        r["running_eq_level"] = r.get("running_min") == r.get("running_max") == n
        if r.get("status") == "STEADY" and not (r["running_eq_level"] and r["prefill_done_before_window"]):
            r["status"] = "INCONCLUSIVE_RUNNING_NE_LEVEL" if not r["running_eq_level"] else "INCONCLUSIVE_PREFILL_NOT_DONE"
        if r.get("step_ms"):
            r["tick_ms"] = r["step_ms"]
            r["cycle_ms"] = round(PP * r["step_ms"], 1)
            r["requests_per_tick"] = round(n / PP, 2)
            r["microbatches_per_s"] = round(1000.0 / r["step_ms"], 2)
            r["wall_clock_tok_s"] = r.get("tok_s")
            r["tick_note"] = ("tick_ms = step_ms = one PP tick (one completed micro-batch of C/2 requests); cycle_ms = 2 ticks = the "
                              "per-request advance time; microbatches_per_s = completed micro-batches per second; wall_clock_tok_s = "
                              "generated tokens / window seconds")
        adm = (r.get("admission") or {}).get("sec")
        if None not in (c0[0], c1[0]) and adm:
            unc = (c1[0] - c0[0]) - (c1[1] - c0[1])
            r["prefill_uncached_tokens"] = int(unc)
            r["prefill_tok_s_uncached"] = round(unc / adm, 1)
            r["prefill_note"] = "admission.prefill_tok_s counts cached prompt tokens; prefill_tok_s_uncached does not (incl. a retry)"
        r["ttft_per_request"] = _ttfts(port, l0)
        if retained_before is not None:
            r["retained_usage_before"] = retained_before
        if a0 and a1 and a1["drafts"] > a0["drafts"]:
            r["accept_rate_per_position_incl_admission"] = {p: round((a1["pos"].get(p, 0) - a0["pos"].get(p, 0)) / (a1["drafts"] - a0["drafts"]), 3)
                                                            for p in sorted(a1["pos"])}
        return r
    save()


if not RERUN:
    # ==================================================================== CELL 7 — E25 NB4: arms, hooks, launches, bodies, analysis (NEW)
    import lzma as _lzma
    import base64 as _b64
    import gzip as _gzip
    NB4 = R.setdefault("nb4", {})
    NB4_EMB = json.loads("{\"nb4_lib.py\": [\"fcab98a257b314192e6c434a3c051cc9faba2cfcc3dbc90d2e2a8acec6cb5f00\", \"{Wp48S^xk9=GL@E0stWa8~^|S5YJf5;PI(R<y`;~Xbna~Q3)_)cEO+Gq7Jd!)j6qqdRyr(MR*WOOWTk{91YI>EXAM{0iU4q%*Zb;tm;xDc9$D@WNy<lpwo%lwy6W#ZSR*SIyyxsU)x*=Ino{;$Lt#MaoB6Yd6(bvFj!<`;|c|64UqN6vBMW!&I^W9*nk>e6hbmN4t^C$nz3A79I+BoUnv!{$DXhG`2wHU1Yv~A<gIO^Gb}Z85<p3}j9-geMv6;l_6g4M&Wo?4-YD^o85b|X83>j)1>GxrW4QB7V!Z(yol-v}TV$n%SUhi~SE~+92~Cq-cM4iEI_#Z33z2RT!fZ=@tF%7Ya6>ES-mxE|U_FSbJ?;Coy&>kLOaSJFeLE>GHF8i^5565p>63JXh+_=gX`G0R5D=lduC&+&TtTl*zBdr#_;iHM2~PvO?n)B>DYWX?wqRU+N>#zwK2tUya46A;fYZMB5z`#T432>VY34kx^--6up?bypF*+_EMiV!;dLPCZ8euG4bJ`bg8j6I%M9gv*4HgJoSlLH>ZF7xvfQYKpzB{JS6tk7L9u(CrzzFwJB}THEg@&m1dCc`npXwr|)VYTikG+~<%6&%6gGhXBuq=W!^?!`~yu@F1$6P}c9H0YTj+32L>$3N<UH95v)=o5E+mJ$;Dc6786h^!$%>>|Y56RyrH3RLwdF+xx4}_S}(QOI_i9AA{@YBG5W*PK0tI!sWXknKr0(1L$WOj?Qfs6N8BmsD$TN<&30m?&bV4dMW*7Aw_Vg`jc$i*p$t90Zz2FE9Q+fGg0%WgCk8pNq$CJ7cz@(F<Rjf1P69+;m~t(v^Ex@1gjI_MUrsl#!V%S}ev!ihM9p+6x(-K0Y>JOI?+G93KL8G#kL3w7VDd0$xc@CiA%yybKeOVH0!t3F3MvrZ{*rxHJ#1SF^L1u;+=4koMfOB@R_&(j!&`{ZNBW8M+TqsgERrK_M-@{jpFN$zCS$wl7Fg^Sh#IO#UIivly*-g^RDn2CY>a&BWVTWE>N;_wa|h8Av{wr03h3KvWs<1`MPvw;DuyCCWg2hHKmBG%>P_w?fOzfVDBi=1E+=(ett9bE3U5c(~HalSQlJQ6;b^7NFdUcI+*fGEfvecS5KP;@BIr>?7LYIlXqZf|q9=82hCn3+d|+}SNh{X7Y1JZW=Rj59lwro)bJURJj2clf-cL2Ly(hm+2OcV_;gR1{(7s;g?lZurQ6e*S;zVt+W`Cl!d>6Bqek9vv!f7dqvid&$km<Muntf5+X|>;-*%>8_}AUr)!;1YiUShilQVfPnL48-QleYIYIf9@uNd9bn-OeGkZFM1CNYZ5=K?y71kALzouqTH4OCXc}YyS5z@k91PF@K<jU_09nd5RDxmig#L#(R(cMJV{Vj^b79osj6a)x(+0Se^FNiL;D<V<{=b6l!vzGaqLHV6`kX$OtC;~Lxv9IE4h?a_jy}wck(<0)*Y=U9OhdLfDf>lxIic3Bd~J8bXd+EoKmp)s=*iM%yIG{s?Kq1xOp_)sQ$>vgIt7k+EIY>^8zZJtlv7JbN_V94Z=&?ginPtWpq>tZ?dxKqoN8u)7JrtfnY<~)OXYf!!oZ9}RU8s8cXbs4D?3bO2|U8`M2=%(OL0mI>T~Wj`D`xh-LUQmlF&q{<&A!&i0~oHaC#t~s3lh@-v(CGMX8@j1-hdKuFKNgsn%KZh>oAZCqOKe=i%EICJVyk(~tqI2q;EKpEl^=h=c|3Sz6Nsb1bS3q}D<?u)WlpR^irq2Gi3ieQp!54eDlu5x3_6L3gsT{t_*<j7<EIwg|@%DlCw&i0Mm-o{hTt8Fr|QCx`@MH`j+x_o-^^l9<j9^}*v&0O;a7IhEROP5k;Ggh$8qksP&h>b4nF=0VaUjn&DSSo<Hy_ddV#yjrUKmguq#5#<<u4@EiJ?(sl@`ZnUTf>C%%tN45Ge>Mpq9$>Q%w}bqy?;m^6=acfSTSwHyVlKygq3KTmZ6aXwmNODz*H>2hYR+G3=9>p^kkJMZO7W}I(~_24wVL1V6=^T3L^Z?L{9ZvhrX6lmHd-u^W`#tyLzwg#LL(nTKk%kdd@d%zZxm{wGZ?fPpqGz%T1_Kw13SS@T51v!IB$Y@A>{GKlvK2W9!&EIgb46n=~ChzOK3@T^CP=Q+*WluV-A^`OapmY%rga-kM^eA#$AF+$r}s|?Q)vqpP*cjH0kTyMt>cfEtvk@yY?E`+A?lmVhJvv49`vogtVkWxKzD_jkS*oqrD5Q`PX?qf>vuA94jcxvH?|Y`pCsm72MS7Rf{s}EH$F2*yi<5aUvYXS1#Rtjs~5EgT~z~A$F}U3`Q5)zjBl1AV|o)GAe8Z$H4jyeYe3wR!*bXkfOq4?U;n`-JfPGZvzGNaNE0x8pEEs+ObU8p>YHD-9vD4w5`)USTPD{*T`E_a2x~`hCy*so=VA}DKmyWan%-@&5Xqd7sDroq2Y++aWCsIhvkN606E@ZOXZ!^ciD9)u!9_t5X+mfzcmr>4`sQR8QII$Py`Y@E39LW{J(-mFc<KEkwUcK(?0ojd>lu{x*<BH%@m%aO+Dk33G+ngy>wW~0ESk^$_IYjV5GPiR+^9!-eEc&=0H>)P*z(}y_L5amvF74yDE0Q^No@oGtgzVL+uIDr=&U$3UK_XLu>JO$-QYlUXh~Ukw1OSWHtkiD>5<p%{z3NL;-48>BiEh^lt>NBxlFvzG!)|TeA5$J$FJ{nE-lq<u6P_&<tRxa>wD+ws0Jb9k36>tM<MCSjmLT>6tVWvr0PJW>>7~3vigNOfUm|SiJ>Xu8(A{#Bem%lhL2Mr4QJN7~Duy)P}G$fY7D?PE^CEO1zgG8KDLit>jbM!h~7Yf5k(o8j*%F|A|t@{a9MHfD6K3Yb*p;IvcsSD!F@Ad372MV3)BB)n_OX2~zQiPs~sB*^MoG0a{dOpy}7NHWzlN!gE7J`|^MTjyfZSZ+fCqMuv0U*PzMBl1juA{9DDWtO5YtNf1{DO|F%N9%5>acN#a}cFU(s$6UJbL>9yOO+*#kaAjvLCtAc#(u@C&V0N!7NV_kU*^NOa-Io+E_juXf>+?5|d^y@NanCPue_b8JRGPRj{uE_y<SzwwKb?kK7B*j?3vfesH=&&d8=vMQ-yZo*#EUuP1}ob^G*f;DitU!9s8}?nnAZdazvmC?Hc)En)N(A|by3V&&~6E(SUn+59i^i|>>psh4sxcgCUC?$^jcXU6p72;Jn|#H1~uA<5Pcx5fqAg%DY>%&?p}#Qya@6XCv<9yYQb0hg|7~(9PwI;koNqEEGJ~&mN!c40mbj)$pG}6F@)HWIwLhI?-vniNbwa=pRMP}P~kw^S_;}%u6p4bCIp#R+j~gDDg75LYP`$cR7-viig`ka3rEv0SXHg#DsFgx=S^kzj=(-7klcEx=77KF3`)>1m>1xF*o29xjs6NtzZ;s16GOI9TEwny3EresFK{L0ht!(4Y{O|zS7t0L2551f2ldjV(doe+?SVR9ffDmnXb&0~e884>&xD(X;ICE86iwMBVn6f`6}^cux_p>=N{I{@fV**#8X}`Tf`L_8Ezy}Ty(YN!EwVf28bS0%s$$-;ZNTZx<>a!?R%|S;^K^X0qe2)u*qc$E(!?lxE393Jt<=XwFTT=N3y(B$v=#s3suGwN=+HcgMM>=6yp$Lpz(zvklI8+|Sz{g{UBktoa7$OB03<^MW}>oMk2VijLb5VpSI#v1#>JZN(1bh)yV&}mK`I|h`<6J2T4Sn{K^{k4fXwY;Hhqg!5pkBbfLXqrcMR(qt_nPfV9OM`8EsSo5Fbps<u^{%KiKAiw1x*_zM&f72ev_Usr8c#N}^A#McF#zIi`$d^mq?*)|$km^qt#|^@us6L`zGTD9^y!mm_Jp$9#2-{it@iYvFd;hOl|AO!mlB4UbWLa0L(@(e)9SI2U{|XX-fBBw;b9#%|BU>bxZjUR@R-U!_i%Ec&s_qiH+1wUJS*A7q&(oRsM7VJvJMPxmC-pjMo`;3`(z7!iE8t_^{{EEFbZ(mil(!&(F?{IX8Y5YS68v#`-i;H1XAR)=+OEaBvpU3{rNk2j~PD)LW1%+s8mwGB^uj13wMk=^oc;FI|@7Vuo{=1P*^)*^_6c7%~|q2>c^jmzfCB^()*(iT1v1R91h=$R})oG;2dM0Y3>ESgI6@tDd%D5EG49HW3Sk9tatUz<fp`HYXa(L7<gdyyAiKHq>r_aiy_@is252moIX$5@L=EzLbq>FUYL9NPbGPTPJ0&85xFs-5@fC~$FxdOmq4r&hA+tZ&4pAQpNjhJrO4RymVYrlxB%BwELb(6VmuzrggB8^Hc<-^x@8hySEYZSuL*lwihLeuwod=y51l#*rc!nV~`rmEq7@3iO@?-6D8?|FBb?lW|m>H?}~ROV>Uj8kT+w#R0KNb?w2}R9YG=!`c<wPAD<B&{OjbaTm!NpUu8+U4(yX(6Kg=fr}e!1R^UROavYhU5aNP+r48^A#;F8rYfKTiRL~+20y2OG-33q%nrKW5h%|c8kt<`wkXTZdcTNULG%!eR{`C7-rlF=<>{iZqF*K$d9?!cD5lO3InL^_6xUDH))eY6ru$5@6tUT??XyATSm+avkYEuZLr=d7Lg;P4)7lqB)c4<+S|OkuGE<ksB8ukUK!DB?75+Zcayu9u-Me|UnhXz3VnWMbtcuvOHW0L<55|BaZ_t)ALUWO`DcNe8pP%7w4I^uM#5`^=7}X_9QSj9EqM(iZ)YZwav=7IrAPKzbD7uI6O3b2DQNS+g^0h~hQn$YRmZWB&TO0g2q|@l9TUC0nl-%mcQIU(9r41LJnMzpm;xurdm~Gb7lQP1?Frq?s^jmTODwdE<f=I;QI<UpO9@;36h)kMwE_%Vo``U*5SKDl?;CdB30F2hf`-YS%U4gC-J$84A`b&?X0GSzw^h!XPR|4z;LI$4QFsO#&cuiDwGEx-|Tu#!@Dh2TIY1Z#Yj+r;Jc!e$>Am#2Xd4Or3a|Gi|c9I52t&c08#|sd?V}_{2uJz1f>*ZoFOb!u?`o6HZ7`<kmkXsSXV-|>Gtxv=IjZ0?IR}l(1vG#JDU>d37Scr*_f4*kw{`U^Ly7-uD&Vc_{#_h>-7vF7r+F3CaI|hB;BoJ+JF0z6Swse0gcuqhS+G%#>nkG37#<yCY=2yja?^6SQT^~aqb@+K(h5zH0;WI#rsa7_P^UZ6yGVZ@U?$K5bdYozcOt^bC@rd%0!t4Jt4fk)CkXQB`D0jZ4IAAE)!j>h(!Ad!;;K4r$iyqu-=SMZ4+eX3emZ8=N;Oj`ki286G4uzFH!25c5r}>V@AH4F|h+h+0>^^J*Hp+7u3%8*QU|9w^4vEz|9a4e8p`zC?6Sd<mieYkYiYq;1#f)mH{A=TPBx?z&W==J@VYE9d$s44gn2?@VAP)T%vK`}N3TNWiCQ(84>h6Twm@YqG>1=!sq*feSv6mlmwSBCXCDeq6pCV0|wO$t!1Os)g;x&%tnF~vYzF4ySbczL+o8?F%#zUMc-7+^-{<o66np<rNKz=-*`4U~4<KKA9KtoQ3k=j6XHHp_NBF+;ygR)hh1~~vCzwcu5>I@nkVt8R6IDAEP`@wMQujaXEf~$wO4h(%_u%!At4-EoXl0L-g2jAj@iqmPzNvu^$O(fPDm>~-|hD;Majm<TbywCKQrcoZ$v?cH*io&*cYdgQBADDPoS@}q1$s3tmfBrAM@?Z6oQH!nvWMJo<(q@)||41|^PRohDjGsmX#_!Av(OnFMwNkZ&4gqN0*JFcxi4$wj(5@&(zhCYbi&~5a5t9zA813BU^3ilUGV%Ym3=gbtwtdc{ZD>=preFKGcY^{}=5z`vU2tHuaCGL_y=5n*aokyH3vP^+0t6|WUrG&-u;(s$|C|@9Bti^7chp_m>2;cf)urZD@sW8`sh$lf1|f8dNTM|=K~l$$URD%gb%}&*MN>TNWQ8T+BmL-%Dx^R?W~;l6@*Jz`d||QXN0vpXT3rG*aQ0ILFyoDhbJ`}qBTSWbm>2waTg@DR4M^tAXc30a>4Tw}JnL#pm(t#bC4a5b&en8&pfyej>jvcj1JjFvyszf%{zGnkyl|wx7#@5>wKiKkU)4|L-A62HBiJrGLw*oYIWQVe@3jV*b9JkJPHE6JQ#_1YMy_h5;$o#Q^g~o->iYVCb5>;n;a3J1OQ`2DUy?nihUOF#vy-AE53|DXgHxc=<`Kgmj()kn%aQ*0+IM1UU@iAO3Gz2x+`(V^TRW>7CXf!pL})B!>FkCZc-rR6+=*4sF}HH1g=C2UAo77Mnhj9@qcvL^iDNJyqVAoBNG1<JFh{7thh!Q}6Lt^3C&Nz|E7jjKbHawlB=KtSuYi~VS~Vt+Vve_DG!a6E`WJJwiE&@d>Vd8auQv1bW3n(5)B{(^(?doF|6kEig-7{FFauN<o!^|M-0M8l>$QFNV3-X2@J;42)n|@{5rVCc@koI_mXtm%7bw}mfF72@q4nFkMMmB=uxeauG<%_5KwAdNFx|%dr<)o)SK~4wG|uykfJNvEH)f}2+Q7m-7yQ}hQt&z;Sw46x?$;Pd$7vx-apYwxbu+dUR-Hl8gn^^dCco?jYv)fuT~Sw@!bJLxHh)=e{8oOU7R8{uCBs}sUHwE2C+Kc&u7VUap8zpZd>70s(b4UzuY7ghlF@&g@E;nA9Qw2XlbPvR%LsW@cNmH92LN4+68!_;vac~o7YC{!_P|CT+gFCX8aOGj@c{)rrXXXZRhea(6?vdidho+HnVgQ)7A#g1bVFkF$otZHD-|Vlx9c#9!X_tJ*W66!fpTcwHNd5&ro0?+*huSF-W*~8ra<*pT%x0J<vX)@bzN^(_65%6n(YUE>WJA3l9K6P{kx{Do7C*Z)>}z|>O#_$q?#Dc*p9%#6uLAa9mzdfXKi{zw`J~wm!GF+<WQPM6wwuX%F=k;4rZaUC6OVfmlrq@o`#7nzt{Ai`Vw#zl=0WdcH_DsdVkT?w~w;f;6?H_-~-M}YZXCft%$ThPMRFjs^ykV8X!OZqUXi4b?A--`HMeA_9V3xo3z5A*|9*XYxdBpieImxo622LyIb3PtI5KND(7Xm`&3&ULDEZ^vkwq4@Ae)&2Bdr#yo&rI*Fk#k@D3P-S(sqB;z9q1PJvrA%!iw|*$bDZq+-;Ht{oKZXdw`FVYKFqlP0*c^{*)vgZeEbSzEM9_{8;VlZt`0hO&mjHN^iAaKz5L*q*O+|99~>2Q6P|FTI|l+#^52<UX}F&b6MAI&ks5)#5svjfVllOXo!FDq7!*yBxuoDj!;NYPD8|-aB7#)9|=q0hKL>Z&PT%!Q{cHI$l2sUi<6h@q&Olt>`wUof{H0`e2+xn$Fn7o{U_LmOXvQ{d0|OJnhp2+^>pTdbA2Njda8LhS1D79b6mBA^;XIYsj_pzvu-7b!oggr^1g?hV@CBf2G0VD`!5!H6D6L`z2ZpYoK`T@;1~tb1Sdf#;^712|7Noz2voG!wmYI^1s{vUcRcThV*eH3^Vosbu0u@jX|Kh01Y@`>t%Ox25IaGxe=^d9V2-ah@yk*grYQPaH9$|lUqn@IKW^NFZKRPDOUe9aJBObs?9)%^-%JtsYs9yAzC**lrY82@Q^pv>a=@{dRKh^Y7=5!c5c9uN`70P+F^(ys?ME)dXH(3oQiUbAQlqNi!#;~&;aF!rAhT3JkJXG3A{~c*)(=5$DDOL-Z$AK+=PelM~NcQ41f75*_W2?Ail#39!;a1$##GVAw+42?Uxuv$2BxwL}RseJZMKIsf1p00=Fm-?mf4F1bAG3QukA0^@9Q#>amnRLM|4<ms9n3Di*O<{C%1cO(pfviK-ac;`5+8C%&Qg(%FhLVczrv|FF<@>~Me_mq01x$X0m4OuASs3?8ot7a}^P5Gdb<O~kGu4X9>d+4n|ZJx@Rwj_Fk>kGOXdvaGP)5!kf@pj?qQ#n<$XJnvX8h=jmhJN^E&!-l5fPVLabh5Ih(`Lz+eCn4C~WLwamzBMXK_?FT10is3Y>;M}-!?YU~Q5-&YEPc0!jR&Uy?6a&glPM}2Aacw%ZkLlrbW4DAXsMIxs2g{~L?HHSgGJ3<7}!*C=qxXaT*SL&Z>0AB2*Y)ck+yeVKuC9U;i$<Bi<x8@MQF*@_+qGT$GH69(hN?dEwlWSw!m?Zlds@h%rL^l9Cza}T@SrvH4<!=j`NoMf8qC#)My;^1qgiVaU@47pHMCsH%iz|yAC06jo>6ps8fg3kKDRyI0HZOdH0jw!DCqhBlaE+fcmNnZT(kY$6K-!jt3zj%ii~Yql~YiS5?hfe(8JARH>pjj?rigRL0g<$pb7w4nP1sAM3P%b;u$~%tIT3EH6qR8BYb642i8Vt{KhPDktwWYfx!=UO{~|7evqFmzy?;-^r0YT|yHKq=Je;^ijMup$`FL*qPzOHKtWAx%iqiZ3kp94PfNlLLXVa{m*x|42!^UA5^Y#8KD^+FITfBhgVFgMWC8KEnT=3b0-5Uf#hD0p#_Gw{#YGv8Y4+^&S-p?+&M7ZoPdKe?dnZomP7ISL1&!$xhfB2U&01Xrkt;Ms7UYm`KihJaYmWpqEZydVQm--Y8)Fzoi8Aw?<k|S|5Pwq3`-2ltX3?-?H{A3hk&<X)xLg?Y&AW4&~36ZX;#deN(+CAm|w!Z*E!^5y{Vg?>9{<Ydls!ieYeDp3vR~YXDsSuAVrj4^G<|1#tBZJy7SZkrZ#|1FI8Iyj-ZmTB6<T>-4)ACO3%>9VEnOPjja{9!Nrrwo~nt-@oCz-^n-UJhUhUV{AektFBCg~pIUB}iHVPP7Zk_&7Zrl7Ul>GhiVs(zCEc6a4nkBV68{=o7nT;#dy7-R>&pyWo5-Yq4dc)6_rLK{6`-aEtWYFvd6@V1fv~^A5c9PsR`_9DMBtM`<9RIbBym-oMN5Ge_hulA&boGVzS)N}r=`F3V#wcQ+0}_#f3a<TP*}2H%eOS067zb-z~`ak4YFe)a}~#@9UT1GbPFg&G-Sz$Uel%Ehh0_pyXc3i3%xhf23M@nbqzFV4I>p6Wq2G5wQ3SOR`wiigGj___Zgp1PF6WJ_|Lc<yJ5ovR^^CTS^^7PcsN<ax(9xh*$+jrXZr^xHrnF)_xrgEsA2B)rf@|rtr81M3~J6mWb9vfZRvO^0ZR2^Ca`%0v&ls)W>}BkJMerAB<o$$1R+r7;qHLcPoWv@IX*xeOBQRx?{J?PDHKb)>#ZEYFE`T(%Xk71L34R%ZU7ck!euU<&Kq2SeMq$V)6zMSh~iL-&Zhukvr<=W^*ClNHAfXR;E*?GB0Q9)($Xg<wUEv{CsF`Nuq%P#L&^qM7_QcYR6)x6?rjmM98nJ0KNKnoWxex%b~fA_FFe7;Ag`qK7fG+m4eumrtq2<R%4q&7&PrI)q2VCYqDmprfd0$@i5p-W%66fMOw73G75QR`^~LB@BurhA107fOjf=D0y}G^(^Vn27skn_scq7MO<XCoFLq{}#WCUhLDrSF8J*^6RQW)lOJRvbl&x&m(4}AYgq!gt}=?ilK?_n~Lq6v*;e#;Jne%5jht0>Hu@qregH_bRti&4m`w{gA69J%49_!=0xunmh(0WXUth!<cP*`OrPg!52d&L~nZ-OR`xgyAUtm8+ujRbb~Z=}yfEVzKa;2BC%@XQU&9;f+`d<xW*?hiH10ADSU5-CJwsRID5;f*p(f@<1cE)nj6sEDu~$TqMkatD&f7==Zl+B+joqU;Xc7TBpnHM4n%@dw)S-CV@Ux&mc;pTWxiSDW$5YvQ4o_6&%2yns`S~N4040FkTSTE;^KcdH|k8JgOh(j4qz~UL?!$ZMrq%)IZAZfcyW&Gr#dhUo|yavPnWbOX!2zX;0zl3WC%)`QIi9zdN7WbnvE}Q@cZ(Q4(Ui;CmMT`CQ`HnA`kc0vrUhD&6csZ?6Lji2~E<XJn>g7R8JpG-88vS9y7$sp4c9>9xa+s3^!&!{5;0NS7&D%lza?!K@t<os8wDk9V8FwsY$Us9&(pNb{VU0Z>QUFD?VdBZznkTS(*p3-uL-_Od|3a3bdje}Lso@v(>zK)3CwkIMLn;2tq@WvBbHB5^W!zoS~|T8ZNi+)HDvi!v*}V-|#XH2=~|JpZRf9|yMHL&Ukm`UH-w*d>WiZ=Xz=f5|G67!X}s6$$e-dTzFqM$qM|n3npHJncC#74S1vR+^Nr<1H_uO>Lb)Dn79tl>$eg$Zj7Nuyi&S^kGdK&plJ#TFyhOcwd}ZdH{=p$Q?s5FOXEJ^)w|UoaEm9Dsj3vK}M+eJL#_(Fg^;!y|x9(z4nUZKkTt4+@Gd2!}o4u=qm1GkQf`{wU*_She*=NORd6U+xOHibLxLRN!8UsxQ7*#v@}Q{Eu9YJnJ**!dRM<@=!a2LbfSVKgxWKFT+zidCZwge;#NBaA0eoDwQqvtqojp43wW{9*cmM;+*PP~!EcfMFgvCrFT+NU_-T*75f3$Hs+Yn0x55s7H^h$8Npw3iA~}tB=v;UuDC?S%;xh+Sxu}-^hGqXq{=V&Edd+<)>!2Ri6qzybZ;@0HPjE6aSs^Y7m$?cWyyfpGBGo0_LQUt_(|0z`9~#-3(DhO=NgFhPi*QXp^|X_Dz6e&uY?ZI!o=eHl0!NRWxH|@FI}|HkZ_r@APSl8@hARHBVgvrj5r3+pM8mm`*J`D=J>-p?2;lkUjx#R6*<k{O0Rjq=iUKctzM?mofBV%>F^nMJ(`cMSf$62WxOmdL`XuLZ#$&hLBaXdJ?srWX0^-O<M3+mN`}kPegPbO+;vS3m!9R5n&yT`~W}A?;>^{gAcFX1=3J64FXA2cn0uDxhPuw@z+9QLTPyp{_f@LCG*OH_ZW@fMT0-#uyEHzw~tvtuO18%fZOOpA<O!Qmjp?)`}9m#T!H0Z9!I#+9VzWEQ4CZ&JcHg^qaN$vt`qNLWZ>m;6!{z?Osgf^|6UO~id5)mW*xM@UaH2i-W^<M-{6e@1}z}{e8FOP<dgMy6!MOKFrT2lo<v#m5U{ZxDF257DAP1s+!{nczq#nwTRWE<g&pbSEJEdFGtnyr_N-D5fqJJsR!OWnU{wZ{cm;sS-#32fZ8ip=|yLi@vMow5d6lI^Rr>IFfpse^5~in~*Rf`ExEMgj?cjQe6QjU6aaFsz#vO>_kkztf0KenD!|Ex;^ldo<%(mRc4)jI7yUE)8Sbh<5=yI~jaeq3E3!9Z=2=37(;N&xV%0`ueC2lB>!x<@1ytemw00saCtoNI8V_!PD?+q%fpkLjS^$kN!FPWd&ktk#3W~M90VWRo^2eU5hi4#3K`JIq(ycks7S!+~uH@Dcr)-xe6Tsu1~II*zre-^FU++Z-ul5@MT%l4=3+gmZN<`)JYlcg(=MBNm(;r$@z*2WY1$zh=I+9$?H?`;aRi(xQX5;%z=hzbg}vl=fLAEW3SPpnz?pGMje0ItbrapKYU+i-`3qnwwN6xOXyrt25HOi5<KZ(Gpqfx0%OPC6NySxiW0mWZq23r-}0MbnC~}Fi>jq<TM9zn$1LPOy&AE;Rl)aT|DqwK>P-E0esoX?YsYqO%S&0ZoVql2b|BG3(M0ho;CocMAZvKDPjuX_uuPHp{YYK#87ejSaThA!6&P3}##be?yinS)^&j$vpLB1X$TpB9L$SIDRBxl>ylrcZ4p{c%dQ|U3&-sF~Vdr!|cVg(M{p&q6YuBMQj?>z)VE+gzb~!NG6v3atTyoG9aQ{vH_nv7pu{av<?LGV~uLx&2v<HL^LCbO|dGsM0#hoN)PjRN*8!{d%m{N!$ec*l9GeuK&M?{IYBCR_A>qj^LfoFvy=QO~{FIRk|b%&_{XT?Tamre9ptv`*mct)wz+#|$3H&hMrdshndpu&!R3c2G8b0?;#aFx`(sVN{7@DYh#bUrlq(sT1??1HF02ZPEmRnW4bs7KK~_c&q0L67Sn4QuvvpTHB9lP1Gz+3w;g)Hp>p2pEY+Zw`sFD}~JmnNDJv5O#=sGVeC>$M_CTNNONbwV1#3phR+sv)}N{{q{FWG@U%X)bRfMhW6`e19-Rgcz?;FBthQD!W@MxZnn;8Nt!fwi9?XC>-%=MG->Az7{y$ztp*rE(-<N=)FGR+S6c<~%#S0Wq(P#h;|?o-Gk}6__+u197zt*pOo9QoBrCV4w^1eiPUO&)v?camj<ZynK$r(er8GDp#s|)e@~JaUa^RrqemMyQWy%P8o_-P@!gM8yU!8Xf*G*S|wm`xwqLroU5%}<;h|H5Jg!?yFFxKWi{9QQN-5XJ74>T=y8ruKV(l{Dq<79GtDTtkg<rjQbsEvAFe)i2qi+emn%e>U3GsZ04C(I#zqA5G_CCfP4q1cPsay#d9jQdbLvE}OERO(F#Mvl(sI{-1H7w?^kRVBieQdc(@CGz|V>Wzc#cpavjQbF`^o3uz-l&T?l6!n7pANJe620cKGQjF>|wozB-5^Z}PwMIM36-i~sUG!^ythzARG4+k@fK$=aR;A9sJzI{h19yF$0*p$_CT-|ghO8)bSzYb_%u(9c16g^L30W`VMG#a=l6*ay1(g>K#d}@!W}C?4wEgT$6_sFU)|W?6X_)w;%4Do#)hZ6U8pm<mt9RA8t@;WzGNWzW7yz1f`e1Z;<tRRk*p-Y7x{X~#c`ks98HY#j&|K-=Lx|pVGD&PiDXv2zH%W9|M>TQ=NFGmWM;KV52C>&#^!qH17rfHG5S2Sv7IVb46#m;ddd6-C16-YCDb}TgAEpaYzlY1gjyfH&%Eh_VDRAr9MuMZphv~^CHACMhd}E4fIPz{tNPB@`2A{|NAY*aXHHLLM4o9G)_k)5T0ziKoUb6@17C9%VHo&?u@RoQzoI*6tZvC%EkCXX(s21ASjUlh{>VOb@=ny@ipq1F}w53t#^cP0PBTRp-V&&;xj*J$Qt0AeiLQd-k^Hn{3ba$}j9D$#>Ppk^5aZ2*F`dQHFsTEp3b#zBq7d$iC!OizIsPjjRsE)4%&AXu4$HRXk`CM&K4&yHGl7BNCu5I*frcso+j@&|b&4#^6{Ao;8{Jy-ISHaAP!wt!zZf+rV-G$^x2i?(sZ9uU6ccomIv#j#FT7K>k4xh^kUO%FROPpjq0qP%0y=-;ZU|l4U^*8m&<fWwUctG0B8Iy|k+(UEJ4%E#~HfL~t7Y9hAd&VV|GX*CDjj$U_Ng4ye9fr-Sw7mujExb3HT5n2=scB&#g+;d344U+1sYkd&O7wx{T}1O$rx@Yyss(oh`TeR_t!2RsrtgPr)iGrx_te3q%4GN8m|_1}1fuiFO@K?|4Xd+{Jr>>_QGT!^Z~p{TvEy9i$B*;C@hBT0?Mr~!M+}Mtq7Q)9ySu|sH11jk43!syL*RPR<8NPo@r)DJ7QE~y#+#ZNVqSNf9MjsP*ykg_q7Cy84jX)lHbo;v=`RS4YxaeGblMuidG%G7D+ddhWnZ|T6zmq)0yPIa2_M2|ZC}`Q&zsjcvV{SdhC5#!L4OOgN?!UKr-aC@LJxc%v@p7=Iqd1TH)2}<5g6t#NesgHWXUkDK7bNyR~&aY1hqB?4_#E^K(0J%@mR5Tz_A3}0_NP(aK&f+j!r2w^bjt15wsxKFET<v&CDX3@viU_rCkD_q_w7iWWFi{dj%~ju@1!bhb5(w#@IrmiEukJSV5pO;#*Lr(~_`Nf#{3h$<dLr@Yg0IDO<`GEM}WZbVe%K;&p0mZ%qHQ6F|rf<NErVP_>Ar=Xfc1_-7d#Lp^X8Nq)>hoxAA%6t#TblK#RRJxPXzA+bE*qJ?jO4_itfuFbmFScUmZ7Kta*?Ke64NIl~m>LZv5eU|FIbT>E<mPrn*pRk9sY%^;M?92Y2BZE$iQ5gRdfv*CF5RLF<T*13}f7y`&tkp3_s$I}Rx0@FMZ7CiL5}Jlz&6FWXliZ4u)$Su;x15HfezIGnx(+nr8Rxp8F@vr87puj4hlk=O!!6S6|60H5dhRnq2YDqDZh&C2fyh+{0hCGq8|Rzm-rBkNZ_eF8ZsKA^>on2iMJrgWHPPbx!frRe;KC!38cArrFWuVMl^n{?TSh101iF%VdDEWDK;X-G<!(>^uoL}lC}-3Wy@=u3oSPK%8EX3oLk2Q(>{!<%iQ8WMrX4LpTDWpM1;ni0()gA=EY4%eIsECTD?OBOfChRWpTHiMc~&xMXJ-DLVN?NC>e)rBh<re)?099X8u|UQyU~%2QIWn)0qBtu@mTi%)TL7FHv^AipITA<0?q$9Z40v$Kdlfm%VAuDVlKH>w5~j48WkhoL>XNh)rb_FQ7Vu6#uB!N`yv-~MlQ0WQK(J5&J}cNzcEwtaS)1D{j@ipXeaB!O4KiAb*=`TAQS8n%>xOk0Zcd>Bdc4x*EdxnsPKk@?C3$;Kn^)@L?JdcK~0~eOPp!5#VMf+zT&orJy-*GT-aOeZbSU9Kek496z3gzZ}^vX92EtXg1p$E425Ya6<ekNZ@?p|aB{5nLSs%3w^J*P&#QKs_Zpl_aKtF=@vCZxu(|`WVOgp#hpf0}r^xgr0CI|66kOrIG?&YP@((83uDmO(D-UbjVfbIgo0#at-hs<R!6%WC;D?{Uemlryt#~I3g?yhwn%6uz&rh-QUDEz7cIwzqp8jE3&z27q`KT{Ja7|7u@)%NJ?*b0cCX;k$gE$2c^qY^p09Ye!y~V%*8z*0_htoe+&=xObqp2I^2Q<yIR@v}Y<yOD3j_TSpibbl>3ST4D8GK+&47;w8dItvkF+T@ur8UQ_kBf5uZSb)~<iw1_)3Z8wy6k|sXB9S8b{mZ-6EZYRFz0{MK6-1BW;Qb<BKH(HGTzch?lW7Kb<Nw6e~{Xufbi@|snBlf^3_u1GgG*+p>8XFlS!Ha=A_^KJe;+^J60X&j+H_jUkJ9{hL1#^Mgk1yYV=0oxMlz;7Smr{{*(x5$qbNXpq8-DXuM&(GnH)KvK>;Kq(dSsSfXzJj?#aVuB+Oq;k|(hD3BBj!5*U(8;zVA!2N^kg_Mv_mM8P#i1oWTmnnG*bYhzAR%Xh60Mo5<i?%A>huRh{O)~XwGaOx}+B}Vxw-MG=4lo;c&G@RIM@fuuGX178S>kKFZo=PVFayK8GQqgj>)i;O`{qB<j{T^w8$u*ipi5SI>^fg@A@!fTbjvOP8tDHcG^E)#Tg?AsCsUI!Lgez$QB*RqKyZ2CH`5UeQD06WmL1D<v{2Qu<zNIDc?LTs?;y46ZcaS<i0N!Nka74rJ+_D2N1JbM71<jXTD}`ag9qS&smCDjE|`1;y=~SJt8rMfT{sgw(&C{qk{1*0JiG#^wUyA4D!_82pi(NHY1#L#ZtgjPCra*apqfZD5Y<y$RCWy)>gXIyPS2~eIG1&wi(DA}BM*tC%H4RSDAPJcK@nJX0X@01-G(EbDQPfG5_Az#E)%M%EC#4w^=g%hL<FePx2%m3#2o0kJF5~LL|D8$=t7me^bZ_qI9zzc%;Zuw2TKrCa%vD6+r82nu=jtkx#IW4N}3TarL*U0e{&q6C`?UbKeLI93%fOY+5EEcNQ9}%r}6*RD^Py=2j_{b;wIiv1ON3Jsp8M0^}`=z=9X16@Fg(;jqj&-Vn)1D(K3h1KdJEpxihO=bJVf->iERHBb>dP>l={KN2Q>CB&AXwzU_O-o;{ABm}=Y;%{4Nw004F^Qwj;AKvTB*tCoj@E(4{)WGav;iHDC%!gIm>0M}kT%ln<?n#s(=1#|3JlbYd<tadT;kGeTBLp5eCB0bj;Yk%(?A?9zIz2h))^bL;55RRs+Z^YjM<@8y&p8^Cy`)BFQ3W^_}fR{oL!E*Z8tfCKJnjz4`iw);OMdXTmdCE1f8H~HUfqZ0y29<4%A-PJnQ_Ai*1%HCd`Of|WNsAEb(4deFVU=~EU$#8EV+)wSh+Xn!wiw+)l>XIeS+n&$Ew7I=em+2gZ$H@1&QpdF{+->rQV%U1*>lPcNddNKZ2IOUF%&f;Cvt1m-8SZOD#_uv@_b!XdFw2)gu8C<iHk;DDP-jq9zJGR0V16h>6ImL$z6vD3t+GwA7tXkJd%ZVyygwYV*&<-g1Y$p&(YpUmxb6$!qmvAG>Jo$O_BU534d2ZMMtGzunWMbZgJjPP>9Mh3$p{1Fh9A^B{fh2b5eS{0WW8Ib22aW{VyH0Ju^ElR;q%dVU+QF?MzhC#$ragRz6o)r_-)d+syrXw(ms?Ac0<~cNxnposJ6n?LaO8xdGK#zYt6Dc7A?w){8n*SsgLt)$BObkC9vI4}wF`22a0+Bu=3~ux^bTk~E5rW?g3)nl;nr;BoqtznJ(WDyehxN(Z{u2T>G66Hq3Z6!4=%)aN{Qt$OflK!;VY#<~@e5jYyNt3RP$PWR2~*;z@3k`$>%G^T#OE?<9W0o@qIiA`+V0?wk&*!wh9QOnowFM85Do(y?8NAxR0w*VbAPRSxj)!CA-ri{%{1G}Fv(c&CYmPo1&;lg^QDxQs!x>vP;x#AxwroJlh={h7Z5`6*aLXH-`m8VsNsWE@9iJ(+t0{#<cW8|B*%26UCL3J!HPSM+%&0l=C|DNeRayxV8X;XAyX@TwRra7rrk+uZKcu9z+`GT<5|4oS1KlnwgBjViy@@RfR0T^c=tE*{Ce`d96x55s%iNVf8AN~<2pr8iS6<8E|!0fg$NC|IGtw))@%EkAZh)6>)`}Y7WEh$9SQoNys^<2<0(x_qqRk$fM6addvgV0aT|ACl2hua#|C=tQdQZYE&tE3+RUyb&esL_sE$(V7K5c1=nX#^C1HlfmfQ$R*Dhcny&vtkS_StQ`$noOUWsd?(<#-EyOKzvA_=>6@9v?P%6As=uOyQ@_Ab14y~Ik#<@`ZF5?eZqUUSlmHx^Y?VF-+%LnHo28d&sV$eknSX~PnPV7_<sKD&jSV>4`G!s?WZI+no@WRuEoXH?$}+SFwLmj%EN@=0_gx(!zC0xaw7wbF?9XQafxxgii=s3(xoy(asJmt2*}i>f0L0hE@t0$9KjWatFQVPWcHkjyT^d#?5k-QJ<6+#lk^DdHx8gJhtVtIt=b`KSs+soZw`_8V>e2`b;qh$Y;jQlUo2#}%%c9!>@%gqP0({Sxm~HD7-46Ucxd6#ofr2ifxrX(SBVMO*+%xFQvSr*U!C$jjlml`;t5$>?zFq5b?BK<1x3^<_zqVS)=5la0-q73090Nwx_W1zAZIO@4WiUY<?zIgivSVv%LS7)2mHqh%?#w97ed}eJigD-aJc-6bp^h65lZ|mkSa!`U*TjC41wuqh5L;f-hKsU($oJu{@noH8st=yaO|SA5e`0-bxau?-HQ@)t=`qJ6C_<-z%kU2L)X9Q?KZ(SB#rPf2UYPgmlWijvffPhDi6ja126;vT@e)OnC^mt&lf5VZEE-yW_aK%J7)vEXV+6AFpNAD58VeZ%U^eoz6&V8e5&AhG5RI0j^HaMfUbx##jlAzG3O#PLPIlf*NN2~r38LOi*H&7F?Vi+xguD>Qo5!;y)C|;7;}hC!?+7%a_0~N8?7e}nOq=f@sv0azbA1+{nlXR5}3#bSYg+iu9l87)+kC9CU7&%xE3n>p;<}_4eX)34La*_ahR?Q<i>DfOI7)V%CW5^!=~FR#vjP&xxS^V3S|92T{f|2$Wj;-h~$Mn3fV})OgW}U=f2PXCZPPY=x}+kj}G|cSB!pDULq-d^OVSA9k2{37x}&%ni%Er2NH1Xjs{~xe4Jd-hpf&qQai0vVPwf-1BAUMbe&z@9FTtFYJHwU{MV(b8m65QiYk|k2~CEjDTlT~eU{!jeTzYIMP=rk4$;i~Lld9tHYof5w<#Fdtyl*9^wt;qqre5@{3~7FU_AV$60a!_15$AOx{L#;!_Ya)z*ccTY;UGf(c2}b^+2{FvF7oT%K-6vO=~bbMw6!|FpXu_$^XM@P4=3LXPmU+t4i2gKYQ$`>;zEuS)4x2fyc8<NKWAp*~}@^b8Te3r95DRsj5*7*spTS2qHRi_dzQcrV}0lQv_aw3>2&%AUUPFk{^M#xSG`0n>Y#n;+FUZJ=3n__zs635AC2`<yZkN0f^(jL>S8pR7R9C7gd7?=_(xPG^P5e<r&m>V^guGb0(f&UL}XDVZhM3sUcpIJ1({NeIz8|tQ!M@^M%FBX8W4zqAa{({X%>|?tC=4YU7ymbYmGkK86)RA>Sz)ev@58EWPs)IC+E*Z4+t5nu6E?stMRP)@x94%Fp#e-&ku&Q&o7+tjuC-5l>bw{TvQHI$#c8nQluuq6_9{z-o1%5-Z%N{&P30HHPnRq{bz%5g(KG-MYVf82YW@C~XQWF~?h(E};ltorAU5dpuXZD#)obw4l0JfX^vaaPRWZecrk7APPf8B6A=nG;bZ^e@Upo?m~nU(&cs09W~U6&JY&U-V-oFUd*x|sIhB%mh&D)B!!7R@$jCVPXnLxto>R=_O;$tk}+M2%~4*7$(+>r8_}K^+@`cy;u{`posC2cq(yEZvQ&Ns?1{6&-m79?L6cc0+Kwo7tnTb<6~scOfl#tf!X%OkU3j(!s{PKR*D5ahv&vpPbQzUh`eS$6Te||nKO+Ds#HAr1F8)!Y^El|d{M*w4sWtAbe}Bxm-u1u*l<6#dYJltKbum(BEQ>!&@=>p;pag3-qCukR_bH47Bp`aJ)c_5}NW88!?+fbkq<RgU^i$_Yjs61fFE89e7$h=rhQ~szO#5bmaqU^4b~+UF2VXBuc}$Ri#2bM$m$B1IYrVAd2O~8EHe&b(36hngNHLXBj{MWJ4x{w{YF12Y5J+j<liA1}#xpe<hi(X0)0BY%E!IyuX6F&-P{RY>2=CGlYag34{0d@|LpI@S2-$Tyx}l1G&A=Q5p5?y$lF8MGA3^?uIWmbl3<KQI2H<Ns#-(q5)!yOCGDD^ZQ(?^sZ1yO$z{w4|Gqr=Rq8hmbu5>1H&>(18vEe_tN{H?o@}Fn^E88U`H@!A{8M4LI|23bc>C~?HN0pOLuY56!={#&jmo5D@>TaA*f)K2(8Q=pK@OtM4bq-B$nxX15jgv5ZFN5AC7is>fros2UGtC76ME=A%57|WqiWtYaI?cKfegnX;e2tlF5qAOZ(?W*@@GCSaVp3VguLlc214>_v$>OuQHcEog>wr}tDBV}jz{RB!pKK%SqznO8r0Wur(xxCFC1<jvZMHeI?OhJpa5Hz7{Eb}j=Y01cICSjCO>j}aYq>6QX&;L<_}!KdUmuKa>bG)-x<q7K7(lSTB<=Y3q_`s<r0|(oNA6c{&stbb1CeAR(8daX%0|YQ0B&p`-9AA^pvsc^;C9FfO@tYt@IvwbxRRl7U8uhwl?*{Z-#WbitxwzHyD}Ci4uvuB-Ge9q927w~D6!#gYgWtXBMx#dX_cBll~hYxMh4tw1%v5Kj|he3@9A0Da(xy7rS(zV%evvECIr_!I^Hj9>&h`Gs3t3^YxzZ@XpB`kdR05iz%?Yy@jMakG?624reZGo4T8A#eWn|@kKnNZV|p@RgUM3s#znlGJ0K+wH@O8pR>E_DvIKe`G}toQ2cmF^MQwHH>R?TcO=6piTDqY@Xk4?ZVMo#$=23ud2*|_KaxS=ktcf2n<lBLqik=R&@FG(_r?MZRK)m2Sln=PY2|S45m$6@D8y#`g+u&eIHb30_o4XzNZ^ER}o%C0`c<P`9{|AOymFZ|K=eVD={24b9n{jcRp>dJ7^UC(X-!W8-q4JJ*vt>mOut1X2axs;ov6RzbY_xg8qQDSwIPWYpB1_st#JxD;JvtH`IPLctU}VhFLH)UjX&7YfPlDH%RlXVmQgjWgJwonR_ghPG&e&XG2mOScJAh316sxIwXA9XoEa|tQC>x^{>wfuTN?^~`#czyCw9Gz)=SOLwI<DP#tVcr7m~grf-;6i4Bb^K~z6rOoi6&*_&!}4y^ZXoR^zwD&HWLkn>Z^%#0t6-mNsQDnq`w6FHGLw%e+1<jniy8d1<L1FKBz7zN99?btYcU-A^@ufJI$Bz;(zaEH&UJD+cw(9hD&F|@dVDo4;jK5WEZq7rf<V#ap^muK>+MHB-{E+IUm7;s_Gu1d=fgw_WxPThK5&tiIC`UhZtH$fzVhDBb3L#@RnP#x}t~05UK4~CG=-0O=^oG2iBFF<ftL2j=LL&g<g+yyjW9g4M}7E88RuW-{gC~gRkRDkOg3nHHa?+k%mJ>4!uJ9FH&UG0+Q=(rRFjdEU(d8%)DB39c6zU?Asja8{2X2(Kl1ilNP)ZagIReH!t6PE^{+xs`jlVp+BJ>A%Q{ZzwLT)gY%>e1gv1#^5Qu5N_R{<7JwMQ?^VcbBf!?N=}Byc-wk-;xHoMwmNI!~P480<?=d9;Fgl~Q|4m|ZX0*vYfkH$CV`z`iUc`?dBjH7#IrAi-@o?^1j^z|q8`v*?ggWDcypF!{S48komaIOrgdl0a&vodISXUv0VZUY{K=t?y*P2eGF(oeNoIx=RyV*g&MxQ(=7p+~21zaeNC$##-u!~wi$x(eu{37d1!1a-km$qsG@*BU}GhL0i`35b(ra9=z*$BPF)$c#Mp8K^+;p1utT9gQ~U1mD|&%q`<)QIM2ZdAr2A#ZCw5;b7J1imWW`6Dt>_~YMBZBo(Xbuyp0e{qjz&XQ%NqN#tPaL^zmUhV_F>EKngDyM82`NZl1Qg<qL5$gV#x1VBSbiP_#?~U36%<YjEWoR`aJlIu2Q1|&*1Z_Xzj)7F0#kT@Hv4q(x-l~-5813gMeLi$HaWc0(k(ObxV{+BrGhE!1ARlSZskp9JVI6<_Ts=1b?GWN`3&UtH2(X|~{B9%cvUgO9EgYy$4-kk6A@<}1{Z?-uH*31B&EKyxcth!IT8^26>em`B`uU{i8#4!u;i%P)6F8+tj8h{g`|5U}<r{O-wF<)~TMagbPo(B{b}6|%q~zSStmnk3zEI1`z52oqE;SD-upI1Hma#EqZ2d~i2|jcB>pXVWq=Ep$1;hu1J08qe*-t-IyBzx6Q{pUFJ{BnXnpB|@p{4>BGr~B!hpZKCX-;7BN^+rD$s>$An0dK&0ze%48aDNBzqHQ_EmQJz_YzT>6G1hVj@dFXXq4T?3&n`M>I~H(TUl=<mDV+>*ms`_`ETF#8k?Bk2PYeK=pf4;=%bl=KGAt{2w|4ZvX2gie3u58>?jY9s0CYpm)JuKZ!@WvfZe7>L8#dYi({Uh%#<;KcQ8WVMPaKnSWAa}k^-hthQs!gE?4rt>B^PzD4b?iq4sQcWWWbdDdhLv7ur_Jv>}64Tkvck92TRYO{F|DqDOgqP5vm=2ri+`YWz2onA)NTOm;A%kFh79EI?r9t(X${H-X{4hYE`|IS$8q@z=&93_FY4PG%HPTz)_;TQ<)JpE91~orn8bjLO0w`&Q^mm!&o(Y!8v17i`Xyd$5?@&61A4TnA?K<mW!uP*32TM$R*)73?k74L`s?$5EPlg=t5;8kal0(b+UuO4lUPQ;kkP7aKMl0Q%VIIeU$Tb4WSQ|5i6tr;XaE?=6tMz5{w=SYN6n35B@#C#z<sSTSoB*JbrO%b-tG+&VPH^qpSK00O%8LpcBzChiDuEEs*k7fv1pdjok)g8!dKN&kk6rMTp|8vO`X_^@bwpg+mdn&=^IMB29Vy)qb;$(RH+LU?KJ&iiH1oV$MW)NJoEVY2r*{3QOYu^z>D);Oz_>`gR^mU^jqv8Ds`_oV=m6Vrx`JSH|J0Zw%ELJgKiIN%4b)_%Bn!G{3f6eJaC&N9TV4>vY8<);Y|11hLAd#Hs%EJP%+4M9)=4u3`&J|-hr3d+lDYS;@!92fZg4m*e2*Fk~pAtG8;JT(pVqx@T_DoF0I=Al^h2nqeCM@9h!$m5-?>hRJvreIPKdl@&Jqdb3m-<d@N3W8@>SKGLf_XM$9_1;nQEg66kfUSsyt6%tu)ctgoDlJ)|tYvE;*Y8|m<^s!ej;8+x8h5Tj<iH=Q&{4tU-1z4Q40QZ#!1Z3X*m^}JXO(Jp>r=5UU`PvDv&v2BhW~wTk?-0Xvs|PuOnD3Y`kNriO6J7DqGC_mN^E6mGYZ3_Svs~>V^nI$2Y_IhxqT^I&(F36Vt<qlMOu_vG^GL2*YMI3`N#Hj7qfhVP61Qx24{QZNlD>2L`ze>7dv2-%w;26^_y=g+f4<xa@_=@JtkH>4(1uUCb4f52$k9CY&TkKKNH!PL1#@>W~bG&Y{eJfXnhfi*GO%Oy!XVZWAK4`RXTPuC`{I7N%Fc|;Vm{ySx)3YA74K3fDfwJc5fRahQF7YTNZesH|^zWr}rM5?7AM+(MnYf`Lzv%N;QXSswN<LTNq)0{%}f4<mv8G_u#foKMLyaPxo7iNwyhz{6n@6Io#9XK)^1y7}ISBxcz^K+&sv;o4??VG{~<{KQ(Vztb2IR)>mtC<<j2uFRD@KQRpV3b;)6|98@jA6p1MsufqSMjrqW!ku#m-B-kx}*ui&l2^tyRCbZtr^Nwhf=qt9W^QEYY2mh->)4|~BD+C>N#oX2u5;TWc*^euRRccY%7lyj;?&NKqfPaK!Yd|O%{dGT}8L^HBr*ud})*`XJknZ(C4LFR`M7QG+^}^RESF=$XXQEii1F}m8CwjGkv!qNF8Dw+HhKBXj$WRx2hoo6Xg>1C)L36x<FA<=cHA;WcpBT=XNQxYygSfYeWLn-NYB;6Vl=YF$&*&?S(!$}*-FrpMQ+pGorDO(9VN+F5jW<jyXu@O6Svp#olraTA-m+{$7@}@1EMnKpli7QOp!90x?*QV!#h?jrRSp7Xd!+Myh-i0zf%}R#E;kMpf5zK6^3q1nl3kX_-JQ7Iw7+}F8TQG-f6q!;>TPD)CKZakNI96=wL~XXKv_kA(@Bpq_84l-5>LMcrJ+&$ha=Qq7oWF97{z<Y$swe{>wnL^z}$1L%NKU23Z~Rt>zzet7p+19?M7IH#~+RkD@vg=>p$OBier--KSa8$!1xe6#7+?^ukS-fi$RC1Ma)!M(++m%^b7~6<SZLwmLvQfgZh<(YZr}@)3FG?3%+r!vL2t=UcW8DVgJ6leL$~o<R)J_Po8rI^viX&dOe^UQSa`&FhyDHe|@&<?F_iGp#R)=_76mfS&B}zno(4~Ksf<pzB~*2;$#XXORk-~qj$P9&UTnMEx8WuU;@?I1b_mFdB4ZmJIH}!ma*P;EebacRXDFygYKE~E(VL)Dr9z4+PU{jLV*f!QWy}M-u$&tpy?VVxp2!Bg6>J0O=`c*nsx+}A3<(bUJJ#D;&36eYPiGA>6i57Ra|?zPByB^tT39(@ZZa*d&Ru8f38*un>+&<_1!MGx(ZhznXSDFqJ)`=a=#y<bD;_3EL&ZkK&*A87~RV~;_E756w2oNe|_=axznO#t_6yXzDoW-owwSX)I(?iKz_?F27aYuZbHYoByRsnDG`FK3L!FA)}IWY8Q}atyWMAo2GwE@V@ibI%}VD4kKgzDUIOgAmcO^m9W45Df3=L~ychiyu3YNZTu5I9IP3eW{Gm4^u^GWTwzO7;Nzm|pkMyBSZ$YZ$&OMrD>C5Erm?g<~@+IGJKz1;IHkghm?kP7moOaI1z0GpAx(g(IeIt9PW5q^|YO1W@hbZw;Umn+~BB4GAp#g)h{YiW>1=)`cJ}zPXeenY7^hBoTohf01QkkR|Q`{*fH!1ZmE#{vPgF+8qGP119ph_B-mghb1J-1SPotyMjaVDT8#3plG+H?JzuEM*`7U874*4bKzc0G98MhKy55WmSqwdjTxmbjG0nbCcI?63DnB`+>z;w4r7y7ZGY1riCgZbd0-r;Tu2wO1au#xVTT=R@i$glZYo!8phk4OzMF_CN5#zmI1Oc~df|KGEs4w%tVpPG9G#UAB%<G$>}s1D1x|a`ywUmp)mZ>9VjedhNHf>jhsOd_x&7l&q8q*m8}ZMlnP9fvJFehniz)NjHRjEd?YpVG;~qe3lFc!xRvKiI0Moe&56H3fT9v8HNhf6G;XT0DYWMFE4Gi@?o7Xwg%S}=DjMsHr2B6hX3tX`#%^$94MBwYNkCSwmX^R>xq6F?hFdtAs%tA{-a6Np#eOrKOEd$+J!&FwkSx7{7?S&l68Qnw?xgI2l;X<PF(?}<bTek;H6p?GgUaMU~z#s`r-Z4q(J~cf>Rl;vp)X)S#b{s$7Jh{80dysikQFf!1>P3L0b#0y(Y0qV3qYgsai~=2hsZRth(A`>^w5#?AKpzb8|5^oNzCr-}Q*Jm8hC>*Kaw3JIw75?J>@53$j)l1Rx}=^wG2pa)b%3N5c>|NH%T<>G?^)!uvQAUq#Gjq|jBW!c-MH00000RU$Gg^L6o!00DuN0jlEzHrzI{vBYQl0ssI200dcD\"], \"nb4_timeline.py\": [\"43a0848def257e70565285b6e172566a0480dc523c76eab100ed536a5a575f5e\", \"{Wp48S^xk9=GL@E0stWa8~^|S5YJf5;L}7rW?cXgXb*@(Krx}@rwH#q5Z`i_CznSd^pXjSWOYFQmMUNN`jlIeV2877X=c8K;UqF8@YR(mmmH!Xg3EJKlBDZ#shV5nX^7d{pA4*qx|RM}h4mj5C=335>JCqM9W8sP7P<BO=MHW5`v7(2C-y-ZQXFcRJI={NrbV{hhg@C%c^MOV(XQm-)VK*q78cC2WD9zLrjlXI+hhr@q>w_9mj^UexyLaJ=$jfCb7!+zA~)W+y*yE}@cO9Qvkv6i&KyvFS}@jHzvO4eFP#)H3oFCO#_emaxdK0NM6{JrIyT3r<vz`ets9E1=&NeLQ=n$tX9-O=)^YzD>_iJV#dKcaCwtR0b5W01IY?>jeN#bAbX_8)!TJeyj+^B^l>5PE*|~fM&DCVsnDPeCZQc9K&=3p}3^3eltd$SIR=a20mM;`seceC71W=pC17CW<FLRIpS?H7um84cvklb(T;e>CeU$K=oAc~D{3P5a1&kQXbvnNiKxmt`YwMqgiB!PJ^FnHtl7cehvKv6bM&$S;bJ5(k!nL4D*MtPe`7DTi57ZNed^02{G{W1#mHFp(H6-zRrEB?11tF_QURqFM1@DywpiwR@S(FgfW$08E439N#A@<9FH4h#KigY?GwJazzGkLrFnCh)W+A3g{^O9i|w-IzwZC^B{N>k*nX!}C8*+d`cwFDq2s3YZDo-S7U`MxL82GDj{)+-I#HE~NYY9K#4B%B<bP)}~(1iJB|ZL-{1wnVyz;`|v<aZp_!8nnm2M+}k*BdCAt4imUT3X57ylwxQKpX%jZ{-$UA}@6`^^1Ngg~le#wO9tzCx7%j%&)X1({+<=RNp+3B=qoq$YtX4`NrKmYNgJG)MJx4aEy~AE3X;caK;`%-FZi_GyP%@F-81cn{xC?{uNWyg3_97)7-GZpIT}zG`2Y3Q~?K|_P;<|KiIdzz%)j^29!28>n$Q3--dj4J?0z5r~{D+E_Ft5`lG|81Q?d!Ci3xoH8`_*i5(>5@JGxjrYYW#O4Z>_F_xezSv2giD^8G)h(AS77SWKeVd+EVvGPB8t~P9)!?u?diT@|}_*&Cex!CxkZtn2ORTwaHTkKcVQmV^Qm`ps$+Zj?A7(XL+>RPW=PIJg0lJOU7qydH&KByV0jGXP36sZ-9>q)9d_Ir$oC!WrDs>+#+N9C<kKvtulk)M?_^|LpT{pUeNMR>_w6Zi6gq_ypKV5!I5B9Yt&G(X8^9DuQ7KIq-0#!etlN@e^GaGD~h*FDioJ3YLi+}5xVYk11^I)V|9f@RiNJ~G_}DV4$G{-`ogyw`MRo;a%z=kU`tqmq7POM)lu0Luun=Dt&<ZAoe)H;oj`EopZCYPm7fJisuZEcRAr=YoG6fx4Gd_13UBy0P^$*xeRzv|e%ao68!aj$YehwIW^KN)-DKB>R6I*TDCV#BpgB=*KDy;CcF*xSE5cceiXiG1ard-HC2)W<He2Xcx{lt66u-76$EH4;22yF}3#G^f)5z&abkiWW^H)0&5z`#&sGRYc!D;_J7>q*PlGfInXbCfHP6L=y;y3^~)9A6RK<RZAJlSk^s~I>f^qhLCY~u>nM~i#l`EuH_tv$<2QYI!5_F&cT({yuBiXV%q!Dp;s6Hcd$pqn=u24n{!^|G_teX{{?{K5f=D!!?}E5IW3h1|9kT$Uu0LH>-#M<rqZ{N?NywdergESZQ+b=?f3<?8#=K_lhPM}|t>HtmN5)yBy)A}IJRqT|wE`0o91aP^EN=+3^$bUn(j=HaQSP%FQ36t4$1?cT;pq^Szc3~fE>?tE3&8#Kgq1wmBeJ3>jp9?)>mF>*-VD`+E_Iy32mv}eKYDM&F86ct^tH{f+_j66dBWY|GF`H9l<quK^of75Duau-_eRHyJkMQ`v0w*J-q_H?;E9}FnqCkwkfuS-resoy^g6=<}bmV3(HcyL{7*Bk#(_TWBjLX`&O^=uNUhebuPnAU7Ir~a05pLvy-UsElZ5$$)G#hD~B$jk2B%e<JZhAw6uWNc-5sb*u#OpUCVuk7igGg!BF-y=R&oU6icunf80pl)s@JIsOxWzChP<DGspq$}vC|CzPY7ZGul`#_)5=E2IF5Hk9n-rTY5b|vQ;)b3mYzGisrf~Ybyau<mu>rl56d{?#=dwOOI`@3z3+-A2&uy3?zV}<pGIS*>!)Gf(fMMt&f^0?fm0#%kNyLS;;=RFc~-_IO=<cVV#HhWM=<~mve^cSND*i0)jLv*yl0&zA8jW*g;Xa3>P$vx7MgS^k)f&XH`Iz0io^`$c;_6B&-hLR!4ID77WuGasVLCb$3!m_k99@alm6AM&ryeaS9Hk?G2F4aFF7S80~II0G#JUm>4>=FeCr`k<FOd>-(zxIt_oThGSb(0B-V}sEX5GcOx85AL2jw`GwOSwXZoTF~Ivsr@M7rf;*kK+38AZ_d1$8GlhNFcZ$EzZgR8LonXd#{@qA^-;N-Fz3xnDWgyys0*f9dya=lraIH8O)wAaRK)_t7gO%g^!fAA2QQcJs~zSW1^$i1jn`~(*1lIhJl(3V+XJ>Yk(K!i@^2ErP^ie8sE!jrQ4)jvAxBS=@YidazoZ+1L@2|aULsMS)kSC*zKw$RY`1QnU!@CVfm%ZWIjo|Q7thy0xOvK*C%g=p1h5FTfM+qcGR(Ih8BL`L&<xgbig}<B6VY@^~53JOCGuHa;o>%^vfTN0sJ>aP%!1Fofm}@{U4UOFB)aL+>~jxNthZ`QFlc5@d*K16bDf%mW2aqs*eHtu~(%7?z~ybgX&2{<<=B8Eb#HwHV=jcyYs<`ddQmmR44_UPKKQ+5^z*=lUor>8Xkbr<a{pAOr!ytR}zo^rHb+BFp)O}4V?Nkq(!4PLagm>54DkiVkaUf9QlHeBuTeg6Dlv%3crQfcXHkrU(;;;ajNq%yYYAoK!As$RikAyKTllgaa32Oz9WaTV*@`F5B3q!$>B?e9!?4=aT5)s00S=@#hBs*$2(mU%HH8hf4DwkF%G8NK`>fa;dm}SIZLJk`>w{{Gppn;k+sKzdi)JXm6SpgW`HNR+sgO{TD^GXY_Y|XA?aKfs#!aTD3dG$g6pT^u>yB`d~2~_rpv+(*Mn0FjbZ;|7)TCh>_zB%)6SHyWLGo9La!VXLB!i=6Dd%X+_ZP&ROUyc&0NW88?&=OGB{0b90xW%SQx8yWkyGRU;md(*OHg$M!PpW)CuZ0*5~=##l2E~M2`?k&9KA<8q6Io)h1%i!7}^xl$E#C3Ldr%CC~^BZvil*S0(^C2n($2l?bexh{dK@|Mc-z8(@JiR+5}zlk2`&9fS|cj#uOF&t>gGYyr^6f+-ud;YL@@1_;UXe?0HxdN9(y^cEWkZP&tdB#?rsb2mtvmR0@FyP%<#-8Hu!V1fsmaxMt(P7Hd(pW$;m$*|!aX~MGI9_uGklDGdI%cM}lp#^<N1fsc~!Gmz;<x-UA0x&u!_j|XjGoj0S$t3ep!0bU%v)OdVLwT#Z0W3aUY~+79-sXk!Q8+<h90FFoa^+-YQL!qWZ6ms4d9UG%eV|xsIP`vPrX9_m>?~bVQ6S&IWjmxs{~CWwEU0d%pV7%-^qm(|^Q2_JW3No6(W<%jgKX{l_HyPvE_i*%aq<_|qE{3wgwCq#z%(Tm`uiD0jrhVo^kcy_an+@aYl{}VpkWSs_Y1tEq0dG0Yztb&Fz=W7E<(^iKtxWLNKU_EEMJ%B|5#0R?Tc0ZC4*VM-L}*#eX#7(JoQ%TcqbG9&7zSVw9fdiEWf2*Y<{qyWed_3o=E~`V;anL_Mo*lqdGqY#FM^=+N+UV0BANo9`cYKgK!>TT1!91cXKvFAF-?`H4rJUjoor@Mt<;BpXbp|1x+bnI`FJLy-zV|t=J>$G!0p!xm*bTk6OO%gqe02-*2>}zcpxid0UJT;h+q5jV20Su4d;MvHU~!L7d_Cw~S^aQ4X{E-p@ey9yw4R`WEScDg!-oFn%9$=vWtySD%T$ESf`UewAYr6#|a37+lz>LE5v}R#aBcUGoR-w4ZsU)&q34_dxOtY5r|X<&PeJ|AZlT5h=DFXdPz8!*=4VrROM4;{P><WiFb(w|jUI{SPoEyDD(~QQB7~-c%dwdly4p1b6{R|G&P_dHwtz$r3e#U#sQ&wMFY)?_UE|f=)c=DhX{#E;V1fKX44^f%3Nq$+8swnj#^pJq|rTNg;Zg=3s=t4WUI`8cAY3evX>j+QB_-D)iGC3MKkZm=SQ4+U&*m+Wp?h;Y%eCSJo0D*4PRQoweU3kl~uN_kaChn|3bbok!ns7f6Qj;Hw@T0AWwvvlAelBo%@;9QvW`Vqb@lUI&iOA3=sDY~WqTaJ=#`=0T;v)AFge<XkuqYTn7=dbZ%#FxqZ+vlNH(t`~8PCwh391TTqcMMU<WtrcV#uX5bE$3~-xE53?NgHuD$_`lKwP+mD&SDd>Lb4#mVp1im`TSy-0FIqWGwyG6%%vPAUNG8k-%?$HqMWsoFTh~P{LBC{kV~A=j;{g>B+=k&D{r(N6nWbAnrPX22UJ5lfe9g%3AlxHcvr{zVgsgXn?}!6;e_|XON9^-dOB-g0`9RG3O>)E4U^kCtpGLtwAQceXLQ8M)Ayle5^o%0K&MA(I=aD{l;GAz!kJ9zP$8JP@hoeY={fzd@Nc<_7(a4)(Bg~3PH)7s;3+RoTDv%%7{v6;Us*r3vYv6AgS~3Ep{05*0CINZdzyJQ>{k{$kDmIP`KyB_58Zu8;_`m>HXr-Y-ECcy|NYzdK2Iwh)yq02?Vz)I7A#zTacV53>_#ywP`^rq$gx{%iCZ`3!p^6w&{fSI7p<gBQ#E9u<TsU4uW*EKFm_bKyO#^xQM!1b3Mv)vLSfLknEf}hFUSA~;6t{Sk7%65J!3~;VV<xkXC48!xi~8xG7-7JMy+8AU90M>^!uEX@*S>BM7U`8e!>u}q-IPbVYolJ<ghc*SsLk!G5P?cWID#AYQ3GTS)%Pg)NS`Lal1(av6A)oO<c?Pw6F=;3jw(X^8$8(=s)BAE+h|l>3CR0$=+VMcwxGPJZ#K;?3)49^imfA}WU8eVD5jfMXf7mMjoX-#Q1bcHCX=Or*d?lfGVVHr=)3X^kOp2kHUT_67u{*v!OK=P+!SfP#|Cr)ZZ!t_gs%7RhFCrgNutcpm^2*Z8i5uL+#^YVse*o1gD*Iz5S^<5vlp2`ok2Y9%VUJ5!}7lpx3K<=p}Er{z)$r0D|Igxc`)SwPhC&?%1#~dhIqOQy>Gh2{swKPu#)9eAoh0u&+W0@c8OL5I7~4~sEC(*FQSgL!?4x0$l+(dn7@7aZ;dwu@>(jFq2(d%*#U27m3v`ZiW-P&;Z3ox_Si~&n`&P<3ZyoBSF6^8-$33iW`vqL6Nj0Ab-!%Dm#p>5NI#btDwc}7D9Rs??qXswCm{R+2ese_SJX&-A3wK1yAR^@y)0HK_5g_NP7xM9+u#RVqYl`sRDRKTPNrI!Lt~Ei{hpF`_RqLlW?SA0;&RhX!VbpWyIomY@k4j->2((9`CuZj90Y(EFe=H${$>pNvp9Nj)bhh6vcybxUn{^C{c$|SSAl`L!^>v8PM@k_J&y&)%(=${8!_7I&j^yg@UsR$zZ_~`qvj^~cWPDWcL+xKYxQC)bpftt`Dxc+QR9i1P&?K-LBE!?n;3-Gv?Y&v02hh2hG`OrXx6r^xW<je+6cbBwN4XF63^2??AA<s0ksq96vOKx6i!5re1r@`j8W@~T0%%=h!}Jhk|-rc6YAf~U{2JLb@sT=a)`M%LaJsya5)cAcowoyTX_1%tXp$I%r?ujTzS=K`Tdi>UQ-v_@JV1Uihp8;>BU~`9#=-2DMprQnk{PvBb|!7F#(Pc5$53L?Sz1*5}pr~QM3YV%%?RlPh=Rk;JGV6ay&j8SzisYdAE1`lC1`Q4)PdsusE%Is%?pyr!UH;3u*&%UpE)`)c5Ek^mZ)7y1UfA-tg9R8)G*1kD5twX~I}rr)g->XZN&<(%Xh6zndGuux;Wj{XACgci?s@Zu}71kJvSdba_fvd-<Bk4t;=CbO9Ao1rdNo=llFWLk^_i(fPW2so}G|e{*{z<SOkehuoV|q_j=nG2mj3CWF_qjq=Ufxbg;F&lc7<uYWChf-Jf?ypF*pMzA@o9_`1X#N^gMgTEJwZyv3UwWX};y-{V8$*_s$91eSZqbLj}a2RQ3G3tJj2IlY22yk1fg?9Pzz`mB~gydIy%qjEiLLR}cy`7jLXyDWi7`YnJlwewul*SE&z9K9;A0tboE&2=2;MJkR3z|s<@p$-V_Kg9y8)~%4oTl0teOkV<>!NzF8A7;GYoXl!O=lbbJ{OP?Ujj#$?xFxL?%m$q21Uq<d9Wz6g%iCNLKRy61-ZTFIfyD}TCU*0n^F^u@l?*xr-E7KIbMILcs0e7E2t%P^>nC@Jatm;BjYm->x_}gzRa_tgo!))t5cF-rRL_(jpDv6MQ9Y*QAA*_6%)KA9=IC88Ru*y9JZr@ep=FDI6${t^}KFCL#j;{e04QxaYtWGt{I-X2-7PUPRwAHfGvfaiz*JlF)qW#6u>2}cDkk5wv2^FdltGVuY1y$lPd`<F1W#b!Uq*zmxC!JMU5b_n!L{aY-z!t$-_B+y?H*U@%~P2*N(kI%b<mtKeW<l9WH02ye3*}^_p5R=2@?8+5Q4*b|lM($i%4k6H~))-7e?B31=d*n}ON1O;U}XxV%~qy!-~N9+AY+Zc+(Ee%&dg;nf^}#^c3Zl>v!QULE1k7&-{}{wdiK!vt8f6eFdNui!?KB}UZ$c~z-k1af3;2VGre2U3`uKjJKf-n>Xp<*vJen(jN{hlb*0Nw%CQ#04`f;drXD!fnt-Z|A9E`9psAy;vo8IxNt0MjYdFmZ!y9LFUfDow;^9iIuSk-Qbo5q{^PVg2wy~b3_z5ieCm>u8EaA!$nY2bW|qoHMHX~CrbBYNS8AW>YBr0C^#`!#U%PF^*0?7FGt|8d;2rt#Wk^Uwm_cWa+*@SkYFRgA)$Ui8hyKlH?JXyAz^>99S?k3Nw)`>ZSz*YqBQ3sE-kW;GP3v*_pq8>fM3Zr8!`Y?C_dt$7y3Y}B3X1Q({u8^DECiux{~5eC#VZ_b4E22fMb=X4j(mfg>{2#z~1^nC!nQxXGZKx+;Hh9c6+NCiQkJ=;>H+_u9#CQDv(D>xls*Y^3UY#RS(9PVVv@sxlf$PMmCibrRXD+10u0=s9~_oc65qGCe=vLr^yJhYUZoK0nWhQS6;YSW5jkj1a%V4(>_6xKYQy&C(qv5h3xO30fi_1BXcx~W(cpo7P`8c0ZUSB^<LYsnX!**KQ~8(310p_3fHxMKkz7!WLY1zCDsrm!3Ypgd7v>F?+s{c+Am3!ES|izl0AH<QK3g>_6YAcj?Wr)d(=CIxxEM*O*dX}z;gK>_9VVyqv&KB<kD^#^KK(D67_YdBjWW(7@WM(pJt#$1og0GZ}Fri0xy5TIc3br(vl%-^k_%n*gj`85*ChW!zq&!eXe_R_=fpp(Pk?ymAfoYT#FP;6#3tYRNSy0F=4UB$TV6zEce&E`gm8glDKQ^=a=-p@U-XHDh}&}I;+x?&KX;VCJ>9rLTC5Jl&b)vuc<=Ues_nL6Q8V9oN4MNW}_tpG4-PBXW7X+MDJwj6(@;Dhl+e~bKtCr<HAY%U3D0wrYOR^!dV#ICZ<k1C;j~QKTUh->C!4fVZTY=;*8v3wcE%5R5`@$94oEy{tTQ3AWAnA>B%AYnXdg{n{xSC(9!UTjOQ%tP9aTFHT=^cEDB9^c=BF3PzEO52!EP}!qLotw$RzvBbM>-$HiwLFvV=g&e!m@!pNbA6}^AWX)=ZL^nhfJ-i+`3O0Rs299Wmg(9hHO$m4E{rJ7RG?&Ns;PMXg|(X%?E9;f4?up&w1lWuN#!B;yxRV&A`?B`a1PadhigHb=IW>8%_azJvGR{_6>G0;78r>XTY4O+Bfib^RD17f9QJvy94mW;s&U3^XX>j#iW`u|;#JDDH@aGO#Y`A}LUb*h&P{O2<}KASSsS(m>k6*`2rLz~yx+kBqto)qL+PHOXhGpUC<gO!KxPLJQ9u4Yq6WIMv9@!sxMgHr@1;^7KKX~Ib|uL|*Kis5|VHelN=0|}iAUx4oEB5W-VQ{1ioq16L_)lN>Owo&Ndgg%2n^(h6wE%FccZ<-Qg@gRSpMAm1^`6PxIKLVMw+}=Hk_EWNMXbD>Owc(&N=`h@DX`mYetKRn2qTg<^r!ECnd4m&_J@1&sR6@{zL2&=0d{V8rF7jub57QbwFw)dmb0143GEWyo?2!lz0%Wl%`eYnmQbZqzRxF3KZu!}tOU;RIkGa2w?>ctkx|_4dp{?;%7$e~Sz!D5g3)@N{)AB5Ge6ic65M?ZsL({EVJ}{1HmJ{^%Iz4q^y^&xR|D&U+K;av}z+o|qn_##ZZci-~_Riv`U{`Bzh7Z}XD&_L4I+OSnUjDkp;PXE7fe2j;HxcFQx8tv6X6@D;JDIeETTL5y8El}XO(_P2$1!7aA}EdmY}akr1E0M&o>S5cL@=ehq5AD{ab=DpQJi8p2kx?JgJh4k1Jtq+rSVDxVg)BMth^!}H=}Sa_2A@~BxW{_yXTCz?a1KA^q^kSU{yth!HY;qg2s;odgE+n?UakN-0(dFoIYstSg#%-&wCI%D!MJPi;0T3-r)J4$AFYv*s*(o=Zis%7DD}PqnA6z`VY30c;E&{@Prt1*}T2%ug981rO^&L2izVU`MpUlj;^*4;JmL-&HeU4JUz^y<+ThYX}Mt=5Vqv5WSmE}oW9|zen+Ou4LMpW1%d6xK)$#Ks;DzZ!$$(bkYnaD6?c}hKclN=u5NruD*C;SW9BvgGtf%4EyWxFl?N}w9M3EQK$ws%7N^9^N|owgxTU`}S=UbrL-k`<GwSSi%=s3GE=mflM(c0U9Ae+w`{M_+uPW7uW7H3LAwwXVquuTI@3rjfeTxaq0F=TJ`Xycns+|B0c%IWUU;H4TIX?eIY8m1#UV`3I5E{uQ*dSYWUlDON1QNX)ZEOx4LGjmQUK3tm=HhCau2(~XhxTzsU`A=F*z!wiS*k1s9J@1k?{W`a3P3@bfk#q?u00Y7jP~3cIKO_fknp^gqyv^@m6_l)P<?EA48tMMXv;%heVm|`Mv^kQmhpErsZcGJT|$omt7f50B4~lq9nq^R91e#51k18u76EM9QY<4~Jg5^N^QMZ?u>h`r=cK;)<jL&7?8}r_YI87ikux<&(&#A>@ms{CJBw&I^2b7;qA^>L#hDu*xeUzZTnxH8O=V2A^7L(|JBjk3PFo4N3UIo7W@oCGi70N|W#m}OV+9!{WrdaGHhw*Io9!>rVPJykLN>tXAr~~??mYGP*RFnzTI~dMwAb^f9yN$;Tft~Zffu=&*>9@dfcpV{1gpicT-xS;j#Y|oc6BFvHk|vf!ECO7$$^2rYuP!WIuVccTpfHGGBkttYy)}nIrH4qjer82hiB3Q?J=<-SYU-JA0@Itp~qX%2)c4N5;lH{#$bkAE7sqaqDM*PEW(qzAfUF|k!oPq8H2Ey(3{F1l`L=t7Tky4kLUeD)o@dJsFxX|N!iQ!--Jw5NbtA&ijzgx=V`We26<{{+GZ|m;O}O7YhN>ijw#9xlX|se>qf*_7XH=+Xdk>o&013U{34rjb*dj;nA@BHMvpoo?;dB_Anou%+h=svwvH=r6rT`dWE&>S*^d%>p2KgO@D&u~-9D8g-)N&)d<%V@(cJ_HO>1(7c!sr+Mw@k>WueDQQ3!gp#9_4P&BvOG!P|@@LG|<iw``4XZde_O=Yig2G5wqtM2K5$tZTE>U!;wzFQ=0@a0&}@(L7?8IUbyag*Z7sewNlzS!YbuMb|g|%!NR*s8k=|_IlS<awY#GS(ruiS!WXkXuQj?C_srenaqp&O)ALKi7iH4#}Rg=1!r-j17m|vj_f@MDcb;TFL}vtEsahB9xd32*`6Mex0RM4&7!QCUg)&o&-w4X&6zYI!v)dxZ#12+GC=7ulb_yNrcnRwS^sJcszr42E$sg0zyRs4?3n>XUhQP*u_76)K>5d#YWs~f>3`SuMe^$@CJ6s~Nfl#N$rQb|xv#*?v+|><cgMr)3@R7q$xoJAYohBV&u~_07I533dcw=%gqs1}Vo<)sIASBk^t>&}S@)Gvr43-6rnFfE7jm-VS*KT2&-V1}tM9nmVF<6T$rVxbNYI9Gx2}NvUa`7Ul>{kY%2$Zwjn+G6nC8G75;GK=$8+zPc_vw&XD;{S=8y)}g~&SY0}qfRO_XB_Qee(Gt`s-YdP&0Na9)y}A`!ut&TsvqJHEY<*TH{2SR<r#+WJRXJ8BMK3q3eA{F{KegI!ZO5~i)OAd;CAbDSOG-;!kP7prMJU8hsIljhpswuc@{*4YlYy>IK~t6#v}b#RHl34a@}phLjYDP6OF`i)t2#>mgrMZRP!Wn<DF0HNCb9N*Np&7|}Y{1RLm;^k6jd@`N_VoYgb&j8I}=VU=n9bTuaa0j1MYy7I4Z3-uZ>S`hl;1pmBFXevz2X|ghyc|q$mD#~(f)FO=K9J3zBI5fo>m({<6QtzJC^F-7+~jOpM~YFvCd`juG3W3G{IxR=QDuI}AI8##H)avZ7_>jI&l$|?0iD&Z>;o(t^=0H56_7+a8v7#mbFp1HKkc<)FTY4ZdvSnRbc*G?pHDbuU!w3i5(Zx0H8ij6Gd!hC{D<0cMumxvIiNUtv2L(Fs3mZXY@_P($|MVAwo5nEKRbVgD055Er2O2`KdP-6SU>o@{V0+Y)kSK_)XYbh1waGVigmTVlR-HkPI3oXg=e4C6}8FeZ}D~dh2Og0Y-<ZP&%qej$|qW(=ftJXcl721sG|ZKsyNg(7`4#h{`f85PXN95@lZ;078CH8>RC)*GjpkI?u?X0m5;XdU-7bHb75w%ort|Yyq}bAKxbtj$YnQEdkSvMbNx_tvk0Z`mo0a(o_)t)B^pyB&pkeg!i7Ee>7D4OkR|i{A(>{`5aq-IenHoEv&gBy$$K&5MbPQh7S7#*+vB48Y?frrfm2Qe4rpbrCO0=Hu}7#NU^zB|O{Umb?8<_PkO)gLb*Clj8YdnRY>Oh}gK8z3*&m<>;3((3OirRCKoP>eoC%W#WvdyX_hO5NInelPncFTDP+Q}7IF-Ww;tZ@LcU|xz+*%!ilp)<1z-3+CVrPSF!G-#6OkhQ}#F<8;pSjbl;iMr(&p9Xt`zIl2=rykGUF5ne(o`XkzQ4!mjz!_OwGYZ0STIJ=_`<r?x7_2d0-?#r5rYN;)K=HT;t=?nf+Mj4Z@RK>+vdDA3F=C{YZ#PfMxasLmo70imSyf&VTBqSKf#RY`gED~*$g2zGX<|d-Ayc66$gqDF9S-$p|R7&7cVmO$JeJgF3iP_3LcF4)5rsK`cqA)n}p?Q6!u3-Y-_WFGuF)kuO!eX+1}Gp?=0RT`BmHm=Xo>99rp18XnoGh%))gyshX^bj--u1!EiH=;@<*l>xv(@QCXEtU$j1I<JmZq&JxXV7nWF;$Mg|W6R$%>?aq1JQ+r1W%YrB`Yb7&R6zM4iVH3)n<|o}oTEGt{oL1s#J|K{A6~Rc^eP%m2s+%{Lnv8tV0?NBvz0Ia9uWiT!_de0$#AP;Yw(nx?BicAfwdFCK;d*Y?&MJgRt`F*ypq%&9aoDAmpD#Br685-(hw@dZWJAf+-ZrQ2`=@0q@d%AI%xkgk#?;2=ggqP7bZNo{LXvk>t}DS2{ew!CU%5SO8Z;h#NzGstLv+kHLW&_u{43TnJeoUd*HmFe8Xs1$)3mxhCw^PA;21!#_!+YoIm%iSjZMbNC%p2w=y2!Pzo;UP7FgB{TP`P58{5kNhR$@A|0V9S_LJxQA&I_dN-|2iuH|^iKMpbKwFvGJP2;`bpr0^*7x#6U&1r}EE>6rrUuM3K@>+ID9{LTkhfOJXOm*su;G-T3$U<d|xw<eZwpc_VCN759+<MK+I`3*7bp7`WNBCti(q1-^i1av&G+jJOe)IY*+<+Ro!Tm7i3NrZqil~=n#<8x-Xa}@Hxpza+ASL*~AYB|#Co99>hCsos0j&^WPdnQ9xY;3*epMEk%mp%|apy%dA6B1^2~bJ#x{d7+Mrd<&sAv`_dW6s2s*4SIKZxmtu)fF#xcBro<z9E4@J>>hNK$hL5~pqbZraiGSq%y**{%;@6&w6pzz!gk&4+2rPMY3{g@yP0vWe`iz_-A<VK+2pl#zG20ph<99OLBR%K6xt=|E=4bdy`Ov}=I@N6`Y4Fcq@#vfp|ymh$QtL#uZL<WfgqbcHih@$RHpA{MztyJgCZ$O9XCUN~zakjstl_oL_TVvwINn3{;o)kU@AKKt=6j^rk%bWkyN#R=N4S93s5QAIZFv1DlhNd7X#NO|+myDgj-pDb!RBTF<HBH?_-UOgn3R?%fFlnD%C%I`;##&E&f&x=%Z;ul3g^CLgGK2>`F?E|R`!=iJRJA*oJP*1oXSr<4v^+sX;ZTZ71r=3rLL=e67b`>hYv&w|b>F>*6NCfZ=*V2uZsJ^hr<hn9qR7V^%TV=gGeE0=o6<5KH)c8iufK7%X7HBK*fPFC&tAuTj>KuMv23YaSp(=l5+3_|z>v$k)Ir{r{NFjBW{p5|w`1Q1gEh{5oyt3WIPS={Jcc<E@^qX5%EasDu>o$I-w8PUxz(#^K3XV2lC7;GryGQu`WRt_zv;8#}?*iFmil3~U8o18>6Y@S5Xv;9T^%jP}wi$}iBf<~I9lf0Dp&HRGMi?|3Usu!ngsIaSPQ`>KR`H<{*qo(1&;{Scr^3=;5%;~5o~S#lyy+FbsS!mlykpPctZS|--LhU$0Q=+-KJSE{84yoa!y`+>FKy%PTDCn!kyb${?iF#1Z31Mmay>0=cLaeCZ`8{*IxB-$VmLNi`Ezf2_Gn)&7>c=GU;Q5_?g2P!*OO{B;-VY#K5G%QJKo^OJX+T&GJ&;sK;S=;RtkkX{({|*4Vc||X_sQbPo{oAH}2}hw=F5ikLz3D54sjt8$q<;^acWaaR5H?)66-9!1vhw5Zdtl%O|be@*#?W2jQbV*{084l~1MBq%yxL&T^~gi-><PWcKU+W-Mj!w%6GcjN#0ZeYcTpTIn%uQvcwY^g=O!djj%w-x`Hwz2rLRCo@-%UOrfFJ2Nl|wjRjc6zB!BCfUkn?6@q4LPfy0rH~5ej&=veT1=Vy1Rz|Ig*xPox2N7Z*{?0jQ>6KJ*i}W1y&bcnNE$`~fp47z$~a@0gdRVFu1&@vohLgf`jpCr$au9tno1`Vkj)f|0{M@$MKzbnAwtFs#MBvf2=XJQRQMjC{LM;*86F2Z{x3yRC>p-9SE>Z8$%IIGuT=Db?<DOxYdNK#iPC@eF;`y2A)25tQ<dr&3;@E@x0hn={dlBuYFI`i1d7Np=--$|OB;5=OIyG#f$!^FXWCgfO08><OV<Ncv%OBr3D~9?K1lsuX<#`0;%G}QX6G_iRK?pQ^V91CXw1)i+bw|Xekbk0rkRk0+~H->w<~Ng$o1tsU7+|L(kZ{4i=;faOqyZg?WLOtpAc+^jfL()KhFQ0l7uI0hzhVwcr||!>E|8BUozWTjBF|4^jNP0U#Wc}rAk~%_<X^4`oo=BdyOiae)HAp*&R@tFor;|??3|uoU4ksUA1p5FCRplG!AUXG5@7&z@b)j<t>b~dz;bMk4A*$HCF%!ZH{}t%wE-aR@dOx3IOwygGnu`_@d}0%g~S;jRb$eZ`cLDFL`!C5T6%-wMkItS#B{rRz;*-DD!f6g;}N}LjHUG0)lMC-#_sRQmu$1zL($F`i1|TbeJ{?G!b`}E8Fkbw*n^bZ_oD1uiOy<$tBoit-1-RP;=^Z`Q#M(kkIM1@ocMbAA6`x6Vvo<*9fv99IWJTdp7)auCbM_;v-1J`eT()qx*&^`z@BY`AtZk4x=D)-glk22%N!jk6-IG#*idPyZ^xc7|V`pG1N|ywu%PZcpjjIR&R~oe;0;Fzc|nK^{E%}7>j}s6JXRv|3G7q;h4l)LH+zJr~ya5Q>g!@s1*z#^YN6xPz1|(4&=G*LOK3AcvWc<f<sv;;1w3H&4n=&TKwU!sgShA+f#yp*IDRW-0Ey#uPD&v=07<;q+84{E<u2Dc66|AzSWyW=*+rW{8aHz2i>4s4u&%C7u7^=&f=4jA%!7gbGD}@xKQa0N01XwXA}dT!T}Tb{Y@$~D7n2m+T^M@CpRXaFH5TqqKF@w{@3K>7qZNjCS*v@yjIoie#j*)o*E2mvb6iVPA2{hT=wGkd`gI)_;ZeLaW196_{ft9u;_=mtGgZN{rSX7!35^$X!l+$8SuO|oQ$F<B{T$ec`#AEQai!oGgF9EpQr1;3u=bY1@#)s{fiILcw~{Ieb#XQ!x3@wH5uztc-rt!oDi0bKN|uLE55uZ{#78<_StG$ZSifEfzVNSfFTUdxS94}*-X<upg^6K%oUIVh-pRORD}HM&v+IY*F^Q7LI8cDwziE$ll;U&mVmJ!J;W?9x<i~kkC-Hxf4O$~L0a^WGK*^l5`||MDgzSj35NWkYd<McF-wKmiDa1`(}5AIDdXCllyl@1o^tbz44Z((@y*R^uHdUoJCh2!%8X7CLf_(Q2fPe?E?h(491w1}>D$K-T8WZi{-b|fVZ6`OYk2bhA^fH&BOK?Z1$gHbrMS&8>s-h#9oK<Y!keEwifEvVwumIHL!!yIAJ6*P#8pp}xOO6P03h=93P$uUI8(iKgHqwDFpO|e>GaHCh>11873L{2RdFf)ciEFmhuvH<TJa8S^hT+d9Y!d$9>dwq|B&b031tX`e;kJeeM&8i?K?(piwi_RL?(cEJw<N?a&c)+&W5EejMIFce48%)7JW}$xVzQug-A?iD1g;2tO8KK*<|Q%y1z?5hS0UTS??;X5`mxgN4@k}0S?3b_Ex7KMsnDy)_rA);vTCS=VVL+imL6A{IQ8kv^$Zr9s67Y;AMUP8MlCW2~O~O#t<sNxv0O)B@rctohz8~u&voP(n$;F3=*m07oD0-8R530Sp2F!Bm{Me9K9y7phQ<Q2^$gby|eiUa=kp-0pljmg72(E_Fa*_^kYD-Jdq$2amFm>f|b}^!im};YN}j&6Sna`I0&uUp$>e*>X@30sl)b0?>O*|UX|IAY5?wA+88^koaOSRkIhIU>)ipI*+%+8y*q4?x%ZXAOAV*V-WqiMU(})`^<RB!=qd~CQ^VcNF!y;^TeF}<=C|h`wKEngtog4JM4BtyHITIyXKqA+%t5oUNsrrLqs!gllH*(j15zH1wypVu?p9XwFflR`HeDuK+6^k{$I@PlYT|FEFbF{vM0k<bvSv2kVBG-OfK}4RkY1C<g`n@sQVh(B9U+O(_-rZH-?@LxQ{c)x&C{cbIK7yBgw?kUF$@dx?oP5j_)cypjQy*hm!8*&)#%*Z^-&Ejte*wx1%DqHYBoD6Xf4DrGJji&%^L9|%ctxF4%R;8c@&Y45pgjr$jxcE%3~#5F3$x<y^xA#{LQZZJ-4`Z5K(|n%)+$hJN|*-{rSj@TBKsTd--OH0W*4klmNe&MF0$LLkrm=ai=k8s_V1dO>a%w^lVw<HPY?h@Ljf&^+G*B6*BSjx$FE7fhQ>}7rqJr>wDD+ptI6T*?h?OB-?DkfitPmGJQfySD(aX7EyUksS3E0<#a6~{b<_dwbCI4^({K#7a|PM!-2dL+Khr7DDW#~_dyejg?GC9wAZR!4j9$NH?ypx+hNoK=qUKxYYQpv*O4p>FXJ^zAGn%EOm>E0zE>w<S?%?j_4;O^hUY=9TdsqHg8}R}w^88WFE}$9XF{~Jys+}{vr2~N&oN9fsPkP*2gNDXmYTjPp{y$#O*k&y+R|X9OFlX=G%f8$1#PtX6t&$7nDY(v;|^iMX4g93&UWq2+v3{oySGp}5Gc{Ik(HDbACh*;Y5~YSQ2X|8;7X$Utd;^00MzyQk<zqLS9L>ybM*Tla?8AyiX24rX#%^3@ScFG%kx}q^@Mh<5?hv>oBU{W1vgksfwfFk4R>A1U4lHfZbzGD!fTbcw$f%$4jerH6}~JbL3{Lyo9g7}*=0#v!?KuRfPk0hdUI4~v&Xdxl+5mJJ9>j%9tC9q3D*$o09w!`#3@axPXsgM>Yy2HM^8(B7lE3_ArX+qGXDhbfISQ=W297u{w*knV$8QCMFztD^H_CR-fZf<zoEtMrPMqS=BJ`K6PkWoOzj#SH&fsA&6Bvfd)ri7^dowa+#s-eWjf_&^-L(S&Yude7@0_JHtIko0^O}<14h-4=7}wmu^^VXLhYnIxo7(x`X$cdIQ3HDW<|^&WHg^S=J^K{JgI2eHWe??K@{rL)dhv?Q&>r1P3_OuCBRUoD{KK!@WjL-_Z;E|kNKNu``_{Z489tC^%tZIpNf>s0(oyl+=j8#V0aCf6v7D?QeP#2<@^gY<4d?{j=Z%Z=eiW1_aKB1v_Iayf&TL~$v9pU1$tPnFG_szR4p8uwj4NhZr3fNDh~JiYp+~f+P1II&hy`$L{FE>Vflf-f<T{=TC#o&&jS3gY@K_P03}y$;P%hOa!$ux;k`l3X4jBywNqMB4aV{4cA21NT5?JT2dtj!@UXK&iY>u0$|a#cV~2r6L4svsBLm`sP9DxkKcAyyFJ}-2mzike6kTCcJ!k|#Rx&zUl>DD^Dk2!`>vaWdV^BN+fs!-My`CA>&t)8j>L$=PtPbADHTCtFh~5y{E6tOzYtRKjsQ$H;T_6t|GdboYvEKZ<{`IN=T^-Y@qt41RBR=?QjbITGik?31+-H{D_y?VS1IvWJ4)MZ)b^~@VL_CxcAagAd+EUTXcOP$x%~XRtE;eVr<JCU;05_#boxRd7!(3nhbUnDY76;0jBUJ~odQD?Y(57KG#5cES4X;L*FbnDAd<Z9fnp|ex^Db6qvAFA%6rX6Hx<>fNV^S=M2S!D0!*8r5GH3y3)<5ueTh$TvlYi<d7)zZ=8v=KywU|w!*PNkZHNG42^YfpZqt|!@90P*184{0^QJ!<n$_jam(x)JRiwb-RMq#7JJuU9w-Jg-LN7q+O>QIn0T#-ooZTZ|`fssGoDmbOGQgKoux*Rq(4+tiDlE#KPRMBqa3GhGLF>JkC2_V2*$>4DQ^D%@__Oea?tnQ>16Zk9SJb~u@!{J53H<VbN)D8(gWQ>hiKh8xn5Rkg_5k1jzo7N-HfWkWVxU#LDB#uo0>l4g9!N146mkOwP{p({8`~{id36p_bCVWLzo>UxUa%j#9Nrbo#WlYPqbUp^NaFP!#7<5$o8!MO81K=Q91`JI3>kV+_Vt23ZM+Au}%7{1P*by&rY4S>F_|z?NMvFT*YJSUc@xH@QG94<Yx)5~$!ALfge%Lw3Du`6@?WMnl3#-3(+yg;o<u<jr=1H&nH3<B>5=pZUC%{s4<qd{W**8ole3fQ&sIEne&a9v=Ba{goc2x%6juEO6|IuUzgA@)3zdcH)t7QLj$Q5n1<;>LFKY!}Ho{1P{U$3+Pu8KD=-g=q4-+uIBA?e8&BxaYUzz`MrJ>gmVn=pr@p#__l>?h*{9?h-$1F$tjP}iiq5EbU=Gn6StIEs9p4_Ri(4+dd?nC6GL1C%Mu;~*?cXC*<9l2Xj9ik(_9);dq{O3&COT%_9sMeCKS=3>>=@tRJIq~ZUQm;Aommn+VzL_!pmJ*=OI{&;k(XxJ=nJCpjZ53`tvlIl?RCB!JO#K|R6_~G9hp@Q%hd<3+taLMU}NBGA`X;}+dEuz3L*p-ux?ksKj+#fAfc43#mm{NjV8k@$NNA-gjq_m+(kA4dJ)hcw*hpRAZudz6cGosZd%o6K<%{0*KF@7=MX{z}H>S06gsS(pwE2z*nYzB_jdsAWVKHH+h;hCS9v+o2_!i6-PNBTK+XhdcJ260s@R^>#$qf5|S&8jd<7{Vv<Vsm<#6uL7t&Er5TGJ>cYLuei9ZX4|rXZ>^HO$UBvh0j>X1oPL7oFLQIu)^0i<ll1h$M^7CNEHYQmL=+3YBj_h=#))J*$}q3B8|{!?FVqlQNZ6=TaC+R(`XJst!5Az95nT|e0-oku+lnx*-pGPm>P_Op9sP|Uyd5kiET<SbChl^{d_W!N6){yX69BRKKhv@6<48r)~4kD&6_$UdS`g(U#g0iFBFm?7Wr6K>fPF~{M{&n61#Y?^j06^E`v*nuu2ms^HcY2jQ^^NPeStYwJI3Ic)hX8Gf~|}6uIx9;4aPtAXv9Ct_N7Hf*Yf3+Bzv<w(kkcJKky$njNRDfXHgS8B^WpvkV$F85)a~vPG4cI}s14Z;9go9i(9QYBhtrf)%abQ+sOVHir&XquS?41}qLm2%ogDQVwVRea>(ha@x@W7x20m2XQ?q&AqJDWM>uQo_J`l?Be4Wpo>P{x_%!n+V2c*ad2#Qf^DQweP1bd$b_`YhZflJ*sGG`BlDvdYXV3b2zV5E;V@XjHnQB63t@fLJUA|)b1naK;&3*Dx*3)E=USv0NLm!z><cbY%Y#*dLMuU-$Y3&y`0dto^p8lEJy~mk#GC8w&5q=wS@4PC`i6Ssi_lSaFX5iO9;R+f)O(Bv>hTFC1%<sIbX>-@5`H7KKb=eS3wQ|fSwU(lBa-WGn<BZV&?9{uo~J`|ZT7z^!g9==uu<zq1$Q1nDa5br%|5l+92wyfLu+`SzAKk=cPJ^20eb>u^%a+Qo0)LEilB4?TqBGEExJHXm!Og5S@tubToL5WJEO*ywG=?J+>}LA(EC6~GY_?tr#IXCh^LGCNOer1o2Ob{EVnjZ@!=-jRkrR!wuo{^W%M}G<8yuQn%LJZV=O^?I~QTWYPP#%6v9;&=WYiS0X4d$EZg}2%6KFGP0+<nr6+uCL!C(1(tM9LN!Ms0<!{F_XIs@tT98RjQ=s}?TLd^@kc-Ph=D>+UVG*UNc!!@$hRK8~k)b&!={Zib_tW4_dpS}fc6Qg>ypF^z%xEQ+MGuWx4nF8BY}y`yn|trz%Fh{VD7Gqy82p^;k*0vFC(E~Fv<=VX1Wx||>1MFaxTPacFKW{Up%+XSwgb)<FNQVhA|gYXpG^JGKI(NQUBY7^sh?<Y1N>nm7p%7%;RHXn^wR{tX1Igp5)Q6{ETf7qn<4b<x+^L7K_U4M5oaee@0gC~6*2Yxm0z}EE7SfAfmfR)AOk1H9KRa=mo$TI%NT=WnS@^bTgG{MGwIUw3aR%$373;Ud|~+IV_ft>M^gvOp+LyIwSpx&G%9yqRKa2RH<?WEUmtg4>q*zZ@vg&Rg{~ZIlQw!n&Pl-Wh1!WU$MC#Az@Y3=M0k0{G8NAX1Kr&LZrC;3ZZ3PPa`J3?m#D89AYuVlgUY>}J@uiIaNgaT;3G(6l~M`+{_7A8`0!@P!II|$J_ggzE_*R#XmWN&t{QVtm*G_zDDd%Owo!4l<<!dHs7QOxoRyptBOKQvxDk%zBz@-q17^Vo`%GbOVXl<+0TCVYzWcl8th_~oXTB7`jw9Rv#Ai&>n`q1k>a;n;<#BE@5LwTSNY8iPX}R{T#1%L1s4LDNFrvj^*Uo1Ok$T=P0^R(OR`HD)h_4*<+p4Fd&Es&8o+z#9_HJp9bGk^NarG1VaPa+%ex92h_siW(6tcj5AS<=)I!h(d)NT<cz#|PmlXejJNSq8s=^@M#jpckW@Dt3cn8KRj)0g#6M|74KtQ$8$nM8%N3s3wWn^i}E&@lOUd^PzqlvK^Lq-gsmVnGbycn%=j$Zl69!~!Bv1|xM7>|45ss@scb)(Bxn2@a`#EaWeCNi)RtTj#J;SwI7WDovi*!Cv^2Bj@&fh|${8D@sm&p4>*Wj&B2|#=Al4&;S4c3N?YXpLCTo00Dw|#ij!Qm5-#;vBYQl0ssI200dcD\"], \"nb4_ppack.py\": [\"6114f505322564170ce134eec91a5e51f16a328f64be45ba08a7faca0eec5127\", \"{Wp48S^xk9=GL@E0stWa8~^|S5YJf5;ClumoLvABXbna~Q3)_)cEP3KbfXE9&SXJMV<=hB6!@)dJk{Re`aCPtYX_C1!aYh_9-NXG4tLHXzLaOnCEN-uVY6*S1s_8BBdD;N<8>(E`kZR(i(1>Kcfij0^X~`&9tJ{S9<6D7pZ4W{bxa-QKQA8vpyYtc<dI5OO?OMj-&)Pc7^!#mRhmRW_r&Wcg_Wrv`toa1_q5hRBg|;VNV!4%C8K`aW#{6vLMQNRHBft2DFF-0*xk<hak(>dTT9YBqDDu{`nLrJCFfvSGB^SnIhVV)?3JUXzi4oI$>`$`hS@dQYOOTpyK*=1Ug`let)V0#eYpDSYOC^=0ua}uio0ZSNAsFrol+O2@W04i^K91gRKB~}Ari2_ThU3}Mpi$l)wL&FZ6)*hIrclU3g+tKfCtCU9d2eK&A7~qYpu_{Pi?i8BYU;jYPT4Wpv;5~{gvOFINWN$UYEm?lI#@AzU-m=L=2n9cJ{aDZXwMl<-1U?-1{Ti9EWI3)g&$5y<<XzlMdbQ)Rzi<NM8ERF>iA=FeF?JifFuBR|8HfXDqMuak)@zg3R2&GJm`dtEgmlu{E<l3&c3iIXu@_(inSCzpZ8=x}yyY2eZ+yoAtuN(Uc_m58|*(TQ<W83hqwvJvjj-nW`T|a?f-15SjRhi6h%UK)C>TViKD*ZBVlG1D|NQ;;$uzBRffe)AfQMRCy>uFQscGyCg*x4e?4UMj=z&zocB_{l%5!axH2Cfdo4Sg5#NjqPFZ2)u>$tBAb13%jb=k6k>faG&905G)y6TBx;})5_BmFx_9h_6ugvY`X&*%yQFw*tpFI#7H>J|;s7kfZkXi{{f&{Y=@{83Nb;L|=f>b@c<8rB25>#@1KwA`uzi=K=H>%`27vWluPbnlbVSrr$W%1HfAty~7QLNDa1`uuiR@bs*=2qw9lFQ(+uSoANKZK{EE-$-$bzq#M>3l&0mf0P4({r%R5WF@??3iG;A=lEchbr$+oQ+DT8(=rr!2^UUHDPF%x1Fkn?sXjEi(S|hN~0jWT*ikiP`g7a}o~~C?a*Pr_w&@_wj`CKj$zwpn+xFm_OtePhQd}v~ucEWGp(z9XIMIIya;M@nElRJu?q`sHZsMrdq;Qgv+Q^5kD61rm?BS?e4UyGbZbv#xZ&xnpkp`g4@HgczH(=q?s!sPONj4k8crBdZF!zxRUh=!B^!9ttXy8p>*~P4W23@;2M-sfU<G4IgK-I^re~bbSnrRhq(4t0$lOzI0Imd6ooX#pR}tedCrG_T&kw+Rki(StU`M2XiNcYenjzU_hCQkiU#$Z9H|-A{<(}op|xXXO_9pM0OG)#25K`Erdn5zeN?l%G)gm8*Ao5H%)-eh$~*^w!Uc6*xU^o2QmtEC@WZtzSx+6%CU4k4fs<e}vo~5lW)F_#@|d0`m}oLGTdEZ%g@;vVCqsgroyLjHnY+HF<g&YJVcsn0(sOro2MG<z1EMaAF$32?i+##kPn<CY)5(L^Y%W(jUwhYpF@OJJzy5LB=T<|2At1^B(#xX$jbV3~?{J3&nHcH|KN5^IzB;&OY+avjYMW@Q8Km7dWLx*DEhn&e7u8H*$rRE`R{_`{#CVn0I_xB8A&pEo)Nnh`IW+H*e7MhvQI@vZkKK~EAcc~AaA2AfJQ<7a2I{PfVv`r|VJs=#U<j+sbo<fp9=iR*id9g5s$EpOAsy7`i3YF4>}uHVlThXn2@O#bsAS|A(Bn#uX`X`j$1Chfax!oDJnThA_#HIW(iP@H9E#feqLQRhp)b@O)J-r9wBL4IUr|AFjh#7u!u-S)gnWKD0A+ccRYIv_+Ti%W;r6d^MOCj1aX{`YLPI$8u*NQO-J-#e>2mJ>rB~LhhS6)w-wAP<##-dmP`#Qk248jLuN8FhU(3Gv*heQhv0ZMNM|luihs}fy7+Jq$m}cm2NR)(ifrhD4uy&O3r9(UW=q`_nGLLJW<Iw0@FU$G;Vh@UGi0-8K3nH7712UidJKrmO;Z@kPBmJlE-gAR{eF;m3DBBArr)YU(lzY&EqlX7;$iWDKOtCYzNi=M=6A^*f-+Y=KQUZ+cnD_)~71hqdrdnTWaG=0Vvl4hZ!<D($lXK;Wvg%Q@;FfLs&6}2GeyzY7TxIeHo*I~g?-(1PVk?^Kru#UUwM`R$n%ggY^5W>6iVSf`mil=L&tQEmimUCHDnvv5@uZJ4tTXPTi9hyH8E%2RwNE)ST`a`q2XDI%y6o}oV8R~8{`DbDHyJjiPvg|R?h+?Q#AWy~mEKfes1s>m>q(V=4+&LeEm~OU5XJCmV{f2V0NrppU(az8dcLAQ?*BUnR7|lWP}EgxAQ~DZX9dJ{aWg?y%TaWGKL>)pT*&X~Y0YW^*u#-5knq}rh!Zx-wK0RUe_-R>7KURTmd|N??{OsTzly?S>F|N{z~HW=Lj}EkVIXwgca^deIs@jRYR4dAmrFsDdnl2EZm2z37iB{neKlS}X4r_qMaLw-szS;rIfq0=<34QxkMTW)yNCdq)QUhb44C}!__bxJ7-$h{PanaNuat94Dq?^ca!9tCx<2NKY=_y<oHWM`_EpXi5&*%-z*aHsyJ`^DWAkyZTqx(z=miO*3q%+{ijXeq%2wQPkN)E9qGT}#Dd+(#)#99s(BGT)X(+LF3b-uDkqNIfdjzLFtCZK?+l8Wd&<LeQRDG%d>J4<1kGl8)sNkTmj;W>;-_#%lMo6s%#=tXSbMYOS)DNv%YLC+or5zAd?mu`$Mg-yW$olGk1GEwgBAjxTqyqkb9e&DIKau&7Z5->aw%p5LpQkY4AFkq<S?lnMCFX!&FO0d9*C@rP&wwTc=}wx4TPk?F*&0Vfu9Vyw+c(d_;@8_vx*6m&VPHnGm!J%BDg^shcisqZFhDQ+h&<*r0YO$+{AV=4XkX0HbRX1QWdmAhlgcuc*ik<jDG9l4v#M3$!Kk=e?53P%!4mO?pGqbZY@QPeukUsL(Xe?w>TWvU#p4Q$HKY(jdUtkWFj3$**YD8~d*?YM(98TO6L>~FtV9e+zgQ$w+z;pEbs05!C|;Ovm3=$%#nmkqkU2pd-L5P>-K+h8x^sdPuz`z+Fj8RzzLukYfpP^PBqZuc-tFb9W?4)o+41RdZSn(j`K|h^{4?@;T^8LyYf-=q=1c0kwS4p^S98P!!IPd(hlPNtTj<1<LnH3SlH`G|*$<ZMbJ>8?tcDtj6buq4pVH)4mLcB9;jZ%T9R-MQ*Nx{kq$ljpuQ%H~j?Ca|{B2huk9GP7w)JSCdgD$E9LBKeyC*$KU5J7XZ?Hfbd;oU31SaOF8-h#q>9f1i2-Lmw6I}(ITCnqFSkCy2MdNoXLnS_ZZs$U<G9Tw%l~NRM$Crymj(R#OQEKVJ<!-Toc$AH7;$d=5INZ*>?Yl|cE68|5+b*pNzW(KZhu1j)Dyeps#9-Acu$)Ne4$079-O(&iZ&u#e3XKQ?&4){%1sDaCTva1Db=y!lis66GRW7#AwgXP^SnqGVJN_TYxz_Ia&K%uB#`$N5zYZQ|*@^Zds!M$$1$^mVd{NWvmNGlqcP53gPSS{3xVZZ$%B1jF#Yp^PMD)E$T!99{-djz`0_x~+-NTL+#RGC1<CORoj_}5R4%L#sWA%w5I$&!4==PXzI7fUDBbQ*~lxwXuKFfnCVce}AGkt1&xXwiT-!JGDJ2A#QL@TF_kg2`|auCdM64MieaHYs71#HK}aF`G~F2=ffYwY@}R+}RR$O<a_o9=ocMV&ng`_vqNT?~tr@*K+Oe<1xKGj0HQc{y1H7i`Lc+FOMs8h(%Uuy2*Si5e<s1p3c=w~x`xOq?%+p;-PjxDw(VyYynb-&>fhi`17Rh{}8znuR}$d^jHz&;}-UN(xKS=_L)wtkqx$>(0~b>0>y>cNVZg*=@Ukyf3%cFmWVEi84Ik9ebGA0z}!J=-E2Y+BAEZsl`CF+aFfm3Bdr-J<F|>D-<#VZ(BKk7a2}hR=R3H+#gR%YklVPf<8ly=-UIRx1ZuHI4NiV>de$8+mZY#=%|i-0XBe1&rjKi!K8fb6Mkj-;xVs~CXB}yOVs0SNUAyOt19Me=EzFdsW(d{T?(qe;C=QHD20m(`;Dpm3_u2w^i%#x9xOjrJqjt0o=T~o_wyZ#QcJi(UlpenwY;80YqhL4RdwqUFx{-Ogmy(nwwU8+jKU;$H86}SSHXF)qHr%ngo<?Qs+kfDHCKg?yvBmc8K<%O+}e9LLJ{+-&l(ayY}2=ZlQe;>*R_?NH`Px5i1M1&-!9y!zcUbuRmy>MDt(X}zLg5dB};I4o}-HQO!$G~k}_jayAK(Cbsc!g(8HEX%Fb;SkH}}GFY4;2J9ZOtj1U6ECTE=%Wy=(NDqt`z@pKJlyfLH-qLx6_4;dFpJNVXy(#+))Ot<A}AR_T_VBZeZ2M}uJB05Pfj&?5NWs%*WsNFGZI%?R1^&*zpKodGUM@{P`5O-d!lt=E~`hnv>f>>W$ZCPb+A}ElL37)fk*kwJ{?FI?Ba%!6MwlPnU7`cvTBnCI{f>{9+!m#e$0L5TY>n~m=m_6l!Vs${qjUiTM4|hth)}5)nOegZO1IlWApGq#JN3lhdcIcqK$CdKVy4`|xH{g)8V~OlwO?9pkcsLc{<G}u_u@IL<g<+`aheD*mb4XP+p>NB}P1s~&=S>wm%EIpqE9EXmJP@fq{3lR@YuFmYuM-!6_f0HfMEAMk!P_rn<X=1+slaCgDAwd}S&Q2C9v)(4=5y(>cDEK_QF-?7tRh)q@$3c@gO1e7)}!6gG*|+tICY}mGLoEE+8k_62H%sTTzt%aJhVDP!(VsQT9S`mj-bI}9>P;#eB5af(j4teL1PcF$0|ONW*P=(Jx+P?=cL#Lz9ljsue=8P+HzI~1kwKW26{SS9d0!ePx<dK6lObVekp;ot{T1Dgu>97na>@&A9ZMHj?VPkl{!YqdF1`|gA0_NFtHYU1-g7F@fX)%%;Mb&oRf|pJs%JzaykIr_I71g5DT%a+j;C>bOgAX_=X^NW{7**PFnsnE>26zV)bBsHMfW(P|qh#XT{KY0dDG@ij+_a)iQw5zo7TLny{!Bg_KvYi@uiuYnD<jGDdgvg)%*~gT+LjXwOxax2dE3Ye1@M|8~E^?Myfsh8d=D`EF=gDEM;c!1)yy3O^9LfvpSot|DUI$WWq&_uEPI6|%a5c~aJUWXENyc^d_*KVaBsb4))7dHi<OUD;6ttU@^`tjvleZ5tc2iWbwZE~=|g((DoDzH>Woh>qu~NG5bacBF$8LOi-ri23A$be4B;*)p_~N5v5N*uelLnsL=p<hLob`fZ6smu!(<RZ77^Q;U<2w)y4M+JTd$_^!~+lvgVTDeMJJ>MD%h%a}5=pL>c?QQOMcN8pH+uk!D@&2r(e&+a(lC5q;HOW*-XiTWS8(KJbAuQ41p8oxg-6?3Z4U+@7rCZ~l8=I;L1IfMVg)Ssy-)f|1pBTH9-jHm{*Nx_57BiDLM@WPGpm=pqZ2Uduy1_XU_%_wp?3K(wHyizns(@|w^ATQ4wpg=4CNPI*L?YSY*2$FlTwRaD}ipG!-?Vlm&O4G^XEwc1NG=6pHsXmeh4pK^`7QjcoephT<Rc`OLAu=H(C)zVd!<P4<z;%FI+D-fT9{Mf?Xcx>GX_-W$@Fuww|1z4Zp4(kB9=WjQGMUDR@z5O=o-N`J7$T($5q!FX{mFI)V%||AzB7$JwK}|qJANFvAH+|uhSCMHaf#CRVJ9Hp!sJ)8$ZA_yNxTd4nwjYDPkF($b!?#ue|^K(ARK9`TMj#t1(h<k;y$~H!*j*j^l^2;2=PLxA|tJPI?T2;ANlT3!Kq$2PhmkS=qGm+gNuEXG9{p1-TQKE+cz%-x8buaLI-bq@tWJo=H7^)BJM$e`5*F-sr4v~@GzP4lW&Qb)uCknB(Z)QY&ex-Ok|cT%Yd2lgrXk0yUKf3J80VAWAKevzKy<S)SYV1N(+n~f**`?1t+0Y9a7NoozKe04Q@kD-2&-pLMlovBo~t#c#t_fvIc)tr+gd;(GCzf4`lzC32esWO17Sq_{nPdchGE1+!b^+@ry2kn&ZFM874jDA`y}YSmZ^rwchU2WMH=Rn7X5uc949vDLvj;uB-7Q%+H#`8ssK_6^E2Qr{^oc9DPW--XdDJV04to^)C>gKxbhCN3he;R#(Sr@T3@IyZ__+zWhCFK$58l8X;t!E~NFqW_wu(l(5p8Gg)>#25DRgkX~gmaF_c&5N}Fez}np|Hk#jL<+I(}zNaiE&!MrP<@2Ae<0chxo)Ywe3t{v+%`T?L(1C3TkD(K~(pb02oH<=nAp7qIk3GPv<lhImq!Rn<0j9e8K)(w|2YbF0o>h~}^y)9<B|+P(S&4sN5WX#g5jBtzIxGq4ta;0!0Q<eqq3D!B@D5D?BEz%n5{+H!iLz;wM1Y;VM9pms*L$Abq-Zzq_!%)X7Dk8VmiT%}5wL7R7(#+~$3ZSPX8M;BGtG)0ON!iNNVdCeC86SS$KAUjkP9{INLYk0KRT<_;WJvk;RW6>&r_3+HNYl%$44M@UwHoH;4ssjJTk&VuoaI%R^^hEd(}pfdZ~6pK4g$6>)2qH;q#$nI{dRKoB*YV|2cGEr5O~){a%XWjTRn@pvlQEhChhHq=Frhe$-m068V-p$qoB?N)=%?s={;lIkqHWnpW>6Deq`lPxD*ZV>exZl!7^oCr8Y+#sr5BtIulxnI8AuyAH9t?GgDS)0fYxLu8`m+}k+j*zQ(^Yg%}>>s>#kDF^YRSFF0c@)bee4qfh_8V?ErU`P3o1!FGDSo$hJg5XzfQ?hk#@BFAi@@5}VZ$p~-^KB7{%6VI^hXHtCnHgo0ysd2B&HU~B#t8zBx}2nZ;Q{*;Y)Sw4e{EA3{D;11x6bcs=~ibGJlph*F-0@7Q@ERA+e<SJhL?Gv?e!O8Ph@`WchFH+9XEtkez2%Rq9teCP7b<#dEQ1#FY4`@l9PxfB3wiM<aSPRjz&2#$E1hsQ#`IyD)xP2+esH61RywHGqB}iNE7%!vNr>AK3*q0$T#U@$y)IQYQOxRf26)?Xnk=-a4$SCd7rtF{vXiM;i`dxqg=vUn2e_ri{~Y6N$RQ2ICIR>6V3waVIqDlLQ7M^xH8Y|l0a%&35%(In&w1=iWL7XCpuDZCq7dKxjetV#9Uw~;CwT!it?nopF5u_ZxqjdQq?br%{hdm)bT9#U=gN2?o~HTYa*KiqD_qpImY*;&)#k!MeIMNXkhWp798q;*1Ubk9lI`kPgZn`$8wplV&$VS_0MTk--AzbtM(@HYjj!y+3)2(UybP(3?7XW8k`>49c?n7kGOM}V>1C(?H93mEDmBTKrIjMfx`4r(<T-1k>U6U)Kbh@Gjl{`_}XIZwGHi<SUM&YuL3P?Ka|1EnP{_(C$dEE!ER5iCowe7kz7-5TgAX~b*ip1T(YxP=B|Q2iasDOety4ibaF^fts%s^z=AjOmLG5b=I|pI+3>@^_+LgMGuH9Srvup&t>X(eELjd3su?M#?Ls-op>974Fl|srXlGm~S&8x8krvXdsHy|n^xO)<zYg)`2ZEwx(|c9(dgT$Ki3afXc}=(Ln=w_$Qyih`?}0LL>j;eD`XN;NhI|7Z59;Y0M3O&l<ewHM;cPe<4&hZX$5PY?_k2{i1xpa(9T^qNcmU5yyDuS)U|@_jJ`j+z3Bi$(Mdj^@^MO}TF+)@-Gszv61{M649AlQCeF3!cNoHvr?P5D)nR{_N1bF>?rTu!WOKSkSei3iN7-a22=4e^iGu;c^cTpG~FT)4>pX`R7<jY;cNE6d&pJT&L52F#BV#dvo(d~=FGdB*}h%_4?!|FIVhsBDGUv>MQm^yf(e^_UN+!LF(0mc4V)e&Z~C_!DmUYx)o8u)u6yNeQn|5=DgAbACJwW?rMB0?K4wnvpyFkUHrb>ARgbC+))?hVhQT>Q;cs~}cZ3NFj-NC298J02tX;k~Gi!X&`dVEMxD_h|{k32BljCP;)m<(imgsz4f@{+2_pFt>z=U9Kni<RNVXP~gI?xB@G$xeMGIjh8J#ML2zi8C^mQUeiyM?He;hVHIy@^a|PKM@xzCb{&GG#g|WFZy|ptPaK&o-79cTL(Q}WZidw-MXCOvDfXHkhD70W|CD<OOUS89qi`U_Hzf&!1Hg~1dFEm>jn<aq3W!N<LrW9)fZViGV6hQd>qXj%?!-P5aY#CG)y8k2_+U_^AkpLT;TP}a+B`bSw0K!{aJW|RH1MT+nJaZpI0XjMLZExq(fjr*mpdSNOxLD#aR!Wd=}rU&(EV0-l73BM03xVmP-$f2u?axu6V`k>NiPIpZkf9Ozt~Hg$*@MHtN|ixN5FO9*Q^7J1UNsW<#s3Mg|~nmxHWI{pS>YF<JFC1S8|#aF!ry$4r5BWF2_}1rXbb0%xJoz3;=sq{o=MiTst5R1CQCOeSmiiWO223A0a|0z)=b`BDcmDUytGA0%!_!E7z|T4jbqOEfGBCeKkH0y+%=Y-Q?3A3ZP!V*a!oulerP*`3%i2cbkfi{f0h_J>7ncWt~WBKGfwXaQ;)SC<FaJx-GF<!ek3XwA70FYjkTE{jnfk7pPhYs9gpBIV=hsYsyGSeyp7&!LsiE$6?SqKm$x++i(2c!wyrl3M&8c&NuFJ$ms(FT3F$gQN$u;+u_>5j>&hK3Kxq`V_NG)d%{CLwSD!zA?U4m2!}ayp)(zJPl-OmT+JmJ<|Mb?tL{Xw93xxieAfrD2i^gN;uVI)AOjMfguMI|S8m;YQlB2n)pU*251p@e5}fPMA|}_9eCk9oI3b;Fx;KY*Hj9KbI^n%c^f>Z%r9=Mxsd*E=F<DdmAyq>2;8Mo{#5s~&P+f)*1CL|F*}}<>wV@e<T28g4_R_D=`C^<PKYo8;5hRVcOi->}CdJ7G$oixfO{emZ&O6$v-92E*X%%73;9rr}i-&}rJx;2Mt%3bd&|ZN~syv)!q1GRFGF&#=6}wSfHr>xo>S1MLa&Rtn@W!qFm6A}yuSM^jJCU6lYs<ghUXB6B#vsp{Qg1^-*6XA0__iQXvA>Q!B)gea@0|2fg-?G{J+g>LW2u7(dV~xfJ+&x%Ur%f~<yivDAh=C9eQRgd|0lGY`PGwSscIwo=XHZ~Wbpi~g4<t-D&4dC+Y<S*zJ}QdMri;;*;S@jw=U>d`Wb8XqL3Kfv|LCk{irCr591+x%sMWtM_M4W#5Ywe$i1!SU)5oSivqo3?2*v;RbPg|xGCxswnVxK>xO+}xGGy{4Q*Uw2HNPLn35neQy+Wi!JtT9AOMxGhwxcjY<!Sn-Ay(Y|I*JGFZajbmh{>~aSPcGhMrR}4!~_u^c`W^cZIBeOzl^MBT=ZH;e=!SXAHC(!LzKW`?Ir*gS$V@SgSqdn;c_0duAii=9=5ZbaEv&Z(>loj-~Gih;32mllzG8;$BmlxUcLDL;ww;IoieVb#@pA&~zDu!39UV6qd6zLhh|Z%s=ZIiI^bsfor8P(f*6~IXO17YTBp)*J589nf8V52EGLX2VI=0WnhDCNLbU*ag)j6V2^7J%#7Vo=4M7o*v9m8Cs>555cn_MeBDPgwBYbkhAQR5{50#%MvX_b;EI-TGICkRY6{YYu1uOh6o}$=*>cn9?jbRmHqBc9)z4mhlQ%cFmV>?hKRTuTWrr7z8m0z%_S=PO(aTWTv5J3AQb%-dN}wvgJ<m$mxiylyujk7!_D~0%oY@oWR}w*!7F%Pjd+s)=mr3-lIr|dvB@xO&cgODyAY8szBc-NmG`dn&6enYqZ52I1P9$8(OE)DfLrivAqK@;72+Jhj_!q}kIBUiTx>D~N5sLS4Cy!__mC%7ymu2fZ@8ChNCv6Sa>d5#`Tb3Qu`{8|F3|w>dhgzC7v%l`;Jbw!|bIh;L#T8V$Vmn?_mvGy^k(s_L09S}DEyv_!?M;~@K*V~Xj8Kv9y6CL2t`0i)odQcq9zdAnIqc%jjIy_BrHt63Ay;MxkJilq2iV|0Qj1DWsd`KeKWCF}u!u*H9*Y*!bzVe|k`72oceFfuqLbG_jS~MM&iUc4%ac}48)kGrz!^zGaFr(Qc3f^OfZ$;MmzHk&rB9t{(e;FY9B{tRfO(`97{YtU%S7cd<_?b-N!@f)giaczgK;4WgvAGn0_3FOQC5GmqhQU`1VHFFJ+5;7w1@=%JN?M|OcDZS+5Awo_w8(Ja6|RJV1*NXtg2~qiPv<eoi=_wG}V#-vD{SL7@DnQHjXtHhxKHHfUT;c>R_wFpG7l7ul!SQrZ<9y8)Ow|#J+n;vn(--{jN*9yPQ_MMznCr3A`)<fMJL_%2jh_KL1Q>fDGMI_pMC|VlXrC0V41DN}~TIkNkuVtuU;Arwoy2AY+2uRHmN<+acbTGy!PmZ5!t8;Ay%%U%~X+j2BCgwx7XIk0q+HeR5#O@X~w_P-yIH{G{=6FV`=kO{=<A!bw1?8md>u;Xy<;z@Wd7S>$YXD9b29o(K^=BBug{y1jq$odvo|XAaKK@3v-@X-Q9iooM-X3&7J%z6S*hiH2YUud86Ey*6VYXYYv|&JG2hbVScHO7}-&BU|iLl4yWu+5$AYSI&-3U+mRtq)M+8L$HbnC0Gq-J~U5wbU=4Xra;P8An&C`I3`h6bYIcMCA<6=dUivy`p8Z%<Qx$`>oaZfLv@o0CD97VVBv2V+E`B{S}r=CGd~38s|!$#?afnRD3Bs&H;@x&tv+>h;<KSC@V@cKN&Vd9ki#lDauqFQEpn8Lqpoo(*54Vq7@qeSWs0VxAue=(TmS0%#%6*u@OPYD+MOr%$ty=hWA%)!p68XhJ~+Yqb2tI0ioDjntNlT}^3H9jh|Jx4-+T=R!dJk>g`9RG5)?acx_`TIeQzLqoTSgQ^=_EdhGB<|<jmUyU|_Buh=G=!5W&w1S{=|!vi4K0kJQb#MYZgZoLv?nKm=+Z0(hHVniW7kqdnfF%Hos&=&|>Yv0P#oXmojInbSM>322DTxhmYR#>Bq>;xc<88hM$l&rhedakJhFY;vSS8Ohnaf8AgR4YQZ@+m--e-ptTy`JymTjSvl#!(58SjPH|9v*etG;=|xDT@f5LXYqO-@krw*ON!4-7no}D;TLEB8ig!7qIns_z^Kz^09O@Ti}~>JQeTtbD5-_Ra(inOXrPj|x~0e12j07WfUnWr{@|$2zx<TwBc8Gpf{+!;hJM;yMj+K9S3J}ZpzY-gFzV}q{=)f_PfksPd4k-0We5VLI6*y1A#Hk4dV$Y{w4_QFGsi;xjg>F`*f}m?3eWr1b4YIBKnzz$S>2M<ZKBo;1W)>bD4Z(tz3+D(FaWaHCUw8qw;^H?_5ofJ-gK@?DYG2COE~so?t=XS8{`Qm6CSmo)G%{~ehL__44V|&6eyi=1R-jUMh*pmA=o9uu)S=C5)b<?Q%LXlz@zB1^4u_#pQ;0NJ+$+5CM-UxctF5uhQ9OO2pwv)_>c#PH8Y`+HXNy1ijbd1oh`U_b!_?<JWUd!JQ9H>oczE&p>-&RjoOLZ?VC4H9&;m7n$xU5GAqeY%atKWmC*)f(>JJCS^ETESW?v6_%yjKrNr5GY;%W{gI$EIW|A?kyO$U<cStYdU?@?xuseW68n=L#A+RctwvR63K1W8f0qGZ+V(5;b6iLsnE^h#y9d~HXwe2@^^Wm%JQF=IirFm`KH9$1}$2Q1RHa{GTFPQ)UWBYXsSdDf*00Fp1hxP#gJ#oVavBYQl0ssI200dcD\"]}")          # build-time files [sha256, lzma + b85]: nb4_lib.py, the hook modules, the P5 sources
    NB4_PIN = json.loads("{\"lib_sha256\": \"fcab98a257b314192e6c434a3c051cc9faba2cfcc3dbc90d2e2a8acec6cb5f00\", \"picks\": {\"nb0\": {\"flags\": \"30bda0eacd17a6d8\", \"ENV_KEYS\": \"0414c653e7d8fe9d\", \"_set_value\": \"3c5acc5a3bf4ccb2\", \"nb0_argv\": \"4e6aa4a58d79bd5b\", \"config_diff\": \"f3f1af0f0dc5b2f5\", \"cg_snapshot\": \"acb2a8225dfca977\", \"_pids\": \"d8abe43eb7422030\", \"_stat\": \"d9d4a286a3a5549d\", \"_io\": \"e2e1fafac3d8aed6\", \"session_procs\": \"2566cdeb40eb3224\", \"_roles\": \"eec0d1af37a4e17d\", \"load_monitor\": \"27fb634926cab0f3\", \"_post\": \"7347da07a25c8ffa\", \"_req_log\": \"178a80ad166d8329\", \"ids_sha\": \"7b53d2d4809efca8\", \"_cached\": \"64e083b7320a7cca\", \"pl_arrays\": \"a22c5f078c8887bd\", \"e23_rows\": \"98e9cd247a79954c\", \"tf_request\": \"8b71ebe5ca2df8bf\", \"cmp_last256\": \"2eabb6d8cd6d6e57\", \"chunk_check\": \"fe12c04751d5fa79\", \"cmp_full\": \"7c3b29901f5b78c6\", \"shift_check\": \"426ff828d7174086\", \"screen\": \"84e30382297c4abf\", \"TF_EST\": \"bc09dd9b1536ca61\", \"tf_suite\": \"1bbb0d4ae8faf09d\", \"scales_facts\": \"ac476b44fef365c6\"}, \"nb1_cell6\": {\"psha\": \"ad40b188f4d211de\", \"P60490\": \"4f95732d99408f82\", \"P60490_REL\": \"f6761f03091ac6c8\", \"P60490_PRE\": \"2305648f9d1f3f2d\", \"P60490_MARKER\": \"ea2b89dc42754f81\", \"NB1_EXT_SRC\": \"c44dcd7fe00cd86f\", \"apply_unified\": \"1a1620a5d4c70476\", \"apply_60490\": \"d575d404a978450a\", \"cgroup_snapshot\": \"79bc0ed4cac95f6f\", \"ENGINE_RX\": \"df67e9a1e10cb20a\", \"engine_procs\": \"dfdf48ffcae013df\", \"clear_engine_procs\": \"3b8ebc9d507169f2\", \"drop_page_cache\": \"1ee09a2433809820\", \"host_ram_gate\": \"924feed8236d7eb4\", \"_boot_gate_rc2\": \"cebeb0eebddd7ecd\", \"BOOT\": \"cf052cedba4cea7b\", \"boot_gate\": \"6cd103cb62ed9237\", \"shm_cleanup\": \"0e88a8da75f20991\", \"boot_left\": \"713d6e27a328e7e3\", \"due\": \"f49a5477fc473bb6\", \"_pick\": \"c0f3cf29a3cdf336\", \"server_caps\": \"e73f6d9a109a3a61\", \"pool_facts\": \"a745fb07f6c29a99\", \"graph_facts\": \"dca642159c6c3d7f\", \"jit_kernels\": \"65c8478076b60627\", \"nccl_transport\": \"7733f15d61ba7c27\", \"rpc_cache_info\": \"b06bf4ae93af3602\", \"rpc\": \"294d614b70e375c7\", \"commit_mode\": \"4b899523b6964302\", \"commit_counter_delta\": \"f1fd6f6a73b15c05\", \"kvu\": \"3b14a679d03c8541\", \"profiler_tools\": \"11296baef8e49aa4\", \"nb1_complete\": \"43eaf27c9fec16fb\", \"lp_compare\": \"0e41d95896e03901\", \"warmup\": \"42dc34a973be60bf\", \"accept_pos\": \"8b0891169d1c00d9\", \"_prefill_done\": \"a51190d28050e9cc\", \"_req_lines\": \"c6761cf76a22143f\", \"_ttfts\": \"0e41274da6cfbf7d\", \"_counters\": \"2d19bfaec47c9d94\", \"window\": \"40c11858c1a106cb\"}}, \"nb0_notebook_sha256\": \"37a427a235bf665ee51c7a01767427a299720b4d80d6676f0ff2accb4510bf2c\", \"nb1_r2_notebook_sha256\": \"62670fce0e5ee950e1d32027522c5b4feffa55c43b061e4c7394668142b2bd3a\", \"nb0_builder_sha256\": \"166b5afd7f6d2a052c9496885b358fbc86f397816ca829ad3081cf21f7de65de\", \"nb1_builder_sha256\": \"15a926ad5e23055602a625009dee6a07cce3426972b8e92017e9f0e7c7bc288d\", \"research\": {\"NB4_RESEARCH_graphs_2026-11-01.md\": \"82f278fc31fe0bc0822c3ff5291ea0d3cf46d39c860f8430b667399f8c0d0e71\", \"NB4_RESEARCH_folders_2026-11-01.md\": \"512169ccf8974283fc68ffa88e15c7775e9436ba638849c7d5de8b05a3d4450b\"}, \"g_override\": null, \"disabled\": [], \"hooks\": {\"nb4_timeline.py\": \"43a0848def257e70565285b6e172566a0480dc523c76eab100ed536a5a575f5e\", \"nb4_p5_engine.py\": \"660e6dd448f3e7945530c7ab25e49cf5802adc7b594791e26baecf6304697d8f\", \"nb4_ppack.py\": \"6114f505322564170ce134eec91a5e51f16a328f64be45ba08a7faca0eec5127\"}, \"hook_source\": {\"nb4_timeline.py\": \"embedded\", \"nb4_p5_engine.py\": \"payload foysalemonshanto/e25-nb4-payload\", \"nb4_ppack.py\": \"embedded\"}, \"worker_ext_sha256\": \"222925706cce2ba0c45c83a7b5e5db86a12fb25285e15e870ee824ff84f29598\", \"nb1r2_b3_rows\": [{\"text\": \"We need solve ARC task. Need infer transformation from examples. User gives Task 0934a4d8. Example 1 input 30x30? output 9x4? Example 2 input 30x30? output 4x5? Example 3 input: \\\"194499276\\\" only one row? Maybe truncated? Need find transformation rule and solve test inputs. But only one test input? It says Example 3 input: 194499276. Maybe that's the test input? Need output. Need infer from\", \"tokens\": [1596, 1144, 11290, 67964, 3274, 13, 14235, 22846, 17454, 494, 9989, 13, 2570, 6486, 5261, 220, 15, 24, 18, 19, 64, 19, 67, 23, 13, 12996, 220, 16, 1879, 220, 18, 15, 87, 18, 15, 30, 2468, 220, 24, 87, 19, 30, 12996, 220, 17, 1879, 220, 18, 15, 87, 18, 15, 30, 2468, 220, 19, 87, 20, 30, 12996, 220, 18, 1879, 25, 328, 16, 24, 19, 19, 24, 24, 17, 22, 21, 1, 1132, 799, 2713, 30, 10380, 57739, 30, 14235, 1423, 17454, 5728, 321, 11290, 1228, 10804, 13, 1921, 1132, 799, 1228, 1879, 30, 1049, 2640, 12996, 220, 18, 1879, 25, 220, 16, 24, 19, 19, 24, 24, 17, 22, 21, 13, 10380, 421, 579, 279, 1228, 1879, 30, 14235, 2468, 13, 14235, 22846, 494], \"margins\": [11.5, 11.25, 0.25, 3.625, 0.375, 2.25, 2.25, 0.875, 4.125, 0.25, 4.25, 1.25, 0.125, 1.0, 1.75, 4.625, 9.0, 9.9375, 10.375, 9.5, 8.875, 10.6875, 14.0, 10.625, 0.125, 0.75, 0.875, 7.5, 4.5, 1.25, 3.5, 0.75, 0.0, 1.75, 4.0, 2.625, 1.25, 9.375, 5.5, 4.25, 2.0, 0.25, 2.625, 0.875, 8.875, 3.75, 4.375, 5.75, 5.5, 4.25, 3.5, 5.5, 0.0, 8.0, 7.25, 7.375, 4.625, 5.875, 1.5, 3.75, 4.875, 9.125, 6.75, 0.25, 0.625, 8.25, 8.25, 12.3125, 11.5, 10.3125, 11.0, 10.1875, 11.625, 9.875, 4.375, 2.25, 0.375, 1.0, 4.875, 0.0, 0.25, 5.75, 2.125, 0.0, 0.25, 2.375, 2.875, 6.75, 9.125, 5.125, 1.875, 0.75, 2.375, 0.0, 0.875, 1.875, 2.125, 0.75, 2.25, 1.625, 9.1875, 7.125, 9.125, 5.625, 1.125, 7.9375, 7.0625, 10.375, 11.375, 10.8125, 10.5625, 11.1875, 11.5, 11.625, 1.0, 1.625, 1.375, 2.625, 0.5, 2.375, 1.75, 2.625, 2.0, 2.625, 0.125, 1.5, 0.25, 1.5], \"error\": null, \"prompt_sha\": \"8df9ed2c640ef5bd\", \"prompt_len\": 2048, \"finish_reason\": \"length\"}, {\"text\": \"We need solve ARC tasks. Need infer transformation rules from examples and apply to test inputs. We have two tasks: 1818057f and 195c6913. Need output transformed test grids. Need analyze carefully.\\n\\nWe need produce final answer likely grids. Need infer rules.\\n\\nTask 1818057f examples:\\nInput/output grids with colors 2,4,8. Output changes some 4s to 8? Let's inspect.\\n\\nExample1 10x12? Input rows:\\n1 222244\", \"tokens\": [1596, 1144, 11290, 67964, 8806, 13, 14235, 22846, 17454, 5426, 494, 9989, 321, 3668, 310, 1228, 10804, 13, 1165, 599, 1330, 8806, 25, 220, 16, 23, 16, 23, 15, 20, 22, 69, 321, 220, 16, 24, 20, 66, 21, 24, 16, 18, 13, 14235, 2468, 22773, 1228, 54455, 13, 14235, 22903, 15060, 13, 271, 1596, 1144, 7936, 1534, 4087, 4222, 54455, 13, 14235, 22846, 5426, 13, 271, 6065, 220, 16, 23, 16, 23, 15, 20, 22, 69, 9989, 25, 198, 2425, 45447, 54455, 440, 7736, 220, 17, 11, 19, 11, 23, 13, 8984, 4203, 1010, 220, 19, 82, 310, 220, 23, 30, 6558, 579, 23331, 13, 271, 12929, 16, 220, 16, 15, 87, 16, 17, 30, 5396, 6761, 25, 198, 16, 220, 17, 17, 17, 17, 19, 19], \"margins\": [11.875, 11.75, 0.5, 3.125, 2.625, 3.0, 3.125, 0.5, 3.0, 0.5, 0.625, 4.875, 1.0, 3.25, 2.0, 5.625, 0.625, 3.875, 0.375, 0.75, 1.375, 6.75, 1.375, 2.125, 12.8125, 11.9375, 11.6875, 13.1875, 12.0625, 8.4375, 10.3125, 10.5, 1.625, 7.25, 13.5625, 12.625, 10.5625, 9.375, 12.0, 12.6875, 11.3125, 11.0625, 2.0, 2.375, 1.625, 0.875, 1.0, 0.125, 0.5, 1.25, 0.125, 1.875, 6.625, 1.625, 1.5, 1.5, 1.0, 4.75, 4.625, 2.0, 0.875, 1.5, 1.0, 0.625, 0.5, 3.375, 1.0, 1.125, 4.5, 12.375, 5.625, 11.125, 12.625, 10.0625, 9.875, 10.125, 11.625, 0.125, 1.375, 0.5, 1.0, 0.75, 0.375, 1.25, 1.0, 1.75, 7.75, 0.75, 7.75, 2.375, 3.625, 1.625, 0.25, 2.0, 3.5, 2.75, 1.125, 2.25, 2.375, 10.8125, 10.75, 0.75, 1.875, 7.5, 2.75, 5.25, 0.25, 2.125, 1.25, 0.0, 7.125, 3.625, 5.5, 4.5, 2.0, 1.5, 0.75, 0.875, 0.0, 6.375, 1.0, 0.75, 10.0, 4.75, 12.625, 11.5625, 8.375, 11.4375], \"error\": null, \"prompt_sha\": \"8750743e9dcadca9\", \"prompt_len\": 2048, \"finish_reason\": \"length\"}, {\"text\": \"We need solve ARC tasks. Need find transformation rule for each task and solve test inputs. We have multiple tasks: 247ef758, 269e22fb, 271d71e2, 28a6681f, 291dc1e1. Need output transformed test inputs. Need infer rules from examples. Need be careful. We can analyze each.\\n\\nWe need produce final answer likely grids for each test input. Need determine exact transformations.\\n\\nLet's parse tasks.\\n\\nTask 247ef758: examples show\", \"tokens\": [1596, 1144, 11290, 67964, 8806, 13, 14235, 1423, 17454, 5728, 364, 1754, 3274, 321, 11290, 1228, 10804, 13, 1165, 599, 5081, 8806, 25, 220, 17, 19, 22, 797, 22, 20, 23, 11, 220, 17, 21, 24, 68, 17, 17, 10481, 11, 220, 17, 22, 16, 67, 22, 16, 68, 17, 11, 220, 17, 23, 64, 21, 21, 23, 16, 69, 11, 220, 17, 24, 16, 7385, 16, 68, 16, 13, 14235, 2468, 22773, 1228, 10804, 13, 14235, 22846, 5426, 494, 9989, 13, 14235, 381, 16097, 13, 1165, 628, 22903, 1754, 13, 271, 1596, 1144, 7936, 1534, 4087, 4222, 54455, 364, 1754, 1228, 1879, 13, 14235, 7995, 4581, 51192, 13, 271, 9764, 579, 4563, 8806, 13, 271, 6065, 220, 17, 19, 22, 797, 22, 20, 23, 25, 9989, 1420], \"margins\": [11.875, 9.75, 1.25, 1.5, 3.625, 2.125, 3.0, 0.125, 5.25, 0.875, 1.625, 5.5, 0.875, 3.125, 2.875, 2.875, 8.5, 7.125, 0.375, 0.375, 1.875, 5.375, 1.625, 3.0, 12.125, 13.3125, 12.9375, 11.8125, 11.8125, 13.5, 12.5, 0.375, 8.875, 11.6875, 10.9375, 12.25, 11.0625, 10.6875, 10.375, 8.3125, 7.375, 10.4375, 11.1875, 9.0, 9.8125, 10.125, 10.25, 10.8125, 10.125, 9.3125, 8.0625, 9.5, 12.4375, 10.4375, 11.0, 11.1875, 11.1875, 9.6875, 10.375, 9.0, 7.75, 7.625, 12.875, 10.8125, 11.375, 9.625, 10.625, 11.0625, 9.875, 2.375, 2.875, 2.125, 0.125, 0.0, 0.875, 0.75, 2.0, 0.625, 3.125, 0.875, 8.375, 3.875, 0.125, 1.25, 4.125, 2.875, 0.375, 0.125, 0.5, 0.125, 0.875, 1.375, 2.5, 2.25, 0.75, 4.0, 1.75, 2.0, 0.875, 0.625, 1.125, 3.375, 4.25, 3.75, 0.875, 0.125, 0.5, 0.875, 6.25, 1.5, 3.125, 10.25, 2.625, 2.25, 3.25, 3.625, 3.625, 6.75, 10.375, 7.625, 12.5, 11.75, 11.25, 11.875, 13.4375, 1.5, 0.0, 0.25], \"error\": null, \"prompt_sha\": \"fc63e297d382cd8f\", \"prompt_len\": 8192, \"finish_reason\": \"length\"}, {\"text\": \"We need solve ARC tasks. Need find transformation rule for each task and solve test inputs. We have multiple tasks with examples and test inputs. Need output transformed grids likely. Need infer rules. This is complex. Need analyze each task carefully. Need produce final answer with outputs for each test input. Need be accurate.\\n\\nWe have tasks:\\n2c181942 (one test input)\\n2d0172a1 (two test inputs? It lists Test input twice)\\n31f7f899 (one test)\\n332f06d7 (one test)\", \"tokens\": [1596, 1144, 11290, 67964, 8806, 13, 14235, 1423, 17454, 5728, 364, 1754, 3274, 321, 11290, 1228, 10804, 13, 1165, 599, 5081, 8806, 440, 9989, 321, 1228, 10804, 13, 14235, 2468, 22773, 54455, 4222, 13, 14235, 22846, 5426, 13, 1061, 369, 6150, 13, 14235, 22903, 1754, 3274, 15060, 13, 14235, 7936, 1534, 4087, 440, 15796, 364, 1754, 1228, 1879, 13, 14235, 381, 12995, 13, 271, 1596, 599, 8806, 25, 198, 17, 66, 16, 23, 16, 24, 19, 17, 318, 588, 1228, 1879, 8, 198, 17, 67, 15, 16, 22, 17, 64, 16, 318, 19186, 1228, 10804, 30, 1049, 11140, 3284, 1879, 10598, 8, 198, 18, 16, 69, 22, 69, 23, 24, 24, 318, 588, 1228, 8, 198, 18, 18, 17, 69, 15, 21, 67, 22, 318, 588, 1228, 8], \"margins\": [10.0, 11.25, 1.0, 0.625, 3.375, 2.5, 2.875, 0.375, 5.125, 0.875, 1.75, 5.875, 0.375, 2.875, 3.625, 4.125, 6.375, 7.5, 0.25, 0.0, 0.25, 6.25, 0.25, 3.75, 2.0, 2.625, 3.125, 2.875, 4.125, 2.75, 0.5, 1.375, 0.625, 0.5, 1.875, 0.125, 1.375, 0.25, 0.0, 9.625, 0.125, 1.375, 1.0, 0.625, 3.125, 4.125, 0.25, 1.875, 0.125, 0.875, 3.875, 1.5, 2.75, 0.0, 2.5, 0.875, 3.0, 4.75, 1.25, 1.0, 0.125, 1.75, 10.625, 1.875, 2.625, 0.125, 3.0, 3.5, 1.125, 0.25, 10.5, 13.375, 13.3125, 12.0625, 10.375, 9.625, 11.0, 0.125, 1.625, 7.5, 1.0, 3.5, 7.75, 7.0, 10.375, 13.0625, 13.1875, 12.0, 11.1875, 11.875, 11.1875, 7.125, 5.75, 5.25, 8.5, 0.625, 0.75, 0.75, 1.375, 7.375, 0.75, 0.0, 4.375, 8.9375, 9.875, 12.8125, 11.8125, 10.6875, 10.1875, 10.5625, 12.125, 4.625, 5.25, 1.125, 0.0, 10.875, 11.6875, 6.875, 13.0, 11.125, 12.0, 10.0625, 11.375, 10.5, 9.25, 9.0, 2.625, 5.25], \"error\": null, \"prompt_sha\": \"49b3958bb234d4fd\", \"prompt_len\": 40000, \"finish_reason\": \"length\"}], \"nb1r2_b3_windows\": {\"8k@3\": {\"status\": \"STEADY\", \"tok_s\": 141.2, \"step_ms\": 33.8, \"mean_acceptance_length\": 3.181, \"running_min\": 3.0, \"running_max\": 3.0}, \"8k@8\": {\"status\": \"STEADY\", \"tok_s\": 254.6, \"step_ms\": 48.3, \"mean_acceptance_length\": 3.077, \"running_min\": 8.0, \"running_max\": 8.0}, \"40k@5\": {\"status\": \"STEADY\", \"tok_s\": 195.6, \"step_ms\": 40.4, \"mean_acceptance_length\": 3.159, \"running_min\": 5.0, \"running_max\": 5.0}, \"40k@8\": {\"status\": \"STEADY\", \"tok_s\": 249.5, \"step_ms\": 49.9, \"mean_acceptance_length\": 3.112, \"running_min\": 8.0, \"running_max\": 8.0}}, \"edits\": {\"K-1 / R1-15 / R2-4\": \"tf_suite: + due(TF_EST[band] + 300) -> SKIPPED_BOOT_BUDGET; WD / DROP removed; runs=\", \"R1-1\": \"CELL 4 levels() + window() steady path: + due() -> SKIPPED_BOOT_BUDGET\", \"R1-6\": \"CELL 4 with_servers + NB1 boot_gate: clear_engine_procs() before every GPU-release wait\", \"R1-7\": \"CELL 2 merlin_install timeout = max(600, left - 100 min)\", \"R1-13\": \"CELL 1 save() retries a dump that races a mutation; CELL 2 the sha thread mutates R under _SAVE_LOCK\", \"R2-3\": \"CELL 4 with_servers: a specific body status is kept (server_status SERVER_DIED)\", \"NB4 pre-build env\": \"CELL 2 merlin_fn_server: base.update(NB4_PREBUILD[name]) before Merlin build() (DRAFT_HEAD)\", \"worker extension\": \"NB1WorkerExt + nb4_env (os.environ read-back per worker) + nb4_p5_stats (agent B's backend)\", \"rebound in CELL 7\": \"merlin_fn_server, merlin_boot, watched, L.template_probe (the RC2 / NB1 definitions stay verbatim before them)\"}}")          # build-time facts: shas of the picks / hooks / research notes, NB1 R2 B3 references
    NB4_EMB_DIR = TMP / "nb4_emb"
    _emb_ok = {}
    for _n, (_want, _blob) in NB4_EMB.items():
        _raw = _lzma.decompress(_b64.b85decode(_blob))
        _emb_ok[_n] = hashlib.sha256(_raw).hexdigest() == _want
        (NB4_EMB_DIR / _n).parent.mkdir(parents=True, exist_ok=True)
        (NB4_EMB_DIR / _n).write_bytes(_raw)
    NB4["embedded_sha_ok"] = _emb_ok
    if not all(_emb_ok.values()):
        raise RuntimeError("NB4_EMBEDDED_SHA_MISMATCH: " + json.dumps(_emb_ok))
    sys.path.insert(0, str(NB4_EMB_DIR))
    import nb4_lib as NL                      # E25 NB4 pure helpers (unit-tested): arms, labels, winner rule, engaged checks, planner


    def nb4_hook_module(name):
        """A hook module: the embedded copy (sha-checked above), else the payload dataset copy whose sha256 the build pinned,
        else None (the arm / analysis is recorded as pending, never faked)."""
        want = (NB4_PIN.get("hooks") or {}).get(name)
        if not want:
            return None, f"{name}: not built at build time (hook pending)"
        p = NB4_EMB_DIR / name
        if not p.is_file():
            p = next((d / name for d in (INP / "datasets" / "foysalemonshanto" / "e25-nb4-payload", INP / "e25-nb4-payload")
                      if (d / name).is_file()), None)
        if p is None or hashlib.sha256(p.read_bytes()).hexdigest() != want:
            return None, f"{name}: missing or sha256 mismatch ({p})"
        import importlib.util as _iu
        spec = _iu.spec_from_file_location(name[:-3], str(p))
        mod = _iu.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            return None, f"{name}: import error {type(e).__name__}: {e}"[:300]
        return mod, str(p)


    NB4_TL, NB4["timeline_module"] = nb4_hook_module("nb4_timeline.py")
    NB4_P5, NB4["p5_module"] = nb4_hook_module("nb4_p5_engine.py")
    NB4_PK, NB4["ppack_module"] = nb4_hook_module("nb4_ppack.py")
    NB4_DISABLED = sorted(set(NB4_PIN.get("disabled") or []) | {x.strip() for x in os.environ.get("NB4_DISABLE", "").split(",") if x.strip()})
    ARMS = NL.make_arms(None, NB4_PIN.get("g_override"), None, None, NB4_DISABLED)
    NB4["disabled"] = NB4_DISABLED
    for _k, _m in (("P", NB4_P5), ("S2", NB4_PK), ("S3", NB4_PK)):   # a patch arm with its module present: booted after its prepare
        if _m is not None and _k not in NB4_DISABLED:
            ARMS[_k].update(integrated=True, why_not=None, pending_prepare=True)
    NB4["arm_problems"] = NL.validate_arms(ARMS)
    NB4["base"] = NL.BASE
    NB4["research"] = NB4_PIN.get("research")
    NB4["nb1r2_b3_windows"] = NB4_PIN.get("nb1r2_b3_windows")
    NB4_PROF_DIR = TMP / "nb4_profile"                     # one profiler dir for every arm (an identical flag value)
    NB4_TIMER = TMP / "nb4_core_timer"                     # T0: the step timer appends to nb4_core_timer.<pid>.jsonl
    (OUT / "timeline").mkdir(exist_ok=True)
    DONE, RECS, GATES = {}, {}, {}
    NB4["ref"] = None                                      # the reference arm (T0, or T0b when T0 is not a clean base)
    NB4["plan_log"] = []
    LAUNCH = {}                                            # launch name -> {"arm", "try", "port", "rung"}
    TRACE_CTX = {"collect": None}
    for _b in ("8k", "40k"):                               # level 9 needs 9 distinct prompts (RC2 prompt(), variant 8 = new)
        PROMPTS[_b] += [prompt(BANDS[_b], v) for v in range(len(PROMPTS[_b]), 9)]
    R["prompts"] = {b: {"n": len(p), "len": len(p[0])} for b, p in PROMPTS.items()}
    _osha = [psha(p)[:16] for p in ORACLE_P + LONG_P]     # RC2's own prompt_sha formula (16 hex)
    NB4["oracle_prompt_identity"] = {"rc2_fp8": [r["prompt_sha"] for r in RC2_REF["oracle_rows"]["rc2_fp8"]] == _osha,
                                     "nb1r2_b3": [r["prompt_sha"] for r in NB4_PIN["nb1r2_b3_rows"]] == _osha}
    if NB4["oracle_prompt_identity"]["rc2_fp8"] and not FAKE:      # recorded diagnostics only (never in need=)
        ROWS["rc2_fp8"] = [{k: r[k] for k in ("text", "tokens", "margins", "error")} for r in RC2_REF["oracle_rows"]["rc2_fp8"]]
    if NB4["oracle_prompt_identity"]["nb1r2_b3"] and not FAKE:
        ROWS["nb1r2_b3"] = [{k: r[k] for k in ("text", "tokens", "margins", "error")} for r in NB4_PIN["nb1r2_b3_rows"]]


    # ---- A0 additions: the engine-core step timer (a dormant vLLM general plugin)
    def _tree_py():
        eng = merlin_install()
        return Path(eng["kv"]["MERLIN_TREE"]), eng["kv"]["MERLIN_VENV"] + "/bin/python", eng


    def core_timer_install():
        """Writes nb4_core_timer.py + its dist-info (entry point vllm.general_plugins) next to the Merlin tree's vllm/ (on the
        server's PYTHONPATH) and checks with the Merlin venv that the entry point resolves. Dormant unless NB4_CORE_TIMER is
        set (T0 only), and only in the EngineCore process. Not a gate: a failure leaves T0 with vLLM's iteration log only."""
        if FAKE:
            return {"ok": False, "why": "local harness: no Merlin tree"}
        tree, py, _ = _tree_py()
        (tree / (NL.CORE_TIMER_MODULE + ".py")).write_text(NL.CORE_TIMER_SRC)
        for rel, txt in NL.CORE_TIMER_DIST.items():
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            (tree / rel).write_text(txt)
        envp = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        envp["PYTHONPATH"] = str(tree)
        cp = subprocess.run([py, "-c", "import importlib.metadata as m, importlib.util as u; "
                             "print([e.value for e in m.entry_points(group='vllm.general_plugins')]); "
                             "print(u.find_spec('nb4_core_timer').origin)"], capture_output=True, text=True, timeout=300, env=envp)
        out = cp.stdout.strip().splitlines()
        ok = cp.returncode == 0 and len(out) >= 2 and "nb4_core_timer:register" in out[-2] and out[-1].startswith(str(tree))
        return {"ok": ok, "rc": cp.returncode, "stdout": out[-2:], "stderr": cp.stderr[-400:], "tree": str(tree),
                "sha256": hashlib.sha256(NL.CORE_TIMER_SRC.encode()).hexdigest()}


    # ---- the patch arms: P (agent B's P5) and S2 (the Q-pack agent), applied before the boot, reverted after
    def _build_env(eng, cc=True):
        kvv = eng["kv"]
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "LD_PRELOAD", "CUDA_HOME", "CUDA_PATH")}
        env.update(CUDA_HOME=eng["cuda_home"], PATH=f"{kvv['MERLIN_VENV']}/bin:{eng['cuda_home']}/bin:{os.environ.get('PATH', '')}",
                   LIBRARY_PATH=f"{SHIM}:{os.environ.get('LIBRARY_PATH', '')}", CUDA_DEVICE_ORDER="PCI_BUS_ID")
        for k in ("CC", "CXX", "CUDAHOSTCXX"):
            env.pop(k, None)
        if cc and kvv.get("MERLIN_CC") and kvv.get("MERLIN_CXX"):
            env.update(CC=kvv["MERLIN_CC"], CXX=kvv["MERLIN_CXX"], CUDAHOSTCXX=kvv["MERLIN_CXX"])
            lib13 = Path(kvv["MERLIN_CC"]).parents[1] / "lib"
            if (lib13 / "libstdc++.so.6").exists():
                env["LD_LIBRARY_PATH"] = f"{lib13}:{env.get('LD_LIBRARY_PATH', '')}"
        return env


    def p5_prepare():
        """Arm P (agent B): (1) the P5 .so rebuilt from the embedded NB3 sources with NB3's build_host_ar.py in the Merlin venv
        (Merlin's GCC 13 first, then the system compiler); (2) nb4_p5_gate.py: 4 processes, both TP pairs concurrently, eager +
        graph replays, bit-exact vs the ordered fp32 reference, while the GPUs are free; (3) the strict cuda_communicator.py
        patch (pre / post sha, marker, py_compile, origin). Any miss -> P5_NOT_INTEGRATED (the boot never starts)."""
        rec = {"ok": False}
        if NB4_P5 is None:
            return dict(rec, why=str(NB4.get("p5_module")))
        if FAKE:
            return dict(rec, why="local harness: no Merlin tree / GPUs")
        tree, py, eng = _tree_py()
        src = NB4_EMB_DIR / "p5_src"                      # embedded with the module, else next to the payload copy
        if not src.is_dir():
            src = Path(str(NB4.get("p5_module"))).parent / "p5_src"
        rec["src"] = str(src)
        bad = [n for n, h in NB4_P5.P5_SRC_SHA.items() if not (src / n).is_file() or hashlib.sha256((src / n).read_bytes()).hexdigest() != h]
        if bad:
            return dict(rec, why=f"P5 sources missing / changed: {bad}")
        outd = TMP / "nb4_p5"
        tries = []
        for mode in ("merlin_cc", "system_cc"):
            if not due(600) and left() < 30 * 60:
                break
            cp = subprocess.run([py, str(src / "build_host_ar.py"), "--out", str(outd), "--build-dir", str(TMP / f"nb4_p5_build_{mode}"),
                                 "--src", str(src)], capture_output=True, text=True, timeout=900, env=_build_env(eng, mode == "merlin_cc"))
            txt = cp.stdout + cp.stderr
            (LOGS / f"p5_build_{mode}.log").write_text(txt)
            bj = next((json.loads(l[len("BUILD_JSON "):]) for l in txt.splitlines() if l.startswith("BUILD_JSON ")), {})
            tries.append({"mode": mode, "rc": cp.returncode, "ok": bj.get("ok"), "so": bj.get("so"), "so_sha256": bj.get("so_sha256"),
                          "error": (bj.get("error") or "")[-600:]})
            if bj.get("ok") and bj.get("so"):
                break
        rec["build"] = tries
        b = next((t for t in tries if t.get("ok") and t.get("so")), None)
        if not b:
            return dict(rec, why="P5 .so build failed (logs/p5_build_*.log)")
        so = b["so"]
        so_sha = hashlib.sha256(Path(so).read_bytes()).hexdigest()
        pay = NB4_P5.write_payload(TMP / "nb4_p5_pay")
        rec["payload"] = pay
        gate_out = OUT / "p5_gate.json"
        g = boot_gate()                                   # the GPUs must be free (the previous arm released them)
        rec["gate_boot_gate"] = {k: g.get(k) for k in ("ok", "why")}
        if g.get("ok") is not True:
            return dict(rec, why="GPUs not free for the P5 gate")
        cp = subprocess.run(NB4_P5.gate_argv(py, pay["nb4_p5_gate.py"]["path"], TMP / "nb4_p5_pay", so, so_sha, gate_out,
                                             max_bytes=NL.P5_MAX_BYTES), capture_output=True, text=True, timeout=1200,
                            env=_build_env(eng, False))
        (LOGS / "p5_gate.log").write_text(cp.stdout + cp.stderr)
        gj = next((json.loads(l[len("NB4_P5_GATE_JSON "):]) for l in (cp.stdout + cp.stderr).splitlines()
                   if l.startswith("NB4_P5_GATE_JSON ")), None)
        rec["gate"] = {"rc": cp.returncode, "verdict": (gj or {}).get("verdict"), "summary": {k: v for k, v in (gj or {}).items()
                                                                                              if k not in ("rows", "sizes")}}
        if cp.returncode != 0:
            return dict(rec, why=f"P5 gate not PASS (rc {cp.returncode}, logs/p5_gate.log)")
        rec["patch"] = NB4_P5.apply_p5_patch(tree, py)
        rec.update(ok=True, env=NB4_P5.server_env(so, so_sha, max_bytes=NL.P5_MAX_BYTES), init_rx=r'NB4_P5 \{.*"event": "init"',
                   counters_rpc="nb4_p5_stats", so=so, so_sha256=so_sha)
        return rec


    def p5_finish():
        """Back to the exact Merlin file (agent B's reverse patch, sha == CC_PRE) for every later arm (T0b)."""
        if NB4_P5 is None or FAKE:
            return {"skipped": True}
        tree, py, _ = _tree_py()
        try:
            return NB4_P5.revert_p5_patch(tree, py)
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}


    def ppack_prepare(variant="Q-pack"):
        """Arms S2 (Q-pack) / S3 (Q-defer) (nb4_ppack.py): apply_ppack_patch(tree, py) (pp_utils.py + model_runner.py, strict
        pre / post sha, marker, py_compile), server_env(variant) = {NB4_PPACK: 1} / {NB4_PPDEFER: 1}; the patched workers log
        'NB4_PPQ {json}' init / stats lines (readback_label, all 4 ranks must agree). Missing pieces / a refused patch ->
        NOT_INTEGRATED."""
        if NB4_PK is None:
            return {"ok": False, "why": str(NB4.get("ppack_module"))}
        if FAKE:
            return {"ok": False, "why": "local harness: no Merlin tree"}
        need = [n for n in ("apply_ppack_patch", "revert_ppack_patch", "server_env", "readback_label") if not hasattr(NB4_PK, n)]
        if need:
            return {"ok": False, "why": f"nb4_ppack.py lacks {need}"}
        tree, py, _ = _tree_py()
        rec = {"patch": NB4_PK.apply_ppack_patch(tree, py)}
        if not (rec["patch"] or {}).get("ok"):
            return dict(rec, ok=False, why="Q-pack patch not ok")
        rec.update(ok=True, env=NB4_PK.server_env(variant), init_rx=r'NB4_PPQ \{.*"event":\s*"init"', variant=variant)
        return rec


    def ppack_finish():
        if NB4_PK is None or FAKE or not hasattr(NB4_PK, "revert_ppack_patch"):
            return {"skipped": True}
        tree, py, _ = _tree_py()
        try:
            return NB4_PK.revert_ppack_patch(tree, py)
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}


    def patch_states():
        """Read-only: the P5 / Q-pack file states in the installed tree (every arm reads back 'not_applied' but its own)."""
        out = {}
        if FAKE or MERLIN_ENG is None or (MERLIN_ENG or {}).get("failed"):
            return out
        tree = Path(MERLIN_ENG["kv"]["MERLIN_TREE"])
        for k, m, fn in (("p5", NB4_P5, "verify_p5_patch"), ("ppack", NB4_PK, "verify_ppack_patch")):
            if m is not None and hasattr(m, fn):
                try:
                    out[k] = getattr(m, fn)(tree).get("state")
                except Exception as e:
                    out[k] = f"ERROR {type(e).__name__}: {e}"[:200]
        return out


    # ---- launches: per-launch flags / env (NB1 hooks), NB0's in-place argv edits, a fresh port + name per try, the pin ladder
    _merlin_fn_server_nb1 = merlin_fn_server       # RC2 CELL 2 (launch.build + host_phase) with NB1's per-boot hooks


    def nb4_plan(arm_key, try_i, rung=0, drop_commit=False):
        a = ARMS[arm_key]
        name, port = NL.launch_name(arm_key, try_i), NL.port_for(arm_key, try_i)
        env = NL.arm_env(a, DIAG_ENV, NB4_TIMER)
        if drop_commit:
            env.pop("VLLM_RECOVERSSM_COMMIT_GRAPH", None)
        LAUNCH[name] = {"arm": arm_key, "try": try_i, "port": port, "rung": rung, "drop_commit": drop_commit}
        NB1_EXTRA[name], NB1_ENV[name], NB1_KV[name] = NL.arm_extra(a, EXT_FLAG, NB4_PROF_DIR), env, NL.pin_for(a, rung)
        NB4_PREBUILD[name] = dict(a.get("prebuild_env") or {})
        return name, port


    def merlin_fn_server(name, kv, port, kv_bytes):
        """E25 NB4: RC2's Merlin plan with NB1's per-launch flags / env, then NB0's in-place edits (seqs 9, this arm's
        compilation config, K4's speculative config: the existing flags' values, no duplicate flags), the help-flag gate on the
        FINAL argv (klee100 K-9), the plan (argv + env + D8 switches) recorded, NB0's load monitor keyed by the launch name."""
        a = ARMS[LAUNCH[name]["arm"]]
        s = _merlin_fn_server_nb1(name, kv, port, kv_bytes)
        if FAKE:                                   # local harness: RC2's recorded Kaggle argv into the fake (NB0 rule)
            tail, out = RC2_REF[kv]["argv_tail"], []
            for i, x in enumerate(tail):
                out.append(str(port) if i and tail[i - 1] == "--port" else str(kv_bytes) if i and tail[i - 1] == "--kv-cache-memory-bytes" else x)
            s.argv = [sys.executable, FAKE, str(CKPT), "--host"] + out + list(NB1_EXTRA.get(name) or [])
        s.argv = nb0_argv(s.argv, {"mtp": True, "seqs": NL.BASE["seqs"], "captures": NL.compile_json(a)})
        if a.get("spec"):
            _set_value(s.argv, "--speculative-config", a["spec"])
        if not FAKE:
            HELP[name] = L.help_check("merlin", s.argv[:s.argv.index("serve") + 1], s.env, s.argv)
            h = HELP[name]
            if isinstance(h, dict) and h.get("ran") is True and h.get("missing"):
                NB4.setdefault("cli_flags", {})[name] = h
                save()
                raise RuntimeError("FAIL_CLI_FLAGS: " + " ".join(h["missing"]))
        env_rec = {k: s.env[k] for k in sorted(s.env) if k.startswith(ENV_KEYS + ("MERLIN_", "NB4_", "TORCH_", "PYTORCH_", "CUDA_", "OMP_", "DRAFT_"))}
        R["setup"].setdefault("nb4_plans", {})[name] = {"arm": a["key"], "change": a["change"], "argv_tail": s.argv[-72:],
                                                        "argv": s.argv, "kv_bytes": kv_bytes, "env": env_rec, "d8": NL.d8_switches(s.env),
                                                        "prebuild_env": NB4_PREBUILD.get(name),
                                                        "diff_vs_rc2_fp8": config_diff(s.argv, s.env, kv_bytes, "fp8")}
        save()
        if not os.environ.get("RC2_CPU_KIT"):
            deadline = time.time() + max(120, min(NL.READY_SEC + 120, left() - 240))
            threading.Thread(target=load_monitor, args=(s, name, deadline), daemon=True).start()
        return s


    def merlin_boot(name, kv, port, timeout, body, need=(), expected_rows=4, require_sampled=True):
        """E25 NB4 (rebinds RC2 CELL 5's merlin_boot; same contract and gates). ONE OOM policy (nb4_lib.retry_decision): a
        launch OOM steps down the pin ladder on non-base arms (B3 -> -0.25 GiB -> the FP8 base pin), a base arm keeps its pin;
        a device fault stops; a RecoverSSM capture failure of T0 retries without the commit env (the C A/B is then dropped); a
        fast non-OOM launch failure is retried once. Every retry = its own launch (fresh name + port: klee100 R1-9 / R2-5); OOM
        / device-fault lines are searched in the WHOLE log (R1-8); a server death is named by what was running (R1-10)."""
        arm_key = LAUNCH[name]["arm"]
        tries, lname, lport, rung = [], name, port, LAUNCH[name]["rung"]
        while True:
            gate = boot_gate()
            if gate["ok"] is not True:
                return {"status": "GPU_NOT_EMPTY", "boot_gate": gate, "tries": tries, "launch_name": lname}
            shm0 = shm_names()
            t0 = time.time()
            BOOT["tag"] = lname
            srv_obj = merlin_fn_server(lname, kv, lport, NB1_KV.get(lname) or NL.BASE["kv_bytes"])
            o = with_servers([srv_obj], timeout, body)
            try:
                full = srv_obj.log() or ""
            except Exception:
                full = ""
            o.update(kv_bytes=NB1_KV.get(lname), pin_rung=rung, boot_gate=gate, shm_cleanup=shm_cleanup(shm0), launch_name=lname,
                     port=lport, ready_sec=getattr(srv_obj, "ready_sec", None), oom_in_log=bool(OOM_ERR.search(full)),
                     xid_in_log=bool(XID.search(full)), oom_lines=[l[:240] for l in full.splitlines() if OOM_ERR.search(l)][:6],
                     commit_env_dropped=LAUNCH[lname].get("drop_commit"))
            tries.append({"launch_name": lname, "port": lport, "kv_bytes": NB1_KV.get(lname), "status": o.get("status"),
                          "oom": o["oom_in_log"], "device_fault": o["xid_in_log"], "ready_sec": o["ready_sec"], "sec": round(time.time() - t0, 1)})
            retry, rung2, drop, why = NL.retry_decision(dict(ARMS[arm_key], env=NB1_ENV.get(lname)), o.get("status"), o["oom_in_log"],
                                                        o["xid_in_log"], time.time() - t0, len(tries), rung, left(),
                                                        NL.EST_MIN[arm_key] * 60 + 300, full[-20000:])
            tries[-1]["retry"] = why
            if not retry:
                break
            rung = rung2
            lname, lport = nb4_plan(arm_key, len(tries), rung, drop or LAUNCH[lname].get("drop_commit"))
            log("NB4 launch retry:", name, "->", lname, "port", lport, "pin", NB1_KV[lname], "|", why)
        o["tries"] = tries
        if o.get("status") == "LAUNCH_FAILED" and o["oom_in_log"]:
            o["status"] = "LAUNCH_FAILED_OOM"
        if arm_key == "P" and str(o.get("status") or "").startswith("LAUNCH_FAILED") and NB4_P5 is not None:
            lab = NB4_P5.p5_arm_label(full)
            o["p5_label"] = lab
            if "init_failed" in " ".join(lab[1]):
                o["status"] = "P5_NOT_INTEGRATED_INIT_FAILED"       # agent B: NB4_P5_INIT_FAILED -> NOT_INTEGRATED
        if o.get("server_status") == "SERVER_DIED" or o.get("status") == "SERVER_DIED":
            o["death"] = NL.death_label(o.get("boot_watchdog"), o["oom_in_log"], o["xid_in_log"], o.get("died_in"))
            if o.get("status") in (None, "DONE", "NOT_RUN", "SERVER_DIED", "BODY_ERROR"):
                o["status"] = o["death"]
        if str(o.get("status", "")).startswith(("LAUNCH_FAILED", "P5_NOT_INTEGRATED")):
            return o
        return gated(o, need=need, expected_rows=expected_rows, require_sampled=require_sampled)


    def watched(body, cap_sec, name):
        """E25 NB4 (rebinds NB1's watched; klee100 R1-2): the body cap is clipped at body start to left() - 480 s (teardown +
        verdict); at cap + 300 s the server is killed (hard backstop); the record carries body_cap_sec / boot_watchdog."""
        def b(ss):
            cap = max(60.0, min(float(cap_sec), left() - 480))
            BOOT.update(deadline=time.time() + cap, fired=None)

            def fire():
                BOOT["fired"] = name
                for x in ss:
                    try:
                        x.kill()
                    except Exception:
                        pass
            tm = threading.Timer(cap + 300, fire)
            tm.daemon = True
            tm.start()
            try:
                o = body(ss)
                LIVE["out"] = o if isinstance(o, dict) else None
            finally:
                tm.cancel()
                BOOT["deadline"] = None
            o["body_cap_sec"] = round(cap)
            if BOOT["fired"]:
                o["boot_watchdog"] = f"FIRED after {cap + 300:.0f} s: server killed"
            return o
        return b


    class GpuSampler(threading.Thread):
        """nvidia-smi every 5 s while a body runs (klee100 R1-14): the highest used MiB per card."""
        def __init__(self, every=5.0):
            super().__init__(daemon=True)
            self.every, self.ev, self.max, self.tot, self.n = every, threading.Event(), {}, {}, 0

        def run(self):
            while not self.ev.is_set():
                for i, u, t in (L.gpu_mem() or []):
                    self.max[i], self.tot[i] = max(self.max.get(i, 0), u), t
                self.n += 1
                self.ev.wait(self.every)

        def rows(self):
            return [[i, self.max[i], self.tot[i]] for i in sorted(self.max)]


    # ---- full traces (klee100 K-8 / R2-10): every stream of every window, the probes' full answers, the template ids
    def nb4_trace_install():
        if getattr(L.Stream, "_nb4_traced", False):
            return
        init = L.Stream.__init__

        def _init(self, *a, **k):
            init(self, *a, **k)
            c = TRACE_CTX["collect"]
            if c is not None:
                c.append(self)
        L.Stream.__init__ = _init
        L.Stream._nb4_traced = True


    def nb4_trace(label, streams):
        path = TRACES / f"{BOOT.get('tag') or 'boot'}_{label}.jsonl"
        try:
            with _SAVE_LOCK, open(path, "a") as fh:
                for i, g in enumerate(streams):
                    row = g.row()
                    body = getattr(g, "body", None) or {}
                    fh.write(json.dumps({"boot": BOOT.get("tag"), "window": label, "i": i, "port": getattr(g, "port", None),
                                         "prompt_tokens": len(body.get("prompt") or []), "prompt_sha": psha(body.get("prompt") or [])[:16],
                                         "seed": body.get("seed"), "max_tokens": body.get("max_tokens"), "temperature": body.get("temperature"),
                                         "text": row["text"], "tokens": row["tokens"], "finish_reason": row["finish_reason"],
                                         "error": row["error"], "done": row["done"], "t": time.time()}) + "\n")
            NB4.setdefault("traces", {})[path.name] = NB4.get("traces", {}).get(path.name, 0) + len(streams)
        except Exception as e:
            NB4.setdefault("trace_errors", []).append(f"{label}: {type(e).__name__}: {e}"[:200])


    def traced(label, fn):
        got = []
        TRACE_CTX["collect"] = got
        try:
            return fn()
        finally:
            TRACE_CTX["collect"] = None
            nb4_trace(label, got)


    nb4_trace_install()


    def chat_probes_full(port, tag="probes"):
        """The lab's chat_probes request (thinking off, T 0, 32 tokens, the same contains-match); the FULL content and
        reasoning of every case go to traces/chat_probes.jsonl (klee100 R2-10)."""
        cases = (("Reply with exactly one number: 7 * 6 =", "42"), ("Reply with exactly one number: 2 + 2 =", "4"),
                 ("Reply with exactly one number: 9 - 3 =", "6"))
        out = []
        for q, want in cases:
            body = {"model": "q27", "messages": [{"role": "user", "content": q}], "temperature": 0, "max_tokens": 32,
                    "chat_template_kwargs": {"enable_thinking": False}}
            rec, trace = {"q": q, "want": want}, {}
            try:
                r = json.loads(L.U.urlopen(L.U.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                                       headers={"Content-Type": "application/json"}), timeout=300).read())
                msg = r["choices"][0]["message"]
                full = (msg.get("content") or "").strip()
                rec.update(content=full[:200], reasoning_chars=len(msg.get("reasoning") or ""), finish=r["choices"][0].get("finish_reason"),
                           ok=re.search(r"(?<!\d)" + re.escape(want) + r"(?!\d)", full) is not None, garbage=L.degenerate(full))
                trace = {"content_full": full, "reasoning_full": msg.get("reasoning")}
            except Exception as e:
                rec.update(error=repr(e)[:200], ok=False)
            out.append(rec)
            try:
                with _SAVE_LOCK, open(TRACES / "chat_probes.jsonl", "a") as fh:
                    fh.write(json.dumps(dict(rec, **trace, boot=BOOT.get("tag"), tag=tag, t=time.time())) + "\n")
            except Exception:
                pass
        return {"ok": all(x["ok"] for x in out), "cases": out}


    _template_probe_lab = L.template_probe


    def nb4_template_probe(port, messages, want_ids, model="q27", timeout=120):
        """The lab's template_probe (unchanged result) + the server's own token ids saved in traces/ (klee100 R2-10)."""
        res = _template_probe_lab(port, messages, want_ids, model=model, timeout=timeout)
        try:
            body = {"model": model, "messages": messages, "add_generation_prompt": True}
            r = json.loads(L.U.urlopen(L.U.Request(f"http://127.0.0.1:{port}/tokenize", data=json.dumps(body).encode(),
                                                   headers={"Content-Type": "application/json"}), timeout=timeout).read())
            p = TRACES / f"template_probe_{BOOT.get('tag') or 'boot'}.json"
            p.write_text(json.dumps({"server_tokens": list(r.get("tokens") or []), "want_tokens": list(want_ids), "result": res}))
            res["server_tokens_file"] = p.name
        except Exception as e:
            res["server_tokens_error"] = repr(e)[:200]
        return res


    L.template_probe = nb4_template_probe


    # ---- the arm body
    def run_item(o, s, label, fn, need_sec=None):
        """Every item: gated on the boot body cap (due(need_sec) -> SKIPPED_BOOT_BUDGET); o["items"] in order; o["died_in"] =
        the first item that began with the server alive and ended with it dead (it names a death, klee100 R1-10)."""
        if need_sec is not None and not due(need_sec):
            o.setdefault("items_skipped", []).append(label)
            return {"status": "SKIPPED_BOOT_BUDGET", "need_sec": round(need_sec), "boot_left_sec": round(boot_left())}
        alive0 = s.alive()
        o.setdefault("items", []).append(label)
        try:
            return fn()
        finally:
            if alive0 and not s.alive() and "died_in" not in o:
                o["died_in"] = label
            if LIVE.get("rec") is not None:
                save()                                     # klee100 R1-12: the live record (o) is saved after every item


    def session_env(s, keys):
        """/proc/<pid>/environ of every process of the server session (workers, EngineCore, the PLE offload process): the
        values of `keys` per process (S: CUDA_DEVICE_MAX_CONNECTIONS must reach every CUDA process)."""
        out = {}
        if s.proc is None or not Path("/proc/self/environ").exists():
            return out
        for p in session_procs({s.proc.pid}):
            try:
                env = dict(x.split("=", 1) for x in (PROC / str(p["pid"]) / "environ").read_bytes().decode(errors="replace").split("\0") if "=" in x)
                out[f"{p['pid']}:{p.get('comm')}"] = {k: env.get(k) for k in keys}
            except Exception as e:
                out[f"{p['pid']}:{p.get('comm')}"] = {"error": repr(e)[:80]}
        return out


    def nb4_facts(s, a, o):
        """The config read-back from the server's own log + the worker extension + the session processes' env (fail closed)
        and the "exactly one change" check vs the reference launch (the base arm's recorded plan)."""
        t = s.log() or ""
        o["caps"], o["pool"], o["graphs"], o["nccl"] = server_caps(t, s.argv), pool_facts(t), graph_facts(t), nccl_transport(t)
        o["workers"] = rpc_cache_info(s.port)
        o["scales"] = scales_facts(t)
        keys = ["CUDA_DEVICE_MAX_CONNECTIONS", "VLLM_RECOVERSSM_COMMIT_GRAPH", "NB4_P5", "NB4_PPACK", "OMP_NUM_THREADS",
                "VLLM_CUSTOM_SCOPES_FOR_PROFILING", "PYTORCH_CUDA_ALLOC_CONF"]
        wr = (rpc(s.port, "nb4_env", ",".join(keys)) or {}).get("results") or []
        o["worker_env"] = wr
        o["session_env"] = session_env(s, keys)
        nd_line = next((l for l in t.splitlines() if "non-default args:" in l), "")
        nd = NL.parse_nondefault(nd_line)
        plan = (R["setup"].get("nb4_plans") or {}).get(BOOT.get("tag")) or {}
        ck = NL.readback_checks(nd, plan.get("env"), a, NB1_KV.get(BOOT.get("tag")) or NL.BASE["kv_bytes"])
        if a.get("env", {}).get("VLLM_RECOVERSSM_COMMIT_GRAPH") and LAUNCH.get(BOOT.get("tag"), {}).get("drop_commit"):
            ck.pop("env VLLM_RECOVERSSM_COMMIT_GRAPH", None)              # T0 relaunched without it (the C A/B is dropped)
        ck["static_scales_applied"] = o["scales"].get("ok") is True
        ck["patch_60490"] = ((R["sections"].get("A0") or {}).get("patch_60490") or {}).get("ok") is True or bool(FAKE)
        ps = patch_states()
        o["patch_states"] = ps
        for pk, arm_k in (("p5", "P"), ("ppack", "S2")):
            if pk in ps:
                ck[f"{pk} patch {'applied' if a['key'] == arm_k else 'not applied'}"] = ps[pk] == ("applied" if a["key"] == arm_k else "not_applied")
        ref = NB4.get("ref")
        refrec = RECS.get(ref) if ref else None
        refplan = (R["setup"].get("nb4_plans") or {}).get((refrec or {}).get("launch_name")) if refrec else None
        if refplan and ref != a["key"]:
            d = NL.diff_vs_ref(refplan.get("argv"), refplan.get("env"), plan.get("argv"), plan.get("env"))
            oc = NL.one_change_ok(d, a, dict(ARMS[ref], env={k: v for k, v in (ARMS[ref].get("env") or {}).items()}),
                                  pin_differs=(NB1_KV.get(BOOT.get("tag")) != refrec.get("kv_bytes")))
            o["diff_vs_ref"] = {"ref": ref, "diff": d, "check": oc}
            ck["one_change_vs_ref"] = oc["ok"]
        o["readback"] = {"nondefault": nd, "checks": ck, "env": plan.get("env"), "d8": plan.get("d8"), "line": nd_line[:6000],
                         "timer_lines": _pick(t, r"NB4 core timer", 4), "commit_lines": _pick(t, r"\[recoverssm-commit\]", 24),
                         "graph_mode_lines": _pick(t, r"resolved cudagraph modes|cudagraph_mode=|Capturing CUDA graphs|Graph capturing finished", 24),
                         "p5_lines": _pick(t, r"NB4_P5|P5_HOST", 12), "int8_lines": _pick(t, r"INT8 weight-only copy", 4)}
        o["config_ok"] = all(ck.values())
        o["memory_census"] = {"rank_load_gib": o["pool"].get("rank_load_gib"), "rank_graph_gib": o["pool"].get("rank_graph_gib"),
                              "kv_tokens": o["pool"].get("kv_tokens"), "workers_mem": [[w.get("rank"), w.get("mem")] for w in
                                                                                         ((o["workers"] or {}).get("ranks") or []) if isinstance(w, dict)]}
        return o


    def _worker_env_map(o, key):
        out = {}
        for w in o.get("worker_env") or []:
            if isinstance(w, dict):
                out[str(w.get("rank"))] = (w.get("env") or {}).get(key)
        return out


    def _start(s, a, o):
        if LIVE.get("rec") is not None:                       # klee100 R1-12: a mid-boot save() keeps the body's record
            LIVE["rec"]["live_body"] = o
        o["census"] = kv_census(s)
        run_item(o, s, "facts", lambda: nb4_facts(s, a, o))
        if a.get("commit_ab") and ((o.get("readback") or {}).get("env") or {}).get("VLLM_RECOVERSSM_COMMIT_GRAPH") == "graph":
            o["commit_flip_to_eager"] = commit_mode(s.port, "eager")     # T0's windows are the base: eager commits
            o["commit_mode_after_flip"] = [(x or {}).get("mode") for x in ((rpc(s.port, "get_recoverssm_commit_stats") or {}).get("results") or [])]
            if any(m != "eager" for m in o["commit_mode_after_flip"] or ["?"]):
                o["readback"]["checks"]["commit flipped to eager"] = False
                o["config_ok"] = False
        o["engaged_log"] = NL.engaged_log(a, s.log() or "", (o.get("readback") or {}).get("env"),
                                          _worker_env_map(o, "CUDA_DEVICE_MAX_CONNECTIONS") if a["key"] == "S" else None,
                                          (o.get("readback") or {}).get("checks"))
        o["probes"] = NL.probe_engine_view(run_item(o, s, "probes", lambda: chat_probes_full(s.port)))
        o["sampled"] = run_item(o, s, "sampled", lambda: sampled_check(s))
        o["gpu_after_ready"] = L.gpu_mem()
        if short_gate_failed(o):
            o["early_stop"] = "SHORT_CORRECTNESS_GATE_FAILED"
            return False
        if o.get("config_ok") is not True:
            o.update(early_stop="CONFIG_MISMATCH", status="CONFIG_MISMATCH",
                     config_mismatch=[k for k, v in o["readback"]["checks"].items() if not v])
            return False
        if not o["engaged_log"]["ok"]:
            o.update(early_stop="NOT_ENGAGED", engaged_why=o["engaged_log"]["evidence"])   # the body stops: no GPU on a no-op arm
            return False
        return True


    def _finish(s, o, smp):
        smp.ev.set()
        try:
            o["gpu_end"], o["gpu_peak_rows"], o["gpu_samples"] = L.gpu_mem(), smp.rows(), smp.n
            o["census_end"] = kv_census(s)
            o["workers_end"] = rpc_cache_info(s.port) if s.alive() else {"error": "server dead"}
            smi = NL.free_by_card([o.get("gpu_after_ready"), o.get("gpu_end"), o.get("gpu_peak_rows")])
            wf = NL.worker_free([o.get("workers"), o.get("workers_end")])
            o["free_mib_by_card_smi"], o["free_mib_by_card_worker"] = smi, wf
            o["free_mib_by_card_all"], o["free_source"] = NL.combine_free(smi, wf)        # klee100 R1-3
            o["jit_compiled_in_body"] = jit_kernels(s.log() or "")
        except Exception as e:
            o["finish_error"] = f"{type(e).__name__}: {e}"[:300]
        return o


    def commit_stats(port):
        return (rpc(port, "get_recoverssm_commit_stats") or {}).get("results")


    def p5_stats(port):
        """Agent B's per-rank stats via the NB4 worker-extension method (between windows only)."""
        r = rpc(port, "nb4_p5_stats", "1")
        out = []
        for x in (r or {}).get("results") or []:
            try:
                out.append(json.loads(x) if isinstance(x, str) else x)
            except Exception:
                out.append({"error": str(x)[:200]})
        return out


    def p5_executed(stats):
        return sum(int(c.get("p5_executed") or 0) for s_ in stats or [] if isinstance(s_, dict) for c in (s_.get("comms") or []))


    def nb4_window(s, band, n, label, k=3):
        """One steady window (NB1 window(): STEADY, real running == level, prefill done, tick / cycle / micro-batches per s /
        wall clock, acceptance per position) + the --cudagraph-metrics rows logged during it (n histogram, FULL / NONE decode
        steps, the micro-batch split) + CPU per server process (lead item 2). Every stream traced in full."""
        lg0 = len(s.log() or "")
        p0, t0 = session_procs({s.proc.pid}) if s.proc is not None else [], time.time()
        w = traced(label, lambda: window(s.port, s, band, n))
        p1, dt = session_procs({s.proc.pid}) if s.proc is not None else [], time.time() - t0
        txt = s.log() or ""
        w["cudagraph"] = NL.cudagraph_table(txt[lg0:], k=k)
        w["split"] = NL.split_label(w["cudagraph"], n)
        w["cpu"] = NL.cpu_per_process(p0, p1, dt, _roles(txt))
        return w


    def windows(o, s, a, prefix=""):
        o.setdefault("steady", {})
        for rep in range(int(a.get("reps") or 1)):
            for band, n in (("8k", 8), ("40k", 8), ("8k", 9), ("40k", 9)):
                key = f"{band}@{n}" + ("" if rep == 0 else f"#{rep + 1}")
                o["steady"][prefix + key] = run_item(o, s, "window_" + prefix + key,
                                                     lambda: nb4_window(s, band, n, prefix + key, a.get("k") or NL.K_SPEC))


    def wait_traces(pdir, n=4, timeout=180):
        """The profiler writes one gzip per worker after /stop_profile; wait until n files exist and their sizes are stable."""
        t0, last, stable = time.time(), None, 0
        while time.time() - t0 < timeout:
            fs = sorted(Path(pdir).rglob("*.pt.trace.json*"))
            sz = [f.stat().st_size for f in fs]
            if len(fs) >= n and sz == last:
                stable += 1
                if stable >= 3:
                    return {"files": len(fs), "sec": round(time.time() - t0, 1)}
            else:
                stable = 0
            last = sz
            time.sleep(2)
        return {"files": len(sorted(Path(pdir).rglob("*.pt.trace.json*"))), "sec": round(time.time() - t0, 1), "timeout": True}


    def timeline_analysis(trace_dir, tag):
        """Agent C's nb4_timeline.analyze on the 4 rank traces: the full report to timeline/<tag>.json, the compact parts in
        the record (clock, summary, stalls, graph modes, cpu activity, NB1 reference)."""
        if NB4_TL is None:
            return {"status": "TIMELINE_MODULE_PENDING", "why": NB4.get("timeline_module")}
        t0 = time.time()
        try:
            rep = NB4_TL.analyze(str(trace_dir), tp=NL.TP)
            rep = NB4_TL.strip_private(rep) if hasattr(NB4_TL, "strip_private") else json.loads(json.dumps(rep, default=lambda x: None))
        except Exception as e:
            return {"status": "TIMELINE_ANALYSIS_ERROR", "error": f"{type(e).__name__}: {e}"[:400], "tb": traceback.format_exc()[-1500:]}
        try:
            (OUT / "timeline" / f"{tag}.json").write_text(json.dumps(rep, default=str))
        except Exception as e:
            rep["write_error"] = repr(e)[:200]
        out = {k: rep.get(k) for k in ("clock", "summary", "stalls", "graph_modes", "cpu_activity", "reference_rank", "forwards_ref_rank",
                                       "stages", "n_hist", "parity", "error")}
        out["nb1_reference"] = {k: v for k, v in (rep.get("nb1_reference") or {}).items() if k != "by_rank"}
        out["ranks"] = sorted((rep.get("cpu_activity") or {}).keys())
        out["file"] = f"timeline/{tag}.json"
        out["sec"] = round(time.time() - t0, 1)
        return out


    def op_placement_all(trace_dir):
        """Lead item 3 (T0): per rank, what runs where (PLE, embeddings, LM head, draft passes, sampler, RecoverSSM commit, the
        broadcast origin), from the rank's own trace."""
        out = {}
        for f in sorted(Path(trace_dir).rglob("*.pt.trace.json*")):
            try:
                with (_gzip.open if f.name.endswith(".gz") else open)(f, "rt", encoding="utf-8") as fh:
                    ev = json.load(fh).get("traceEvents") or []
                out[f.name] = NL.op_placement(ev)
            except Exception as e:
                out[f.name] = {"error": repr(e)[:200]}
        return out


    def nb4_profile(port, a, band, n):
        """One profiled window at the compare band (lead / folder sweep item 6): `n` decoding streams of PROMPTS[band] after
        settling (the lab's profiled_load: its own window, T 1.0 seeds 4321+, bounded /start_profile {20, 40}); traces kept in
        traces/profile_<arm>_<band>@<n>; analysed by nb4_timeline.analyze."""
        tag = f"{a['key']}_{band}@{n}"
        shutil.rmtree(NB4_PROF_DIR, ignore_errors=True)
        NB4_PROF_DIR.mkdir(parents=True, exist_ok=True)
        rec = L.profiled_load(port, PROMPTS[band], n, NB4_PROF_DIR, secs=6.0, max_new=MAX_NEW[band], spec=True,
                              prefill_tok_s=PREFILL["tok_s"])
        rec.update(band=band, level=n, delay_iters=20, max_iters=40)
        rec["wait_traces"] = wait_traces(NB4_PROF_DIR)
        rec.pop("trace", None)                                 # the lab's trace_summary verdict is not used (wrong graph-mode rule)
        kept = TRACES / f"profile_{tag}"
        kept.mkdir(exist_ok=True)
        rec["kept"] = []
        for f in sorted(NB4_PROF_DIR.rglob("*.pt.trace.json*")):
            shutil.copy2(f, kept / f.name)
            rec["kept"].append([f.name, f.stat().st_size])
        rec["timeline"] = timeline_analysis(kept, tag)
        if a["key"] == "T0":
            rec["op_placement"] = op_placement_all(kept)
        return rec


    def core_timer_rows():
        rows, files = [], sorted(TMP.glob(NB4_TIMER.name + ".*.jsonl"))
        dst = OUT / "core_timer"
        dst.mkdir(exist_ok=True)
        for f in files:
            try:
                shutil.copy2(f, dst / f.name)
                for l in f.read_text().splitlines():
                    try:
                        rows.append(json.loads(l))
                    except Exception:
                        pass
            except Exception:
                pass
        return rows, [f.name for f in files]


    def t0_extras(s, o):
        """T0: the engine-core side (the step timer rows, vLLM's iteration log: n per step in order = the parity sequence)."""
        rows, files = core_timer_rows()
        return {"core_timer_files": files, "core_timer_rows": len(rows), "core_timer": NL.core_timer_summary(rows),
                "iteration": NL.iteration_summary(s.log() or "")}


    def commit_ab(o, s):
        """Arm C inside T0's boot (lead item 4, NB1 commit_ab pattern): a greedy 64-token row in eager, the counters, mode ->
        graph, 8k@8 + 40k@8 windows (n = 4 per micro-batch, captured with CAPS16), the counters, the same greedy row in graph
        mode (bit-identity claim), mode -> eager. Stage 1 only (stage-0 commits are PP-deferred)."""
        ab = {"windows": {}}
        ids = PROMPTS["2k"][2]
        ab["greedy_eager"] = g_e = nb1_complete(s.port, ids, 64, temperature=0.0, logprobs=2)
        ab["stats_before"] = commit_stats(s.port)
        ab["set_graph"] = commit_mode(s.port, "graph")
        for band in ("8k", "40k"):
            lab = f"{band}@8"
            ab["windows"][lab] = run_item(o, s, "C_window_" + lab, lambda: nb4_window(s, band, 8, "C_" + lab))
        ab["stats_after"] = commit_stats(s.port)
        ab["greedy_graph_vs_eager"] = lp_compare(g_e, nb1_complete(s.port, ids, 64, temperature=0.0, logprobs=2))
        steps = sum(int((w or {}).get("steps") or 0) for w in ab["windows"].values())
        ab["split"] = NL.commit_split(ab["stats_before"], ab["stats_after"], steps or None)
        ab["counter_delta"] = commit_counter_delta(ab["stats_before"], ab["stats_after"]) if ab["stats_before"] and ab["stats_after"] else None
        ab["reset_eager"] = commit_mode(s.port, "eager")
        return ab


    def body_arm(a, is_ref, ref_key):
        """One arm (= one boot): short gates + read-back + the change visible (_start), the greedy oracle (vs the base), TF
        2k / 8k / 40k (T0: fresh + repeat = the in-boot A/A), warm-up, the commit / P5 counters around the 4 windows (x reps),
        the profiled windows at the compare bands, T0's extras + the C runtime A/B. A greedy FAIL_HIGH_MARGIN vs the base
        keeps TF (diagnosis) and stops."""
        k = a["key"]

        def body(ss):
            s, o, smp = ss[0], {"arm": k, "is_ref": is_ref, "ref": ref_key}, GpuSampler()
            smp.start()
            try:
                if _start(s, a, o):
                    refs = ("rc2_fp8", "nb1r2_b3") + (() if is_ref else (ref_key.lower(),))
                    o["oracle"] = run_item(o, s, "oracle", lambda: oracle(s.port, key=k.lower(), refs=refs), need_sec=300)
                    vs = ((o.get("oracle") or {}).get("vs_" + ref_key.lower()) or {}).get("status") if ref_key and not is_ref else None
                    runs = ("fresh", "repeat") if k == "T0" else ("fresh",)
                    o["tf"] = run_item(o, s, "tf_2k-8k-40k", lambda: tf_suite(k, s, ["2k", "8k", "40k"], runs=runs))
                    o["reference_stored"] = bool(ROWS.get(k.lower())) and all(
                        (((TFR.get(k) or {}).get(b) or {}).get("fresh") or {}).get("arr") is not None for b in ("2k", "8k", "40k"))
                    if vs == "FAIL_HIGH_MARGIN":
                        o["early_stop"] = "ORACLE_FAIL_VS_BASE"
                        return o
                    o["warmup"] = run_item(o, s, "warmup", lambda: warmup(s.port, 9), need_sec=180)
                    o["commit_stats_before"] = commit_stats(s.port)
                    o["p5_before"] = p5_stats(s.port) if k == "P" else None
                    windows(o, s, a)
                    o["commit_stats_after"] = commit_stats(s.port)
                    steps = sum(int((w or {}).get("steps") or 0) for w in (o.get("steady") or {}).values() if isinstance(w, dict))
                    o["commit_split"] = NL.commit_split(o["commit_stats_before"], o["commit_stats_after"], steps or None)
                    if k == "P":
                        o["p5_after"] = p5_stats(s.port)
                        o["p5_delta"] = {"executed": p5_executed(o["p5_after"]) - p5_executed(o["p5_before"]),
                                         "per_step": NB4_P5.engaged_per_step(o["p5_before"], o["p5_after"], max(1, steps)) if NB4_P5 else None}
                        o["p5_label"] = NB4_P5.p5_arm_label(s.log() or "", o["p5_after"]) if NB4_P5 else None
                    if k in ("S2", "S3") and NB4_PK is not None:
                        o["ppack_label"] = NB4_PK.readback_label(s.log() or "", a.get("variant"))
                    o["profiles"] = {}
                    for pb in a.get("profiles") or []:
                        band, n = pb.split("@")
                        o["profiles"][pb] = run_item(o, s, "profile_" + pb, lambda: traced("profile_" + pb, lambda: nb4_profile(
                            s.port, a, band, int(n))), need_sec=420)
                    if k == "T0":
                        o["profiler_tools"] = profiler_tools()
                        o["t0"] = run_item(o, s, "t0_extras", lambda: t0_extras(s, o))
                        if a.get("commit_ab") and o.get("commit_flip_to_eager") is not None:
                            o["commit_ab"] = run_item(o, s, "commit_ab", lambda: commit_ab(o, s), need_sec=480)
                        else:
                            o["commit_ab"] = {"status": "NOT_INTEGRATED", "why": "T0 booted without the commit-graph env"}
            except Exception as e:
                o.update(status="BODY_ERROR", body_error=f"{type(e).__name__}: {e}"[:800], body_tb=traceback.format_exc()[-2500:])
            finally:
                _finish(s, o, smp)
            return o
        return body


    def run_arm(k):
        """One arm through merlin_boot (boot gate: GPU empty, engine processes gone, page cache dropped, Merlin's host-RAM
        rule; evidence + oracle gates vs the base) with the per-boot body cap; the reference arm stores its rows."""
        a = ARMS[k]
        ref = NB4.get("ref")
        is_ref = ref is None
        name, port = nb4_plan(k, 0)
        BOOT["tag"] = name
        need = () if is_ref else (ref.lower(),)
        o = merlin_boot(name, "fp8", port, min(NL.READY_SEC, max(300, left() - 600)),
                        watched(body_arm(a, is_ref, ref), NL.CAP_MIN[k] * 60, name), need=need, expected_rows=4)
        LIVE["out"] = o
        if LIVE.get("rec") is not None:
            LIVE["rec"].pop("live_body", None)                 # klee100 R1-12: the full record replaces the live copy
        try:
            why = str((o.get("boot_gate") or {}).get("why") or "")
            if o.get("status") == "GPU_NOT_EMPTY" and why.startswith(("HOST_RAM", "ENGINE_PROCS")):
                o["status"] = "HOST_RAM_GATE_FAILED" if why.startswith("HOST_RAM") else "ENGINE_PROCS_LEFT"
            ln = o.get("launch_name") or name
            o.update(arm=k, change=a["change"], plan_tail=((R["setup"].get("nb4_plans") or {}).get(ln) or {}).get("argv_tail"),
                     load_monitor=LOADMON.get(ln))
            srv = (o.get("servers") or {}).get(ln) or {}
            o["device_fault"] = bool(o.get("xid_in_log")) or bool(srv.get("xid"))
            o["oom_in_body"] = not str(o.get("status", "")).startswith("LAUNCH_FAILED") and bool(o.get("oom_in_log"))
        except Exception as e:
            keep_after_error(o, e)
        return o


    def nb4_arm_step(k):
        """step() around one arm. A patch arm (P, S2): its prepare step first (P: build + gate + patch; S2: patch) -> the arm
        is integrated only when it is ok; the revert always runs after the boot (finally). The first clean base becomes the
        reference."""
        a = ARMS[k]
        if a.get("patch"):
            prep_fn, fin_fn = (p5_prepare, p5_finish) if a["patch"] == "p5" else ((lambda: ppack_prepare(a["variant"])), ppack_finish)
            prep = step(k, f"{k.lower()}_prepare", 8 if k == "P" else 2, prep_fn)
            NB4[f"{k.lower()}_prepare"] = {x: prep.get(x) for x in ("status", "ok", "why", "build", "gate", "patch", "so", "so_sha256", "error")}
            new = NL.make_arms(prep if k == "S2" else None, NB4_PIN.get("g_override"), prep if k == "P" else None,
                               prep if k == "S3" else None, NB4_DISABLED)[k]
            if prep.get("status") != "DONE" or prep.get("ok") is not True:
                ARMS[k] = dict(new, integrated=False, why_not=(("P5_NOT_INTEGRATED: " if k == "P" else "NOT_INTEGRATED: ")
                                                              + str(prep.get("why") or prep.get("error") or prep.get("status")))[:400])
                try:
                    NB4[f"{k.lower()}_finish"] = fin_fn()
                except Exception as e:
                    NB4[f"{k.lower()}_finish"] = {"error": repr(e)[:200]}
                rec = {"status": ("P5_NOT_INTEGRATED" if k == "P" else "NOT_INTEGRATED"), "why": ARMS[k]["why_not"]}
                RECS[k] = rec
                save()
                return rec
            ARMS[k] = new
            NB4["arm_problems"] = NL.validate_arms(ARMS)
            if NB4["arm_problems"]:
                NB4[f"{k.lower()}_finish"] = fin_fn()
                rec = {"status": "CONFIG_MISMATCH", "why": NB4["arm_problems"]}
                RECS[k] = rec
                return rec
            try:
                rec = step(k, NL.launch_name(k, 0), NL.EST_MIN[k], lambda: run_arm(k))
            finally:
                NB4[f"{k.lower()}_finish"] = fin_fn()
        else:
            rec = step(k, NL.launch_name(k, 0), NL.EST_MIN[k], lambda: run_arm(k))
        RECS[k] = rec
        if NB4.get("ref") is None and k in NL.BASE_ARMS and NL.base_ok(rec):
            NB4["ref"] = k
        save()
        return rec


    # ---- analysis: correctness vs the base, engaged + mechanism, speed vs base with the A/A spread, labels
    def _tf(k, band, run="fresh"):
        x = ((TFR.get(k) or {}).get(band) or {}).get(run)
        return x if x and x.get("arr") is not None else None


    def tf_vs(ref, k, bands=("2k", "8k", "40k")):
        out = {}
        for b in bands:
            r, c = _tf(ref, b), _tf(k, b)
            if not r or not c:
                out[b] = {"label": "INCONCLUSIVE_MISSING_TF"}
                continue
            ful, lst = cmp_full(r["arr"], c["arr"]), cmp_last256(r, c)
            out[b] = {"label": NL.tf_label(ful), "label_last256": NL.tf_label(lst),
                      "full": {x: ful.get(x) for x in ("status", "n", "expected", "mean_abs_dlogprob", "p99_abs_dlogprob", "max_abs_dlogprob",
                                                       "rank1_agreement", "first_window_over_screen_mean", "bins_4096", "chunk")},
                      "last256": {x: lst.get(x) for x in ("status", "mean_abs_dlogprob", "p99_abs_dlogprob", "rank1_agreement")},
                      "shift": shift_check(r["arr"], c["arr"])}
        return out


    def _windows_of(k):
        return {w: v for w, v in ((RECS.get(k) or {}).get("steady") or {}).items() if isinstance(v, dict)}


    def _w_brief(v):
        v = v or {}
        return dict({x: v.get(x) for x in ("status", "tok_s", "wall_clock_tok_s", "tick_ms", "cycle_ms", "microbatches_per_s", "gen_tokens",
                                            "running_min", "running_max", "running_eq_level", "mean_acceptance_length",
                                            "accept_rate_per_position_incl_admission", "preemptions", "kv_usage_max",
                                            "prefill_done_before_window")},
                    split=(v.get("split") or {}), cudagraph_by_mode=(v.get("cudagraph") or {}).get("decode_by_mode"),
                    cpu_total_cores=(v.get("cpu") or {}).get("total_cores"))


    def arm_brief(rec):
        rec = rec or {}
        return {"status": rec.get("status"), "correctness": rec.get("correctness"), "early_stop": rec.get("early_stop"),
                "launch_name": rec.get("launch_name"), "port": rec.get("port"), "kv_bytes": rec.get("kv_bytes"), "pin_rung": rec.get("pin_rung"),
                "ready_sec": rec.get("ready_sec"), "tries": rec.get("tries"), "death": rec.get("death"), "died_in": rec.get("died_in"),
                "boot_watchdog": rec.get("boot_watchdog"), "config_ok": rec.get("config_ok"), "config_mismatch": rec.get("config_mismatch"),
                "readback_checks": (rec.get("readback") or {}).get("checks"), "d8": (rec.get("readback") or {}).get("d8"),
                "diff_vs_ref": (rec.get("diff_vs_ref") or {}).get("check"), "engaged_log": rec.get("engaged_log"),
                "worker_env": rec.get("worker_env"), "windows": {w: _w_brief(v) for w, v in (rec.get("steady") or {}).items() if isinstance(v, dict)},
                "profiles": {b: {"status": (p or {}).get("status"), "timeline_summary": ((p or {}).get("timeline") or {}).get("summary"),
                                 "stalls": ((p or {}).get("timeline") or {}).get("stalls"),
                                 "graph_modes": ((p or {}).get("timeline") or {}).get("graph_modes")}
                             for b, p in (rec.get("profiles") or {}).items() if isinstance(p, dict)},
                "commit_split": rec.get("commit_split"), "p5_delta": rec.get("p5_delta"), "p5_label": rec.get("p5_label"),
                "graph_capture": {"captured": (rec.get("graphs") or {}).get("captured"),
                                  "rank_graph_gib": (rec.get("pool") or {}).get("rank_graph_gib")},
                "free_mib_by_card": rec.get("free_mib_by_card_all"), "free_source": rec.get("free_source"),
                "oom_in_log": rec.get("oom_in_log"), "oom_in_body": rec.get("oom_in_body"), "device_fault": rec.get("device_fault"),
                "error": rec.get("error"), "body_error": rec.get("body_error"), "postprocess_error": rec.get("postprocess_error")}


    def _tl(rec, band):
        return (((rec or {}).get("profiles") or {}).get(band) or {}).get("timeline")


    def nb4_analysis():
        ref = NL.ref_arm(RECS)
        an = {"ref": ref, "arms": {}, "aa": {}, "winners": [], "nb1r2_b3_windows": NB4.get("nb1r2_b3_windows"),
              "rule": "WIN = correctness PASS vs base AND every decisive window gain > max(2 %, A/A spread), no window SLOWER, "
                      "equal micro-batch splits only AND the mechanism proven (trace / counters); " + NL.WIN_NOTE}
        t0w, t0bw = _windows_of("T0"), _windows_of("T0b")
        an["aa"]["speed"] = NL.aa_spread(t0w, t0bw)
        an["aa"]["tf_t0_fresh_vs_repeat"] = {b: (lambda r, c: {"label": NL.tf_label(cmp_full(r["arr"], c["arr"], bins=False)),
                                                                "last256": cmp_last256(r, c)} if r and c else None)(_tf("T0", b), _tf("T0", b, "repeat"))
                                             for b in ("2k", "8k", "40k")}
        an["aa"]["tf_t0b_vs_t0"] = tf_vs("T0", "T0b") if RECS.get("T0b") and RECS.get("T0") else None
        an["aa"]["oracle_t0b_vs_t0"] = (((RECS.get("T0b") or {}).get("oracle") or {}).get("vs_t0") or {}).get("status")
        an["t0_vs_nb1r2_b3_oracle"] = (((RECS.get("T0") or {}).get("oracle") or {}).get("vs_nb1r2_b3") or {}).get("status")
        an["t0_vs_rc2_fp8_oracle"] = (((RECS.get("T0") or {}).get("oracle") or {}).get("vs_rc2_fp8") or {}).get("status")
        t0 = RECS.get("T0") or {}
        an["timeline"] = {"profiles": {b: (p or {}).get("timeline") for b, p in (t0.get("profiles") or {}).items() if isinstance(p, dict)},
                          "op_placement": {b: (p or {}).get("op_placement") for b, p in (t0.get("profiles") or {}).items() if isinstance(p, dict)},
                          "engine_core": t0.get("t0")}
        ab = t0.get("commit_ab") or {}
        if ab.get("windows"):
            an["C"] = NL.c_verdict(ab.get("windows"), t0w, an["aa"]["speed"], ab.get("greedy_graph_vs_eager"), ab.get("split"),
                                   t0.get("correctness"))
            an["C"]["counter_delta"] = ab.get("counter_delta")
            an["C"]["captured_n"] = NL.commit_graph_ns("\n".join((t0.get("readback") or {}).get("commit_lines") or []))
        else:
            an["C"] = {"label": "NOT_INTEGRATED" if t0 else "SKIPPED_BUDGET", "why": ab.get("why") or ab.get("status") or "T0 not run"}
        for k in NL.ARM_ORDER:
            a, rec = ARMS[k], RECS.get(k)
            e = {"change": a["change"], "integrated": a.get("integrated"), "why_not": a.get("why_not"), "diagnostic": a.get("diagnostic")}
            corr = speed = mech = None
            if rec is not None and not str(rec.get("status", "")).startswith(("SKIPPED", "P5_NOT", "NOT_INT")):
                if ref and k != ref:
                    tfl = tf_vs(ref, k)
                    ors = ((rec.get("oracle") or {}).get("vs_" + ref.lower()) or {}).get("status")
                    corr = NL.correctness(rec.get("correctness"), ors, {b: v["label"] for b, v in tfl.items()})
                    e["tf_vs_ref"] = tfl
                elif k == ref:
                    corr = {"verdict": "REFERENCE"}
                if k not in NL.BASE_ARMS:
                    speed = NL.speed_verdict(_windows_of(k), t0w, t0bw, decisive=tuple(a.get("decisive") or NL.WINDOWS))
                    cb = (a.get("profiles") or ["8k@8"])[0]
                    mech = NL.mechanism(a, {"tl": _tl(rec, cb), "tl_ref": _tl(t0, cb),
                                            "cg": {w: (v or {}).get("cudagraph") for w, v in _windows_of(k).items()},
                                            "log": "\n".join(((rec.get("readback") or {}).get("int8_lines") or [])),
                                            "p5_label": rec.get("p5_label"), "p5_delta": rec.get("p5_delta"),
                                            "ppack_label": rec.get("ppack_label"),
                                            "accept": ((_windows_of(k).get("8k@8") or {}).get("accept_rate_per_position_incl_admission") or {})})
                elif k == "T0":
                    mech = NL.mechanism(a, {"tl": _tl(rec, "8k@8"), "iteration_lines": ((rec.get("t0") or {}).get("iteration") or {}).get("lines"),
                                            "core_timer_rows": (rec.get("t0") or {}).get("core_timer_rows")})
            e.update(correctness=corr, speed=speed, mechanism=mech, brief=arm_brief(rec) if rec else None,
                     label=NL.final_label(a, rec, corr, speed, (rec or {}).get("engaged_log") if rec else None, mech))
            an["arms"][k] = e
            if str(e["label"]["label"]).startswith("WIN"):
                an["winners"].append(k)
        if str(an["C"].get("label", "")).startswith("WIN"):
            an["winners"].append("C")
        return an
    save()


if not RERUN:
    # ==================================================================== CELL 8 — E25 NB4 main: gates, install, T0 -> S -> S2 -> S3 -> G -> P -> T0b, verdict
    def nb4_main():
        """Gates before any GPU (inputs, merge, arms valid incl. the never-do list, embedded files), install, #60490 (fail
        closed), the core timer (not a gate), the sha256 gate, then the arms in the planner's order (nb4_lib.next_arm: lead
        order, T0b protected / moved earlier when the next arm + T0b do not both fit, T0b as the base when T0 is not clean,
        optional K4 / INT8 only with budget)."""
        m = R["sections"]["A0"]["merge"]
        GATES["INPUTS"] = (bool(MERLIN_OK), None if MERLIN_OK else json.dumps(R["inputs"])[:400])
        GATES["MERGE"] = (m.get("ok") is True, m.get("why") or m.get("bad"))
        GATES["ARMS_VALID"] = (not NB4["arm_problems"], NB4["arm_problems"] or None)
        GATES["EMBEDDED"] = (all(NB4["embedded_sha_ok"].values()), NB4["embedded_sha_ok"])
        if all(ok for ok, _ in GATES.values()):
            rec = step("A0", "merlin_install", 10, lambda: {"install": (lambda e: {"ok": True, "tree": e["kv"].get("MERLIN_TREE")} if not FAKE else {"fake": True})(merlin_install())})
            GATES["MERLIN_INSTALL"] = (rec.get("status") == "DONE", rec.get("error"))
        if all(ok for ok, _ in GATES.values()):
            rec = step("A0", "patch_60490", 3, apply_60490)
            GATES["PATCH_60490"] = (rec.get("status") == "DONE" and rec.get("ok") is True, rec.get("error"))
        if all(ok for ok, _ in GATES.values()):
            NB4["core_timer"] = step("A0", "core_timer_install", 2, core_timer_install)
            if NB4["core_timer"].get("ok") is not True:              # T0 keeps vLLM's iteration log; the timer is recorded absent
                ARMS["T0"]["core_timer"] = False
            NB4["arms"] = {k: {x: a.get(x) for x in ("change", "integrated", "why_not", "compile", "extra", "env", "prebuild_env", "spec",
                                                     "profiles", "decisive", "ladder", "reps", "mechanism")} for k, a in ARMS.items()}
            save()
        if all(ok for ok, _ in GATES.values()):
            ok, why = sha_gate(max(60, min(1500, left() - 110 * 60)))      # klee100 R1-7: the wait never starves the arms
            R["setup"]["sha_gate"] = {"ok": ok, "why": why}
            GATES["SHA256"] = (ok, why)
            save()
        if all(ok for ok, _ in GATES.values()):
            while True:
                k, why = NL.next_arm(ARMS, DONE, left())
                NB4["plan_log"].append({"t_min": round((time.time() - T_START) / 60, 1), "left_min": round(left() / 60, 1), "next": k, "why": why})
                save()
                if k is None:
                    if "no clean base" in why:
                        for k2 in NL.ARM_ORDER:
                            if k2 not in DONE and ARMS[k2].get("integrated"):
                                RECS[k2] = DONE[k2] = {"status": "SKIPPED_NO_BASE", "why": why}
                    break
                DONE[k] = nb4_arm_step(k)
                if ABORT["why"]:
                    break


    def nb4_verdict(main_error):
        try:
            an = nb4_analysis()
        except Exception as e:
            an = {"error": f"{type(e).__name__}: {e}"[:600], "tb": traceback.format_exc()[-2500:], "arms": {}}
        labels = {k: (v.get("label") or {}).get("label") for k, v in (an.get("arms") or {}).items()}
        labels["C"] = (an.get("C") or {}).get("label")
        return {"labels": labels, "winners": an.get("winners"), "main_error": main_error,
                "gates": {k: {"ok": v[0], "why": v[1]} for k, v in GATES.items()}, "analysis": an,
                "plan_log": NB4.get("plan_log"), "ref": NB4.get("ref"), "core_timer": NB4.get("core_timer"),
                "prepare": {k: NB4.get(k) for k in ("p_prepare", "p_finish", "s2_prepare", "s2_finish", "s3_prepare", "s3_finish")},
                "disabled": NB4.get("disabled"),
                "modules": {"timeline": NB4.get("timeline_module"), "p5": NB4.get("p5_module"), "ppack": NB4.get("ppack_module")},
                "research": NB4.get("research"), "minutes": round((time.time() - T_START) / 60, 1), "abort": ABORT["why"],
                "notes": ["speed numbers are engine end-to-end; microbenchmarks are never counted (PLAN section 0)",
                          "G and P run on the user's instruction as diagnostics (expected gain below the PLAN 5 % rule)",
                          "a WIN_SPEED needs paired ARC accuracy + PLAN §4.1-7 evidence before adoption"]}


    if not RERUN and not os.environ.get("RC2_UNIT"):
        MAIN_ERROR, VERDICT = None, {"labels": {}, "label": "NOT_REACHED"}
        try:                                       # klee100 R1-11: whatever happens, the verdict, save() and the fallback are written
            nb4_main()
        except Exception as e:
            MAIN_ERROR = {"error": f"{type(e).__name__}: {e}"[:800], "tb": traceback.format_exc()[-2500:]}
            R["errors"].append("CELL 8 main: " + MAIN_ERROR["error"])
        finally:
            try:
                VERDICT = nb4_verdict(MAIN_ERROR)
            except Exception as e:
                VERDICT = {"label": "VERDICT_ERROR", "why": f"{type(e).__name__}: {e}"[:400], "main_error": MAIN_ERROR,
                           "tb": traceback.format_exc()[-2500:]}
            finally:
                R["verdict"] = VERDICT
                R["minutes"] = round((time.time() - T_START) / 60, 1)
                R["host_phase"] = HOST_PHASE
                save()
                try:
                    (OUT / "nb4_verdict.json").write_text(json.dumps(VERDICT, indent=1, default=str))
                except Exception as e:
                    print("nb4_verdict.json not written:", repr(e)[:200], flush=True)
                write_fallback_submission()
                print(json.dumps({k: VERDICT.get(k) for k in ("labels", "winners", "main_error", "gates", "ref", "minutes")}, default=str)[:8000])
                print("@@NB4 " + json.dumps(VERDICT.get("labels"), default=str) + "@@", flush=True)
