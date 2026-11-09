# Audit C (independent rules and presentation review), 2026-11-10 ~05:15 BDT
Verbatim agent report (read-only; nothing edited, nothing changed on Kaggle).

Word count: WRITEUP_kaggle_body.md 1,133 whitespace tokens (1,077 without markdown-only tokens); about 1,110 with title/subtitle; limit 1,500.
Read-only Kaggle API checks: notebook foysalemonshanto/q27-xhigh-e25-nb6-submit is_private false, version 1, GPU on, internet off; dataset q27-e15b-payload Apache 2.0; models p1..p5 private.

MUST-FIX
1. Submission notebook depends on five private Kaggle Models qwen3-8-flash-next-w4a16-merlin-p1..p5 (is_private true for p1, p2, p5 checked; p3, p4 same batch). Public notebook not reproducible; conflicts with "real, working Kaggle code submission ... code and methods open source" and rules line 12 (only Writeup-attached resources become public). Fix (user on Kaggle): make p1-p5 public before 05:59 (licence field already Apache 2.0) or attach to the Writeup; also check e23-merlin-env and qwen38-flash-next-fp8ple-overlay (API gives no is_private, probably public; overlay description says "Private mirror ... License: Qwen Community License 1.0").
2. No licence statement for team code in Writeup or PDFs (grep licen/MIT/CC0/open source: nothing). Suggested Writeup line: "Licence: our notebook code, the ARC-AGI-2 adapter (arc2_duck.py) and our analysis scripts are released under MIT-0. Franzen's Duck fork is Apache-2.0. The model weights keep their upstream licences (Qwen3.8-Flash-Next; Intel AutoRound W4A16; albucino FP8-PLE overlay)."
3. Team/author mismatch: ARC-AGI-2 team Hack2Publish has 5 members incl. Raihan RT; Paper Track team "FOYSAL" has 4 incl. Tarek Rahman; Writeup/PDFs list 3 authors as Team Hack2Publish. Suggested line 1: "Authors: Md. Hamid Hosen, MD. Foysal Emon Shanto, Md Esfer Abdus Sami (Kaggle Paper Track team FOYSAL, with Tarek Rahman; ARC-AGI-2 submissions under team Hack2Publish)." ARC-AGI-2 team disqualified/unranked = eligibility risk not fixable by editing.
4. Companion PDF "Code and data ... available from the authors on request" contradicts open-source requirement; add a Writeup line with a public URL only if one exists, else accept risk.

SHOULD-FIX
5. Main claim and 54.17 not in first 150 words (claim at word 155, 54.17 ~word 520). Insert after line 1: "Main result: a Qwen3.8-Flash-Next tool agent on 4x L4 scored 54.17 on the ARC-AGI-2 public leaderboard (submission 57851995). We show that a faster 3-bit build of the same agent verified as many answers but was right less often (61/79 vs 50/79 on matched tasks), and we propose verified-answer precision as the acceptance test for any speed or compression change."
6. Figure numbers conflict with PDF (PDF Fig 1 = LB history, Fig 2 = pipeline); Writeup order 1, 3, 2. Rename captions: pipeline -> Figure 2 (PDF Figure 2); verification -> Figure 3; LB history -> Figure 1; drop "(media gallery)".
7. "The 15 train-verified wrong programs" ambiguous; replace lead-in: "We read all 15 train-verified wrong programs from our two smaller quality runs (W4A16 on 15 tasks: 2; 3-bit on 30 tasks: 13):"
8. Define "3-bit": "a faster "3-bit" build of the same agent (mixed W3/W2 AutoRound experts, about 3 bits per weight; called 3-bit below)."
9. Prior work numbers vs PDF Table 1: "scored 75.7-87.2 at about 680-1,240 tok/s (one of them with a different model)".
10. Credit third-party weights: "4-bit routed experts (Intel's W4A16 AutoRound checkpoint) and albucino's FP8 n-gram embedding (PLE) overlay".
11. Add a short References line with links (Duck harness, Franzen fork, Merlin, notnitsuj writeup, Qwen3.8-Flash-Next).
12. Notebook name "Q27": add "; the "q27" prefix is historical: the notebook serves Qwen3.8-Flash-Next, not the 27B model."
13. Check PDFs under Files are publicly downloadable; add main PDF as Project Link if allowed.
14. PDF presentation (needs rebuild; skip if short on time): A.8 table has no caption; internal jargon in A.6/A.8 (E28 levers, RERUN_PROD, CAPS16, C12, I8, Q-defer, measured F); companion PDF has no date line. Notebook first markdown cell has internal notes; leave it (editing creates v2).

OK: word count ~1,110/1,500; Main Track; project link is real public notebook v1 matching 57851995; PDFs have required sections; both PDFs have AI-use statements; titles/authors identical across Writeup and PDFs; page counts 16/18 match; checked numbers match (79/61/50, p 0.08/0.043, 24/26 vs 22/28, 65.0 vs 53.75, 63 vs 57, 49.58, 13/13, 107/143/161, 738k); main PDF cross-refs resolve; no overfull lines/orphan headings on pages 3, 5, 9, 14; payload dataset Apache 2.0.
