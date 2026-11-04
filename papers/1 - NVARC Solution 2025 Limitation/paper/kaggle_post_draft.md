# DRAFT: Kaggle discussion post (not posted; needs Hamid's approval of the text)

**Title:** Where the NVARC pipeline saturates on the public ARC-AGI-2 eval: the candidate pool, not selection

We built 34 notebooks on the NVARC 2025 solution and re-scored all 41 distinct results they produced, using the public evaluation set (120 tasks, 172 test inputs). For two full runs, the logs also report where the answer ranks among the decoded candidates for every test input. A short summary of what we measured:

- **Selection headroom is small.** A correct grid was in the candidate pool for 55 and 56 of 172 test inputs. Picking the right candidate every time (an oracle) would raise the score from 33.50 to 41.17 and from 35.83 to 40.50, i.e. only +4.67 to +7.67. Every in-pool answer was ranked within the top 20. Submitting the kgmon top-2 instead of the probmul top-2 changes the score by -1.67 to +1.33.
- **About 68 % of test inputs never get a correct candidate** (117 and 116 of 172). For 48 and 47 of them, the candidates are on average more than 90 % cell-correct, yet none is exact.
- **Low diversity.** There are about 14 candidates per test input but only about 7.3-7.6 distinct grids. In one case all 16 candidates are the same wrong grid.
- **Re-runs re-solve the same test inputs.** The union of five full runs grows 51 -> 57 -> 58 -> 61 -> 62 test inputs, and 37 of them are solved in all five. The union of all 41 submissions is 64 / 172 (46.50).
- **Run-to-run noise.** Two runs of the same code with identical seeds solved 49 and 45 test inputs. On the LB, one notebook version scored 27.36-32.64 over 10 submissions.

One practical note: the dual selector in these notebooks submits the probmul top-2, so kgmon never changes the submitted pair.

What we did *not* measure: more samples or views, a diversity term in the search, stochastic sampling or a larger model. These are where any remaining headroom would have to come from.

Paper (arXiv link: TODO) and scripts/data: TODO link.
