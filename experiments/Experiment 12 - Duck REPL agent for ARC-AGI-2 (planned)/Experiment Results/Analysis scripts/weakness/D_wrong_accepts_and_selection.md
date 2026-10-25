# Reader D: wrong accepts, attempt_2, prompt pressure

The reader re-ran the unsubmitted functions with a scratch replay and scored them against the solutions.

## Q1: the four wrong accepts

**Common pattern:** in every case the model ran its transform on the test input before submitting, but only checked that the output matched *its own rule*. Each failure comes from an arbitrary tie-break that fit the training pairs and that the test input exercises.

- **EVAL6 b5ca7ac4_b1.**
  - **True rule:** border-8 boxes slide left and border-2 boxes slide right; the box nearest the wall lands first and later boxes stack on it.
  - **Implemented:** an overlap graph, 2-coloured with "topmost box of a chain → inner lane" (L2492-2499).
  - It called this "a bit arbitrary" (L2512) and dropped the column idea (L2350, L2538).
  - **Test:** it printed its own prediction, "-> expect cols 12,17,12,17" (L4246-4260), then wrote "looks structurally correct… Submitting".
- **EVAL30 b5ca7ac4_b0.** A fixed asymmetric greedy order, called "strange" and "odd" (L2060, L2071), but kept as "a fixed rule that fits all the data" (L2744). It hand-simulated the test: "matches my prediction perfectly".
- **EVAL30 4c3d4a41_b0.**
  - **True rule:** shift each column up by its overlap with the staircase.
  - **Implemented:** a hard-coded "remove exactly one top 5" (L819, L841). It had considered the right rule (L843).
  - **Test:** needs a shift of 3 and a shift of 0, neither seen in training. It waved this off ("In the training, that doesn't happen…", L844).
- **EVAL30 3dc255db_b0.** Ejects dots from the container's "tip"; with several tips it picks the one with the most free space (L1227-1246). On the test chevron this picks the wrong tip. "I'll trust this" (L3619), then "I'm confident. Submitting" (L3854).

## Q2: attempt_2

- **How it is filled:** `select_attempts` (arc2_duck.py L553-581) uses accepted outputs, then failed-submission outputs, then the test input.
- **What happened:** in all 56 query slots (EVAL6 8 + EVAL30 48), attempt_2 was the test input and was never correct. Every branch that submitted did so exactly once and was accepted.
- **Root gap:** a function that passes `check()` but is never submitted is invisible to selection.
- **Verified rescues by replay:**
  - **EVAL6 b5ca7ac4_b0** `transform2`: train 3/3 and exact on the test. As attempt_2 next to b1's wrong output, the task would have been rescued.
  - **EVAL6 62593bfd_b1:** exact on both test inputs.
  - **Net:** harvesting check-exact functions would take EVAL6 from 4/8 to 7/8 queries (5/6 tasks).
- **EVAL30:** no cross-branch rescue, because b1 never ran.

## Q3: prompt pressure

- **Against overfitting:** arc2_duck.py L229-231 ("make it general… check the rule fits the structure of the test inputs"). The model used it only as a self-consistency check.
- **Toward overfitting:**
  - The only stated acceptance criterion is train-exactness (L180, the nudge L281).
  - "Prefer the simplest rule" made it pick the simplest tie-break.
  - Nothing asks it to list alternative rules that fit training and compare them on the test. 62593bfd_b1 did this on its own and was right.
  - Acceptance ends the branch, so no alternative is recorded.

## Main-session correction

The reader's "minor" point is **not a bug**: the ARC-AGI-3 nudge text appears in the transcript only because Duck logs it before the adapter rewrites it. The vendor builds it at tool_agent.py L6983-7038, marks it as a `nudge`, and the adapter's `_append_context_message` (arc2_duck.py L344) replaces the game tail before the message is sent. This reader was launched before the correction was sent.
