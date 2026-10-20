# Kaggle discussion 745787 — "Qwen 3.8 token generation speed" (ARC Prize 2026 ARC-AGI-2)
URL: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/745787
Copied by the user on 2026-10-05. Verbatim text of the thread as pasted:

Darren Amadeus Martin · 477th in this Competition · Posted a day ago
Qwen 3.8 token generation speed
Has anybody got a decent aggregate token generation speed for qwen 3.8 27b or flash next ? In arc agi 3, everybody seems to be using qwen 3.8 flash next since rtx pro 6000 has a very high bandwidth. However 4 L4 seems to be very slow when running these models due to lower bandwidth and slow gpu communication. Has anybody got like an aggregate above 500 tok/s ? Would like to hear how you optimize the inference

CPMP (4th): That's the problem of long competitions. New, stronger models can appear. We got 70+ before Qwen 3.8, and this model reshuffled everything. Next 4 weeks will be hot on LB.

Anil Thomas (8th): doesn't qwen3.8 come with its own challenges? the generations at xhigh effort are very long (median around 64K from what I have seen). from this arcprize.org page, it appears that you need xhigh for good results. it would be challenging to squeeze 240 puzzles (probably around 330 queries) into the time budget. if you turn off thinking, the results are abysmal. if someone has tried submissions solely based on qwen3.8, please post. it will save the rest of us countless GPU-hours!

Anil Thomas (8th): I would think the folks that have posted recent scores above 45 (I am not in this list yet) are doing something super-clever. of course, it is also possible that I am doing something wrong.

juliencst (1593rd): Currently exploring submissions solely on qwen 3.8, and reaching scores of 8.xx I have a few promising experiments locally, will see if that translates to the lb

Russell Kirk (610th): [image, not copied]

Anil Thomas (8th): hm… I would have expected something around 30. back-of-the-envelope calculation assuming 330 queries, 350 tokens/s, mean generation of 64k tokens and 42% accuracy for a full run: 42*12*60*60/(64*1024*330/350) ~ 30. 350 tokens/s on a 4x L4 seems plausible for the FP8 quantized model.

Russell Kirk (610th): Don't assume I implemented it correctly. In ARC-AGI-3, I tried MTP and I set it up incorrectly and didn't get any gain :D

Russell Kirk (610th): https://www.kaggle.com/code/russcore/arcagi2-qwen38-freeform-direct  i shared the notebook, i don't plan to follow that route atm -- and it may have mistakes. But maybe it'll save you some GPU.

Anil Thomas (8th): 'turn_max_tokens': 40960 — that is too low. try setting it to 64k and also increase the total context length.

Darren Amadeus Martin: wow, thank you for the notebook, how was the token generation speed in your experience ?

Darren Amadeus Martin: I did not know that the foundation have benchmarked qwen 3.8. with $0.4469/task and $2.55 per million output tokens, shouldnt it be roughly 170k tokens per task (considering 2 pass per task, maybe 85k-90k ish tokens). Can you tell me the source of 64k median ?

NguyenThanhNhan (49th): @anlthms I managed to submit twice with qwen 3.8 27b fp8 + mtp at 65k tokens and xhigh reasoning effort. Next stop is flash next, but the queue for 4xL4 has been insane

Darren Amadeus Martin: May I ask what did you do on this setup ? With 65k tokens/task (assuming you only did 1 pass), 240 task and 12 hours, you'd need roughly 360 tok/s on aggregate. Not only that, you'd also need to fit the kv cache.

NguyenThanhNhan (49th): @darrenamadeusmartin I could only do 1 pass. I had a custom intervention logic to force the model to stop reasoning and commit with a solution grid after 49k tokens. Therefore, not all 240 tasks would generate 65k tokens. Those 2 submissions used pure reasoning, which is also quite token-inefficient. I think you could solve many more tasks within 12 hours by implementing a simple harness.
Btw, here's my sglang server config:
  "--kv-cache-dtype", "fp8_e4m3",
  "--attention-backend", "flashinfer",
  "--linear-attn-backend", "triton",
  "--chunked-prefill-size", "2048",
  "--max-prefill-tokens", "4096",
  "--mamba-radix-cache-strategy", "extra_buffer",
  "--mamba-full-memory-ratio", "0.5",
  "--mamba-ssm-dtype", "float32",
  "--speculative-algorithm", "EAGLE",
  "--speculative-num-steps", "3",
  "--speculative-eagle-topk", "1",
  "--speculative-num-draft-tokens", "4",

Darren Amadeus Martin: Thank you for your explanation. How much concurrency do you use ? If you don't mind, can you share your setup notebook? Ive been having trouble trying to get decent speed

Darren Amadeus Martin: your words made me want to assume that your whole pipeline is LLM+harness haha. LB movement currently is very interesting as many have reached beyond 45 on LB
