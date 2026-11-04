# Qwen 3.8 Flash Next is All you Need

Author: darrenamadeusmartin | topic id 755564 | votes 6 | created 2026-11-03 23:19:52.500000
URL: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/writeups/Domdolus-Tolus-Solution

# I. Opening
We were basically idle for most of the competition: we tried several ways to improve our score but failed to even get past 35% on the public leaderboard. What gave us hope was the announcement of the second Milestone Prize in ARC-AGI-3 and Qwen 3.8 27B managing to score above 40 on the semi-private eval, as released by the ARC Prize Foundation. This gave us hope that by using the right harness with a much more advanced variant of Qwen 3.8, the Flash-Next variant, we might have a good shot at a gold medal or even the leaderboard prize. Our code is almost entirely written by Claude Opus 5.5 with Claude Code. On a side note, this writeup is also written with the help of Claude.

# II. Framework & Budget
We used Claude Code and connected it to the Google Colab MCP (a connector that lets Claude Code create and run notebook cells on Colab by itself) so that it could test new ideas on notebooks with an L4. The only thing we asked Claude Code to do was to optimize inference speed down to the kernel level, as we have no experience in this field. We also gave it ideas of our own, such as which model to use, KV cache quantization, and harness design. We used an RTX PRO 6000 notebook on Colab for quick accuracy tests. If a new setting or a new kernel-level optimization proved to improve our score or inference speed, we pushed the notebook to Kaggle and ran a full 6-hour validation on the public evaluation data. On the Kaggle side, we pushed our notebooks with GitHub and the Kaggle library, while also using the new Kaggle MCP to stream the notebook logs. Our budget went to one month of a Claude Max account, which cost roughly ¥21,250, and to Google Colab compute, where we ended up spending ¥33,679. That sums up to ¥54,929, or approximately $350.

# III. Pipeline
In short, our pipeline uses a simple REPL-based harness on a 2-bit Qwen 3.8 Flash-Next with a patched vLLM to increase its inference speed. The full details are as follows.

### 1. Model
Qwen 3.8 Flash-Next is a Mixture-of-Experts (MoE) model: every layer has many small feed-forward networks called experts, and a router picks only a few of them for each token, so the model is big but only a small part of it is used for every token. Most of its weights sit in these experts. Its layers are a mix of two types: Gated DeltaNet layers, which keep a fixed-size memory of the conversation (the DeltaNet state) instead of looking back at every earlier token, and Qwen Sparse Attention (QSA) layers, which do look back at the earlier tokens. It also uses hyper-connections, which keep several parallel copies of the hidden state and mix them with small mixer matrices between layers, and a very large n-gram table, a lookup table of extra embeddings picked by the last few tokens, which we keep in host RAM.

For the experts and the n-gram table, we used the Q2_0 checkpoint from ISTA-DASLab, and for the rest of the model, we used the official BF16 base model. ISTA-DASLab used a quantization method called GSQ [1]. In 2-bit, every weight can only take one of 4 fixed values (levels), multiplied by a scale shared by its group of weights. Instead of rounding each weight to its nearest level, GSQ uses calibration data to choose the level of every weight so that each block's output stays as close as possible to the base model itself. The model card, which can be accessed [here](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), also mentions a method called RCO [2], which chooses how many bits each part of the model gets; it did not end up in the model we used, as those parts were imported from the base model. Since the file was in GGUF format (the file format of llama.cpp), we needed to convert it to safetensors so that we could use vLLM instead of llama.cpp. Fortunately, Q2_0 stores the same kind of numbers as vLLM's own 2-bit format, so the conversion was only a repacking, without any extra rounding. The shared experts, routers, embeddings, norms, and vision tower were kept in BF16, while the attention and DeltaNet projections, the hyper-connection mixers, the LM head (the last layer, which gives a score to every token in the vocabulary), and the n-gram table are stored in FP8 (8-bit floating point).

### 2. Harness Design
Our harness is heavily inspired by the DUCK harness by Tufa Labs, which can be accessed [here](https://www.kaggle.com/code/jeroencottaar/taaf-duck-harness-kaggle), by the approach of Daniel Franzen (@dfranzen), the winner of ARC-AGI-3 Milestone 2, and by the agentic coding REPL from the pivotools team (Dibya Chakravorty, Debsankha Manik, and Bernhard Altaner), which showed that simply giving LLMs a REPL increases their accuracy on ARC-AGI-2 [3]. Our harness is built directly on their open-source code, which can be accessed [here](https://github.com/gutfeeling/arc-agi-2-submission). The model is given the puzzles in image format, and it can only access the text format by calling the REPL (a live Python session where the model runs code and reads the output) in a tool call. We also gave the model several helper functions, such as bounding boxes and a check function that tests a candidate function on the training examples. It can only give its final answer if it creates a function that passes all training examples. If it finishes without giving such a function, or if the function gives an error, we ask the model for a correct function and let it continue reasoning. We limit each task to 2 tries, which are independent attempts at the same task, with a maximum of 200k decoded tokens each. If it fails to give a function and runs out of budget, we use its best function to produce the output. If both tries give the same answer, we use one of the functions generated along the way that produces a different answer, to enforce diversity.

Compared to the RTX PRO 6000 in ARC-AGI-3, 4 L4s are much slower and can only reach a reliable generation speed at a high level of concurrency. This is because generating one token means reading the active weights from memory, and that cost is shared by every task generating in the same step, so the more tasks run together, the more tokens we get for the same work. However, the attention layers have to keep the keys and values of every earlier token in GPU memory (the KV cache), so their memory grows with every token, while the DeltaNet state always stays the same size. Using a 262k context at high concurrency is definitely impossible, which is why we used a short-term memory window that works as follows. The model is first given the system prompt including the images, and then it generates tokens until the context reaches 64k tokens. When it reaches 64k tokens, we cut the middle of the attention cache and keep only the system prompt and the last generated tokens, 48k tokens in total, while preserving the DeltaNet state so that the model doesn't forget the previous generations entirely. In the end, we managed to run 70 tasks concurrently. In total, the final notebook produces roughly 30 million tokens in the 12-hour run (estimated from our 6-hour validation).

### 3. Parallel Layout
We split the model over the 4 cards with tensor parallelism (TP), where each card holds a quarter of every dense layer, and expert parallelism (EP), where each card holds a quarter of the experts. Both of these are standard vLLM options. The 4 L4s can only talk to each other over PCIe, which is slow, so most of our changes here remove duplicated memory or communication.

The QSA layers only have 2 KV heads, so with 4 cards, every head is stored twice, on 2 cards, which wastes memory. We used decode context parallel (DCP) to stop this. In short, the 2 cards that share 1 attention head each hold only half of the conversation instead of the whole thing, each computes attention over its own half, and then they combine their partial attention outputs into the final result. In QSA, only a selected part of the earlier tokens is used to generate the next token, and a small scoring network called the indexer picks those tokens. Normal vLLM finishes the indexer before sending the query to the other card, which wastes a little time, so we send the query in the background while the indexer runs. We also changed the transfer of the partial attention outputs between the 2 cards from FP32 to BF16, so less data has to travel between the cards.

For sampling, which is choosing the next token from the output scores (logits, one score for every token in the vocabulary), normal vLLM gathers the logits on all 4 cards, so sampling is done 4 times; we assign each request to 1 card, which removes the redundant sampling. Lastly, since we only use a top-k of 20, meaning only the 20 most likely tokens can be picked, the rest of the vocabulary doesn't matter for sampling, so we made each card send only its 64 best tokens and their ids instead of its whole part of the vocabulary.

### 4. KV-Cache Capacity
The KV-cache capacity is how many tokens of all running conversations fit in GPU memory at the same time, and it decides how many tasks we can run together, so we tried to squeeze as much as we could out of it. In the final notebook, vLLM reports a capacity of about 4.4 million tokens, which is roughly 63k tokens per task at 70 tasks.

We stored the KV cache in 4 bits instead of the usual 8 bits, which greatly increased our capacity. A few channels of the keys have very large values, which makes plain 4-bit rounding inaccurate, so we first multiply the keys by a Hadamard matrix (a fixed matrix of +1 and −1 values), which spreads those large values evenly over all channels. The query is rotated the same way, so the attention scores stay the same. Every token also stores its own scale and offset, so its 4-bit values fit its own range. The indexer also keeps a small compressed key for every 4 tokens in BF16, so we replaced that with FP8 (the FP8 index keys).

vLLM stores the cache in pages, which are fixed-size blocks of tokens, and by default, 12 QSA layers share each page, which can waste memory in the empty part of each conversation's last page. Switching to 4 layers per group made the pages 3 times smaller and reduced this waste. The DeltaNet state, which is normally stored in FP32, was also changed to FP16 with stochastic rounding, which freed memory without hurting accuracy. Stochastic rounding rounds each value up or down at random, more often towards the closer side, so the rounding errors cancel out over many steps instead of adding up.

The rest of the memory trims are:
- One shared top-k buffer instead of separate ones;
- A rotary position cache sized to the actual window instead of the model's maximum length;
- FP8 mixer weights padded to multiples of 64 instead of 256/128;
- Token embeddings stored in host RAM, as only a few rows are needed per step, so reading them over PCIe is cheap;
- A GPU memory utilization (the share of GPU memory vLLM is allowed to use) of 0.945, which is close to the limit;
- An indexer scratch buffer of 128 MB instead of 512 MB;
- Images capped at 2 MP and video turned off, so less memory has to be reserved for the vision tower;
- Expandable segments, a PyTorch memory setting that reduces the unused gaps between memory blocks.

### 5. Recompute Prevention
LLM inference has two kinds of work: prefill, where the model reads the prompt, and decode, where it generates new tokens one by one. Prefill is expensive, and every time the model calls the REPL, the harness sends the whole conversation back as a new request, so without any cache, vLLM would prefill the whole conversation again after every tool call. vLLM's prefix cache prevents this by reusing the cache of the part that is the same as an earlier request. However, in a hybrid model like Flash-Next, a cache hit also needs a saved DeltaNet state at exactly that point, as the state cannot be rebuilt without reading all the tokens again. So we implemented several things to prevent recomputation.

Prefix cache hits can now land every 800 tokens instead of every 6,400, which increases the hit rate. DeltaNet states are kept at every 800-token boundary instead of only at the end of each prompt, which prevents recomputing the previous replies. DeltaNet states only exist at every block, or every 1,504 tokens, so a follow-up call had to recompute everything after the last saved state; by parking the exact final state of each call, which we call exact resume, we prevented that recomputation too. When the harness sends the conversation back as text, retokenizing it (turning the text back into tokens) might give slightly different tokens, which causes a mismatch and another recomputation, so we reuse the model's own token ids wherever the text is unchanged (token-id reuse). We also keep only the newest DeltaNet state of each conversation and store it in RAM, to stop old DeltaNet states from eating up the memory. At high concurrency, when many conversations reach their peak length at the same time, some of the cache might get evicted (removed from GPU memory to make space), which causes recomputation, so we copy what is about to be evicted to RAM (RAM offload); if it's needed again, it is restored instead of recomputed.

We also saw a collapse where the speed suddenly dropped and prefill kept rising without any recovery. It was caused by vLLM evicting the least recently used conversation, which is often the one that has waited the longest and is about to be called next. That conversation then had to be recomputed, which evicted another one, causing a chain of evictions and prefill-heavy work. To prevent this, our collapse fix holds back some calls when memory is under pressure and releases the held calls one by one to avoid heavy recomputation. This also gave us stable inference.

### 6. Decode Kernels and Graphs
We also did some kernel-level optimizations. A kernel is a single program that runs on the GPU, a GEMM is a matrix multiplication kernel, and fusing means doing two operations in one kernel, so the result in between doesn't have to be written to memory and read back. Our changes are:
- Switching the MoE GEMM to the indexed type, which works directly on the list of tokens routed to each expert instead of first reordering them into groups;
- Splitting the long input of the hyper-connection GEMM into chunks (split-K): this GEMM has a long input but a small output, so only a few parts of the GPU had work, and the chunks let more of the GPU work at the same time before the results are added up;
- Removing a redundant Hadamard rotation of the query;
- Computing the cut positions of the short-term memory window once per step instead of in every QSA layer;
- Top-k on BF16 instead of FP32;
- A fast Hadamard rotation with a butterfly transform, which applies the rotation in a few cheap add-and-subtract passes instead of a full matrix multiplication;
- Removing redundant Gated DeltaNet copies, and using fewer tiles for the DeltaNet state copies;
- Fused kernels for the hyper-connection down and SiLU, for the QSA output gate, and for the selected-token shift-back;
- Sparse CUDA graphs: a CUDA graph records all kernel launches of one step once and replays them, which removes the launch overhead of the many small kernels, but a graph only works for the batch size it was recorded with, so we recorded graphs for a sparse set of sizes that also covers the mixed steps where decode and prefill happen at the same time;
- Using NCCL instead of vLLM's custom all-reduce: all-reduce is the step where the cards add up their partial results after each layer, and vLLM's custom version is made for cards with a fast direct link, so over the L4s' PCIe, NVIDIA's NCCL library was faster.

### 7. Fast Server Startup
Server startup used to take as long as 13 minutes, but we managed to get it under 4 minutes by reading the utility packages (our offline copy of vLLM and its libraries) into RAM in parallel, prefetching the model in the background, making each card read only its own experts, loading the n-gram table once into /dev/shm (a folder that lives in RAM) without copying it, and restoring prebuilt Triton, Humming, and FlashInfer kernel caches so the kernels don't have to be compiled again at every start.

# IV. Submission
Because of sampling and temperature, we submitted some notebooks several times to get a luckier submission. However, in the final days of the competition, despite faster kernels, our score went down, which might have been caused by the server issues.

Our first submission already used the same 2-bit model, the REPL harness with images, and DCP, along with an FP8 KV cache, the FP16 DeltaNet state, the prefix reuse, the RAM offload, and most of the memory trims. It ran 55 tasks at once with a single try of up to 100k decoded tokens per task, and a much shorter short-term memory window: the middle of the attention cache was cut at 32k tokens, keeping 16k. Its context window, which is the longest a conversation is allowed to grow, counting the parts that were already cut, was 262k tokens, and if a conversation reached it, the harness trimmed the conversation itself down to about 16k tokens. Every row below lists what changed compared to the submission before it, and notebooks we submitted more than once share one row with all of their scores. Our full submission history is as follows:

| # | Public LB | What changed |
|---|---|---|
| 1 | 51.67 | First submission (see above) |
| 2 | 65.14 | 3 rounds: the task as is, then transposed, then mirrored, each starting from the best code so far |
| 3 | 58.75 | 2 tries, grids given as text |
| 4 | 66.81 / 65.42 | 2 tries, grids as images only |
| 5 | 63.33 | Smaller KV pages, newest DeltaNet state only, sampling on 1 card, embeddings in RAM; 60 tasks at once |
| 6 | 66.25 | 4-bit KV cache, DeltaNet state in RAM, FP8 index keys; 100 tasks at once |
| 7 | 62.50 | + exact resume; 100 tasks at once |
| 8 | 71.53 | Harness fixes; 80 tasks at once |
| 9 | 59.72 | RAM offload off, token-id reuse, informed try 2; 80 tasks at once |
| 10 | 71.67 | 3 tries, RAM offload back on; 80 tasks at once |
| 11 | 72.92 / 73.47 | + fast server startup |
| 12 | 78.47 | 48-64k short-term memory window, prompt fixes, 3 plain tries, 100k tokens; 60 tasks at once |
| **13** | **82.08** / 80.00 | 2 tries, 150k tokens |
| 14 | 77.64 | + faster kernels and communication (no collapse fix) |
| 15 | 79.44 | 70 tasks at once, collapse fix, 200k tokens, 1M window |
| **16** | **83.47** / 72.36 / 74.58 / 79.19 / 78.06 | + sparse CUDA graphs, kernel trims, no token numbers in the prompt |

Some of the harness changes in the table need a short explanation:
- Harness fixes: long tool outputs are cut at whole lines with a note on what was left out, the bounding-box helpers use the most common color of the grid as the background by default, and the conversation is kept after every rejected final answer;
- Informed try 2: the second try only starts after the first one has ended, and it is told to find a different rule, along with the first try's best code;
- Prompt fixes: we removed contradictions between the prompt, the chat template, and the harness, such as the format of the final answer;
- Plain tries: tries on the task as is, without the transposed or mirrored versions;
- 1M window: the context window was raised from 262k to 1M tokens, so the harness never has to trim a conversation, and only the short-term memory window is used;
- No token numbers in the prompt: the model is no longer told its token budget or the size of its window.

# V. Insights
Below are some insights on what we tried and what we learned in this competition.

### 1. Improved NVARC's Solution
We asked Fable 5 to port last year's winning NVARC solution to vLLM, which handles many requests in parallel much better. NVARC fine-tunes a small model on every task at test time and then searches for answers with a depth-first search (DFS). The port ended up making the pipeline faster, which allowed a lower DFS threshold (so more candidate answers are explored) and more training epochs, but it did not translate well to the leaderboard.

### 2. ARC is a Vision Problem
We found a very interesting paper that uses a small neural network for ARC-AGI-2 by turning every input and output into an image with different canvas sizes [4]. We tried to imitate this pipeline and trained it on NVARC's winning data on Kaggle's TPU, which is fast, hoping to get a higher score, but it failed miserably. We did not finish the training, as it seemed to plateau quickly before showing any hope of a high score.

### 3. GSQ-RCO and Qwen 4
The algorithm developed by ISTA-DASLab is a very interesting way of quantizing, as it offers very high-quality quantizations close to the base model at the same bit width. This model was the reason our main pipeline works. Their quantization method might be the future of more accurate and accessible quantizations, albeit at a higher cost than simple GPTQ. There were also rumors about Qwen 4, and we even planned to apply the same quantization if Qwen 4 arrived before the competition ended, but that did not happen.

### 4. 4-bit KV Cache
Since DeepSeek V4.1 Flash already uses a 4-bit KV cache [5] and our model is already quantized below 4 bits, we were sure that using a 4-bit KV cache would not hurt our pipeline, and it worked like a charm. We managed to increase capacity without hurting accuracy at all. We haven't tried the same settings on other pipelines, but we could also argue that since the LLM's reasoning and results are verified with code, the quality of its answers is protected by the code safeguard.

### 5. Short-Term Memory
Our short-term memory approach, where the middle part is cut every time the conversation reaches a certain length, helped us increase concurrency. We also realized that increasing the length of the window and how many tokens we save helps the model retain its recent reasoning. This might also be explained by Qwen's DeltaNet, which acts like a short, fuzzy memory. However, just like the KV cache argument, we could argue that the lack of quality loss is helped by the code safeguard.

### 6. Short Experiments on Flash-Next
We found a paper saying that the Mamba state, which is similar to Qwen's DeltaNet state, can be used for test-time tuning, where the model uses the Mamba states of previous successful answers to make itself smarter [6]. We tried the same approach: if an answer managed to create a transform function, we used the average of those states to start a new generation instead of a clean DeltaNet state. However, this did not affect the intelligence at all in our experiment. We also tried the 4-bit model with an 8-bit KV cache, using RAM to store the KV cache, as we had seen on Reddit that this is possible, because only some of Flash-Next's layers need a KV cache, as seen [here](https://huggingface.co/Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound). However, we did not manage to get the pipeline working in time, even though the 4-bit model seemed smarter and reached the same accuracy with fewer tokens.

### 7. Other Model Variants
We also tried other GSQ-RCO models such as Qwen 3.8 27B, which can be accessed [here](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ), Qwen 3.6 35B-A3B, which can be accessed [here](https://huggingface.co/ISTA-DASLab/Qwen3.6-35B-A3B-2Bit-GSQ), and a Swift variant of Qwen 3.8 Flash-Next, which is claimed to be smarter and to use fewer tokens and can be accessed [here](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), but all of them fell short under the same settings, so we did not proceed with them. We also tried DeepSeek V4.1 Flash after the release of the 1017 version, which can be accessed [here](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), but unfortunately we couldn't get it to run at a decent speed on the L4s, so we abandoned that idea too.

# VI. Final Notebook
For the private leaderboard, we chose the 2 notebooks with the highest public leaderboard scores, 83.47 and 82.08. Their differences can be seen in the table in Section IV. The full notebook, along with the models, the utility notebook, and the server loading cache, is attached and can be accessed [here](https://www.kaggle.com/code/darrenamadeusmartin/domdolus-tolus-arc-agi-2/).

Please note that the shared notebook was made for Kaggle's old Docker image, so some updates need to be made in order to run it on the new image, such as rebuilding the utility packages and the kernel caches for the new image's Python and CUDA versions.

The notebook does not use fixed input paths. Its first cells search the whole /kaggle/input folder for the files they need: the competition data by the name of its test file (arc-agi_test_challenges.json), the vLLM utility by its manifest file (UTILITY_MANIFEST.json), the model by the folder that holds its config.json and .safetensors files, and the kernel caches by their file names. This way, the notebook can still find everything if Kaggle changes where the attached inputs are placed. If a search finds none or more than one of the same input, the notebook stops and lists what it found instead of guessing.

The notebook also uses a time limit of 12 hours minus 5 minutes, counted from the start of the notebook's kernel. Kaggle's 12 hours count from the start of the session, which is a little earlier, so the 5 minutes keep us safely inside the limit. One minute before this limit, the tries are stopped, and a background timer then stops the harness and the vLLM server and writes `submission.json` from the answers so far. The submission file is also rewritten every time a try finishes, so a valid file exists at all times.

# VII. Message for the Host and Staff
We would like to express our sincere gratitude to the ARC Prize team and all the Kaggle staff for this amazing competition, and we hope it continues in the future. We also hope that the ARC Prize team and Kaggle staff might look into some of our suggestions below:

- For notebooks with multiple editors, it would be very helpful if the other editors could see the live logs of a running notebook, even if they are not the one who pushed it.
- It would also be very helpful if Kaggle had an MCP that lets agents run code live in Kaggle environments, similar to the Google Colab MCP.
- The competition might not have ended smoothly, but we genuinely hope for more transparency on vital issues, as many important messages were posted outside the Kaggle discussion forums, which left those without an X or Discord account less informed about these issues.

# VIII. Closing
We would like to thank the Qwen team for the amazing model and ISTA-DASLab for an extraordinary quantization method. We also want to thank Tufa Labs, Daniel Franzen, and the pivotools team, who gave us a lot of inspiration for building a harness for this competition, and the pivotools team especially, as our harness is built on their open-source code. A member of our team posted a controversial discussion topic, and we would like to apologize for any bad impression that our posts on the official discussion forums and Discord may have given. Despite that, we are very grateful for the support from the whole discussion forum, where we got a lot of inspiration too. We genuinely hope that ARC Prize continues in the future with a more stable competition infrastructure, giving more people access to compete and exchange knowledge with one another. As a team, we are very grateful to have reached rank 4 on the public leaderboard. At the time this writeup was finished, the private LB was not out yet, but we hope to keep the same rank. Lastly, we want to congratulate all teams on their achievements, and if anything in this writeup is unclear, feel free to ask in the comments below.

# References
[1] A. Dadgarnia et al., "GSQ: Highly-accurate low-precision scalar quantization for LLMs via Gumbel-Softmax sampling," arXiv preprint arXiv:2604.18556, 2026. [Online]. Available: [https://arxiv.org/abs/2604.18556](https://arxiv.org/abs/2604.18556).<br>
[2] M. Helcig and D. Alistarh, "Model compression with exact budget constraints via Riemannian manifolds," arXiv preprint arXiv:2605.00649, 2026. [Online]. Available: [https://arxiv.org/abs/2605.00649](https://arxiv.org/abs/2605.00649).<br>
[3] D. Chakravorty, D. Manik, and B. Altaner, "Agentic coding improves ARC AGI 2 performance across models," pivotools blog, Feb. 2026. [Online]. Available: [https://pivotools.github.io/pivotools-quarto-blog/posts/agentic_coding_arc_agi/](https://pivotools.github.io/pivotools-quarto-blog/posts/agentic_coding_arc_agi/).<br>
[4] K. Hu et al., "ARC is a vision problem!," arXiv preprint arXiv:2511.14761, 2025. [Online]. Available: [https://arxiv.org/abs/2511.14761](https://arxiv.org/abs/2511.14761).<br>
[5] DeepSeek-AI et al., "DeepSeek-V4.1-Flash: Pushing the limits of KV cache compression," arXiv preprint arXiv:2609.19969, 2026. [Online]. Available: [https://arxiv.org/abs/2609.19969](https://arxiv.org/abs/2609.19969).<br>
[6] T. Y. Liu et al., "PICASO: Permutation-invariant context composition with state space models," in Proc. Int. Conf. Learn. Represent. (ICLR), 2025. [Online]. Available: [https://arxiv.org/abs/2502.17605](https://arxiv.org/abs/2502.17605).<br>