# Qwen 3.8 token generation speed
URL /competitions/arc-prize-2026-arc-agi-2/discussion/745787 | author Darren Amadeus Martin | posted 2026-10-04 12:09:21.948000 | votes 2 | comments 16 | forum ARC Prize 2026 - ARC-AGI-2

<p>Has anybody got a decent aggregate token generation speed for qwen 3.8 27b or flash next ? In arc agi 3, everybody seems to be using qwen 3.8 flash next since rtx pro 6000 has a very high bandwidth. However 4 L4 seems to be very slow when running these models due to lower bandwidth and slow gpu communication. Has anybody got like an aggregate above 500 tok/s ? Would like to hear how you optimize the inference</p>

- [CPMP | 2026-10-04 14:21:17.423000 | votes 3 | id 3532725]
  <p>That's the problem of long competitions. New, stronger models can appear. We got 70+ before Qwen 3.8, and this model reshuffled everything.</p>
  <p>Next 4 weeks will be hot on LB.</p>
  - [Anil Thomas | 2026-10-04 17:09:30.640000 | votes 1 | id 3532778]
    <p>doesn't qwen3.8 come with its own challenges? the generations at xhigh effort are  very long (median around 64K from what I have seen). from <a href="https://arcprize.org/results/alibaba-qwen3-8-27b" target="_blank">this arcprize.org page</a>, it appears that you need xhigh for good results. it would be challenging to squeeze 240 puzzles (probably around 330 queries) into the time budget. if you turn off thinking, the results are abysmal.</p>
    <p>if someone has tried submissions solely based on qwen3.8, please post. it will save the rest of us countless GPU-hours!</p>
    - [Anil Thomas | 2026-10-04 17:13:01.580000 | votes 0 | id 3532779]
      <p>I would think the folks that have posted recent scores above 45 (I am not in this list yet) are doing something super-clever. of course, it is also possible that I am doing something wrong. </p>
    - [juliencst | 2026-10-04 18:04:46.587000 | votes 2 | id 3532789]
      <p>Currently exploring submissions solely on qwen 3.8, and reaching scores of 8.xx
      I have a few promising experiments locally, will see if that translates to the lb</p>
    - [Russell Kirk | 2026-10-04 18:09:23.527000 | votes 1 | id 3532792]
      <p><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F27681700%2Fca25897f0c65302134645e183ba315b7%2FScreenshot%20from%202026-10-04%2014-08-40.png?generation=1791137360157769&amp;alt=media" alt=""></p>
    - [Anil Thomas | 2026-10-04 18:58:34.207000 | votes 0 | id 3532807]
      <p>hm… I would have expected something around 30.</p>
      <p>back-of-the-envelope calculation assuming 330 queries, 350 tokens/s, mean generation of 64k tokens and 42% accuracy for a full run:
      <code>42*12*60*60/(64*1024*330/350) ~ 30</code></p>
      <p>350 tokens/s on a 4x L4 seems plausible for the FP8 quantized model.</p>
    - [Russell Kirk | 2026-10-04 19:01:18.590000 | votes 0 | id 3532808]
      <p>Don't assume I implemented it correctly.  In ARC-AGI-3, I tried MTP and I set it up incorrectly and didn't get any gain :D</p>
    - [Russell Kirk | 2026-10-04 19:13:56 | votes 3 | id 3532813]
      <p><a href="https://www.kaggle.com/code/russcore/arcagi2-qwen38-freeform-direct" target="_blank">https://www.kaggle.com/code/russcore/arcagi2-qwen38-freeform-direct</a></p>
      <p>i shared the notebook, i don't plan to follow that route atm -- and it may have mistakes. But maybe it'll save you some GPU.</p>
    - [Anil Thomas | 2026-10-05 00:09:42.160000 | votes 1 | id 3532869]
      <blockquote>
        <p>'turn_max_tokens': 40960</p>
      </blockquote>
      <p>that is too low. try setting it to 64k and also increase the total context length.</p>
    - [Darren Amadeus Martin | 2026-10-05 00:40:59.867000 | votes 0 | id 3532877]
      <p>wow, thank you for the notebook, how was the token generation speed in your experience ?</p>
    - [Darren Amadeus Martin | 2026-10-05 00:43:00.773000 | votes 0 | id 3532878]
      <p>I did not know that the foundation have benchmarked qwen 3.8. with  $0.4469/task and $2.55 per million output tokens, shouldnt it be roughly 170k tokens per task (considering 2 pass per task, maybe 85k-90k ish tokens). Can you tell me the source of 64k median ?</p>
    - [NguyenThanhNhan | 2026-10-05 04:12:49.653000 | votes 2 | id 3532916]
      <p><a href="/anlthms" target="_blank">@anlthms</a> I managed to submit twice with qwen 3.8 27b fp8 + mtp at 65k tokens and xhigh reasoning effort.
      <img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1134433%2Fbbfa41cbacb4961c46014f523e4183c1%2FScreenshot%202026-10-05%20at%2011.08.30.png?generation=1791173423946159&amp;alt=media" alt=""></p>
      <p>Next stop is flash next, but the queue for 4xL4 has been insane 🥲</p>
    - [Darren Amadeus Martin | 2026-10-05 04:19:02.280000 | votes 0 | id 3532920]
      <p>May I ask what did you do on this setup ? With 65k tokens/task (assuming you only did 1 pass), 240 task and 12 hours, you'd need roughly 360 tok/s on aggregate. Not only that, you'd also need to fit the kv cache.</p>
    - [NguyenThanhNhan | 2026-10-05 04:40:12.070000 | votes 1 | id 3532928]
      <p><a href="/darrenamadeusmartin" target="_blank">@darrenamadeusmartin</a> 
      I could only do 1 pass. I had a custom intervention logic to force the model to stop reasoning and commit with a solution grid after 49k tokens. Therefore, not all 240 tasks would generate 65k tokens.</p>
      <p>Those 2 submissions used pure reasoning, which is also quite token-inefficient. I think you could solve many more tasks within 12 hours by implementing a simple harness.</p>
      <p>Btw, here's my sglang server config:</p>
      <pre><code>        "--kv-cache-dtype",
              "fp8_e4m3",
              "--attention-backend",
              "flashinfer",
              "--linear-attn-backend",
              "triton",
              "--chunked-prefill-size",
              "2048",
              "--max-prefill-tokens",
              "4096",
              "--mamba-radix-cache-strategy",
              "extra_buffer",
              "--mamba-full-memory-ratio",
              "0.5",
              "--mamba-ssm-dtype",
              "float32",
              "--speculative-algorithm",
              "EAGLE",
              "--speculative-num-steps",
              "3",
              "--speculative-eagle-topk",
              "1",
              "--speculative-num-draft-tokens",
              "4",
      </code></pre>
    - [Darren Amadeus Martin | 2026-10-05 08:41:16.627000 | votes 0 | id 3533007]
      <p>Thank you for your explanation. How much concurrency do you use ? If you don't mind, can you share your setup notebook? Ive been having trouble trying to get decent speed</p>
  - [Darren Amadeus Martin | 2026-10-05 00:40:25.723000 | votes 0 | id 3532876]
    <p>your words made me want to assume that your whole pipeline is LLM+harness haha. LB movement currently is very interesting as many have reached beyond 45 on LB</p>