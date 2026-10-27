# LLM Inference Engineering — Hands-on Labs

A series of from-scratch experiments into how large language models actually run at inference time: where the time goes, why, and how serving systems make it faster. Each lab measures something on real hardware, explains the result from first principles (memory bandwidth, FLOPs, attention cost), and is written up as a short report.

**Stack:** Python · PyTorch · Hugging Face Transformers · Apple Silicon GPU (MPS) · matplotlib

## Labs

| # | Lab | What I did | Key finding |
|---|-----|------------|-------------|
| 1 | [Prefill vs Decode (no KV cache)](labs/01-prefill-vs-decode/) | Wrote GPT-2's autoregressive generation loop by hand (no `model.generate()`), timed every forward pass with GPU sync, and compared 39- vs 156-token prompts | Without a KV cache, decode cost grows with sequence length (~35 ms → ~80 ms per token over 200 steps); a 4× longer prompt raised prefill only 2.75×, which points to a fixed per-pass cost from reading the weights |

*More labs coming: KV cache, GPU profiling, batching, quantization.*

## Highlights

![Step time per generated token, 39-token prompt](labs/01-prefill-vs-decode/results/step_time_39tok.png)

- **Measured correctly:** synchronized the GPU before stopping each timer (MPS runs asynchronously) and excluded one-time kernel setup with a warm-up pass
- **Checked correctness:** the hand-written decode loop produces the same output as `model.generate()` with greedy decoding
- **Explained the numbers:** back-of-envelope roofline estimate (~7 ms weight reads at 68 GB/s vs ~15 ms of compute at ~2.6 TFLOPS) to explain why prefill scales sub-linearly
- **Recorded open questions:** an unexplained slowdown at decode step 1, and hypotheses to test with a profiler in later labs

## Repository layout

```
.
├── labs/
│   └── 01-prefill-vs-decode/
│       ├── README.md              # summary, results, how to run
│       ├── profile_no_cache.py    # experiment code
│       ├── report.pdf             # full write-up
│       └── results/               # plots and terminal output
├── docs/                          # course plan / reference material
└── requirements.txt
```

## Running the labs

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python labs/01-prefill-vs-decode/profile_no_cache.py
```

Runs on Apple Silicon (MPS) and falls back to CPU otherwise.
