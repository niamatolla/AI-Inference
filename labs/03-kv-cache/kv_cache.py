from transformers import AutoTokenizer, AutoModelForCausalLM,DynamicCache
import torch
import time
import statistics
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path


#AutoRegressive loop variables
max_new_tokens=200
temperature=1.0 

#Define the pretrained model ID
model_id="gpt2"

#Load the tokenizer
tokenizer=AutoTokenizer.from_pretrained(model_id)

#Load the model
model=AutoModelForCausalLM.from_pretrained(model_id)

#Verify model Config
print(model.config)

#Move to MPS
device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
print("Running on:", device)

#Tokenize the input text into PyTorch tensors
prompt=(
    " Inference only runs the forward pass, "
    "the weights are frozen, to produce new tokens we dont have yet. "
    "We produce them once at a time, and each one depends on the one before it"
)

input_ids=tokenizer.encode(prompt, return_tensors="pt").to(device)

#Feed the tokens to the model and generate new tokens : TTFT+199*TPOT
#outputs=model.generate(**inputs, max_new_tokens=200)

#Number of timed repeats per version; we plot the median step time per position
#so one-off MPS hiccups don't show up as spikes in the curves.
n_runs=3

#KV cache size, measured: the real bytes the cache occupies right now.
#Sum over every layer of (key elements + value elements), times bytes per element.
def kv_cache_bytes_measured(cache) -> int:
    total_elements=0
    for layer in cache.layers:
        total_elements += layer.keys.numel() + layer.values.numel()
    bytes_per_element=cache.layers[0].keys.element_size()
    return total_elements * bytes_per_element

#KV cache size, from the formula:
#2 (K and V) x n_layers x n_heads x head_dim x seq_len x bytes
#and since n_heads x head_dim = d_model, this is 2 x n_layers x d_model x seq_len x bytes
def kv_cache_bytes_formula(seq_len: int, bytes_per_element: int) -> int:
    n_layers=model.config.n_layer
    n_heads=model.config.n_head
    head_dim=model.config.n_embd // n_heads
    return 2 * n_layers * n_heads * head_dim * seq_len * bytes_per_element

#Both loops run a fixed max_new_tokens (no EOS early stop) so A and B
#cover the same positions and their curves line up step for step.

#Version A: No KV cache
#Run the autoregressive loop without KV cache
#Returns (step times in seconds, full token sequence)
def run_no_cache(max_new_tokens: int,input_ids:torch.Tensor,temperature: float):
    times=[]
    for i in range(max_new_tokens):
        with torch.no_grad():
            start_time = time.perf_counter()
            #use_cache=False: GPT-2 builds a cache by default; A must not
            outputs=model(input_ids, use_cache=False)
            #Get logits for the very last token in the sequence
            next_token_logits=outputs.logits[:,-1,:] / temperature
            #Take the most likely token (Greedy)
            next_token=torch.argmax(next_token_logits, dim=-1, keepdim=True)

        #Sync before stopping the clock
        if device.type == "mps":
            torch.mps.synchronize()
        end_time = time.perf_counter()
        times.append(end_time-start_time)

        # Append the new token to the sequence
        input_ids=torch.cat([input_ids, next_token], dim=1)

    return times, input_ids

#Version B: With KV cache
#Run the autoregressive loop with KV cache
#Returns (step times in seconds, full token sequence)
def run_with_cache(max_new_tokens: int,input_ids:torch.Tensor,temperature: float):
    times=[]
    cache=DynamicCache()
    for i in range(max_new_tokens):
        if(i == 0):
            model_input=input_ids
        else:
            model_input=next_token
        with torch.no_grad():
            start_time = time.perf_counter()
            outputs=model(model_input,past_key_values=cache,use_cache=True)
            #Get logits for the very last token in the sequence
            next_token_logits=outputs.logits[:,-1,:] / temperature
            # Get cached key-value
            cache = outputs.past_key_values
            #Take the most likely token (Greedy)
            next_token=torch.argmax(next_token_logits, dim=-1, keepdim=True)

        #Sync before stopping the clock
        if device.type == "mps":
            torch.mps.synchronize()
        end_time = time.perf_counter()
        times.append(end_time-start_time)

        # Append the new token to the sequence
        input_ids=torch.cat([input_ids, next_token], dim=1)
    print("Cache type:", type(cache))
    print("Layer count:", len(cache))
    print("Methods and attributes:", dir(cache))
    print("Layer 0 keys:", cache.layers[0].keys.shape )
    print("The number format of layer 0 keys:", cache.layers[0].keys.dtype)
    print("Bytes per number:", cache.layers[0].keys.element_size())

    #Measured vs formula
    seq_len=cache.layers[0].keys.shape[2]
    bytes_per_element=cache.layers[0].keys.element_size()
    measured=kv_cache_bytes_measured(cache)
    formula=kv_cache_bytes_formula(seq_len, bytes_per_element)
    print(f"KV cache at {seq_len} tokens: measured {measured:,} B, formula {formula:,} B, "
          f"match: {measured == formula}")
    print(f"Per token: measured {measured/seq_len:,.0f} B ({measured/seq_len/1024:.1f} KB), "
          f"formula {formula/seq_len:,.0f} B")

    return times, input_ids

#Warm-up: a few untimed steps of BOTH loops so MPS one-time setup is not counted.
run_no_cache(max_new_tokens=5, input_ids=input_ids, temperature=temperature)
run_with_cache(max_new_tokens=5, input_ids=input_ids, temperature=temperature)

#Timed runs: repeat each version n_runs times, keep the median per position
runs_A, runs_B = [], []
for r in range(n_runs):
    t_A, ids_A = run_no_cache(max_new_tokens=max_new_tokens, input_ids=input_ids, temperature=temperature)
    t_B, ids_B = run_with_cache(max_new_tokens=max_new_tokens, input_ids=input_ids, temperature=temperature)
    runs_A.append(t_A)
    runs_B.append(t_B)
    print(f"Run {r+1}/{n_runs} done")
times_A = np.median(np.array(runs_A), axis=0).tolist()
times_B = np.median(np.array(runs_B), axis=0).tolist()

#Correctness check: greedy decoding means A and B must produce identical tokens
print("A and B generated identical tokens:", torch.equal(ids_A, ids_B))

#Decode the output token IDs into readable text
decoded_text=tokenizer.batch_decode(ids_B, skip_special_tokens=True)
print(decoded_text)

for name, times in [("A (no cache)", times_A), ("B (KV cache)", times_B)]:
    decode=times[1:]
    print(f"--- {name} ---")
    print(f"Prefill (TTFT): {times[0]*1000:.2f}ms")
    print(f"Decode: median {statistics.median(decode)*1000:.2f} ms, mean {sum(decode)/len(decode)*1000:.2f} ms")
    print(f"Step 1: {times[1]*1000:.2f}ms, last step: {times[-1]*1000:.2f} ms")


#Plot step time (y) vs token position (x) for A and B on the same axes
series=[
    ("A: no cache (recompute)", times_A, "#2a78d6"),
    ("B: KV cache (reuse)",     times_B, "#eb6834"),
]

fig, ax=plt.subplots(figsize=(10, 5))
for label, times, color in series:
    step_ms=np.array(times)*1000
    positions=np.arange(len(step_ms))
    ax.plot(positions, step_ms, color=color, linewidth=2, label=label)
    #Mark step 0 (prefill) so it stands apart from the decode steps
    ax.scatter([0], [step_ms[0]], color=color, s=64, zorder=3,
               edgecolor="white", linewidth=2)
    #Direct label at the end of each line
    ax.annotate(f"{step_ms[-1]:.1f} ms", (positions[-1], step_ms[-1]),
                xytext=(6, 0), textcoords="offset points", va="center", color="#52514e")

ax.set_title(f"Step time per token: no cache vs KV cache (GPT-2, median of {n_runs} runs)")
ax.set_xlabel("Token position (0 = prefill, 1+ = decode steps)")
ax.set_ylabel("Step time (ms)")
ax.set_ylim(bottom=0)
ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left")

fig.tight_layout()
(Path(__file__).parent / "results").mkdir(exist_ok=True)
fig.savefig(Path(__file__).parent / "results" / "step_time_cache_vs_no_cache.png", dpi=150)
plt.show()
