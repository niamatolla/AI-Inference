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

#Warm-up: one untimed forward pass so MPS one time setup is not counted
with torch.no_grad():
    model(input_ids)
if device.type == "mps":
    torch.mps.synchronize()

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

    return times, input_ids

times_A, ids_A = run_no_cache(max_new_tokens=max_new_tokens, input_ids=input_ids, temperature=temperature)
times_B, ids_B = run_with_cache(max_new_tokens=max_new_tokens, input_ids=input_ids, temperature=temperature)

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


#Add Version B on the same axes.
median=statistics.median(times_A[1:])
#Plot step time (y) vs token position (x)
step_ms=np.array(times_A)*1000
positions=np.arange(len(step_ms))

fig, ax=plt.subplots(figsize=(10, 5))
ax.plot(positions, step_ms, color="#2a78d6", linewidth=2)
#Highlight step 0 (prefill) so it stands apart from the decode steps
ax.scatter([0], [step_ms[0]], color="#2a78d6", s=64, zorder=3)
ax.annotate(f"Prefill (TTFT) {step_ms[0]:.1f} ms", (0, step_ms[0]),
            xytext=(10, 10), textcoords="offset points", color="#52514e")
ax.axhline(median*1000, color="#52514e", linewidth=1, linestyle="--")
ax.annotate(f"Decode median {median*1000:.1f} ms", (len(step_ms)-1, median*1000),
            xytext=(0, 6), textcoords="offset points", ha="right", color="#52514e")

ax.set_title("Step time per generated token (no KV cache)")
ax.set_xlabel("Token position (0 = prefill, 1+ = decode steps)")
ax.set_ylabel("Step time (ms)")
ax.set_ylim(bottom=0)
ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
(Path(__file__).parent / "results").mkdir(exist_ok=True)
fig.savefig(Path(__file__).parent / "results" / "step_time_no_cache.png", dpi=150)
plt.show()
