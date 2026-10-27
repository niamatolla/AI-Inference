from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import statistics
import matplotlib.pyplot as plt
import numpy as np

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

four_time_longer_prompt=(
    "Inference is the process where a trained language model uses its "
    "existing knowledge to generate new text. During inference, the "
    "model only performs the forward pass, meaning the input is passed "
    "through the neural network to calculate predictions and determine "
    "which token should come next. The model's weights remain frozen, "
    "so they are not updated or modified during this process. Unlike "
    "training, where the model learns by adjusting its weights based on "
    "errors, inference simply uses the knowledge the model has already "
    "learned. When generating a response, the model produces one new "
    "token at a time. Each token is selected based on the input context "
    "and the tokens generated before it. After generating a token, the "
    "model adds it to the existing sequence and uses the updated context "
    "to predict the next token. This process repeats until generation "
    "stops."
)

input_ids=tokenizer.encode(four_time_longer_prompt, return_tensors="pt").to(device)

#Feed the tokens to the model and generate new tokens : TTFT+199*TPOT
#outputs=model.generate(**inputs, max_new_tokens=200)

#Warmu-up: one untimed forward pass so MPS one time setup is not counted
with torch.no_grad():
    model(input_ids)
if device.type == "mps":
    torch.mps.synchronize()

times=[]
#AutoRegressive loop
for i in range(max_new_tokens):
    with torch.no_grad():
        start_time = time.perf_counter()
        outputs=model(input_ids)
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

    #Stop if end of sequence token is generated
    if next_token.item() == tokenizer.eos_token_id:
        break



#Decode the output token IDs into readable text
decoded_text=tokenizer.batch_decode(input_ids, skip_special_tokens=True)
print(decoded_text)

print(f"Prefill (TTFT): {times[0]*1000:.2f}ms")
decode=times[1:]
median=statistics.median(decode)
mean=sum(decode)/len(decode)
print(f"Decode: median {median*1000:.2f} ms, mean {mean*1000:.2f} ms")
print(f"Step 1: {times[1]*1000:.2f}ms, last step: {times[-1]*1000:.2f} ms")

#Plot step time (y) vs token position (x)
step_ms=np.array(times)*1000
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
fig.savefig("Results/step_time_no_cache.png", dpi=150)
plt.show()
