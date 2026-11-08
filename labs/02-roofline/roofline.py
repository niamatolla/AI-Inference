#Roofline for the M1 and the T4 (FP32), with GPT-2 decode/prefill intensities
#and the Lab 1 timings placed on the M1 roofline
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#Hardware numbers (FP32) from the table in README.md
MACHINES={
    "Apple M1 (8-core GPU)": {"peak_flops":2.61e12,"spec_bw":68.3e9,"measured_bw":57.79e9},
    "NVIDIA T4": {"peak_flops":8.1e12,"spec_bw":320e9,"measured_bw":231.27e9},
}

BYTES_PER_ELEM=4        #FP32, to match the peaks and the bandwidth benchmark

#GPT-2 small
D_MODEL=768
N_LAYERS=12
VOCAB=50257
N_PARAMS=124_439_808                      #total parameters (weights read every pass)
LAYER_MATMUL_PARAMS=N_LAYERS*12*D_MODEL**2 #QKV 3d^2 + attn proj d^2 + MLP 8d^2 per layer
LM_HEAD_PARAMS=D_MODEL*VOCAB              #tied embedding used as the output projection

#[768,768] matmul intensities from arithmetic_intensity.py (full bytes: X + W + Y)
def matmul_intensity(S,d=D_MODEL,b=BYTES_PER_ELEM):
    return (2*S*d*d)/((S*d+d*d+S*d)*b)

DECODE_AI=matmul_intensity(1)
PREFILL_AI={S:matmul_intensity(S) for S in [40,128,512]}

#Lab 1 timings (M1, FP32, no KV cache) from labs/01-prefill-vs-decode/README.md
#(label, tokens in the forward pass, measured seconds)
LAB1_STEPS=[
    ("prefill, 39-tok prompt",39,20.35e-3),
    ("decode step 1, 39-tok prompt",40,35e-3),
    ("last decode step, 39-tok prompt",238,78e-3),
    ("prefill, 156-tok prompt",156,55.92e-3),
    ("decode step 1, 156-tok prompt",157,87e-3),
    ("last decode step, 156-tok prompt",355,154e-3),
]

def gpt2_pass_flops(L):
    #Without a KV cache every step is a full forward pass over all L tokens
    matmuls=2*L*(LAYER_MATMUL_PARAMS+LM_HEAD_PARAMS)
    #QK^T and attention*V: 2*L*L*d each, per layer
    attention=N_LAYERS*4*L*L*D_MODEL
    return matmuls+attention

def gpt2_pass_bytes(L):
    #Lower bound: read every weight once + write the logits for all L positions
    return (N_PARAMS+L*VOCAB)*BYTES_PER_ELEM

def roof(ai,peak_flops,bw):
    return np.minimum(peak_flops,ai*bw)

#Ridge points
print("--- Ridge points (FP32) ---")
for name,m in MACHINES.items():
    print(f"{name:<22} spec: {m['peak_flops']/m['spec_bw']:6.1f} FLOP/B   "
          f"measured: {m['peak_flops']/m['measured_bw']:6.1f} FLOP/B")

print(f"\nDecode intensity: {DECODE_AI:.2f} FLOP/B")
for S,ai in PREFILL_AI.items():
    print(f"Prefill S={S:<4} intensity: {ai:.2f} FLOP/B")

#Lab 1 vs the M1 roofline
m1=MACHINES["Apple M1 (8-core GPU)"]
print("\n--- Lab 1 on the M1 roofline (measured bandwidth) ---")
print(f"{'step':<34} | {'L':>4} | {'ms':>6} | {'AI':>6} | {'GFLOP/s':>8} | {'roof':>8} | {'% roof':>6}")
lab1_points=[]
for label,L,t in LAB1_STEPS:
    ai=gpt2_pass_flops(L)/gpt2_pass_bytes(L)
    achieved=gpt2_pass_flops(L)/t
    ceiling=roof(ai,m1["peak_flops"],m1["measured_bw"])
    lab1_points.append((ai,achieved))
    print(f"{label:<34} | {L:>4} | {t*1e3:>6.1f} | {ai:>6.1f} | {achieved/1e9:>8.1f} | "
          f"{ceiling/1e9:>8.1f} | {achieved/ceiling*100:>5.1f}%")

#Naive view: count decode step 1 as if it produced 1 token with a KV cache
naive_flops=2*(LAYER_MATMUL_PARAMS+LM_HEAD_PARAMS)
naive_ai=naive_flops/(N_PARAMS*BYTES_PER_ELEM)
naive_achieved=naive_flops/35e-3
naive_ceiling=roof(naive_ai,m1["peak_flops"],m1["measured_bw"])
print(f"\nDecode step 1 counted as 1 token: AI {naive_ai:.2f} FLOP/B, "
      f"{naive_achieved/1e9:.1f} GFLOP/s vs roof {naive_ceiling/1e9:.1f} GFLOP/s "
      f"({naive_achieved/naive_ceiling*100:.0f}%)")
print(f"Memory-bound time for one pass of weights: "
      f"{N_PARAMS*BYTES_PER_ELEM/m1['measured_bw']*1e3:.1f} ms (measured: 35 ms)")

#Plot: one panel per machine, log-log
INK="#0b0b0b"
MUTED="#52514e"
GRID="#e5e4e0"
DECODE_COLOR="#2a78d6"   #categorical slot 1
PREFILL_COLOR="#eb6834"  #categorical slot 2
LAB1_COLOR="#1baf7a"     #categorical slot 3

ai_axis=np.logspace(-1,3,400)
fig,axes=plt.subplots(1,2,figsize=(13,5.5),sharey=True)

for ax,(name,m) in zip(axes,MACHINES.items()):
    peak=m["peak_flops"]/1e9
    ridge=m["peak_flops"]/m["measured_bw"]

    ax.plot(ai_axis,roof(ai_axis,m["peak_flops"],m["spec_bw"])/1e9,
            color=MUTED,linewidth=1.5,linestyle="--",label="Roof, spec bandwidth")
    ax.plot(ai_axis,roof(ai_axis,m["peak_flops"],m["measured_bw"])/1e9,
            color=INK,linewidth=2,label="Roof, measured bandwidth")

    #Ridge point on the measured roof
    ax.scatter([ridge],[peak],color=INK,s=64,zorder=4)
    ax.annotate(f"ridge {ridge:.0f} FLOP/B",(ridge,peak),xytext=(-8,10),
                textcoords="offset points",ha="right",color=INK)
    ax.annotate(f"peak {peak/1e3:.2f} TFLOP/s",(ai_axis[-1],peak),xytext=(0,6),
                textcoords="offset points",ha="right",color=MUTED)

    #Vertical markers: where the [768,768] matmul sits for decode and prefill
    ax.axvline(DECODE_AI,color=DECODE_COLOR,linewidth=1.5,linestyle=":",label="Decode (S=1)")
    ax.text(DECODE_AI,1.3,"  decode",rotation=90,va="bottom",color=INK,fontsize=9)
    for i,(S,ai) in enumerate(PREFILL_AI.items()):
        ax.axvline(ai,color=PREFILL_COLOR,linewidth=1.5,linestyle=":",
                   label="Prefill (S=40, 128, 512)" if i==0 else None)
        ax.text(ai,1.3,f"  S={S}",rotation=90,va="bottom",color=INK,fontsize=9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1,1000)
    ax.set_ylim(1,2e4)
    ax.set_title(name,color=INK)
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.grid(which="major",color=GRID,linewidth=0.8)
    ax.spines[["top","right"]].set_visible(False)

#Lab 1 measurements only exist for the M1
ax=axes[0]
ax.scatter([p[0] for p in lab1_points],[p[1]/1e9 for p in lab1_points],
           color=LAB1_COLOR,s=64,edgecolor="white",linewidth=1.5,zorder=5,
           label="Lab 1 pass (actual work: L tokens)")
ax.scatter([naive_ai],[naive_achieved/1e9],facecolor="white",edgecolor=LAB1_COLOR,
           linewidth=2,s=64,zorder=5,label="Lab 1 decode step 1, counted as 1 token")

axes[0].set_ylabel("Attainable GFLOP/s")
axes[0].legend(loc="upper left",frameon=False,fontsize=9)
fig.suptitle("Roofline, FP32: GPT-2 [768,768] matmul intensities and Lab 1 timings",color=INK)
fig.tight_layout()

out=Path(__file__).parent/"results"/"roofline_M1_T4.png"
fig.savefig(out,dpi=150)
print(f"\nSaved {out}")
