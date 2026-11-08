#Arithmetic intensity of GPT-2's [768,768] matmul: Y[S,d] = X[S,d] @ W[d,d]
#S=1 is decode (matrix-vector), S>1 is prefill (matrix-matrix)

D_MODEL=768
BYTES_PER_ELEM=4       #FP32: 4 bytes per number
PROMPT_LENGTHS=[1,40,128,512]

def matmul_flops(S,d):
    #S*d output elements each a dot product of length d
    #each multiply-add counts as 2 FLOPs
    return 2*S*d*d

def weight_bytes(d,b):
    #The weights are read once per pass no matter how many tokens use them
    return d*d*b

def full_bytes(S,d,b):
    #Read input X [S,d] + read weights W [d,d] + write output Y [S,d]
    return (S*d+d*d+S*d)*b

print(f"Matmul: X[S,{D_MODEL}] @ W[{D_MODEL},{D_MODEL}], {BYTES_PER_ELEM} bytes/element\n")
print(f"{'S':>5} | {'FLOPs':>12} | {'weight bytes':>12} | {'full bytes':>12} | "
      f"{'AI weights':>10} | {'AI full':>8} | {'AI full / S':>11}")
print("-"*90)

for S in PROMPT_LENGTHS:
    flops=matmul_flops(S,D_MODEL)
    w_bytes=weight_bytes(D_MODEL,BYTES_PER_ELEM)
    f_bytes=full_bytes(S,D_MODEL,BYTES_PER_ELEM)

    #Weights-only intensity the d*d cancels and you get S
    ai_weights=flops/w_bytes
    #Full intensity also counts moving the activations X and Y
    ai_full=flops/f_bytes

    print(f"{S:>5} | {flops:>12,} | {w_bytes:>12,} | {f_bytes:>12,} | "
          f"{ai_weights:>10.2f} | {ai_full:>8.2f} | {ai_full/S:>11.2f}")
