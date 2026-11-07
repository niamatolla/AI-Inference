import torch
import time
import statistics

#Configuration
#Pick the GPU: CUDA (T4 ) and MPS (Apple Silicon), else CPU
if torch.cuda.is_available():
    device=torch.device("cuda")
elif torch.backends.mps.is_available():
    device=torch.device("mps")
else:
    device=torch.device("cpu")
print(f"Running on: {device.type.upper()}")

def sync():
    #GPUs run asynchronously wait until all queued work has finished
    if device.type=="cuda":
        torch.cuda.synchronize()
    elif device.type=="mps":
        torch.mps.synchronize()

TENSOR_SIZE_GB=1        #decimal GB (10^9 bytes), same unit as datasheets
NUM_ITERATIONS=50
NUM_RUNS=5              #repeat the timed loop and report the median

#Calculate the number of elements for 32 bit floats (4 bytes each)
num_elements=int(TENSOR_SIZE_GB*1e9/4)

print(f"Allocating 2 x {TENSOR_SIZE_GB} GB tensors on {device} ...")
#Allocate source AND destination once, so the loop times only data movement
tensor_src=torch.randn(num_elements,dtype=torch.float32,device=device)
tensor_dest=torch.empty_like(tensor_src)

#Use the real byte count of the tensor not the config value
tensor_bytes=tensor_src.numel()*tensor_src.element_size()
#Each copy reads the source once and writes the destination once
bytes_per_copy=2*tensor_bytes

#Warm up runs to initialize the GPU
for _ in range(3):
    tensor_dest.copy_(tensor_src)
sync()

print(f"Starting GPU benchmark: {NUM_RUNS} runs x {NUM_ITERATIONS} copies...")
bandwidths=[]
for r in range(NUM_RUNS):
    start_time=time.perf_counter()

    #The copy loop copy_ writes into the existing buffer and  no new allocation
    for _ in range(NUM_ITERATIONS):
        tensor_dest.copy_(tensor_src)

    #Wait for the GPU to finish all the copies before stopping the timer
    sync()

    total_time=time.perf_counter()-start_time
    bandwidth_gb_s=bytes_per_copy*NUM_ITERATIONS/total_time/1e9
    bandwidths.append(bandwidth_gb_s)
    print(f"  run {r+1}: {total_time:.4f} s -> {bandwidth_gb_s:.2f} GB/s")

print("\n--- GPU Benchmark Results ---")
print(f"Tensor size:        {tensor_bytes/1e9:.2f} GB")
print(f"Data moved per run: {bytes_per_copy*NUM_ITERATIONS/1e9:.2f} GB")
print(f"Median Bandwidth:   {statistics.median(bandwidths):.2f} GB/s")
print(f"Min / Max:          {min(bandwidths):.2f} / {max(bandwidths):.2f} GB/s")
