import torch
import subprocess
import os
import time

def get_all_gpus():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free",
         "--format=csv,nounits,noheader"]
    )
    gpus = []
    for line in result.decode("utf-8").strip().split("\n"):
        gpu_id, free = line.split(",")
        gpus.append((int(gpu_id.strip()), int(free.strip())))
    return gpus

def wait_for_best_gpu(min_vram=4000, check_interval=30):
    """Wait until some GPU has at least min_vram MB free."""
    while True:
        gpus = get_all_gpus()
        if not gpus:
            print("No GPUs found, retrying in 30s...")
            time.sleep(check_interval)
            continue

        # Sort by free VRAM
        gpus.sort(key=lambda x: x[1], reverse=True)
        best_id, best_free = gpus[0]

        if best_free >= min_vram and best_id != 7:
            print(f"✅ Using GPU {best_id} with {best_free} MB free VRAM.")
            return f"cuda:{best_id}"
        else:
            print(f"⏳ Max free VRAM = {best_free} MB (need ≥ {min_vram}). Retrying in {check_interval}s...")
            time.sleep(check_interval)

if __name__ == "__main__":
    device = wait_for_best_gpu(min_vram=20000, check_interval=2)  # wait until 20GB free
    os.system(f"CUDA_VISIBLE_DEVICES={device.split(':')[-1]} sh scripts/pretraining/pretrain_simclr.sh")