
import torch
print("CUDA available:", torch.cuda.is_available(), "| device =", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

