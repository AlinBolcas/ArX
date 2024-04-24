import torch
print(torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
