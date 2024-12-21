import torch
from torch.utils.data import TensorDataset

# Dummy data para testar
train_data = torch.randn(1682, 6, 256, 256)
train_labels = torch.randint(0, 2, (1682, 1, 256, 256))  # Dados binários como exemplo

print(f"Train data shape: {train_data.shape}")
print(f"Train labels shape: {train_labels.shape}")

dataset = TensorDataset(train_data, train_labels)
print("Dataset criado com sucesso!")
