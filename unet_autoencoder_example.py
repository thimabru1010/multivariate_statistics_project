import torch
import torch.nn as nn
import torch.optim as optim
from unet_autoencoder import Unet_Autoencoder

# Example usage
def main():
    # PyTorch uses channels-first format (C, H, W)
    
    # Create standard UNet autoencoder (4 levels, moderate compression)
    standard_unet = Unet_Autoencoder(
        input_shape=(6, 128, 128),  # Changed to PyTorch format (C, H, W)
        depth=5,
        initial_filters=32
    )
    standard_unet.summary()
    
    latent_size = standard_unet.get_latent_space_size()
    print(f"\nStandard UNet latent space size: {latent_size}")
    print('-' * 50)
    
    # Create more compressed UNet autoencoder (5 levels, higher compression)
    compressed_unet = Unet_Autoencoder(
        input_shape=(6, 256, 256),
        depth=5,
        initial_filters=64
    )
    compressed_unet.summary()
    
    latent_size = compressed_unet.get_latent_space_size()
    print(f"\nCompressed UNet latent space size: {latent_size}")
    print('-' * 50)
    
    # Create less compressed UNet autoencoder (3 levels, lower compression)
    less_compressed_unet = Unet_Autoencoder(
        input_shape=(6, 256, 256),
        depth=8,
        initial_filters=8
    )
    less_compressed_unet.summary()
    
    latent_size = less_compressed_unet.get_latent_space_size()
    print(f"\nLess Compressed UNet latent space size: {latent_size}")
    print('-' * 50)
    
    # Example of how to use the model (create optimizer, loss function)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = standard_unet.to(device)
    
    # Training configuration
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    
    # Example data (replace with your actual data)
    dummy_input = torch.randn(4, 6, 256, 256, device=device)
    dummy_target = torch.randn(4, 6, 256, 256, device=device)
    
    # Example forward and backward pass
    outputs = model(dummy_input)
    loss = criterion(outputs, dummy_target)
    print(f"Example forward pass shape: {outputs.shape}")
    print(f"Example loss: {loss.item()}")

if __name__ == "__main__":
    main()
