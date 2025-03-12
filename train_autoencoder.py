import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm
from utils import extract_training_patches, prepare_training_and_testing_data
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import mean_squared_error
from unet_autoencoder import Unet_Autoencoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from joblib import dump
import torchvision.transforms.functional as TF
import random

class AutoencoderDataset(Dataset):
    """Dataset class with data augmentation for autoencoder tasks"""
    
    def __init__(self, images, apply_augmentation=False):
        """
        Args:
            images: Input images in tensor format (B, C, H, W)
            apply_augmentation: Whether to apply data augmentation
        """
        self.images = images
        self.apply_augmentation = apply_augmentation
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        image = self.images[idx]
        
        if self.apply_augmentation:
            # Apply data augmentation
            image = self.augment(image)
            
        # For autoencoder, input and target are the same
        return image, image
    
    def augment(self, image):
        """Apply multiple data augmentations to image"""
        # Convert to correct format if needed
        if not torch.is_tensor(image):
            image = torch.from_numpy(image)
            
        # Clone to avoid modifying originals
        img = image.clone()
        
        # 1. Random horizontal flip (50% probability)
        if random.random() > 0.5:
            img = TF.hflip(img)
            
        # 2. Random vertical flip (30% probability)
        if random.random() > 0.7:
            img = TF.vflip(img)
            
        # 3. Random rotation (90/180/270 degrees) with 20% probability
        if random.random() > 0.8:
            angle = random.choice([90, 180, 270])
            img = TF.rotate(img, angle)
            
        # The following augmentations apply only to input image
        
        # 4. Random brightness adjustment (40% probability)
        if random.random() > 0.6:
            brightness_factor = random.uniform(0.8, 1.2)  # ±20% brightness
            
            # Split channels
            img_visual = img[:3]  # First 3 channels (RGB/IRRG)
            img_extra = img[3:]   # Additional channels
    
            img_visual = TF.adjust_brightness(img_visual, brightness_factor)
            
            # Recombine
            img = torch.cat([img_visual, img_extra], dim=0)
            
        # 5. Random contrast adjustment (30% probability)
        if random.random() > 0.7:
            contrast_factor = random.uniform(0.8, 1.2)  # ±20% contrast
            
            # Split channels
            img_visual = img[:3]  # First 3 channels (RGB/IRRG)
            img_extra = img[3:]   # Additional channels
    
            img_visual = TF.adjust_contrast(img_visual, contrast_factor)
            
            # Recombine
            img = torch.cat([img_visual, img_extra], dim=0)
            
        # 6. Random Gaussian noise (20% probability)
        if random.random() > 0.5:
            noise = torch.randn_like(img) * 0.15  # 5% noise
            img = img + noise
            img = torch.clamp(img, 0, 1)  # Ensure values stay in valid range
            
        # 7. Random pixel masking (30% probability)
        # Note: This augmentation should only be applied to the input image, not the target
        if random.random() > 0.5:
            # Determine mask dimensions
            _, h, w = img.shape
            num_pixels = int(h * w * 0.20)  # Mask 10% of pixels
            
            # Generate random pixel coordinates
            mask_rows = torch.randint(0, h, (num_pixels,))
            mask_cols = torch.randint(0, w, (num_pixels,))
            
            # Set those pixels to zero
            for c in range(img.shape[0]):
                img[c, mask_rows, mask_cols] = 0.0
        
        return img

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_images, batch_targets in val_loader:
            batch_images = batch_images.to(device)
            batch_targets = batch_targets.to(device)

            outputs = model(batch_images)
            loss = criterion(outputs, batch_targets)
            val_loss += loss.item()

            # Store predictions and targets for metrics
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_targets.cpu().numpy())

    val_loss /= len(val_loader)

    # Calculate reconstruction metrics
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Mean Squared Error
    mse = np.mean((all_preds - all_targets) ** 2)
    # PSNR (Peak Signal-to-Noise Ratio)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')

    print(f"Validation Metrics - MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")

    return val_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, patience, save_path, device):
    model.train()
    best_loss = float('inf')
    epochs_no_improve = 0
    
    # Create directory for saving models if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(num_epochs):
        epoch_loss = 0

        for batch_images, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_images = batch_images.to(device)
            batch_targets = batch_targets.to(device)

            # Forward
            outputs = model(batch_images)
            loss = criterion(outputs, batch_targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        val_loss = validate_model(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], LR: {optimizer.param_groups[0]['lr']:.6f}, "
              f"Training Loss: {epoch_loss:.6f}, Validation Loss: {val_loss:.6f}")

        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            print(f"Validation loss improved. Saving model to {save_path}")
            torch.save(model.state_dict(), os.path.join(save_path, "best_autoencoder.pth"))
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save a sample reconstruction image every few epochs
        if (epoch + 1) % 5 == 0:
            save_reconstruction_samples(model, val_loader, epoch + 1, save_path, device)

def save_reconstruction_samples(model, val_loader, epoch, save_path, device, num_samples=3):
    """Save sample reconstructions to visualize progress"""
    model.eval()
    images, targets = next(iter(val_loader))
    
    sample_indices = random.sample(range(len(images)), min(num_samples, len(images)))
    
    with torch.no_grad():
        images = images[sample_indices].to(device)
        reconstructions = model(images).cpu()
        images = images.cpu()
    
    plt.figure(figsize=(12, 4 * num_samples))
    for i in range(len(sample_indices)):
        # Original image - show only RGB channels
        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(np.clip(images[i][:3].permute(1, 2, 0).numpy(), 0, 1))
        plt.title(f"Original - Sample {i+1}")
        plt.axis('off')
        
        # Reconstructed image - show only RGB channels
        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.imshow(np.clip(reconstructions[i][:3].permute(1, 2, 0).numpy(), 0, 1))
        plt.title(f"Reconstructed - Sample {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    reconstruction_dir = os.path.join(save_path, "reconstructions")
    os.makedirs(reconstruction_dir, exist_ok=True)
    plt.savefig(os.path.join(reconstruction_dir, f"epoch_{epoch}.png"))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an autoencoder using UNet architecture')
    parser.add_argument('--pca', action='store_true', help='Apply PCA to data')
    parser.add_argument('--patch_size', type=int, default=128, help='Selects the size of the patches')
    parser.add_argument('--labels_path', type=str, default='data/Potsdam/5_Labels_all', help='Path to labels')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--unet_depth', type=int, default=4, help='Depth of the UNet architecture')
    parser.add_argument('--unet_filters', type=int, default=32, help='Number of filters in the first layer')
    args = parser.parse_args()
    
    labels_path = 'data/Potsdam/5_Labels_all'
    
    # Use the same filenames as in the original script
    filenames_label = ['top_potsdam_2_14_label.tif', 'top_potsdam_6_12_label.tif', 'top_potsdam_5_13_label.tif']
    print(filenames_label)
    
    # Train test split
    train_filenames_label = ['top_potsdam_2_14_label.tif', 'top_potsdam_6_12_label.tif']
    print(len(train_filenames_label))
    
    map_rgb2cat = {(255, 255, 255): 0, (0, 0, 255): 1, (0, 255, 255): 2, (0, 255, 0): 3, (255, 255, 0): 4, (255, 0, 0): 5}
    colors = list(map_rgb2cat.keys())
    print(colors)

    # Only load the image data, discard the labels for autoencoder training
    train_data, _, _ = prepare_training_and_testing_data(train_filenames_label, map_rgb2cat, labels_path, block_size=1)
    
    if args.pca:
        exp_folder = 'NoPatches'
        features_folder = 'PCA_Channels'
        print('Processing PCA...')
        print(train_data.shape)
        original_shape = train_data.shape
        pca = PCA(n_components=0.95)
        pca.fit(train_data.reshape(train_data.shape[0], -1))

        # Save PCA model using joblib
        dump(pca, 'data/AutoEncoder/pca_model.joblib')
        
        train_data = pca.transform(train_data.reshape(train_data.shape[0], -1))
        
        train_data = train_data.reshape(original_shape[0], original_shape[1], -1)
        
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = explained_variance_ratio.cumsum()
        print(f"Explained variance: {cumulative_variance[-1]}")
    
    print(train_data.shape)
    train_data = train_data.reshape(6000, 6000, -1)
    train_data = train_data[:, :, :3]
    
    # Extract patches for training
    train_data = extract_training_patches(train_data, args.patch_size, overlap=0.2)
    
    # Transpose to have channels first for PyTorch
    train_data = train_data.transpose(0, 3, 1, 2)
    print(f"Train data shape: {train_data.shape}")
    
    # Set up training parameters
    num_epochs = args.epochs
    learning_rate = args.lr
    batch_size = args.batch_size
    in_channels = train_data.shape[1]  # Number of input channels
    patience_early_stopping = 10  # Number of epochs for early stopping
    save_path = "data/AutoEncoder"  # Path to save the best model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Convert to tensor
    all_data = torch.Tensor(train_data)
    
    # Create train/val indices
    dataset_size = len(all_data)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Create datasets with appropriate augmentation settings
    train_dataset = AutoencoderDataset(
        all_data[train_indices],
        apply_augmentation=True  # Apply augmentation to training data
    )
    
    val_dataset = AutoencoderDataset(
        all_data[val_indices],
        apply_augmentation=False  # No augmentation for validation data
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model - for autoencoder, input and output channels should be the same
    model = Unet_Autoencoder(
        input_shape=(in_channels, args.patch_size, args.patch_size),  # Changed to PyTorch format (C, H, W)
        depth=args.unet_depth,
        initial_filters=args.unet_filters
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,
        patience=5,
        min_lr=1e-5
    )
    
    # Train the model
    train_model(
        model, 
        train_loader, 
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs,
        patience_early_stopping,
        save_path,
        device
    )
