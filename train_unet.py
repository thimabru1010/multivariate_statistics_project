import numpy as np
import os
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from osgeo import gdal
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from skimage.util.shape import view_as_windows
from scipy.stats import mode
from tqdm import tqdm
from utils import extract_training_patches, extract_testing_patches, load_train_data, load_classes,\
    labels2groups, pixels2histogram
import matplotlib.pyplot as plt
from utils import load_tif_image, extract_training_patches, extract_testing_patches,\
    rgb_to_categories, plot_classes_histogram, balance_dataset, create_block_index_matrix, prepare_training_and_testing_data
import argparse
import matplotlib.patches as mpatches
from sklearn.metrics import f1_score, accuracy_score, jaccard_score, fbeta_score
from unet import UNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from joblib import dump
import torchvision.transforms.functional as TF
import random

class SegmentationDataset(Dataset):
    """Dataset class with data augmentation for segmentation tasks"""
    
    def __init__(self, images, masks, apply_augmentation=False):
        """
        Args:
            images: Input images in tensor format (B, C, H, W)
            masks: Target masks in tensor format (B, 1, H, W)
            apply_augmentation: Whether to apply data augmentation
        """
        self.images = images
        self.masks = masks
        self.apply_augmentation = apply_augmentation
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        
        if self.apply_augmentation:
            # Apply data augmentation
            image, mask = self.augment(image, mask)
            
        return image, mask
    
    def augment(self, image, mask):
        """Apply multiple data augmentations to image and mask pair"""
        # Convert to correct format if needed
        if not torch.is_tensor(image):
            image = torch.from_numpy(image)
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask)
            
        # Clone to avoid modifying originals
        img = image.clone()
        msk = mask.clone()
        msk = msk.unsqueeze(0)  # Add channel dimension to mask
        
        # 1. Random horizontal flip (50% probability)
        if random.random() > 0.5:
            img = TF.hflip(img)
            msk = TF.hflip(msk)
            
        # 2. Random vertical flip (30% probability)
        if random.random() > 0.7:
            img = TF.vflip(img)
            msk = TF.vflip(msk)
            
        # 3. Random rotation (90/180/270 degrees) with 20% probability
        if random.random() > 0.8:
            angle = random.choice([90, 180, 270])
            img = TF.rotate(img, angle)
            msk = TF.rotate(msk, angle)
            
        # The following augmentations apply only to input image, not the mask
        
        # 4. Random brightness adjustment (40% probability)
        if random.random() > 0.6:
            brightness_factor = random.uniform(0.8, 1.2)  # ±20% brightness
            
            # Dividir canais
            img_visual = img[:3]  # Primeiros 3 canais (RGB/IRRG)
            img_extra = img[3:]   # Canais adicionais (podem ser outros tipos de dados)
    
            img_visual = TF.adjust_brightness(img_visual, brightness_factor)
            
            # Recombinar
            img = torch.cat([img_visual, img_extra], dim=0)
            
        # 5. Random contrast adjustment (30% probability)
        if random.random() > 0.7:
            contrast_factor = random.uniform(0.8, 1.2)  # ±20% contrast
            
            # Dividir canais
            img_visual = img[:3]  # Primeiros 3 canais (RGB/IRRG)
            img_extra = img[3:]   # Canais adicionais
    
            img_visual = TF.adjust_contrast(img_visual, contrast_factor)
            
            # Recombinar
            img = torch.cat([img_visual, img_extra], dim=0)
            
        # 6. Random Gaussian noise (20% probability)
        if random.random() > 0.8:
            noise = torch.randn_like(img) * 0.05  # 5% noise
            img = img + noise
            img = torch.clamp(img, 0, 1)  # Ensure values stay in valid range
        
        return img, msk.squeeze(0)

def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_images, batch_masks in val_loader:
            batch_images = batch_images.to("cuda" if torch.cuda.is_available() else "cpu")
            batch_masks = batch_masks.to("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.long)

            outputs = model(batch_images)
            loss = criterion(outputs, batch_masks.squeeze(1))
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            targets = batch_masks.cpu().numpy()

            all_preds.append(preds)
            all_targets.append(targets)

    val_loss /= len(val_loader)

    # Flatten arrays for metric computation
    all_preds = np.concatenate([p.flatten() for p in all_preds])
    all_targets = np.concatenate([t.flatten() for t in all_targets])

    # Metrics computation
    f1 = f1_score(all_targets, all_preds, average="weighted")
    acc = accuracy_score(all_targets, all_preds)
    miou = jaccard_score(all_targets, all_preds, average="weighted")

    print(f"Validation Metrics - F1 Score: {f1:.4f}, Accuracy: {acc:.4f}, mIoU: {miou:.4f}")

    return val_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,\
    num_epochs, patience, save_path, device):
    model.train()
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        epoch_loss = 0

        for batch_images, batch_masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_images = batch_images.to(device)
            batch_masks = batch_masks.to(device, dtype=torch.long)  # CrossEntropyLoss exige targets long

            # Forward
            outputs = model(batch_images)  # Outputs shape: (N, num_classes, H, W)
            loss = criterion(outputs, batch_masks.squeeze(1))  # Targets shape: (N, H, W)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        val_loss = validate_model(model, val_loader, criterion)

        scheduler.step(val_loss)
        
        # Fix the learning rate display
        print(f"Epoch [{epoch+1}/{num_epochs}], LR: {optimizer.param_groups[0]['lr']:.6f}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            print(f"Validation loss improved. Saving model to {save_path}")
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster Potsdam dataset')
    parser.add_argument('--pca', action='store_true', help='Apply PCA to data')
    parser.add_argument('--patch_size', type=int, default=128, help='Selects the size of the patches')
    parser.add_argument('--labels_path', type=str, default='data/Potsdam/5_Labels_all', help='Path to labels')
    args = parser.parse_args()
    
    labels_path = 'data/Potsdam/5_Labels_all'
    
    # filenames_label = np.random.choice(filenames_label, 3)
    filenames_label = ['top_potsdam_2_14_label.tif', 'top_potsdam_6_12_label.tif', 'top_potsdam_5_13_label.tif']
    print(filenames_label)
    
    # Train test split
    train_filenames_label = ['top_potsdam_2_14_label.tif', 'top_potsdam_6_12_label.tif']
    print(len(train_filenames_label))
    
    # map_rgb2cat = {'(255, 255, 255)': 0, '(255, 0, 0)': 1, '(0, 255, 255)': 2, '(0, 255, 0)': 3, '(0, 0, 255)': 4, '(255, 255, 0)': 5}
    map_rgb2cat = {(255, 255, 255): 0, (0, 0, 255): 1, (0, 255, 255): 2, (0, 255, 0): 3, (255, 255, 0): 4, (255, 0, 0): 5}
    # Converta as chaves do dicionário para tuplas de inteiros
    colors = list(map_rgb2cat.keys())
    print(colors)

    #! Only reads data
    train_data, train_labels, _ = prepare_training_and_testing_data(train_filenames_label, map_rgb2cat, labels_path, block_size=1)
    
    if args.pca:
        exp_folder = 'NoPatches'
        features_folder = 'PCA_Channels'
        print('Processing PCA...')
        print(train_data.shape)
        original_shape = train_data.shape
        pca = PCA(n_components=0.95)
        pca.fit(train_data.reshape(train_data.shape[0], -1))

        # Save PCA model using joblib
        dump(pca, 'data/UNet/pca_model.joblib')
        
        train_data = pca.transform(train_data.reshape(train_data.shape[0], -1))
        
        train_data = train_data.reshape(original_shape[0], original_shape[1], -1)
        
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = explained_variance_ratio.cumsum()
        # print(cumulative_variance, explained_variance_ratio)
        print(f"Explained variance: {cumulative_variance[-1]}")
    
    print(train_data.shape, train_labels.shape)
    train_data = train_data.reshape(6000, 6000, -1)
    train_labels = train_labels.reshape(6000, 6000)
    
    train_data = extract_training_patches(train_data, args.patch_size, overlap=0.2)
    train_labels = extract_training_patches(train_labels, args.patch_size, overlap=0.2)
    
    train_data = train_data.transpose(0, 3, 1, 2)
    print(f"Train data shape: {train_data.shape}, Train labels shape: {train_labels.shape}")
    
    classes, _ = np.unique(train_labels, return_counts=True)
    
    #! Parâmetros UNet

    num_epochs = 50
    learning_rate = 1e-3
    batch_size = 32
    num_classes = len(classes)  # Número de classes para classificação multiclasse
    patience_early_stopping = 10  # Número de épocas para early stopping
    save_path = "data/UNet"  # Caminho para salvar o melhor modelo
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # First split into train/val WITHOUT augmentation
    all_data = torch.Tensor(train_data)
    all_labels = torch.Tensor(train_labels)
    
    # Create train/val indices
    dataset_size = len(all_data)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Create datasets with appropriate augmentation settings
    train_dataset = SegmentationDataset(
        all_data[train_indices], 
        all_labels[train_indices],
        apply_augmentation=True  # Apply augmentation ONLY to training data
    )
    
    val_dataset = SegmentationDataset(
        all_data[val_indices], 
        all_labels[val_indices],
        apply_augmentation=False  # NO augmentation for validation data
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = UNet(in_channels=6, out_channels=num_classes).to(device)

    # Definição da função de perda e otimizador
    criterion = nn.CrossEntropyLoss(ignore_index=5)  # Para segmentação multiclasse
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,
        patience=5,
        min_lr=1e-5, 
        verbose=True
    )
    
    #! Treino
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

