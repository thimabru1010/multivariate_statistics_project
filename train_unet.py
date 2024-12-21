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
from utils import extract_patches, load_train_data, load_classes,\
    labels2groups, pixels2histogram
import matplotlib.pyplot as plt
from utils import load_tif_image, extract_training_patches, extract_testing_patches,\
    rgb_to_categories, plot_classes_histogram, balance_dataset, create_block_index_matrix, prepare_training_and_testing_data
import argparse
import matplotlib.patches as mpatches
from sklearn.metrics import f1_score, accuracy_score, jaccard_score
from unet import UNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, save_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            print(f"Validation loss improved. Saving model to {save_path}")
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
if __name__ == '__main__':
    labels_path = 'data/Potsdam/5_Labels_all'
    filenames_label = os.listdir(labels_path)
    
    # filenames_label = np.random.choice(filenames_label, 3)
    filenames_label = ['top_potsdam_2_14_label.tif', 'top_potsdam_6_12_label.tif', 'top_potsdam_5_13_label.tif']
    print(filenames_label)
    
    # Train test split
    # train_filenames_label, test_filenames_label = train_test_split(filenames_label, test_size=0.2)
    train_filenames_label = ['top_potsdam_2_14_label.tif', 'top_potsdam_6_12_label.tif']
    test_filenames_label = ['top_potsdam_5_13_label.tif']
    print(len(train_filenames_label), len(test_filenames_label))
    
    # map_rgb2cat = {'(255, 255, 255)': 0, '(255, 0, 0)': 1, '(0, 255, 255)': 2, '(0, 255, 0)': 3, '(0, 0, 255)': 4, '(255, 255, 0)': 5}
    map_rgb2cat = {(255, 255, 255): 0, (0, 0, 255): 1, (0, 255, 255): 2, (0, 255, 0): 3, (255, 255, 0): 4, (255, 0, 0): 5}
    # Converta as chaves do dicionário para tuplas de inteiros
    colors = list(map_rgb2cat.keys())
    print(colors)

    block_size = 128
    
    train_data, train_labels, _ = prepare_training_and_testing_data(train_filenames_label, map_rgb2cat, labels_path, block_size=block_size, overlap=0.2)
    # test_data, test_labels, _ = prepare_training_and_testing_data(test_filenames_label, map_rgb2cat, labels_path, train=False, block_size=block_size)
    train_data = train_data.transpose(0, 3, 1, 2)
    train_labels = np.expand_dims(train_labels, axis=1)
    print(f"Train data shape: {train_data.shape}, Train labels shape: {train_labels.shape}")
    
    classes, _ = np.unique(train_labels, return_counts=True)
    
    #! Parâmetros UNet
    # Parâmetros gerais
    num_epochs = 50
    learning_rate = 0.001
    batch_size = 16
    num_classes = len(classes)  # Número de classes para classificação multiclasse
    patience = 5  # Número de épocas para early stopping
    save_path = "data/UNet/best_model.pth"  # Caminho para salvar o melhor modelo

    # Assumindo que os dados já estão carregados como tensores
    # Tensores: `images` (NxCxHxW) e `masks` (NxHxW)
    # Substitua `images` e `masks` pelos tensores reais do dataset Potsdam
    # images = torch.randn(100, 3, 256, 256)  # Exemplo
    # masks = torch.randint(0, num_classes, (100, 256, 256))  # Exemplo

    # Divisão do conjunto de dados
    dataset = TensorDataset(torch.Tensor(train_data), torch.Tensor(train_labels))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instanciar o modelo
    model = UNet(in_channels=6, out_channels=num_classes).to("cuda" if torch.cuda.is_available() else "cpu")

    # Definição da função de perda e otimizador
    criterion = nn.CrossEntropyLoss(ignore_index=5)  # Para segmentação multiclasse
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    #! Treino
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, save_path)