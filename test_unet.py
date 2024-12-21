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
    rgb_to_categories, plot_classes_histogram, balance_dataset, create_block_index_matrix,\
        prepare_training_and_testing_data, reconstruct_patches, categories_to_rgb
import argparse
import matplotlib.patches as mpatches
from sklearn.metrics import f1_score, accuracy_score, jaccard_score
from unet import UNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def test_model(model, save_path, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(save_path))  # Carrega o melhor modelo salvo
    model.to(device)

    print("Testing the model...")
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_images, batch_masks in test_loader:
            batch_images = batch_images.to(device)
            batch_masks = batch_masks.to(device, dtype=torch.long)

            outputs = model(batch_images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            targets = batch_masks.cpu().numpy()

            all_preds.append(preds)
            all_targets.append(targets)

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)[:, 0]
    print(preds.shape, targets.shape)
    preds[targets == 5] = 5
    # Flatten arrays for metric computation
    all_preds = np.concatenate([p.flatten() for p in all_preds])
    all_targets = np.concatenate([t.flatten() for t in all_targets])
    
    all_preds = all_preds[all_targets != 5]
    all_targets = all_targets[all_targets != 5]

    # Metrics computation
    f1 = f1_score(all_targets, all_preds, average="weighted")
    acc = accuracy_score(all_targets, all_preds)
    miou = jaccard_score(all_targets, all_preds, average="weighted")

    print(f"Test Results - F1 Score: {f1:.4f}, Accuracy: {acc:.4f}, mIoU: {miou:.4f}")
    return preds
    
if __name__ == '__main__':
    labels_path = 'data/Potsdam/5_Labels_all'
    
    test_filenames_label = ['top_potsdam_5_13_label.tif']
    print(len(test_filenames_label))
    
    # map_rgb2cat = {'(255, 255, 255)': 0, '(255, 0, 0)': 1, '(0, 255, 255)': 2, '(0, 255, 0)': 3, '(0, 0, 255)': 4, '(255, 255, 0)': 5}
    map_rgb2cat = {(255, 255, 255): 0, (0, 0, 255): 1, (0, 255, 255): 2, (0, 255, 0): 3, (255, 255, 0): 4, (255, 0, 0): 5}
    # Converta as chaves do dicionário para tuplas de inteiros
    colors = list(map_rgb2cat.keys())
    print(colors)

    block_size = 128
    
    test_data, test_labels, _ = prepare_training_and_testing_data(test_filenames_label, map_rgb2cat, labels_path, train=False, block_size=block_size)
    test_data = test_data.transpose(0, 3, 1, 2)
    test_labels = np.expand_dims(test_labels, axis=1)
    print(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")
    
    classes, _ = np.unique(test_labels, return_counts=True)
    
    #! Parâmetros UNet
    batch_size = 16
    num_classes = len(classes)  # Número de classes para classificação multiclasse
    save_path = "data/UNet/best_model.pth"  # Caminho para salvar o melhor modelo
    
    test_dataset = TensorDataset(torch.Tensor(test_data), torch.Tensor(test_labels))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = UNet(in_channels=6, out_channels=num_classes).to("cuda" if torch.cuda.is_available() else "cpu")
    
    preds = test_model(model, save_path, test_loader)
    
    img_test_preds = reconstruct_patches(preds, original_shape=(6000, 6000))
    
    img_test_label_rgb = load_tif_image('data/Potsdam/5_Labels_all/top_potsdam_5_13_label.tif').transpose(1, 2, 0)
    
    map_rgb2cat = {(255, 255, 255): 0, (0, 0, 255): 1, (0, 255, 255): 2, (0, 255, 0): 3, (255, 255, 0): 4, (255, 0, 0): 5}
    map_cat2rgb = {v: k for k, v in map_rgb2cat.items()}
    label_names = ['Impervious surfaces', 'Building', 'Low vegetation', 'Tree', 'Car', 'Background']
    map_rgb2names = {k: label_names[v] for k, v in map_rgb2cat.items()}
    img_test_preds_rgb = categories_to_rgb(img_test_preds, map_cat2rgb)
    # print(img_test_label.shape)
    # img_test_label_rgb = categories_to_rgb(img_test_label, map_cat2rgb)
    
    # Plot predicted reconstructed image, reconstructed labels and original labels
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(img_test_preds_rgb)
    ax[0].set_title('Predicted labels')
    ax[0].axis('off')
    ax[1].imshow(img_test_label_rgb)
    ax[1].set_title('Original labels')
    ax[1].axis('off')
    # Put a legend to the right of the current axis
    colors = list(map_rgb2cat.keys())
    patches = [mpatches.Patch(color=np.array(color)/255, label=f'{map_rgb2names[color]}') for color in colors]
    fig.legend(handles=patches, loc='upper right', bbox_to_anchor=(1, 0.85), ncol=1)
    plt.tight_layout(rect=[0, 0, 0.85, 0.9]) 
    plt.show()
    fig.savefig(f'data/UNet/predicted_reconstructed_original_labels.png', dpi=300)
    plt.close()
    