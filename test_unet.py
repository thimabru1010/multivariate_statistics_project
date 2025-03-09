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
from sklearn.metrics import f1_score, accuracy_score, jaccard_score, fbeta_score
from unet import UNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from joblib import load

def test_model(model, test_loader, device, targets_original_shape):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_images, batch_masks in tqdm(test_loader, desc="Testing"):
            batch_images = batch_images.to(device)
            batch_masks = batch_masks.to(device, dtype=torch.long)

            # Forward pass
            outputs = model(batch_images)

            # Get predictions
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            targets = batch_masks.squeeze(1).cpu().numpy()
            
            all_preds.append(preds)
            all_targets.append(targets)
    
    # Combine all predictions and targets
    all_preds = np.concatenate([p.reshape(-1) for p in all_preds])
    all_targets = np.concatenate([t.reshape(-1) for t in all_targets])
    
    all_preds_rec = all_preds.copy()
    all_preds_rec[all_targets == 5] = 5
    all_targets_rec = all_targets.copy()
    
    all_preds = all_preds[all_targets != 5]
    all_targets = all_targets[all_targets != 5]
    
    # Calculate metrics
    test_accuracy = np.mean(all_preds == all_targets)
    test_f1_score = f1_score(all_targets, all_preds, average='weighted')
    test_f2_score = fbeta_score(all_targets, all_preds, beta=2, average='weighted')
    test_miou = jaccard_score(all_targets, all_preds, average='weighted')
    
    print('Test Metrics:')
    print(f'  Accuracy: {test_accuracy:.4f}')
    print(f'  F1 Score: {test_f1_score:.4f}')
    print(f'  F2 Score: {test_f2_score:.4f}')
    print(f'  mIoU: {test_miou:.4f}')
    
    # Return metrics and predictions
    metrics = {
        'accuracy': test_accuracy,
        'f1_score': test_f1_score,
        'f2_score': test_f2_score,
        'miou': test_miou
    }
    
    return all_preds_rec.reshape(targets_original_shape).squeeze(1), all_targets_rec.reshape(targets_original_shape).squeeze(1), metrics
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster Potsdam dataset')
    parser.add_argument('--pca', action='store_true', help='Apply PCA to data')
    parser.add_argument('--patch_size', type=int, default=128, help='Selects the size of the patches')
    parser.add_argument('--labels_path', type=str, default='data/Potsdam/5_Labels_all', help='Path to labels')
    args = parser.parse_args()
    
    labels_path = 'data/Potsdam/5_Labels_all'
    
    test_filenames_label = ['top_potsdam_5_13_label.tif']
    print(len(test_filenames_label))
    
    # map_rgb2cat = {'(255, 255, 255)': 0, '(255, 0, 0)': 1, '(0, 255, 255)': 2, '(0, 255, 0)': 3, '(0, 0, 255)': 4, '(255, 255, 0)': 5}
    map_rgb2cat = {(255, 255, 255): 0, (0, 0, 255): 1, (0, 255, 255): 2, (0, 255, 0): 3, (255, 255, 0): 4, (255, 0, 0): 5}
    # Converta as chaves do dicionário para tuplas de inteiros
    colors = list(map_rgb2cat.keys())
    print(colors)
    
    test_data, test_labels, _ = prepare_training_and_testing_data(test_filenames_label, map_rgb2cat, labels_path, train=False, block_size=1)
    
    if args.pca:
        exp_folder = 'NoPatches'
        features_folder = 'PCA_Channels'
        
        pca = load('data/UNet/pca.joblib')
        
        original_shape = test_data.shape
        test_data = pca.transform(test_data.reshape(test_data.shape[0], -1))
        test_data = test_data.reshape(original_shape[0], original_shape[0])
    
    print(test_data.shape)
    test_data = test_data.reshape(6000, 6000, -1)
    test_labels = test_labels.reshape(6000, 6000)
    
    # Prepare test data and loader
    test_data = extract_testing_patches(test_data, args.patch_size)
    test_labels = extract_testing_patches(test_labels, args.patch_size)
    print(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")
    
    classes, _ = np.unique(test_labels, return_counts=True)
    
    #! Parâmetros UNet
    batch_size = 16
    num_classes = len(classes)  # Número de classes para classificação multiclasse
    save_path = "data/UNet"  # Caminho para salvar o melhor modelo
    
    test_dataset = TensorDataset(torch.Tensor(test_data), torch.Tensor(test_labels))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    #! Teste
    # Carregar o melhor modelo
    model = UNet(in_channels=6, out_channels=num_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pth')))
    
    # Prepare test data and loader
    test_data = torch.Tensor(test_data.transpose(0, 3, 1, 2))
    test_labels = torch.Tensor(np.expand_dims(test_labels, axis=1))
    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Test the model
    predictions, targets, metrics = test_model(model, test_loader, device, test_labels.shape)
    
    # Save results
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    np.save(os.path.join(save_path, 'preds.npy'), predictions)
    np.save(os.path.join(save_path, 'labels.npy'), targets)
    
    # Save metrics to a text file
    with open(os.path.join(save_path, 'metrics.txt'), 'w') as f:
        for key, value in metrics.items():
            f.write(f'{key}: {value}\n')
    
    print('Test results saved!')
    
    img_test_preds = reconstruct_patches(predictions, original_shape=(6000, 6000))
    img_test_label_rgb = load_tif_image('data/Potsdam/5_Labels_all/top_potsdam_5_13_label.tif').transpose(1, 2, 0)
    
    map_rgb2cat = {(255, 255, 255): 0, (0, 0, 255): 1, (0, 255, 255): 2, (0, 255, 0): 3, (255, 255, 0): 4, (255, 0, 0): 5}
    map_cat2rgb = {v: k for k, v in map_rgb2cat.items()}
    label_names = ['Impervious surfaces', 'Building', 'Low vegetation', 'Tree', 'Car', 'Background']
    map_rgb2names = {k: label_names[v] for k, v in map_rgb2cat.items()}
    img_test_preds_rgb = categories_to_rgb(img_test_preds, map_cat2rgb)
    
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
    fig.savefig(os.path.join(save_path, 'predicted_reconstructed_original_labels.png'), dpi=300)
    plt.close()
    