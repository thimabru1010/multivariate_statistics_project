import numpy as np
import os
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from osgeo import gdal
import sys
import signal
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from skimage.util.shape import view_as_windows
from scipy.stats import mode
from tqdm import tqdm
from utils import extract_patches, load_train_data, load_classes,\
    labels2groups, pixels2histogram, calculate_miou
import matplotlib.pyplot as plt
from utils import load_tif_image, extract_training_patches, extract_testing_patches,\
    categories_to_rgb, rgb_to_categories, plot_classes_histogram, balance_dataset, create_block_index_matrix,\
        reconstruct_patches, reconstruct_reduced_patches
import re
import argparse
import matplotlib.patches as mpatches
from sklearn.metrics import f1_score, accuracy_score

if __name__ == '__main__':
    sufix = 'Baseline_10x10'
    print(f'Experiment: {sufix}')
    test_preds = np.load(f'data/lda_results/{sufix}/preds.npy')#.reshape((360000, 10, 10))
    img_test_label = load_tif_image('data/Potsdam/5_Labels_all/top_potsdam_5_13_label.tif').transpose(1, 2, 0)
    print(test_preds.shape)
    print(img_test_label.reshape(-1).shape)
    print(f"Classes: {np.unique(test_preds)}")
    print(f"Classes: {np.unique(img_test_label)}")
    
    block_size = 10
    # if sufix == 'Baseline':
    #     block_size = 1
    # else:
    #     block_size = 10
    
    # Decode preds that were reduced by PCA
    # Convert (360000,) to (360000, 10, 10)
    if block_size > 1:
        img_decoded = np.zeros_like(img_test_label.mean(axis=-1))
        img_decoded_patches = extract_testing_patches(img_decoded, block_size)
        for i in range(img_decoded_patches.shape[0]):    
            img_decoded_patches[i, :, :] = np.repeat(test_preds[i], block_size*block_size).reshape(block_size, block_size)
        
        img_test_preds = reconstruct_patches(img_decoded_patches, original_shape=(6000, 6000))
    else:
        img_test_preds = test_preds.reshape((6000, 6000))
    
    # Save an image for each class
    for i in range(6):
        img_test_preds_class = np.zeros_like(img_test_preds)
        img_test_preds_class[img_test_preds == i] = i
        plt.imsave(f'data/lda_results/{sufix}/test_preds_class_{i}.png', img_test_preds_class)
    
    # Convert to RGB
    map_rgb2cat = {(255, 255, 255): 0, (0, 0, 255): 1, (0, 255, 255): 2, (0, 255, 0): 3, (255, 255, 0): 4, (255, 0, 0): 5}
    print(img_test_label.shape)
    # img_test_label = rgb_to_categories(img_test_label, map_rgb2cat)
    
    # Assign background class to preds
    # img_test_preds[img_test_label == 5] = 5

    # Create reverse dictionary
    map_cat2rgb = {v: k for k, v in map_rgb2cat.items()}
    img_test_preds_rgb = categories_to_rgb(img_test_preds, map_cat2rgb)
    
    # Defina a cor alvo
    target_color = np.array([255, 0, 0])
    # Crie uma máscara para os pixels que correspondem à cor alvo
    mask = np.all(img_test_label == target_color, axis=-1)
    # Atribua a cor alvo aos pixels correspondentes em img_test_preds_rgb
    img_test_preds_rgb[mask] = target_color

    # Save image
    plt.imsave(f'data/lda_results/{sufix}/test_preds.png', img_test_preds_rgb)
    
    # Reconstruct reduced labels
    test_labels_patches = np.load(f'data/lda_results/{sufix}/labels.npy')
    
    # Calculate Classification metrics
    
    # Filter background class
    test_preds_metrics = test_preds.reshape(-1)
    test_labels_metrics = test_labels_patches.reshape(-1)
    
    test_preds_metrics = test_preds_metrics[test_labels_metrics != 5]
    test_labels_metrics = test_labels_metrics[test_labels_metrics != 5]
    
    f1 = f1_score(test_labels_metrics, test_preds_metrics, average='weighted')
    accuracy = accuracy_score(test_labels_metrics, test_preds_metrics)
    miou = calculate_miou(test_labels_metrics, test_preds_metrics, num_classes=5)
    
    print(f'F1-Score: {f1}')
    print(f'Accuracy: {accuracy}')
    print(f'mIoU: {miou}')
    
    
    if block_size > 1:
        img_decoded = np.zeros_like(img_test_label.mean(axis=-1))
        img_decoded_patches = extract_testing_patches(img_decoded, block_size)
        for i in range(img_decoded_patches.shape[0]):    
            img_decoded_patches[i, :, :] = np.repeat(test_labels_patches[i], block_size*block_size).reshape(block_size, block_size)
        img_test_labels = reconstruct_patches(img_decoded_patches, original_shape=(6000, 6000))
    else:
        img_test_labels = test_labels_patches.reshape((6000, 6000))
    
    img_test_labels_rgb = categories_to_rgb(img_test_labels, map_cat2rgb)
    img_test_labels_rgb[mask] = target_color
    # plt.imsave(f'data/lda_results/test_labels_30x30_rec.png', img_test_labels_rgb)
    
    # img_label = load_tif_image('data/Potsdam/5_Labels_all/top_potsdam_5_13_label.tif').transpose(1, 2, 0)
    
    # Plot predicted reconstructed image, reconstructed labels and original labels
    fig, ax = plt.subplots(1, 3, figsize=(25, 6))
    ax[0].imshow(img_test_preds_rgb)
    ax[0].set_title('Predicted labels')
    ax[0].axis('off')
    ax[1].imshow(img_test_labels_rgb)
    ax[1].set_title('Reconstructed labels')
    axis = ax[1].axis('off')
    ax[2].imshow(img_test_label)
    ax[2].set_title('Original labels')
    ax[2].axis('off')
    # Put a legend to the right of the current axis
    colors = list(map_rgb2cat.keys())
    patches = [mpatches.Patch(color=np.array(color)/255, label=f'Class {map_rgb2cat[color]}: {color}') for color in colors]
    fig.legend(handles=patches, loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=3)
    plt.tight_layout()
    plt.show()
    fig.savefig(f'data/lda_results/{sufix}/predicted_reconstructed_original_labels.png', dpi=300)
    plt.close()
    
    