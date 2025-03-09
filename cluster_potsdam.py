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
    labels2groups, pixels2histogram
import matplotlib.pyplot as plt
from utils import load_tif_image, extract_training_patches, extract_testing_patches,\
    rgb_to_categories, plot_classes_histogram, balance_dataset, create_block_index_matrix, prepare_training_and_testing_data
import re
import argparse
import matplotlib.patches as mpatches
from sklearn.metrics import f1_score
import cv2

if __name__ == '__main__':
    # Create argparse
    parser = argparse.ArgumentParser(description='Cluster Potsdam dataset')
    parser.add_argument('--pca', action='store_true', help='Apply PCA to data')
    parser.add_argument('--patch_size', type=int, default=1, help='Selects the size of the patches')
    parser.add_argument('--labels_path', type=str, default='data/Potsdam/5_Labels_all', help='Path to labels')
    parser.add_argument('--n_clusters', type=int, default=5, help='Selects the size of the patches')
    args = parser.parse_args()
    
    filenames_label = os.listdir(args.labels_path)
    
    pca_flag = args.pca
    
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

    # Crie uma imagem de exemplo com as cores
    color_squares = np.array(colors).reshape(1, len(colors), 3).astype(np.uint8)

    # Plote a figura com os quadrados de cores
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.imshow(color_squares)
    ax.set_xticks(range(len(colors)))
    ax.set_xticklabels(map_rgb2cat.values(), rotation=45, ha='right')
    # Crie patches para a legenda
    patches = [mpatches.Patch(color=np.array(color)/255, label=f'Class {map_rgb2cat[color]}: {color}') for color in colors]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.axis('off')
    plt.tight_layout()
    # plt.show()
    # fig.savefig('data/rgb2cat.png', dpi=300)
    plt.close()

    # block_size = args.patches_size
    train_data, train_labels, train_idx_patches = prepare_training_and_testing_data(train_filenames_label, map_rgb2cat, args.labels_path, block_size=args.patch_size)
    test_data, test_labels, test_idx_patches = prepare_training_and_testing_data(test_filenames_label, map_rgb2cat, args.labels_path, train=False, block_size=args.patch_size)
    
    np.save('data/kmeans_results/train_data.npy', train_data)
    
    print(train_data.shape, train_labels.shape)
    print(test_data.shape, test_labels.shape)
    
    
    label_names = ['Impervious surfaces', 'Building', 'Low vegetation', 'Tree', 'Car', 'background']
    # label_names = ['Impervious surfaces', 'Building', 'Low vegetation', 'Tree', 'Car']
    plot_classes_histogram(train_labels, label_names, show=False)
    
    if args.patch_size > 1:
        #! Retalhos, input = (n_samples, n_patches * n_features)
        exp_folder = 'Patches'
        if pca_flag:
            features_folder = 'PCA_Pixels'
            print('Processing PCA...')
            print(train_data.shape)
            pca = PCA(n_components=0.95)
            pca.fit(train_data.reshape(train_data.shape[0], -1))
            
            train_data = pca.transform(train_data.reshape(train_data.shape[0], -1))
            test_data = pca.transform(test_data.reshape(test_data.shape[0], -1))
            
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = explained_variance_ratio.cumsum()
            # print(cumulative_variance, explained_variance_ratio)
            print(f"Cumulative variance: {cumulative_variance[-1]}")
        else:
            features_folder = 'All_Pixels'
            train_data = train_data.reshape(train_data.shape[0], -1)
            test_data = test_data.reshape(test_data.shape[0], -1)
            train_idx_patches = train_idx_patches.reshape(train_idx_patches.shape[0], -1)
            test_idx_patches = test_idx_patches.reshape(test_idx_patches.shape[0], -1)
        
        train_labels = np.array(mode(train_labels.reshape(train_labels.shape[0], -1), axis=1).mode)
        test_labels = np.array(mode(test_labels.reshape(test_labels.shape[0], -1), axis=1).mode)
        train_idx_patches = np.array(mode(train_idx_patches.reshape(train_idx_patches.shape[0], -1), axis=1).mode)
        test_idx_patches = np.array(mode(test_idx_patches.reshape(test_idx_patches.shape[0], -1), axis=1).mode)
    else:
        exp_folder = 'NoPatches'
        #! Sem retalhos, input = (n_samples, n_channels), channels = features
        if pca_flag: #! PCA is applied on the 7 channels
            features_folder = 'PCA_Channels'
            print('Processing PCA...')
            print(train_data.shape)
            pca = PCA(n_components=0.95)
            pca.fit(train_data.reshape(train_data.shape[0], -1))
            
            train_data = pca.transform(train_data.reshape(train_data.shape[0], -1))
            test_data = pca.transform(test_data.reshape(test_data.shape[0], -1))
            
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = explained_variance_ratio.cumsum()
            # print(cumulative_variance, explained_variance_ratio)
            print(f"Explained variance: {cumulative_variance[-1]}")
        else:
            features_folder = 'All_Channels'
            train_data = train_data.reshape(train_data.shape[0], -1)
            test_data = test_data.reshape(test_data.shape[0], -1)
            # train_idx_patches = train_idx_patches.reshape(train_idx_patches.shape[0], -1)
            # test_idx_patches = test_idx_patches.reshape(test_idx_patches.shape[0], -1)
            train_idx_patches = np.expand_dims(train_idx_patches, axis=1)
            test_idx_patches = np.expand_dims(test_idx_patches, axis=1)
    
    # Remove background of training data
    train_data = train_data[train_labels != 5]
    train_idx_patches = train_idx_patches[train_labels != 5]
    train_labels = train_labels[train_labels != 5]
    
    # Standardize idx data to 0-1
    train_idx_patches = train_idx_patches / np.max(train_idx_patches)
    test_idx_patches = test_idx_patches / np.max(test_idx_patches)
    print(train_idx_patches.shape)
    print(test_idx_patches.shape)
    
    if pca_flag:
        train_data = np.concatenate([train_data, np.expand_dims(train_idx_patches, axis=1)], axis=1)
        test_data = np.concatenate([test_data, np.expand_dims(test_idx_patches, axis=1)], axis=1)
    else:
        print(train_data.shape, train_idx_patches.shape)
        train_data = np.concatenate([train_data, train_idx_patches], axis=1)
        test_data = np.concatenate([test_data, test_idx_patches], axis=1)
        
    print(f"Reduced shape: {train_data.shape}")
    print(f"Reduced labels shape: {train_labels.shape}")
    
    print(f"Experiment: {exp_folder}")
    print(f"Features: {features_folder}")
    
    classes, _ = np.unique(train_labels, return_counts=True)
    # kmeans = KMeans(n_clusters=len(classes), random_state=0)
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0)
    
    kmeans.fit(train_data)
    
    # Faz previsões para classificar todos os pixels
    train_preds = kmeans.predict(train_data)
    print(f"Predictions shape: {train_preds.shape}")
    
    # Faz uma seleção do cluster mais comum proporcionalmente a cada classe
    # cluster2label = {}
    # clusters = np.unique(train_preds)
    # print(clusters)
    # preds_tmp = train_preds.copy()
    # # Count the number of pixels per class
    # pixels_per_class = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
    # for clss in list(range(5)):
    #     pixels_per_class[str(clss)] = np.sum(train_labels == clss)
    # print(pixels_per_class)
    # for cluster in clusters:
    #     cluster_labels = train_labels[preds_tmp == cluster]
    #     # Print proportion normalized of each label
    #     unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    #     print(unique_labels, counts)
    #     print(unique_labels.shape, counts.shape)
    #     for clss in unique_labels:
    #         counts[clss] = counts[clss] / pixels_per_class[str(clss)]
    #     most_common_label = np.argmax(counts)
    #     # most_common_label = mode(cluster_labels).mode
    #     train_preds[preds_tmp == cluster] = most_common_label
    #     cluster2label[cluster] = most_common_label
    #     print(f'Cluster: {cluster} - Most common label: {most_common_label}')
    #     print(dict(zip(unique_labels, counts)))
    
    # Faz uma seleção dos clusters mais comuns para cada classe
    cluster2label = {}
    clusters = np.unique(train_preds)
    print(clusters)
    preds_tmp = train_preds.copy()
    for cluster in clusters:
        cluster_labels = train_labels[preds_tmp == cluster]
        # Print proportion normalized of each label
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        counts = counts / np.sum(counts)
        most_common_label = mode(cluster_labels).mode
        train_preds[preds_tmp == cluster] = most_common_label
        cluster2label[cluster] = most_common_label
        print(f'Cluster: {cluster} - Most common label: {most_common_label}')
        print(dict(zip(unique_labels, counts)))
        
    # Metrics
    train_accuracy = np.mean(train_preds == train_labels)
    train_f1_score = f1_score(train_labels, train_preds, average='weighted')
    
    print(f'Train Accuracy: {train_accuracy}')
    print(f'Train F1 Score: {train_f1_score}')
    
    print(cluster2label)
    # Fazer previsões para classificar todos os pixels
    print(f"Test Labels: {np.unique(test_labels)}")
    test_preds = kmeans.predict(test_data)

    # Apply map to test data
    for cluster, label in cluster2label.items():
        test_preds[test_preds == cluster] = label
    
    output_folder = f'data/kmeans_results/{exp_folder}/{features_folder}/clusters={args.n_clusters}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    np.save(os.path.join(output_folder, 'preds.npy'), test_preds)
    np.save(os.path.join(output_folder, 'labels.npy'), test_labels)
    print('Predictions saved!')
    