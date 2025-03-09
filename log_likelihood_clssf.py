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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import scipy.stats as stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster Potsdam dataset')
    parser.add_argument('--pca', action='store_true', help='Apply PCA to data')
    parser.add_argument('--patch_size', type=int, default=1, help='Selects the size of the patches')
    parser.add_argument('--labels_path', type=str, default='data/Potsdam/5_Labels_all', help='Path to labels')
    args = parser.parse_args()
    
    labels_path = 'data/Potsdam/5_Labels_all'
    filenames_label = os.listdir(labels_path)
    
    pca_flag = args.pca
    
    # filenames_label = np.random.choice(filenames_label, 3)
    filenames_label = ['top_potsdam_2_14_label.tif', 'top_potsdam_6_12_label.tif', 'top_potsdam_5_13_label.tif']
    print(filenames_label)
    
    # Train test split
    # train_filenames_label, test_filenames_label = train_test_split(filenames_label, test_size=0.2)
    train_filenames_label = ['top_potsdam_2_14_label.tif', 'top_potsdam_6_12_label.tif']
    test_filenames_label = ['top_potsdam_5_13_label.tif']
    print(len(train_filenames_label), len(test_filenames_label))
    
    map_rgb2cat = {(255, 255, 255): 0, (0, 0, 255): 1, (0, 255, 255): 2, (0, 255, 0): 3, (255, 255, 0): 4, (255, 0, 0): 5}
    # Converta as chaves do dicionário para tuplas de inteiros
    colors = list(map_rgb2cat.keys())
    print(colors)

    block_size = args.patch_size
    
    train_data, train_labels, train_idx_patches = prepare_training_and_testing_data(train_filenames_label, map_rgb2cat, labels_path, block_size=block_size)
    test_data, test_labels, test_idx_patches = prepare_training_and_testing_data(test_filenames_label, map_rgb2cat, labels_path, train=False, block_size=block_size)
    
    print(train_data.shape, train_labels.shape)
    print(test_data.shape, test_labels.shape)
    
    label_names = ['Impervious surfaces', 'Building', 'Low vegetation', 'Tree', 'Car', 'background']
    plot_classes_histogram(train_labels, label_names, show=False)
    
    if args.patch_size > 1:
        exp_folder = 'Patches'
        if pca_flag:
            print('Processing PCA...')
            data = np.concatenate([train_data, test_data], axis=0)
            pca = PCA(n_components=0.95)
            pca.fit(data.reshape(data.shape[0], -1))
            
            train_data = pca.transform(train_data.reshape(train_data.shape[0], -1))
            test_data = pca.transform(test_data.reshape(test_data.shape[0], -1))
            
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = explained_variance_ratio.cumsum()
            # print(cumulative_variance, explained_variance_ratio)
            print(f"Explained variance: {cumulative_variance[-1]}")
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
    
    # Normalize idx data
    train_idx_patches = train_idx_patches / np.max(train_idx_patches)
    test_idx_patches = test_idx_patches / np.max(test_idx_patches)
    print(train_idx_patches.shape, train_data.shape)
    print(test_idx_patches.shape, test_data.shape)
    
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
    
    #! Algorithm starts here
    # Group pixels by class
    means = {}
    covs = {}
    for clss in classes:
        class_pixels = train_data[train_labels == clss]
        means[clss] = np.mean(class_pixels, axis=0)
        covs[clss] = np.cov(class_pixels.T)
    
    classes_likelihoods = []
    print(classes)
    for clss in classes:
        # Calculamos a Log-Verossimilhança para cada classe
        classes_likelihoods.append(stats.multivariate_normal.logpdf(test_data, mean=means[clss], cov=covs[clss]))
    
    #! A classe com maior verossimilhança é a classe predita
    test_preds = np.argmax(classes_likelihoods, axis=0)
    
    output_folder = f'data/log_likelihood/{exp_folder}/{features_folder}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    np.save(os.path.join(output_folder, 'preds.npy'), test_preds)
    np.save(os.path.join(output_folder, 'labels.npy'), test_labels)
    print('Predictions saved!')