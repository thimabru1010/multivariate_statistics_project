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
from utils import extract_patches, load_train_data, filter_class, load_classes,\
    labels2groups, pixels2histogram
import matplotlib.pyplot as plt
from utils import load_tif_image, extract_patches, RGB2Categories
import re

def preprocess(filenames, map_rgb2cat):
    for filename in filenames_label:
        basename = filename.split('_label.tif')[0]
        label = load_tif_image(os.path.join(labels_path, filename))
        irrg = load_tif_image(os.path.join('data/Potsdam/3_Ortho_IRRG/', f'{basename}_IRRG.tif'))
        dsm_basename = 'dsm' + basename[3:]
        dsm_basename = re.sub(r'_(\d)(?=(_|\b))', r'_0\1', dsm_basename)
        print(dsm_basename)
        # dsm_height = load_tif_image(os.path.join('data/Potsdam/1_DSM_normalisation/', f'{dsm_basename}_normalized_lastools.jpg'))
        
        label = label.transpose(1, 2, 0)
        irrg = irrg.transpose(1, 2, 0)
        
        label = RGB2Categories(label, map_rgb2cat)
        print(label.shape)
        print(irrg.shape)
        # print(dsm_height.shape)

        label_patches = extract_patches(label, 10, 0).squeeze(axis=1)
        irrg_patches = extract_patches(irrg, 10, 0).squeeze(axis=1)
        # dsm_height_patches = extract_patches(dsm_height, 10, 0).squeeze(axis=1)
        
        train_data.append(irrg_patches)
        train_labels.append(label_patches)
    return np.array(train_data), np.array(train_labels)


if __name__ == '__main__':
    labels_path = 'data/Potsdam/5_Labels_all'
    filenames_label = os.listdir(labels_path)
    
    filenames_label = np.random.choice(filenames_label, 3)
    
    # Train test split
    train_filenames_label, test_filenames_label = train_test_split(filenames_label, test_size=0.2)
    
    train_data = []
    train_labels = []
    map_rgb2cat = {'(255, 255, 255)': 0, '(0, 255, 0)': 1, '(0, 255, 255)': 2, '(0, 0, 255)': 3, '(255, 255, 0)': 4}
    train_data, train_labels = preprocess(train_filenames_label, map_rgb2cat)
    test_data, test_labels = preprocess(test_filenames_label, map_rgb2cat)
        