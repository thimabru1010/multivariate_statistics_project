import numpy as np
import os
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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
from imblearn.over_sampling import SMOTE

if __name__ == '__main__':
    # reset_signal_handlers()
    # Load the data
    data_path = os.path.join('data', 'meta.csv')
    df = pd.read_csv(data_path)
    
    print(df['Grid'].value_counts())
    
    df = df[df.Grid == 1]
    df = df[df.Season == 'Summer']
    print(df.shape)
    # Get random sample from train_data
    df = df.sample(frac=5e-2)
    print('Dataset size:', df.shape)
    
    data, labels = load_train_data(df['Path'].values.tolist())
    # data = data.reshape(data.shape[0], -1)
    # labels = labels.reshape(labels.shape[0], -1)
    print(data.shape, labels.shape)
    
    filtered_data, filtered_labels = filter_class(data, labels, 12)
    filtered_data, filtered_labels = filter_class(filtered_data, filtered_labels, 15)
    filtered_data, filtered_labels = filter_class(filtered_data, filtered_labels, 17)
    filtered_data, filtered_labels = filter_class(filtered_data, filtered_labels, 16)
    filtered_data, filtered_labels = filter_class(filtered_data, filtered_labels, 33,\
        percentage_limit=0.9)
    data, labels = filtered_data, filtered_labels
    print(data.shape, labels.shape)
    
    print(data.shape)
    print(labels.shape)
    
    data = extract_patches(data, 10, 0).squeeze(axis=1)
    labels = extract_patches(labels, 10, 0).squeeze(axis=1)
    
    # pca = PCA(n_components=0.95)
    # data = pca.fit_transform(data.reshape(data.shape[0], -1))
    
    # explained_variance_ratio = pca.explained_variance_ratio_
    # cumulative_variance = explained_variance_ratio.cumsum()
    # # print(cumulative_variance, explained_variance_ratio)
    # print(f"Explained variance: {cumulative_variance[-1]}")
    
    labels = np.array(mode(labels.reshape(labels.shape[0], -1), axis=1).mode)
    print(labels[0])
    
    # print(f"Reduced shape: {data.shape}")
    # print(f"Reduced labels shape: {labels.shape}")
    
    classes_df = load_classes('data/classes.csv')
    grouped_classes_df = pd.read_csv('data/class_groups.csv')
    classes_df = pd.merge(classes_df.reset_index(), grouped_classes_df, on='Class')
    
    labels = labels2groups(classes_df, labels)
    groups_df = grouped_classes_df[['Group_ID', 'Group', 'Group_Color']].drop_duplicates()
    groups_df.set_index('Group_ID', inplace=True)
    
    # Plot histogram of classes
    # Create an histogram of the Group classes
    fig, ax = plt.subplots()
    unique_classes, counts = np.unique(labels, return_counts=True)
    print(dict(zip(unique_classes, counts)))
    weights = np.ones_like(labels.reshape(-1)) / len(labels.reshape(-1))
    n, bins, patches = ax.hist(labels.reshape(-1), bins=np.unique(labels).shape[0], weights=weights)

    # Substitua os rótulos do eixo x pelos nomes das classes
    ax.set_xticks(bins[:-1] + (bins[1] - bins[0]) / 2)
    # print([groups_df.loc[cls, "Group"] for cls in unique_classes])
    ax.set_xticklabels([groups_df.loc[cls, "Group"] for cls in unique_classes], rotation=45, ha='right')

    ax.set_title('Histogram of classes')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Frequency')

    plt.tight_layout()
    # plt.show()
    # 1/0

    data = data.reshape(data.shape[0], -1)
    labels = labels.reshape(-1)
    print(data.shape, labels.shape)
    unique_classes, counts = np.unique(labels, return_counts=True)
    print(dict(zip(unique_classes, counts)))

    # Normalize data
    # data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    # print(f"Max: {np.max(data)}, Min: {np.min(data)}")
    # Apply PCA to reduce the number of images (samples)
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.5)
    
    # classes, counts = np.unique(train_labels, return_counts=True)
    # class2count = dict(zip(classes, counts))
    # total_labels = np.sum(counts)
    # print(total_labels)
    # class_weights = np.array([total_labels / class2count[label] for label in classes])
    # print(np.sum(class_weights))
    # class_weights /= class_weights.sum()
    # class2weights = dict(zip(classes, class_weights))
    # print(class2weights)
    
    lda = LDA(n_components=3)
    lda.fit(train_data, train_labels)
    print(lda.score(test_data, test_labels))
    test_preds = lda.predict(test_data)
    
    print(test_preds.shape)
    accuracy = np.sum(test_preds == test_labels) / test_labels.shape[0]
    
    print(f"Accuracy: {accuracy}")