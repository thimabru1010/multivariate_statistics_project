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
    df = df.sample(frac=1e-2)
    print('Dataset size:', df.shape)
    
    data, labels = load_train_data(df['Path'].values.tolist())
    # data = data.reshape(data.shape[0], -1)
    # labels = labels.reshape(labels.shape[0], -1)
    print(data.shape, labels.shape)

    # data = extract_patches(data, 10, 0).squeeze(axis=1)
    # labels = extract_patches(labels, 10, 0).squeeze(axis=1)
    
    filtered_data, filtered_labels = filter_class(data, labels, 12, 0.2)
    filtered_data, filtered_labels = filter_class(data, labels, 13, 0.2)
    filtered_data, filtered_labels = filter_class(data, labels, 14, 0.2)
    filtered_data, filtered_labels = filter_class(filtered_data, filtered_labels, 15, 0.2)
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
    
    print(data.shape)
    print(labels.shape)
    
    pca = PCA(n_components=0.95)
    data = pca.fit_transform(data.reshape(data.shape[0], -1))
    
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = explained_variance_ratio.cumsum()
    # print(cumulative_variance, explained_variance_ratio)
    print(f"Explained variance: {cumulative_variance[-1]}")
    
    labels = np.array(mode(labels.reshape(labels.shape[0], -1), axis=1).mode)
    print(labels[0])
    
    print(f"Reduced shape: {data.shape}")
    print(f"Reduced labels shape: {labels.shape}")
    
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
    weights = np.ones_like(labels.reshape(-1)) / len(labels.reshape(-1))
    n, bins, patches = ax.hist(labels.reshape(-1), bins=np.unique(labels).shape[0], weights=weights)

    # Substitua os rótulos do eixo x pelos nomes das classes
    ax.set_xticks(bins[:-1] + (bins[1] - bins[0]) / 2)
    ax.set_xticklabels([groups_df.loc[cls, "Group"] for cls in unique_classes], rotation=45, ha='right')

    ax.set_title('Histogram of classes')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Normalize data
    data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    # print(f"Max: {np.max(data)}, Min: {np.min(data)}")
    # Apply PCA to reduce the number of images (samples)
    data = data.reshape(labels.shape[0], -1)
    labels = labels.reshape(labels.shape[0], -1)
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.5)
    
    classes, counts = np.unique(labels, return_counts=True)
    # class2count = dict(zip(classes, counts))
    # total_labels = np.sum(counts)
    # print(total_labels)
    # class_weights = np.array([total_labels / class2count[label] for label in classes])
    # print(np.sum(class_weights))
    # class_weights /= class_weights.sum()
    # class2weights = dict(zip(classes, class_weights))
    # print(class2weights)
    
    # Implementa o KMeans para encontrar os clusters
    kmeans = KMeans(n_clusters=20, random_state=0)
    
    # Faz o fit dos dados de treino
    print(train_data.shape)
    # 1/0
    kmeans.fit(train_data)
    
    # Faz previsões para classificar todos os pixels
    train_preds = kmeans.predict(train_data)
    print(f"Predictions shape: {train_preds.shape}")
    
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
        print(dict(zip(unique_labels, counts)))
        most_common_label = mode(cluster_labels).mode
        train_preds[preds_tmp == cluster] = most_common_label
        cluster2label[cluster] = most_common_label
        print(f'Cluster: {cluster} - Most common label: {most_common_label}')
    
    print(cluster2label)
    # Fazer previsões para classificar todos os pixels
    test_preds = kmeans.predict(test_data)
    print(f"Predictions shape: {test_preds.shape}")

        
    accuracy = np.mean(test_preds == test_labels)
    print(f'Test Accuracy: {accuracy}')
    
    np.save('data/kmeans_preds.npy', test_preds)
    np.save('data/kmeans_labels.npy', test_labels)
    print('Predictions saved!')
    # results = pd.DataFrame({'Predictions': test_preds, 'Labels': test_labels})
    # results.to_csv('data/results.csv', index=False)
