import numpy as np
import os
import pandas as pd
from sklearn.mixture import GaussianMixture
from osgeo import gdal
import sys
import signal
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from skimage.util.shape import view_as_windows
from scipy.stats import mode
from tqdm import tqdm
from utils import extract_patches, load_train_data

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

    data = extract_patches(data, 10, 0).squeeze(axis=1)
    labels = extract_patches(labels, 10, 0).squeeze(axis=1)
    
    # # Exclude samples with few labels
    # valid_samples = np.sum(labels != 0, axis=(1, 2)) > 0.5 * labels.size
    # data = data[valid_samples]
    # labels = labels[valid_samples]
    
    print(data.shape)
    print(labels.shape)
    
    pca = PCA(n_components=0.95)
    data = pca.fit_transform(data.reshape(data.shape[0], -1))
    
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = explained_variance_ratio.cumsum()
    print(cumulative_variance, explained_variance_ratio)
    
    labels = np.array(mode(labels.reshape(labels.shape[0], -1), axis=1).mode)
    print(labels[0])
    
    print(f"Reduced shape: {data.shape}")
    print(f"Reduced labels shape: {labels.shape}")

    # Normalize data
    data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    print(f"Max: {np.max(data)}, Min: {np.min(data)}")
    # Apply PCA to reduce the number of images (samples)
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.5)
    
    classes, counts = np.unique(labels, return_counts=True)
    class2count = dict(zip(classes, counts))
    # print(class2count)
    total_labels = np.sum(counts)
    print(total_labels)
    class_weights = np.array([total_labels / class2count[label] for label in classes])
    print(np.sum(class_weights))
    class_weights /= class_weights.sum()
    class2weights = dict(zip(classes, class_weights))
    print(class2weights)
    
    # Inicializar o GMM com os parâmetros conhecidos
    gmm = GaussianMixture(n_components=len(classes), covariance_type='full', max_iter=50,\
        init_params='kmeans', verbose=2, verbose_interval=5, weights_init=class_weights)
    
    # Ajustar o modelo com refinamento EM
    gmm.fit(train_data)
        
    # max_iterations = 10  # Número máximo de iterações para refinamento
    # for iter in range(max_iterations):
    #     # Fazer previsões no conjunto de treino
    #     train_preds = gmm.predict(train_data)
        
    #     # Reatribuir os rótulos para os clusters no conjunto de treino
    #     clusters = np.unique(train_preds)
    #     cluster2label = {}
    #     for cluster in clusters:
    #         cluster_labels = train_labels[train_preds == cluster]
    #         most_common_label = mode(cluster_labels).mode[0]
    #         cluster2label[cluster] = most_common_label
    #         train_preds[train_preds == cluster] = most_common_label

    #     accuracy = np.mean(train_preds == train_labels)
    #     print(f'Train Accuracy: {accuracy} - Iteration {iter}')
    #     if accuracy >= 0.95:
    #         break
    
    #     classes_preds, counts_preds = np.unique(cluster2label.values(), return_counts=True)
    #     class2count = dict(zip(classes, counts))
    #     missing_classes = set(classes) - set(classes_preds)
    #     adjusted_weights = []
    #     for cls in classes:
    #         if cls in missing_classes:
    #             # Peso muito alto para classes ausentes
    #             adjusted_weights.append(0.8)  # Força a classe ausente a ser priorizada
    #         else:
    #             # Peso proporcional às previsões existentes
    #             adjusted_weights.append(1 / class2count[cls])
        
    
    #     adjusted_weights /= adjusted_weights.sum()
    #     class2weights = dict(zip(classes, adjusted_weights))
    #     print(f"New adjusted weights: {class2weights}")
        
    #     # Reajustar os pesos do GMM
    #     gmm.weights_ = adjusted_weights
    #     # print(f"Novos pesos ajustados: {adjusted_weights}")

    #     gmm.fit(train_data)


    # Fazer previsões para classificar todos os pixels
    predicoes = gmm.predict(test_data)
    print(f"Predictions shape: {predicoes.shape}")
    
    # Map the predictions to the original classes by finding the most common class in each cluster
    clusters = np.unique(predicoes)
    cluster2label = {}
    for cluster in clusters:
        # print(f'Cluster {cluster}: {np.sum(predicoes == cluster)}')
        cluster_labels = test_labels[predicoes == cluster]
        most_common_label = mode(cluster_labels).mode
        cluster2label[cluster] = most_common_label
        predicoes[predicoes == cluster] = most_common_label
        print(f'Cluster: {cluster} - Most common label: {most_common_label}')
        
    accuracy = np.mean(predicoes == test_labels)
    print(f'Accuracy: {accuracy}')
    
    results = pd.DataFrame({'Predictions': predicoes, 'Labels': test_labels})
    results.to_csv('data/results.csv', index=False)
