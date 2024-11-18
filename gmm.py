import numpy as np
import os
import pandas as pd
from sklearn.mixture import GaussianMixture
from osgeo import gdal
import sys
import signal
from sklearn.model_selection import train_test_split

def reset_signal_handlers():
    signal.signal(signal.SIGFPE, signal.SIG_DFL)  # Reset floating-point exceptions
    signal.signal(signal.SIGSEGV, signal.SIG_DFL)  # Reset segmentation faults
    
def load_tif_image(tif_path):
    gdal_header = gdal.Open(str(tif_path))
    return gdal_header.ReadAsArray()

def load_train_data(train_paths):
    train_data = []
    label_data = []
    for folder_path in train_paths:
        basename = folder_path.split('/')[-1]
        rgb_path = os.path.join('data', folder_path, basename + '_10m_RGB.tif')
        label_path = os.path.join('data', folder_path, basename + 'labels.tif')
        
        img = load_tif_image(rgb_path).transpose(1, 2, 0)
        label = load_tif_image(label_path).transpose(1, 2, 0)
        
        # print(img.shape)
        train_data.append(img)
        label_data.append(label)
    return np.stack(train_data, axis=0), np.stack(label_data, axis=0)
        

if __name__ == '__main__':
    reset_signal_handlers()
    # Load the data
    data_path = os.path.join('data', 'meta.csv')
    df = pd.read_csv(data_path)
    
    print(df['Grid'].value_counts())
    
    df = df[df.Grid == 1]
    df = df[df.Season == 'Summer']
    print(df.shape)
    # Get random sample from train_data
    df = df.sample(frac=1e-2)
    print(df.shape)
    
    data, labels = load_train_data(df['Path'].values.tolist())
    print(data.shape)
    print(labels.shape)
    
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.5)
    
    n_classes = np.unique(labels)
    means = []
    covs = []
    weights = []
    for clss in n_classes:
        pixels_clss = train_data[train_labels == clss]
        
        means.append(np.mean(pixels_clss, axis=0))
        covs.append(np.cov(pixels_clss, rowvar=False))
        weights.append(len(pixels_clss) / len(train_data))
    
    # Inicializar o GMM com os parâmetros conhecidos
    gmm = GaussianMixture(n_components=n_classes, covariance_type='full', max_iter=100)

    # Configurar parâmetros iniciais no GMM
    gmm.means_init = np.array(means)
    gmm.weights_init = np.array(weights)
    gmm.precisions_init = np.linalg.inv(np.array(covs))
    
    # Ajustar o modelo com refinamento EM
    gmm.fit(train_data)

    # Fazer previsões para classificar todos os pixels
    predicoes = gmm.predict(test_data)
    
    

    # Remodelar a previsão para o formato da imagem original (512x512, neste exemplo)
    # segmentacao = predicoes.reshape(imagem_rgb.shape[:2])
