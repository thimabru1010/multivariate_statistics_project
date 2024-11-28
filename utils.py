import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from skimage.util.shape import view_as_windows
from osgeo import gdal
import os
from tqdm import tqdm
import pandas as pd

def extract_patches(image: np.ndarray, patch_size: int, overlap: float) -> np.ndarray:
    stride = int((1-overlap)*patch_size)
    if len(image.shape) == 4:
        window_shape_array = (1, patch_size, patch_size, image.shape[3])
        print(image.shape)
        return np.array(view_as_windows(image, window_shape_array, step=(1, stride, stride, 1))).reshape((-1,) + window_shape_array)
    elif len(image.shape) == 3:
        window_shape_array = (1, patch_size, patch_size)
        print(image.shape)
        return np.array(view_as_windows(image, window_shape_array, step=(1, stride, stride))).reshape((-1,) + window_shape_array)
    
# def reset_signal_handlers():
#     signal.signal(signal.SIGFPE, signal.SIG_DFL)  # Reset floating-point exceptions
#     signal.signal(signal.SIGSEGV, signal.SIG_DFL)  # Reset segmentation faults
    
def load_tif_image(tif_path):
    gdal_header = gdal.Open(str(tif_path))
    return gdal_header.ReadAsArray()

def load_train_data(train_paths):
    train_data = []
    label_data = []
    for folder_path in train_paths:
        basename = folder_path.split('/')[-1]
        rgb_path = os.path.join('data', folder_path, f'{basename}_10m_RGB.tif')
        label_path = os.path.join('data', folder_path, f'{basename}_labels.tif')

        img = load_tif_image(rgb_path).transpose(1, 2, 0)
        label = load_tif_image(label_path)

        # print(img.shape)
        train_data.append(img)
        label_data.append(label)
    return np.stack(train_data, axis=0), np.stack(label_data, axis=0)

def filter_class(data, labels, target_class, percentage_limit=0.7):
    # Inicialize listas para armazenar os patches que não excedem o limite de porcentagem
    filtered_data = []
    filtered_labels = []

    # Iterar sobre cada patch
    for i in tqdm(range(data.shape[0])):
        class_percentage = np.sum(labels[i] == target_class) / labels[i].size
        # print(class_percentage)
        
        # Se a porcentagem da classe alvo for menor ou igual ao limite, mantenha o patch
        if class_percentage <= percentage_limit:
            # print('here')
            filtered_data.append(data[i])
            filtered_labels.append(labels[i])
            # filtered_flag = True

    # Converta as listas para arrays NumPy
    filtered_data = np.array(filtered_data)
    filtered_labels = np.array(filtered_labels)

    print("Original number of patches:", data.shape[0])
    print("Filtered number of patches:", filtered_data.shape[0])
    return filtered_data, filtered_labels

def load_classes(data_path):
    classes_df = pd.read_csv(data_path, sep='\t')
    classes_df = classes_df.set_index('ID')
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan',
 'black', 'yellow', 'magenta', 'lime', 'teal', 'indigo', 'maroon', 'navy', 'peru', 'gold',
 'darkorange', 'darkgreen', 'darkred', 'darkblue', 'darkmagenta', 'darkcyan', 'darkgray',
 'darkolivegreen', 'cyan', 'darkslategray', 'darkgoldenrod', 'darkseagreen', 'darkslateblue']
    classes_df['Color'] = colors
    return classes_df

def labels2groups(classes_df, labels):
    group_labels = classes_df[['ID', 'Group_ID']].values
    # print(group_labels.shape)
    labels_tmp = labels.copy()
    for label in group_labels:
        # print(label)
        labels_tmp[labels == label[0]] = label[1]
        # print(np.unique(labels))
    labels_groups = labels_tmp.copy()
    return labels_groups

def pixels2histogram(data):
    # data = np.mean(data, axis=3)
    return np.array([np.histogram(patch, bins=256, range=(0, 1), density=True)[0] for patch in data])

def histogram2pixels(data):
    return np.array([np.repeat(patch, 256) for patch in data])

def normalize_to_255(image):
    normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
    return (normalized * 255).astype(np.uint8)