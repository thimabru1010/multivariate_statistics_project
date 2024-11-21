import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from skimage.util.shape import view_as_windows
from osgeo import gdal
import os

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


# # Load the data
# data_path = os.path.join('data', 'meta.csv')
# df = pd.read_csv(data_path)

# print(df['Grid'].value_counts())

# df = df[df.Grid == 1]
# df = df[df.Season == 'Summer']
# print(df.shape)
# # Get random sample from train_data
# df = df.sample(frac=5e-2)
# print('Dataset size:', df.shape)

# data, labels = load_train_data(df['Path'].values.tolist())
# # data = data.reshape(data.shape[0], -1)
# # labels = labels.reshape(labels.shape[0], -1)
# print(data.shape, labels.shape)

# # Plot 3 rgb image and segmentation labels side by side
# fig, axs = plt.subplots(3, 2, figsize=(10, 10))
# for i in range(3):
#     axs[i, 0].imshow(data[i])
#     axs[i, 1].imshow(labels[i])
# plt.show()


# import numpy as np
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# from utils import load_train_data