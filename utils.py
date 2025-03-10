import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from skimage.util.shape import view_as_windows
from osgeo import gdal
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import re

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

def extract_training_patches(image: np.ndarray, patch_size: int, overlap: float) -> np.ndarray:
    """
    Extract patches from an image with any number of dimensions.
    
    Args:
        image: Input array of any shape
        patch_size: Size of patches (same for all spatial dimensions)
        overlap: Overlap fraction between patches (0 to 1)
    
    Returns:
        Array of patches
    """
    stride = int((1-overlap)*patch_size)
    
    # Create window shape - patch_size for first dimensions, full size for channels
    window_shape = tuple([patch_size] * (len(image.shape)-1)) if len(image.shape) > 2 else (patch_size, patch_size)
    if len(image.shape) > 2:
        window_shape += (image.shape[-1],)
    
    # Create step size - stride for spatial dims, full size for channels
    step = tuple([stride] * (len(image.shape)-1)) if len(image.shape) > 2 else (stride, stride)
    if len(image.shape) > 2:
        step += (image.shape[-1],)
    
    print(f"Image shape: {image.shape}")
    print(window_shape)
    patches = view_as_windows(image, window_shape, step=step)
    
    # Reshape to (-1, *window_shape)
    output_shape = (-1,) + window_shape
    return np.array(patches).reshape(output_shape)

def extract_testing_patches(image, patch_size):
    """
    Extracts and sorts patches from a large remote sensing image.
    
    Parameters:
        image_path (str): Path to the remote sensing image.
        patch_size (int): Size of the patches (square patches).
        output_dir (str): Directory to save the sorted patches.
    """
    # Get image dimensions
    h, w = image.shape[:2]
    
    patches = []

    # Extract patches
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            
            # Skip incomplete patches on edges
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue
            
            # Compute metric (e.g., average intensity)
            patches.append(patch)
    return np.array(patches)
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

def filter_class(data, labels, target_class, percentage_limit=0.5):
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

def rgb_to_categories(img_ref_rgb, label_dict):
    # Convert Reference Image in RGB to a single channel integer category
    w = img_ref_rgb.shape[0]
    h = img_ref_rgb.shape[1]
    # c = img_train_ref.shape[2]
    cat_img_train_ref = np.full((w, h), -1, dtype=np.uint8)
    print(label_dict[(255, 255, 0)])
    for i in range(w):
        for j in range(h):
            r = img_ref_rgb[i][j][0]
            g = img_ref_rgb[i][j][1]
            b = img_ref_rgb[i][j][2]
            rgb_key = (r, g, b)
            # rgb_key = str(rgb)
            # print(rgb_key)
            # print(rgb_key == (255, 255, 0))
            # print(rgb_key, (255, 255, 0))
            cat_img_train_ref[i][j] = label_dict[rgb_key]
    return cat_img_train_ref

def categories_to_rgb(cat_img, label_dict):
    w = cat_img.shape[0]
    h = cat_img.shape[1]
    c = 3
    img_rgb = np.zeros((w, h, c), dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            label = cat_img[i][j]
            rgb = label_dict[label]
            img_rgb[i][j] = rgb
    return img_rgb

def plot_classes_histogram(labels, label_names, show=True):
    fig, ax = plt.subplots()
    unique_classes, counts = np.unique(labels, return_counts=True)
    print(dict(zip(unique_classes, counts)))
    weights = np.ones_like(labels.reshape(-1)) / len(labels.reshape(-1))
    n, bins, patches = ax.hist(labels.reshape(-1), bins=np.unique(labels).shape[0], weights=weights)

    # Substitua os rótulos do eixo x pelos nomes das classes
    ax.set_xticks(bins[:-1] + (bins[1] - bins[0]) / 2)
    # print([groups_df.loc[cls, "Group"] for cls in unique_classes])
    ax.set_xticklabels(label_names, rotation=45, ha='right')

    ax.set_title('Histogram of classes')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Frequency')

    plt.tight_layout()
    if show:
        plt.show()
    # 1/0
    
def balance_dataset(data, labels):
    classes, counts = np.unique(labels, return_counts=True)
    print(classes, counts)
    min_samples = np.min(counts)
    balanced_data = []
    balanced_labels = []
    for c in classes:
        idx = np.where(labels == c)[0]
        idx = np.random.choice(idx, min_samples, replace=False)
        balanced_data.append(data[idx])
        balanced_labels.append(labels[idx])
    return np.concatenate(balanced_data, axis=0), np.concatenate(balanced_labels, axis=0)

def oversample_dataset(data, labels):
    classes, counts = np.unique(labels, return_counts=True)
    print(classes, counts)
    max_samples = np.max(counts)
    balanced_data = []
    balanced_labels = []
    for c in classes:
        idx = np.where(labels == c)[0]
        idx = np.random.choice(idx, max_samples, replace=True)
        balanced_data.append(data[idx])
        balanced_labels.append(labels[idx])
    return np.concatenate(balanced_data, axis=0), np.concatenate(balanced_labels, axis=0)

def create_block_index_matrix(image_shape, block_size):
    # Obtenha os índices das posições dos elementos
    indices = np.indices(image_shape[:2])
    
    # Divida os índices pelo tamanho do bloco para obter os índices dos blocos
    block_indices = indices // block_size
    
    # Combine os índices dos blocos para obter um índice único para cada bloco
    block_matrix = block_indices[0] * (image_shape[1] // block_size) + block_indices[1]
    
    return block_matrix

def reconstruct_reduced_patches(data, index_matrix, block_size):
    # Obtenha o número de blocos ao longo do eixo x e y
    n_blocks_x = data.shape[1] // block_size
    n_blocks_y = data.shape[2] // block_size
    
    # Inicialize uma matriz para armazenar os patches reconstruídos
    reconstructed_data = np.zeros((data.shape[0], n_blocks_x * block_size, n_blocks_y * block_size, data.shape[3]))
    
    # Iterar sobre cada patch
    for i in range(data.shape[0]):
        for j in range(n_blocks_x):
            for k in range(n_blocks_y):
                # Obtenha o índice do bloco
                block_idx = j * n_blocks_y + k
                
                # Obtenha a posição do bloco
                block_x = block_idx // (data.shape[2] // block_size)
                block_y = block_idx % (data.shape[2] // block_size)
                
                # Reconstrua o patch
                reconstructed_data[i, block_x*block_size:(block_x+1)*block_size, block_y*block_size:(block_y+1)*block_size] = data[i, j, k]
    return reconstructed_data

def reconstruct_patches(patches, original_shape):
    print(patches.shape)
    patch_size = patches.shape[1]
    img = np.zeros(original_shape)
    idx = 0
    for i in range(0, original_shape[0] - patch_size + 1, patch_size):
        for j in range(0, original_shape[1] - patch_size + 1, patch_size):
            img[i:i + patch_size, j:j + patch_size] = patches[idx]
            idx += 1
    return img

def prepare_training_and_testing_data(filenames, map_rgb2cat, labels_path, train=True, block_size=20, overlap=0):
    train_data = []
    train_labels = []
    indexes_data = []
    for filename in filenames:
        basename = filename.split('_label.tif')[0]
        label = load_tif_image(os.path.join(labels_path, filename))
        irrg = load_tif_image(os.path.join('data/Potsdam/3_Ortho_IRRG/', f'{basename}_IRRG.tif'))
        dsm_basename = 'dsm' + basename[3:]
        dsm_basename = re.sub(r'_(\d)(?=(_|\b))', r'_0\1', dsm_basename)
        print(dsm_basename)
        dsm_height = load_tif_image(os.path.join('data/Potsdam/1_DSM_normalisation/', f'{dsm_basename}_normalized_lastools.jpg'))
        # mag_fourier = np.load(os.path.join('data/Potsdam/RGB_Fourier/', f'magnitude_{basename}_RGBIR.npy'))
        # phase_fourier = np.load(os.path.join('data/Potsdam/RGB_Fourier/', f'phase_{basename}_RGBIR.npy'))
        ndvi = np.load(os.path.join('data/Potsdam/NDVI/', f'ndvi_normalized_{basename}_RGBIR.npy'))
        canny = np.load(os.path.join('data/Potsdam/Edges/', f'canny_{basename}_RGBIR.npy'))
        
        label = label.transpose(1, 2, 0)
        irrg = irrg.transpose(1, 2, 0)
        
        # Normalize images
        irrg = (irrg - np.min(irrg)) / (np.max(irrg) - np.min(irrg))
        dsm_height = (dsm_height - np.min(dsm_height)) / (np.max(dsm_height) - np.min(dsm_height))
        # mag_fourier = (mag_fourier - np.min(mag_fourier)) / (np.max(mag_fourier) - np.min(mag_fourier))
        # phase_fourier = (phase_fourier - np.min(phase_fourier)) / (np.max(phase_fourier) - np.min(phase_fourier))
        canny = (canny - np.min(canny)) / (np.max(canny) - np.min(canny))
        ndvi = (ndvi - np.min(ndvi)) / (np.max(ndvi) - np.min(ndvi))
        
        label = rgb_to_categories(label, map_rgb2cat)
        # index_matrix = np.indices(label.shape[:2]).transpose(1, 2, 0)
        index_matrix = create_block_index_matrix(label.shape[:2], block_size)
        print(label.shape)
        print(irrg.shape)
        print(dsm_height.shape)
        print(index_matrix.shape)
        print(ndvi.shape)
        
        if block_size > 1:
            if train:
                label_patches = extract_training_patches(label, block_size, overlap)
                irrg_patches = extract_training_patches(irrg, block_size, overlap)
                dsm_height_patches = np.expand_dims(extract_training_patches(dsm_height, block_size, overlap), axis=-1)
                index_matrix_patches = extract_training_patches(index_matrix, block_size, overlap)
                canny_patches = np.expand_dims(extract_training_patches(canny, block_size, overlap), axis=-1)
                ndvi_patches = np.expand_dims(extract_training_patches(ndvi, block_size, overlap), axis=-1)
                print("Train")
                print(label_patches.shape)
                print(irrg_patches.shape)
                print(dsm_height_patches.shape)
                print(index_matrix_patches.shape)
                print(canny_patches.shape)
                print(ndvi_patches.shape)
            else:
                label_patches = extract_testing_patches(label, block_size)
                irrg_patches = extract_testing_patches(irrg, block_size)
                dsm_height_patches = np.expand_dims(extract_testing_patches(dsm_height, block_size), axis=-1)
                index_matrix_patches = extract_testing_patches(index_matrix, block_size)
                canny_patches = np.expand_dims(extract_testing_patches(canny, block_size), axis=-1)
                ndvi_patches = np.expand_dims(extract_testing_patches(ndvi, block_size), axis=-1)
                print("Test")
                print(label_patches.shape)
                print(irrg_patches.shape)
                print(dsm_height_patches.shape)
                print(index_matrix_patches.shape)
                print(canny_patches.shape)
                print(ndvi_patches.shape)
        else:
            label = label.reshape(-1)
            irrg = irrg.reshape(-1, irrg.shape[-1])
            dsm_height = np.expand_dims(dsm_height.reshape(-1), axis=-1)
            index_matrix = index_matrix.reshape(-1)
            canny = np.expand_dims(canny.reshape(-1), axis=-1)
            ndvi = np.expand_dims(ndvi.reshape(-1), axis=-1)
            return np.concatenate([irrg, dsm_height, ndvi, canny], axis=-1), label, index_matrix
            
        indexes_data.append(index_matrix_patches)
        train_data.append(np.concatenate([irrg_patches, dsm_height_patches, ndvi_patches, canny_patches], axis=-1))
        train_labels.append(label_patches)
    return np.concatenate(train_data, axis=0), np.concatenate(train_labels, axis=0), np.concatenate(indexes_data, axis=0)

def calculate_iou_segmentation(pred_mask, true_mask, class_id):
    """
    Calcula o Intersect over Union (IoU) para segmentação semântica.
    
    Args:
        pred_mask (numpy.ndarray): Máscara predita (array 2D com valores binários ou de classes).
        true_mask (numpy.ndarray): Máscara verdadeira (array 2D com valores binários ou de classes).
        
    Returns:
        float: Valor do IoU entre as máscaras.
    """
    # Cria máscaras binárias apenas para a classe de interesse
    pred_binary = (pred_mask == class_id)
    true_binary = (true_mask == class_id)
    
    # Calcula interseção e união
    intersection = np.logical_and(pred_binary, true_binary).sum()
    union = np.logical_or(pred_binary, true_binary).sum()
    
    # Evita divisão por zero
    iou = intersection / union if union > 0 else 0.0
    return iou

def calculate_miou(pred_mask, true_mask, num_classes):
    ious = []
    for class_id in range(num_classes):
        iou = calculate_iou_segmentation(pred_mask, true_mask, class_id)
        ious.append(iou)
    return np.mean(ious)
