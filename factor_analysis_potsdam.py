import numpy as np
import os
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from osgeo import gdal
from scipy.stats import mode
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import load_tif_image, extract_training_patches, extract_testing_patches,\
    rgb_to_categories, plot_classes_histogram, prepare_training_and_testing_data
import argparse
import matplotlib.patches as mpatches
import time
import seaborn as sns
from matplotlib.colors import ListedColormap

def perform_factor_analysis(data, n_components=None, rotation='varimax'):
    """
    Perform factor analysis on the given data.
    
    Args:
        data: Input data matrix (samples x features)
        n_components: Number of factors to extract (None for automatic selection)
        rotation: Rotation method ('varimax', 'quartimax', 'promax', etc.)
        
    Returns:
        transformed_data: Data transformed into factor space
        model: Fitted factor analysis model
    """
    print(f"Performing Factor Analysis with {n_components if n_components else 'auto'} components...")
    
    # Create and fit the factor analysis model
    model = FactorAnalysis(n_components=n_components, 
                          rotation=rotation,
                          random_state=42)
    
    # Fit the model and transform data
    start_time = time.time()
    transformed_data = model.fit_transform(data)
    end_time = time.time()
    
    print(f"Factor Analysis completed in {end_time - start_time:.2f} seconds")
    
    # Calculate variance explained
    components = model.components_
    variance = np.sum(components**2, axis=1)
    total_variance = np.sum(variance)
    explained_variance_ratio = variance / total_variance
    
    # Print variance explained
    cumulative_variance = np.cumsum(explained_variance_ratio)
    print(f"Number of factors extracted: {model.components_.shape[0]}")
    print(f"Total variance explained: {cumulative_variance[-1]:.4f}")
    
    for i, var in enumerate(explained_variance_ratio):
        print(f"Factor {i+1}: {var:.4f} of variance ({cumulative_variance[i]:.4f} cumulative)")
    
    return transformed_data, model

def visualize_factor_loadings(model, feature_names=None, output_folder=None):
    """Visualize factor loadings with detailed heatmap and individual factor plots"""
    components = model.components_
    n_factors = components.shape[0]
    n_features = components.shape[1]
    
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(n_features)]
    
    # 1. Create a heatmap of all factor loadings
    plt.figure(figsize=(12, max(8, n_factors * 0.5)))
    sns.heatmap(components, annot=True, fmt='.2f', cmap='coolwarm', 
                xticklabels=feature_names, yticklabels=[f"Factor {i+1}" for i in range(n_factors)])
    plt.title('Factor Loadings Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'factor_loadings_heatmap.png'), dpi=300)
    plt.close()
    
    # 2. Bar plots for each factor
    plt.figure(figsize=(14, n_factors * 3))
    for i in range(n_factors):
        plt.subplot(n_factors, 1, i+1)
        bars = plt.bar(feature_names, components[i], color=plt.cm.viridis((i+1)/n_factors))
        
        # Highlight significant loadings (abs value > 0.3)
        threshold = 0.3
        for j, v in enumerate(components[i]):
            if abs(v) > threshold:
                bars[j].set_color('red' if v > 0 else 'blue')
                plt.annotate(f'{v:.2f}', xy=(j, v), ha='center', va='bottom' if v > 0 else 'top',
                           fontweight='bold')
                
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=-threshold, color='blue', linestyle='--', alpha=0.5)
        plt.title(f"Factor {i+1} Loadings")
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Loading Value')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'factor_loadings_detail.png'), dpi=300)
    plt.close()
    
    # 3. Save the loading values to CSV
    loadings_df = pd.DataFrame(components, 
                               columns=feature_names,
                               index=[f"Factor {i+1}" for i in range(n_factors)])
    loadings_df.to_csv(os.path.join(output_folder, 'factor_loadings.csv'))
    
    return loadings_df

def visualize_latent_space(transformed_data, labels=None, label_names=None, output_folder=None):
    """Visualize the data in latent factor space"""
    # Pick the first three factors for visualization
    n_factors = min(3, transformed_data.shape[1])
    
    # Create scatterplots for the first few factors
    if n_factors >= 2:
        plt.figure(figsize=(10, 8))
        
        # If we have labels, color by class
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.get_cmap('tab10', len(unique_labels))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(transformed_data[mask, 0], transformed_data[mask, 1], 
                           c=[colors(i)], label=label_names[i] if label_names else f"Class {label}",
                           alpha=0.7)
            plt.legend()
        else:
            plt.scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.7)
            
        plt.xlabel('Factor 1')
        plt.ylabel('Factor 2')
        plt.title('Data Distribution in Factor Space (Factors 1 vs 2)')
        plt.grid(linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_folder, 'factor_space_2d.png'), dpi=300)
        plt.close()
    
    # 3D visualization if we have at least 3 factors
    if n_factors >= 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.get_cmap('tab10', len(unique_labels))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(transformed_data[mask, 0], transformed_data[mask, 1], transformed_data[mask, 2],
                          c=[colors(i)], label=label_names[i] if label_names else f"Class {label}",
                          alpha=0.7)
            ax.legend()
        else:
            ax.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], alpha=0.7)
            
        ax.set_xlabel('Factor 1')
        ax.set_ylabel('Factor 2')
        ax.set_zlabel('Factor 3')
        ax.set_title('Data Distribution in Factor Space (Factors 1-3)')
        plt.savefig(os.path.join(output_folder, 'factor_space_3d.png'), dpi=300)
        plt.close()
    
    # Create a correlation matrix between factors
    factor_df = pd.DataFrame(transformed_data, columns=[f"Factor {i+1}" for i in range(transformed_data.shape[1])])
    corr_matrix = factor_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Factors')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'factor_correlations.png'), dpi=300)
    plt.close()

def analyze_factor_scores(transformed_data, labels=None, label_names=None, output_folder=None):
    """Analyze the distribution of factor scores across classes"""
    if labels is None:
        return
    
    n_factors = transformed_data.shape[1]
    n_classes = len(np.unique(labels))
    
    # Calculate stats
    factor_stats = []
    for factor_idx in range(n_factors):
        class_stats = {}
        for class_idx in np.unique(labels):
            factor_scores = transformed_data[labels == class_idx, factor_idx]
            class_stats[class_idx] = {
                'mean': np.mean(factor_scores),
                'std': np.std(factor_scores),
                'min': np.min(factor_scores),
                'max': np.max(factor_scores)
            }
        factor_stats.append(class_stats)
    
    # Save stats to CSV
    stats_data = []
    for factor_idx, factor_stat in enumerate(factor_stats):
        for class_idx, stat in factor_stat.items():
            stats_data.append({
                'Factor': f"Factor {factor_idx+1}",
                'Class': label_names[class_idx] if label_names else f"Class {class_idx}",
                'Mean': stat['mean'],
                'StdDev': stat['std'],
                'Min': stat['min'],
                'Max': stat['max']
            })
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(os.path.join(output_folder, 'factor_score_stats.csv'), index=False)
    
    # Create boxplots for each factor by class
    for factor_idx in range(min(5, n_factors)):  # Limit to first 5 factors to avoid too many plots
        plt.figure(figsize=(10, 6))
        factor_data = []
        for class_idx in np.unique(labels):
            factor_scores = transformed_data[labels == class_idx, factor_idx]
            factor_data.append(factor_scores)
        
        plt.boxplot(factor_data, labels=[label_names[i] if label_names else f"Class {i}" 
                                       for i in np.unique(labels)])
        plt.title(f'Factor {factor_idx+1} Score Distribution by Class')
        plt.ylabel('Factor Score')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'factor_{factor_idx+1}_by_class.png'), dpi=300)
        plt.close()

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Factor Analysis on Potsdam dataset')
    parser.add_argument('--patch_size', type=int, default=1, help='Size of the patches')
    parser.add_argument('--labels_path', type=str, default='data/Potsdam/5_Labels_all', help='Path to labels')
    parser.add_argument('--n_factors', type=int, default=None, help='Number of factors to extract')
    parser.add_argument('--rotation', type=str, default='varimax', help='Rotation method for factor analysis')
    args = parser.parse_args()
    
    # Define data paths
    labels_path = args.labels_path
    filenames_label = ['top_potsdam_2_14_label.tif', 'top_potsdam_6_12_label.tif', 'top_potsdam_5_13_label.tif']
    
    # Define RGB to category mapping
    map_rgb2cat = {(255, 255, 255): 0, (0, 0, 255): 1, (0, 255, 255): 2, 
                  (0, 255, 0): 3, (255, 255, 0): 4, (255, 0, 0): 5}
    
    # Class names for visualization
    label_names = ['Impervious surfaces', 'Building', 'Low vegetation', 'Tree', 'Car', 'Background']
    
    # Load the data
    print("Loading data...")
    data, labels, indices = prepare_training_and_testing_data(
        filenames_label, map_rgb2cat, labels_path, block_size=args.patch_size)
    
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}, Indices shape: {indices.shape}")
    
    # Create output folder
    exp_folder = 'Patches' if args.patch_size > 1 else 'NoPatches'
    output_folder = f'data/factor_analysis_results/{exp_folder}/factors={args.n_factors or "auto"}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Reshape data if needed
    if args.patch_size > 1:
        # Reshape for patch-based analysis
        data = data.reshape(data.shape[0], -1)
        indices = indices.reshape(indices.shape[0], -1)
        
        # Mode for labels in patches
        labels = np.array(mode(labels.reshape(labels.shape[0], -1), axis=1).mode)
        indices = np.array(mode(indices.reshape(indices.shape[0], -1), axis=1).mode)
    else:
        # Flatten features for pixel-based analysis
        data = data.reshape(data.shape[0], -1)
        # The indices data is already flat for pixel-based analysis
    
    # Normalize indices to 0-1
    indices = indices / np.max(indices)
    
    # Remove background class if needed
    mask = labels != 5
    data = data[mask]
    indices = indices[mask]
    labels = labels[mask]
    
    # Concatenate indices with data
    if len(indices.shape) > 1:
        # For patch-based, indices might be multi-dimensional
        data = np.concatenate([data, indices], axis=1)
    else:
        # For pixel-based, indices might be 1D
        data = np.concatenate([data, indices.reshape(-1, 1)], axis=1)
    
    # Plot class distribution
    plot_classes_histogram(labels, label_names[:5], show=False)
    plt.savefig(os.path.join(output_folder, 'class_distribution.png'))
    
    # Extract feature names (can be customized based on actual features)
    feature_names = []
    # Add RGB channel names
    for channel in ['IR', 'R', 'G']:
        feature_names.append(channel)
    # Add DSM
    feature_names.append('DSM')
    # Add NDVI
    feature_names.append('NDVI')  
    # Add Canny edge
    feature_names.append('Canny')
    # Add Spatial Index
    feature_names.append('SpatialIdx')
    
    # Perform Factor Analysis
    transformed_data, fa_model = perform_factor_analysis(
        data, 
        n_components=args.n_factors,
        rotation=args.rotation
    )
    
    # Visualize and analyze factor loadings
    loadings_df = visualize_factor_loadings(fa_model, feature_names, output_folder)
    
    # Visualize data in latent space
    visualize_latent_space(transformed_data, labels, label_names, output_folder)
    
    # Analyze factor scores by class
    analyze_factor_scores(transformed_data, labels, label_names, output_folder)
    
    # Save transformed data and model info
    np.save(os.path.join(output_folder, 'factor_scores.npy'), transformed_data)
    np.save(os.path.join(output_folder, 'labels.npy'), labels)
    
    # Save factor loadings
    np.save(os.path.join(output_folder, 'factor_loadings.npy'), fa_model.components_)
    
    print(f"Factor analysis results saved to {output_folder}")
    print("Factor Loadings:")
    print(loadings_df.round(2))
