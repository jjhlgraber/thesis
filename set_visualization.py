import torch

from setGame import SetGame
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding

from collections import defaultdict

import os


ATTRIBUTE_NAMES = ["number", "color", "pattern", "shape"]


def compute_dim_reduct(embeddings, n_components=2):

    # 1. PCA
    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(embeddings)

    # 2. t-SNE
    tsne = TSNE(n_components=n_components, perplexity=30, random_state=42)
    tsne_embeddings = tsne.fit_transform(embeddings)

    # 3. Isomap
    isomap = Isomap(n_components=n_components)
    isomap_embeddings = isomap.fit_transform(embeddings)

    # 4. Locally Linear Embedding (LLE)
    lle = LocallyLinearEmbedding(n_components=n_components, random_state=42)
    lle_embeddings = lle.fit_transform(embeddings)

    reduction_methods = [
        ("PCA", pca_embeddings),
        ("t-SNE", tsne_embeddings),
        ("Isomap", isomap_embeddings),
        ("LLE", lle_embeddings),
    ]
    return reduction_methods


def compute_all_DR(test_data, symbol_seq, symbol_seq_flat):
    test_data_DR = defaultdict(lambda: defaultdict())
    symbol_seq_DR = defaultdict(lambda: defaultdict())
    symbol_seq_flat_DR = defaultdict()

    for n_components in [2, 3]:
        for seq_pos in [0, 1, 2]:
            test_data_DR[n_components][seq_pos] = compute_dim_reduct(
                test_data[:, seq_pos], n_components=n_components
            )
            symbol_seq_DR[n_components][seq_pos] = compute_dim_reduct(
                symbol_seq[:, seq_pos], n_components=n_components
            )
        symbol_seq_flat_DR[n_components] = compute_dim_reduct(
            symbol_seq_flat, n_components=n_components
        )

    return test_data_DR, symbol_seq_DR, symbol_seq_flat_DR


def plot_embed(reduction_methods, labels, dim):
    num_methods = len(reduction_methods)

    if dim == 2:
        fig, axes = plt.subplots(
            nrows=1, ncols=num_methods, figsize=(5 * num_methods, 5)
        )
    elif dim == 3:
        fig, axes = plt.subplots(
            nrows=1,
            ncols=num_methods,
            figsize=(5 * num_methods, 5),
            subplot_kw=dict(projection="3d"),
        )
    else:
        raise ValueError("Embeddings must have 2 or 3 dimensions for plotting.")

    # Handle single plot case
    if num_methods == 1:
        axes = [axes]

    # Iterate over the reduction methods and plot the embeddings
    for i, (method_name, embeddings) in enumerate(reduction_methods):
        ax = axes[i]

        # Get unique labels and assign a color to each label
        unique_labels = torch.unique(labels).numpy()
        colors = plt.get_cmap("Accent", len(unique_labels))

        # Plot each label separately
        for label in unique_labels:
            indices = labels == label
            if dim == 2:
                ax.scatter(
                    embeddings[indices, 0],
                    embeddings[indices, 1],
                    c=colors(label),
                    label=label,
                    alpha=0.7,
                )
            elif dim == 3:
                ax.scatter(
                    embeddings[indices, 0],
                    embeddings[indices, 1],
                    embeddings[indices, 2],
                    c=colors(label),
                    label=label,
                    alpha=0.7,
                )

        ax.set_title(method_name)
        ax.legend()

    plt.tight_layout()
    plt.show()


def save_tensors_to_dict(data_dict, save_path):
    """Saves tensors within a nested dictionary to disk.

    Args:
      data_dict: A nested dictionary containing tensors.
      save_path: The directory to save the tensors.
    """
    os.makedirs(save_path, exist_ok=True)
    for key1, value1 in data_dict.items():
        if isinstance(value1, dict):
            for key2, tensor_value in value1.items():
                torch.save(tensor_value, os.path.join(save_path, f"{key1}_{key2}.pt"))
        else:
            torch.save(value1, os.path.join(save_path, f"{key1}.pt"))


def load_tensors_to_dict(load_path):
    """Loads tensors from disk into a nested dictionary.

    Args:
      load_path: The directory containing the saved tensors.

    Returns:
      A nested dictionary containing the loaded tensors.
    """
    loaded_data = defaultdict(lambda: defaultdict())
    for filename in os.listdir(load_path):
        if filename.endswith(".pt"):
            keys = filename[:-3].split("_")
            if len(keys) == 2:
                key1, key2 = keys
                loaded_data[int(key1)][int(key2)] = torch.load(
                    os.path.join(load_path, filename)
                )
            else:
                key = keys[0]
                loaded_data[int(key)] = torch.load(os.path.join(load_path, filename))
    return loaded_data
