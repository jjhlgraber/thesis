import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ltn
import itertools
from itertools import product
import networkx as nx
import random
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider

from simple_abstractor import SimpleAbstractorEncoder


class AbstractorOrderModel(nn.Module):
    def __init__(self, input_dim, abstractor_kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = abstractor_kwargs["object_dim"]
        hidden_size = 32
        self.relu = nn.ReLU()
        self.dense1 = torch.nn.Linear(self.input_dim, hidden_size)
        self.dense2 = torch.nn.Linear(hidden_size, hidden_size)
        self.dense3 = torch.nn.Linear(hidden_size, self.embedding_dim)
        self.embedder = nn.Sequential(
            self.dense1, self.relu, self.dense2, self.relu, self.dense3
        )

        self.abstractor_encoder = SimpleAbstractorEncoder(**abstractor_kwargs)
        flatten = False
        if flatten:
            self.aggregate = nn.Flatten()
            output_dim = self.abstractor_encoder.symbol_dim * 2  # sequence_length
        output_dim = self.abstractor_encoder.symbol_dim
        self.hidden_dense = nn.Linear(in_features=output_dim, out_features=32)

        self.final_layer = nn.Linear(in_features=32, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        x = self.embedder(x)
        y = self.embedder(y)
        x = torch.cat([x, y], dim=-1).view(-1, 2, self.embedding_dim)
        x = self.abstractor_encoder(x)
        # print(x.shape)
        # print(self.flatten(x).shape)
        # print(torch.mean(x, axis=1).shape)
        # assert False
        x = torch.mean(x, axis=1)
        x = self.hidden_dense(x)
        x = self.relu(x)
        x = self.final_layer(x)
        x = self.sigmoid(x)
        return x


# Relation R
class BaselineRelationalModel(torch.nn.Module):
    def __init__(
        self, num_objects, object_dim, hidden_size=32, final_size=8, learn_embed=False
    ):
        super(BaselineRelationalModel, self).__init__()
        self.elu = torch.nn.ELU()
        self.sigmoid = torch.nn.Sigmoid()

        # self.embed = torch.nn.Embedding(
        #     num_embeddings=num_objects, embedding_dim=object_dim
        # )
        # torch.nn.init.normal_(self.embed.weight, mean=0.0, std=1.0)
        # self.embed.weight.requires_grad = learn_embed

        self.dense1_x = torch.nn.Linear(object_dim, hidden_size)
        self.dense2_x = torch.nn.Linear(hidden_size, hidden_size)
        self.dense3_x = torch.nn.Linear(hidden_size, final_size)

        self.dense1_y = torch.nn.Linear(object_dim, hidden_size)
        self.dense2_y = torch.nn.Linear(hidden_size, hidden_size)
        self.dense3_y = torch.nn.Linear(hidden_size, final_size)

    def forward(self, x, y):
        # x = self.embed.weight[x]
        x = self.elu(self.dense1_x(x))
        x = self.elu(self.dense2_x(x))
        x = self.elu(self.dense3_x(x))

        # y = self.embed.weight[y]
        y = self.elu(self.dense1_y(y))
        y = self.elu(self.dense2_y(y))
        y = self.elu(self.dense3_y(y))
        out = self.sigmoid(torch.sum(torch.multiply(x, y), dim=-1))
        return out


# deprecated
class BaselineRelationalIndependentEmbedModel(nn.Module):
    def __init__(self, num_objects):
        super(BaselineRelationalIndependentEmbedModel, self).__init__()
        self.logits = nn.Parameter(
            data=torch.zeros(num_objects, num_objects), requires_grad=True
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        logit = self.logits[x, y]
        out = self.sigmoid(logit)
        return out


class BaselineRelationalIndependentModel(nn.Module):
    def __init__(self, num_objects):
        super(BaselineRelationalIndependentModel, self).__init__()
        self.num_objects = num_objects
        self.sigmoid = nn.Sigmoid()
        self.embedding_dim = num_objects**2
        self.lin = torch.nn.Linear(self.embedding_dim, 1, bias=False)

    def forward(self, x, y):
        onehot_pair = torch.zeros(x.shape[0], self.embedding_dim)
        flat_indices = (self.num_objects * x + y).squeeze()
        onehot_pair[torch.arange(x.shape[0]), flat_indices] = 1

        logit = self.lin(onehot_pair)

        out = self.sigmoid(logit)
        return out


# Relation R
class BaselineRelationalModelConcat(torch.nn.Module):
    def __init__(
        self, num_objects, object_dim, hidden_size=16, final_size=1, learn_embed=False
    ):
        super(BaselineRelationalModelConcat, self).__init__()
        self.elu = torch.nn.ELU()
        self.sigmoid = torch.nn.Sigmoid()

        seq_len = 2
        self.object_dim_concat = seq_len * object_dim

        # self.embed = torch.nn.Embedding(
        #     num_embeddings=num_objects, embedding_dim=object_dim
        # )
        # torch.nn.init.normal_(self.embed.weight, mean=0.0, std=1.0)
        # self.embed.weight.requires_grad = learn_embed
        # self.lin = torch.nn.Linear(2 * object_dim, 1)

        self.dense1 = torch.nn.Linear(self.object_dim_concat, hidden_size)
        self.dense2 = torch.nn.Linear(hidden_size, hidden_size)
        self.dense3 = torch.nn.Linear(hidden_size, final_size)

        # self.dense1_y = torch.nn.Linear(object_dim, hidden_size)
        # self.dense2_y = torch.nn.Linear(hidden_size, hidden_size)
        # self.dense3_y = torch.nn.Linear(hidden_size, final_size)

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=-1)

        xy = self.elu(self.dense1(xy))
        xy = self.elu(self.dense2(xy))
        xy = self.elu(self.dense3(xy))
        out = self.sigmoid(xy)

        return out


def plot_mp(heatmap_data, pos_examples=[], neg_examples=[]):
    if pos_examples is None:
        pos_examples = []
    if neg_examples is None:
        neg_examples = []

    def plot_heatmap(frame_index):
        data = heatmap_data[frame_index]
        plt.imshow(data, cmap="PuBu", vmin=0, vmax=1)
        plt.colorbar()
        plt.title(f"Heatmap Frame {frame_index}")

        # Mark specific grids (if provided)
        for row, col in pos_examples:
            plt.gca().add_patch(
                plt.Rectangle(
                    (col - 0.475, row - 0.475),
                    0.95,
                    0.95,
                    edgecolor="yellow",
                    fill=False,
                    lw=1,
                )
            )

        for row, col in neg_examples:
            plt.gca().add_patch(
                plt.Rectangle(
                    (col - 0.475, row - 0.475),
                    0.95,
                    0.95,
                    edgecolor="red",
                    fill=False,
                    lw=1,
                )
            )

        plt.show()

    num_frames = len(heatmap_data)
    slider = IntSlider(min=0, max=num_frames - 1, step=1, description="Frame")
    interact(plot_heatmap, frame_index=slider)


def adjacency_anti_transitive(n):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    while True:
        added_edge = False
        adjacency = nx.adjacency_matrix(G).toarray()
        sorted_non_edges = sorted(nx.non_edges(G), key=lambda k: random.random())
        for u, v in sorted_non_edges:
            if G.has_edge(v, u):
                continue
            if (
                not np.logical_and(adjacency[u], adjacency[:, v]).any()
                and not np.logical_and(adjacency[u], adjacency[v]).any()
                and not np.logical_and(adjacency[:, u], adjacency[:, v]).any()
            ):
                G.add_edge(u, v)
                added_edge = True
                break
        if not added_edge:
            break

    adjacency = nx.adjacency_matrix(G).toarray()
    adjacency = torch.tensor(adjacency)
    return adjacency


def adjacency_triangular_lattice(n, periodic=True):
    lattice = nx.triangular_lattice_graph(n // 2, n // 2, periodic=periodic)
    adjacency = nx.adjacency_matrix(lattice).toarray()[np.arange(n), :][:, np.arange(n)]
    adjacency += np.eye(adjacency.shape[0]).astype(int)

    adjacency = torch.tensor(adjacency)
    return adjacency


def adjacency_lattice(n, dim, periodic=True, distance=2):
    div = int(n ** (1.0 / dim))
    # rest = n % dim
    grid_dims = dim * [div]
    # grid_dims[0] += rest
    grid_dims

    lattice = nx.grid_graph(grid_dims, periodic=periodic)
    for u, v in lattice.edges():
        for i in range(1, distance + 1):
            if i != 1:  # Avoid duplicate edges
                neighbors = [
                    (u[0] + i, u[1]),
                    (u[0] - i, u[1]),
                    (u[0], u[1] + i),
                    (u[0], u[1] - i),
                ]
                for neighbor in neighbors:
                    if neighbor in lattice.nodes:
                        lattice.add_edge(u, neighbor)

    adjacency = nx.adjacency_matrix(lattice).toarray()
    adjacency += np.eye(adjacency.shape[0]).astype(int)

    adjacency = torch.tensor(adjacency)
    return adjacency


def adjacency_total_order(n):

    adjacency = torch.triu(torch.ones(n, n))

    return adjacency


def adjacency_from_string(N=1000, string="O"):
    # Make a plot with "HELLO" text; save as PNG
    fig, ax = plt.subplots(figsize=(4, 1))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis("off")
    ax.text(0.5, 0.5, string, va="center", ha="center", size=85)
    fig.savefig("hello.png")
    plt.close(fig)

    # Open this PNG and draw random points from it
    from matplotlib.image import imread

    data = imread("hello.png")[::-1, :, 0].T
    mask = data < 1
    indices = np.random.choice(mask.sum(), N, replace=False)
    X = np.argwhere(mask)[indices].astype(float)

    X[:, 0] /= data.shape[0]
    X[:, 1] /= data.shape[0]
    X = X[np.argsort(X[:, 0])]
    X = X[np.argsort(X[:, 1])]

    pos = {node: (i, j) for node, (i, j) in enumerate(X)}
    G = nx.random_geometric_graph(X.shape[0], 0.1, pos=pos)
    adjacency = nx.adjacency_matrix(G).toarray()
    adjacency += np.eye(adjacency.shape[0]).astype(int)
    adjacency = torch.tensor(adjacency)
    return adjacency


def get_pairs_adjacency(adjacency):
    pos_pairs = torch.argwhere(adjacency == 1)
    neg_pairs = torch.argwhere(adjacency == 0)
    return pos_pairs, neg_pairs


def get_samples_adjacency(pos_pairs, neg_pairs, train_split_pos, train_split_neg):
    pos_n_train_points = int(pos_pairs.size(0) * train_split_pos)
    neg_n_train_points = int(neg_pairs.size(0) * train_split_neg)

    pos_sample_indices = np.random.choice(
        len(pos_pairs), pos_n_train_points, replace=False
    )
    neg_sample_indices = np.random.choice(
        len(neg_pairs), neg_n_train_points, replace=False
    )

    pos_sample = pos_pairs[pos_sample_indices]
    neg_sample = neg_pairs[neg_sample_indices]

    return pos_sample, neg_sample


def get_constants(num_objects, object_dim, feature_type="onehot"):
    if feature_type == "gaussian":
        learn_embed = False
        embed = torch.nn.Embedding(num_embeddings=num_objects, embedding_dim=object_dim)
        torch.nn.init.normal_(embed.weight, mean=0.0, std=1.0)
        embed.weight.requires_grad = learn_embed
        constants_tensor = embed.weight
    elif feature_type == "index":
        constants_tensor = torch.arange(num_objects).unsqueeze(-1)
    else:
        assert num_objects == object_dim
        constants_tensor = torch.eye(num_objects)

    constants = [
        ltn.Constant(features, trainable=False) for features in constants_tensor
    ]

    return constants, constants_tensor


def get_pairs_total_order(num_objects):
    object_pairs = torch.Tensor(
        list(itertools.permutations(range(num_objects), r=2))
    ).int()
    object_order_relations = object_pairs[:, 0] < object_pairs[:, 1]

    pos_pairs = object_pairs[object_order_relations]
    neg_pairs = object_pairs[object_order_relations == False]

    return pos_pairs, neg_pairs


def get_samples_total_order(
    num_objects, pos_pairs, neg_pairs, pos_chain_depth, train_split_pos, train_split_neg
):

    if pos_chain_depth:
        pos_sample = []
        for j in range(pos_chain_depth):
            pos_sample += [[i, i + j + 1] for i in range(num_objects - (j + 1))]

        pos_sample = torch.tensor(pos_sample)

        indices_pos_chain = torch.zeros(len(pos_pairs))

        for pair in pos_sample:
            indices_pos_chain += (pos_pairs == pair).all(-1).int()

        indices_pos_chain = ~indices_pos_chain.bool()

        pos_pairs_to_sample_from = pos_pairs[indices_pos_chain]
    else:
        pos_sample = torch.tensor([]).int()
        pos_pairs_to_sample_from = pos_pairs

    neg_n_train_points = int(neg_pairs.size(0) * train_split_neg)
    pos_n_train_points = int(neg_pairs.size(0) * train_split_pos) - len(pos_sample)
    if pos_n_train_points < 0:
        pos_n_train_points = 0

    pos_sample_indices = np.random.choice(
        len(pos_pairs_to_sample_from), pos_n_train_points, replace=False
    )
    neg_sample_indices = np.random.choice(
        len(neg_pairs), neg_n_train_points, replace=False
    )

    pos_sample = torch.cat((pos_sample, pos_pairs_to_sample_from[pos_sample_indices]))
    neg_sample = neg_pairs[neg_sample_indices]
    if pos_sample.shape[0] == 0:
        pos_sample = None
    if neg_sample.shape[0] == 0:
        neg_sample = None

    return pos_sample, neg_sample


def get_eye_3D(num_objects):
    triples = product(range(num_objects), repeat=3)
    triples_with_id = [triple for triple in triples if len(set(triple)) < 3]

    eye_3D = torch.ones(num_objects, num_objects, num_objects)
    for i, j, k in triples_with_id:
        eye_3D[i, j, k] = 0
    eye_3D = eye_3D.bool()
    return eye_3D


def get_sat(heatmap, ground_truth, mask=None, p=1):
    if mask is None:
        mask = torch.ones_like(ground_truth).bool()
    xs = 1 - torch.abs(ground_truth - heatmap)
    xs = torch.pow(1.0 - xs, p)
    numerator = torch.sum(torch.where(~mask, torch.zeros_like(xs), xs))
    denominator = torch.sum(mask)
    sat = (1.0 - torch.pow(torch.div(numerator, denominator), 1 / p)).item()
    return sat
