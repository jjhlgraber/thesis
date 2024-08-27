import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, fixed

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
        self.flatten = nn.Flatten()
        output_dim = self.abstractor_encoder.symbol_dim * 2  # sequence_length
        self.hidden_dense = nn.Linear(in_features=output_dim, out_features=32)

        self.final_layer = nn.Linear(in_features=32, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        x = self.embedder(x)
        y = self.embedder(y)
        x = torch.cat([x, y], dim=-1).view(-1, 2, self.embedding_dim)
        x = self.abstractor_encoder(x)
        x = self.flatten(x)
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
