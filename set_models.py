import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


class SetCNNEmbedder(nn.Module):
    def __init__(self, n_input_channel=4, hidden_dim=128, embed_dim=64):
        super().__init__()
        self.n_input_channel = n_input_channel
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(self.n_input_channel, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=4, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.fc1 = nn.Linear(32, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.embed_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.amax(dim=(-1, -2))
        x = torch.flatten(x, -1)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


class SetClassifierLayer(nn.Module):
    def __init__(
        self,
        label_choice,
        embed_dim,
        seq_len=3,
        n_features=4,
        n_feature_states=3,
        hidden_sizes=[],
        # pointwise=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.label_choice = label_choice
        self.seq_len = seq_len
        self.n_features = n_features
        self.n_feature_states = n_feature_states
        self.hidden_sizes = hidden_sizes
        # self.pointwise = pointwise
        # if self.pointwise:
        #     self.original_seq_len = self.original_seq_len
        #     self.seq_len = 1

        seq_len_for_pairwise = self.seq_len if self.seq_len > 2 else 1

        self.get_final_layer_functions = {
            "features": dict(
                n_labels=self.seq_len * self.n_features, n_classes=self.n_feature_states
            ),
            # "features_pointwise": dict(n_labels=self.n_features, n_classes=self.n_feature_states),
            "pairwise_sim": dict(
                n_labels=seq_len_for_pairwise * self.n_features, n_classes=2
            ),
            "triple_sim": dict(n_labels=self.n_features, n_classes=2),
            "triple_dissim": dict(n_labels=self.n_features, n_classes=2),
            "triple_valid_features": dict(n_labels=self.n_features, n_classes=2),
            "triple_multiclass": dict(
                n_labels=self.n_features, n_classes=self.n_feature_states
            ),
            "is_set": dict(n_labels=1, n_classes=2),
        }

        try:
            self.n_labels = self.get_final_layer_functions[self.label_choice][
                "n_labels"
            ]
            self.n_classes = self.get_final_layer_functions[self.label_choice][
                "n_classes"
            ]
        except KeyError:
            raise ValueError(
                f"Invalid label_choice: {self.label_choice}. "
                f"Valid choices are: {list(self.get_final_layer_functions.keys())}"
            )

        self._set_classifier()

    def forward(self, x):
        logits = self.classifier(x)
        return logits

    def _set_classifier(
        self,
    ):
        # binary classification with n labels
        input_size = self.seq_len * self.embed_dim
        if self.n_classes < 3:
            output_size = self.n_labels
            self._set_layers(input_size, output_size)
            self.classifier = self.binary_classifier
            self.criterion_fn = nn.BCEWithLogitsLoss()
            self.criterion = self.binary_criterion
            self.accuracy = torchmetrics.Accuracy("binary")
        # rest
        else:
            output_size = self.n_labels * self.n_classes
            self._set_layers(input_size, output_size)
            self.classifier = self.multiclass_classifier
            self.criterion_fn = nn.CrossEntropyLoss()
            self.criterion = self.multiclass_criterion
            self.accuracy_fn = torchmetrics.Accuracy(
                "multiclass", num_classes=self.n_classes
            )
            self.accuracy = self.multiclass_accuracy

    def _set_layers(self, input_size, output_size):
        self.layers = nn.Sequential()
        for hidden_size in self.hidden_sizes:
            self.layers.append(nn.Linear(input_size, hidden_size))
            self.layers.append(nn.ReLU())
            input_size = hidden_size
        self.layers.append(nn.Linear(input_size, output_size))

    def binary_classifier(self, x):
        logits = self.layers(x).squeeze(-1)
        return logits

    def multiclass_classifier(self, x):
        logits = self.layers(x).view(-1, self.n_labels, self.n_classes)
        # if self.label_choice == "features_pointwise":
        #     logits = self.layers(x).view(
        #         -1, self.original_seq_len * self.n_labels, self.n_classes
        #     )
        # else:
        #     logits = self.layers(x).view(-1, self.n_labels, self.n_classes)

        logits = self.layers(x).view(-1, self.n_labels, self.n_classes)
        return logits

    def binary_criterion(self, logits, labels):
        loss = self.criterion_fn(logits, labels.float())
        return loss

    def multiclass_criterion(self, logits, labels):
        loss = self.criterion_fn(logits.view(-1, self.n_classes), labels.view(-1))
        return loss

    def multiclass_accuracy(self, logits, labels):
        accuracy = self.accuracy_fn(logits.argmax(-1), labels)
        return accuracy


class SetSequenceModel(pl.LightningModule):
    def __init__(
        self,
        base_embedder=None,
        contextual_embedder=None,
        aggregate_seq=nn.Flatten(),
        classifier=None,
        seq_len=3,
        seq_len_final_layer=3,
        label_choice="is_set",
        n_features=4,
        n_feature_states=3,
        hidden_sizes_classifier=[],
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "base_embedder",
                "contextual_embedder",
                "aggregate_seq",
                "classifier",
            ]
        )  # Save init parameters
        self.seq_len = seq_len
        self.seq_len_final_layer = seq_len_final_layer
        self.label_choice = label_choice
        self.n_features = n_features
        self.n_feature_states = n_feature_states
        self.hidden_sizes_classifier = hidden_sizes_classifier
        self.lr = lr

        self.base_embedder = base_embedder
        self.contextual_embedder = contextual_embedder
        self.embedder = nn.Sequential()
        if self.base_embedder:
            self.embedder.append(nn.Flatten(start_dim=0, end_dim=1))
            self.embedder.append(self.base_embedder)
            self.embedder.append(
                nn.Unflatten(dim=0, unflattened_size=(-1, self.seq_len))
            )
            self.embed_dim = self.base_embedder.embed_dim
        if self.contextual_embedder:
            if isinstance(self.contextual_embedder, nn.Sequential):
                self.embedder = nn.Sequential(*self.embedder, *self.contextual_embedder)
                self.embed_dim = self.contextual_embedder[-1].symbol_dim
            else:
                self.embedder.append(self.contextual_embedder)
                self.embed_dim = self.contextual_embedder.symbol_dim

        self.aggregate_seq = aggregate_seq

        if classifier:
            self.final_layer = classifier
            self.label_choice = classifier.label_choice
            self.embed_dim = classifier.embed_dim
            self.seq_len = classifier.seq_len
            self.n_features = classifier.n_features
            self.n_feature_states = classifier.n_feature_states
            self.hidden_sizes_classifier = classifier.hidden_sizes
        else:
            self.final_layer = SetClassifierLayer(
                label_choice=self.label_choice,
                embed_dim=self.embed_dim,
                # seq_len=self.seq_len,
                seq_len=self.seq_len_final_layer,
                n_features=self.n_features,
                n_feature_states=self.n_feature_states,
                hidden_sizes=self.hidden_sizes_classifier,
            )
        self.criterion = self.final_layer.criterion
        self.accuracy = self.final_layer.accuracy

    def forward(self, sequence):
        # TODO fix in the datamodules instead!
        if self.seq_len == 1:
            sequence = sequence.unsqueeze(1)
        if self.embedder:
            sequence = self.embedder(sequence)
        x = self.aggregate_seq(sequence)
        logits = self.final_layer(x)
        return logits

    def training_step(self, batch, batch_idx):
        input_seq, labels = batch
        logits = self(input_seq)
        loss = self.criterion(logits, labels)
        acc = self.accuracy(logits, labels)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        input_seq, labels = batch
        logits = self(input_seq)
        loss = self.criterion(logits, labels)
        acc = self.accuracy(logits, labels)
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        input_seq, labels = batch
        logits = self(input_seq)
        loss = self.criterion(logits, labels)
        acc = self.accuracy(logits, labels)
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class MaxPoolModule(nn.Module):
    def __init__(self, pool_dim=-2):
        super(MaxPoolModule, self).__init__()
        self.pool_dim = pool_dim

    def forward(self, x):
        return x.max(dim=self.pool_dim).values


class MeanPoolModule(nn.Module):
    def __init__(self, pool_dim=-2):
        super(MeanPoolModule, self).__init__()
        self.pool_dim = pool_dim

    def forward(self, x):
        return x.mean(dim=self.pool_dim)


class FirstPoolModule(nn.Module):
    def __init__(self):
        super(FirstPoolModule, self).__init__()

    def forward(self, x):
        return x[:, 0]


class SetVectorEmbedder(nn.Module):
    def __init__(self, n_features=4, hidden_sizes=[], embed_dim=64):
        super().__init__()
        self.n_features = n_features
        self.hidden_sizes = hidden_sizes
        self.embed_dim = embed_dim

        self.layers = nn.Sequential()
        input_size = self.n_features
        for hidden_size in self.hidden_sizes:
            self.layers.append(nn.Linear(input_size, hidden_size))
            self.layers.append(nn.ReLU())
            input_size = hidden_size
        self.layers.append(nn.Linear(input_size, self.embed_dim))

    def forward(self, x):
        embed = self.layers(x)
        return embed
