import os
import torch
from torch.utils.data import Dataset
from matplotlib import image as mpimg
import matplotlib.pyplot as plt
from itertools import product
import pickle


class SetCardBaseDataset(Dataset):
    def __init__(
        self,
        image_embedder=None,
        data_dir: str = "./data",
        features_used: list = [0, 1, 2, 3],
        feature_states_used: list = [0, 1, 2],
        use_official_cards: bool = True,
    ):
        self.index_to_features = ["number", "color", "pattern", "shape"]
        self.features_to_index = {"number": 0, "color": 1, "pattern": 2, "shape": 3}
        features_used.sort()
        self.features_used = features_used
        self.n_features = len(features_used)
        if self.n_features < 1:
            raise ValueError("features_used should be non empty")

        self.features = ["number", "color", "pattern", "shape"]

        # self.colors = ["red", "green", "purple"]
        # self.patterns = ["empty", "striped", "solid"]
        # self.shapes = ["diamond", "oval", "squiggle"]
        # self.numbers = ["one", "two", "three"]

        self.n_image_channels = 4
        self.n_states_per_features = len(feature_states_used)
        (self.height, self.width) = (70, 50)

        self.n_states_per_features = len(feature_states_used)
        self.feature_states_used = feature_states_used
        # TODO purpose still relevant?
        # if len(self.feature_states_used) == 2:
        #     self.use_only_pair_feature_states()

        self.use_official_cards = use_official_cards
        if self.use_official_cards:
            filename = "all-cards.png"
            file_path = os.path.join(data_dir, filename)
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Image file not found: {file_path}")

            self.n_cards = 81
            self.n_rows_cols = 9
            (self.leftmargin, self.topmargin) = (4, 8)
            (self.vspace, self.hspace) = (1, 0)

            self.card_images = self.get_official_images_of_cards(file_path)
            self.card_feature_vectors = self.vector_representation()

        else:
            assert self.features_used == [
                0,
                1,
                2,
                3,
            ], "use all feature for the custom cards"
            self.card_feature_vectors = torch.LongTensor(
                list(product(self.feature_states_used, repeat=4))
            )[:, self.features_used]
            self.n_cards = self.card_feature_vectors.size(0)

            load_tensor_file_path = os.path.join(
                data_dir,
                "custom_setcards.pt",
            )
            load_dict_file_path = os.path.join(
                data_dir,
                "features_to_index.pkl",
            )
            with open(load_dict_file_path, "rb") as f:
                features_to_index = pickle.load(f)
            card_indices = [
                features_to_index[tuple(card)]
                for card in self.card_feature_vectors.tolist()
            ]

            self.card_images = torch.load(load_tensor_file_path)[card_indices]

        if image_embedder:
            self.precompute_embeddings(image_embedder)
            self.set_card_representations_type("embeddings")
        else:
            self.card_embeds = None
            self.set_card_representations_type("images")

    def __len__(self):
        return len(self.card_feature_vectors)

    def __getitem__(self, idx):
        card = self.card_representations[idx]
        card_features = self.card_feature_vectors[idx]
        return card, card_features

    def use_only_pair_feature_states(self):
        i, j = self.feature_states_used
        indices = torch.logical_or(
            self.card_feature_vectors == i, self.card_feature_vectors == j
        ).all(dim=-1)
        self.n_cards = indices.sum().item()
        self.card_feature_vectors = self.card_feature_vectors[indices]
        self.card_images = self.card_images[indices]

    def precompute_embeddings(self, image_embedder):
        with torch.no_grad():
            self.card_embeds = image_embedder(self.card_images)

    def set_card_representations_type(self, card_representations_type):
        if card_representations_type == "images":
            self.card_representations_type = "images"
            self.card_representations = self.card_images
        elif card_representations_type == "embeddings":
            self.card_representations_type = "embeddings"
            if self.card_embeds is None:
                raise AssertionError(
                    "Can not set card_representations_type to 'embeddings' if the embeddings are not precomputed. Try self.precompute_embeddings(image_embedder)."
                )
            self.card_representations = self.card_embeds
        elif card_representations_type == "feature_vectors":
            self.card_representations_type = "feature_vectors"
            self.card_representations = self.card_feature_vectors.float()
        else:
            raise AssertionError(
                "card_representations_type has to be either 'embeddings' or 'images'."
            )

    def get_official_images_of_cards(self, file_path):
        im = mpimg.imread(file_path)
        cards = torch.from_numpy(im).permute(2, 1, 0)
        card_images = torch.empty(
            self.n_cards, self.n_image_channels, self.height, self.width
        )

        shift_to_bottomright = torch.Tensor([self.height, self.width]).int()
        for i in range(9):
            for j in range(9):
                topleft = torch.Tensor(
                    [
                        self.leftmargin + i * (self.height + self.hspace),
                        self.topmargin + j * (self.width + self.vspace),
                    ]
                ).int()

                bottomright = topleft + shift_to_bottomright
                card_image = cards[
                    :, topleft[0] : bottomright[0], topleft[1] : bottomright[1]
                ]
                card_images[self.grid_to_index(i, j)] = card_image

        return card_images

    def vector_representation(self):
        """Pre-computes representations for all cards."""
        # find dimensions
        card_feature_vectors = (
            torch.arange(self.n_cards)
            .repeat(self.n_features)
            .view(self.n_features, self.n_cards)
        )

        for i, feature in enumerate(self.features_used):
            if feature == 0 or feature == 3:
                card_feature_vectors[i] = card_feature_vectors[i] // self.n_rows_cols
            else:
                card_feature_vectors[i] = card_feature_vectors[i] % self.n_rows_cols
            if feature % 2 == 0:
                card_feature_vectors[i] = card_feature_vectors[i] % 3
            else:
                card_feature_vectors[i] = card_feature_vectors[i] // 3
        return card_feature_vectors.T

    def index_to_grid(self, idx):
        row = idx // self.n_rows_cols
        col = idx % self.n_rows_cols
        return row, col

    def grid_to_index(self, row, col):
        return row * self.n_rows_cols + col

    def plot_card(self, card_index):
        fig = plt.figure(figsize=(3, 2))
        card = self.card_images[card_index].permute(1, 2, 0)
        plt.imshow(card)
        plt.show()


class SetTriplesDataset(Dataset):
    def __init__(
        self,
        setcard_dataset,
        label_choice="is_set",
        balanced_subset=False,  # can pass int for max number of triples in dataset
        balance_positions=False,
        max_size_subset=None,
    ):
        super().__init__()
        self.setcard_dataset = setcard_dataset
        self.label_choice = label_choice
        self.balanced_subset = balanced_subset
        self.balance_positions = balance_positions
        self.max_size_subset = max_size_subset

        self.seq_len = 3  # triples

        triples_imbalanced = torch.combinations(
            torch.arange(self.setcard_dataset.n_cards), self.seq_len
        )
        if self.balance_positions:
            perm_indices = torch.argsort(
                torch.rand(
                    triples_imbalanced.shape,
                    generator=torch.Generator().manual_seed(42),
                ),
                dim=-1,
            )
            self.triples = torch.gather(triples_imbalanced, dim=-1, index=perm_indices)
        else:
            self.triples = triples_imbalanced

        if self.balanced_subset:
            is_set_labels = self.triple_is_set(self.triples)
            sets_indices = torch.where(is_set_labels)[0]
            if isinstance(self.balanced_subset, int):
                perm = torch.randperm(
                    sets_indices.size(0), generator=torch.Generator().manual_seed(42)
                )
                sets_indices = sets_indices[perm[: self.balanced_subset // 2]]
            nonsets_indices = torch.where(~is_set_labels)[0]
            perm = torch.randperm(
                nonsets_indices.size(0), generator=torch.Generator().manual_seed(42)
            )
            idx = perm[: sets_indices.size(0)]
            nonsets_indices_subset = nonsets_indices[idx]

            indices_subset = torch.cat((sets_indices, nonsets_indices_subset))
            self.triples = self.triples[indices_subset]

        self.label_functions = {
            "features": self.card_features,
            # "features_pointwise": self.card_features,
            "pairwise_sim": self.pairwise_similarity_flat,
            "triple_sim": self.all_features_similar,
            "triple_dissim": self.all_features_dissimilar,
            "triple_valid_features": self.feature_wise_validity,
            "triple_multiclass": self.generate_multiclass_labels,
            "is_set": self.triple_is_set,
        }

        self.set_labels(self.label_choice)

        self.triples_hidden_states = None
        self.use_hidden_states = "init done"
        self._get_cards = self.get_from_setcard_dataset

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        cards = self._get_cards(idx)
        labels = self.labels[idx]
        return cards, labels

    def set_get_cards(self, use_hidden_states: bool):
        self.use_hidden_states = use_hidden_states
        if self.use_hidden_states:
            self._get_cards = self.get_hidden_states
        else:
            self._get_cards = self.get_from_setcard_dataset

    def get_from_setcard_dataset(self, idx):
        triples_indices = self.triples[idx]
        cards = self.setcard_dataset.card_representations[triples_indices]
        return cards

    def get_hidden_states(self, idx):
        cards = self.triples_hidden_states[idx]
        return cards

    def precompute_abstractor_hidden_states(
        self, module, initial_S=None, batch_size=128
    ):
        module.eval()
        with torch.no_grad():
            all_hidden_states = []
            for i in range(0, len(self.triples_hidden_states), batch_size):
                batch_hidden_states = self.triples_hidden_states[i : i + batch_size]
                batch_hidden_states = module(batch_hidden_states)
                all_hidden_states.append(batch_hidden_states)
            self.triples_hidden_states = torch.cat(all_hidden_states, dim=0)

    def set_labels(self, label_choice):
        self.label_choice = label_choice
        if self.label_choice not in self.label_functions:
            raise ValueError(
                f"Invalid value for self.label_choice. Choose from {list(self.label_functions.keys())}"
            )
        label_function = self.label_functions[self.label_choice]
        self.labels = label_function(self.triples).long()

    def card_features(self, triple_indices):
        features_seq = self.setcard_dataset.card_feature_vectors[triple_indices].view(
            -1, self.setcard_dataset.n_features * self.seq_len
        )
        return features_seq

    def pairwise_similarity(self, triple_indices):
        card1 = self.setcard_dataset.card_feature_vectors[triple_indices[:, 0]]
        card2 = self.setcard_dataset.card_feature_vectors[triple_indices[:, 1]]
        card3 = self.setcard_dataset.card_feature_vectors[triple_indices[:, 2]]
        sim_12 = torch.eq(card1, card2)
        sim_13 = torch.eq(card1, card3)
        sim_23 = torch.eq(card2, card3)
        similarities_pairwise = torch.stack((sim_12, sim_13, sim_23), dim=-2)
        return similarities_pairwise

    def pairwise_similarity_flat(self, triple_indices):
        similarities_pairwise_flat = self.pairwise_similarity(triple_indices).view(
            -1, self.setcard_dataset.n_features * self.seq_len
        )
        return similarities_pairwise_flat

    def all_features_similar(self, triple_indices):
        similarities_pairwise = self.pairwise_similarity(triple_indices)
        all_similar = torch.all(similarities_pairwise, dim=-2)
        return all_similar

    def all_features_dissimilar(self, triple_indices):
        similarities = self.pairwise_similarity(triple_indices)
        all_dissimilar = torch.all(~similarities, dim=-2)
        return all_dissimilar

    def feature_wise_validity(self, triple_indices):
        similarities = self.pairwise_similarity(triple_indices)
        return torch.logical_or(
            torch.all(similarities, dim=-2), torch.all(~similarities, dim=-2)
        )

    def triple_is_set(self, triple_indices):
        features_validity = self.feature_wise_validity(triple_indices)
        is_set = torch.all(features_validity, dim=-1)
        return is_set

    def generate_multiclass_labels(self, triple_indices):
        all_sim = self.all_features_similar(triple_indices)
        all_dissim = self.all_features_dissimilar(triple_indices)
        labels = torch.zeros(triple_indices.shape[0], self.setcard_dataset.n_features)
        labels[all_sim] = 1
        labels[all_dissim] = 2
        return labels

    def plot_triple(self, triple_indices):
        fig, axarr = plt.subplots(1, 3, figsize=(3, 2))
        label = ["A", "B", "C"]
        cards = self.setcard_dataset.card_images[triple_indices].permute(0, 2, 3, 1)
        for i, c in enumerate(cards):
            axarr[i].imshow(c)
            axarr[i].axis("off")
            axarr[i].set_title("%s" % label[i])
        fig.tight_layout()
        plt.show()


# deprecated!!!
class SetPairsDataset(Dataset):
    def __init__(self, setcard_dataset, label_choice="is_set"):
        super().__init__()
        self.setcard_dataset = setcard_dataset
        self.label_choice = label_choice

        self.seq_len = 2  # pairs

        self.pairs = torch.combinations(
            torch.arange(self.setcard_dataset.n_cards), self.seq_len
        )

        label_functions = {
            "features": lambda: self.card_features(self.pairs),
            "pairwise_sim": lambda: self.pairwise_similarity(self.pairs),
        }

        # Get the label function based on the choice, defaulting to an error
        label_function = label_functions.get(
            self.label_choice, ValueError("Invalid value for self.label_choice")
        )

        self.labels = label_function().long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pair_indices = self.pairs[idx]
        cards = self.setcard_dataset.card_representations[pair_indices]
        labels = self.labels[idx]
        return cards, labels

    def card_features(self, pair_indices):
        labels = self.setcard_dataset.card_feature_vectors[pair_indices].view(
            -1, self.setcard_dataset.n_features * self.seq_len
        )
        return labels

    def pairwise_similarity(self, pair_indices):
        card1 = self.setcard_dataset.card_feature_vectors[pair_indices[:, 0]]
        card2 = self.setcard_dataset.card_feature_vectors[pair_indices[:, 1]]
        sim_12 = torch.eq(card1, card2)
        return sim_12

    # def pairwise_similarity_flat(self, pair_indices):
    #     similarities_pairwise_flat = self.pairwise_similarity(pair_indices).view(
    #         -1, self.setcard_dataset.n_features * self.seq_len
    #     )
    #     return similarities_pairwise_flat
