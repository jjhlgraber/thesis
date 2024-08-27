import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from pytorch_lightning import LightningDataModule

from set_data import (
    SetCardBaseDataset,
    SetTriplesDataset,
    SetPairsDataset,
    # SetTriplesPrecomputedDataset,
)

from collections import Counter


class SetCardDataModule(LightningDataModule):
    def __init__(
        self,
        setcard_dataset=None,
        batch_size: int = 32,
        num_workers: int = 4,
        image_embedder=None,
        data_dir: str = "./data",
        val_split: float = 0.2,
        test_split: float = 0.1,
    ):
        super().__init__()
        self.setcard_dataset = "features"
        self.setcard_dataset = setcard_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_embedder = image_embedder
        self.data_dir = data_dir
        self.val_split = val_split
        self.test_split = test_split
        self.train_split = 1 - (self.val_split + self.test_split)

        self.train_sampler = None
        self.val_sampler = None
        self.test_sampler = None

        self._assure_existence_setcard_dataset()

    def setup(self, stage=None):
        self._set_split_datasets(self.setcard_dataset)

    def _assure_existence_setcard_dataset(self):
        if self.setcard_dataset is None:
            self.setcard_dataset = SetCardBaseDataset(
                image_embedder=self.image_embedder,
                data_dir=self.data_dir,
            )

    def _set_split_datasets(self, dataset):
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [self.train_split, self.val_split, self.test_split]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            shuffle=self.train_sampler is None,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            shuffle=False,
            num_workers=self.num_workers,
        )


class SetTriplesDataModule(SetCardDataModule):
    def __init__(
        self,
        setcard_dataset=None,
        batch_size: int = 32,
        num_workers: int = 4,
        image_embedder=None,
        data_dir: str = "./data",
        val_split: float = 0.2,
        test_split: float = 0.1,
        label_choice="is_set",
        balanced_sampling: bool = False,
        balanced_subset: int = False,
        balance_positions: bool = False,
    ):
        super().__init__(
            setcard_dataset=setcard_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            image_embedder=image_embedder,
            data_dir=data_dir,
            val_split=val_split,
            test_split=test_split,
        )
        self.label_choice = label_choice
        self.balanced_sampling = balanced_sampling
        self.balanced_subset = balanced_subset
        self.balance_positions = balance_positions

        self._assure_existence_setcard_dataset()

        self.triples_dataset = SetTriplesDataset(
            self.setcard_dataset,
            label_choice=self.label_choice,
            balanced_subset=self.balanced_subset,
            balance_positions=self.balance_positions,
        )

    def setup(self, stage=None):
        self._set_split_datasets(self.triples_dataset)

        if self.balanced_sampling:
            if self.train_split:
                self.train_sampler = self._get_balanced_sampler(self.train_dataset)
            if self.val_split:
                self.val_sampler = self._get_balanced_sampler(self.val_dataset)
            if self.test_split:
                self.test_sampler = self._get_balanced_sampler(self.test_dataset)

    def set_labels_dm(self, label_choice):
        self.label_choice = label_choice
        self.triples_dataset.set_labels(self.label_choice)

    def _get_balanced_sampler(self, dataset):
        labels = dataset.dataset.labels[dataset.indices]

        if len(labels.shape) == 1:
            label_tuples = [label.tolist() for label in labels]
        else:
            label_tuples = [tuple(label.tolist()) for label in labels]
        class_counts = Counter(label_tuples)

        weights = {label: 1.0 / count for label, count in class_counts.items()}
        sample_weights = [weights[label_tuple] for label_tuple in label_tuples]

        num_samples = len(dataset.indices)

        return WeightedRandomSampler(sample_weights, num_samples, replacement=True)


class SetPairsDataModule(SetCardDataModule):
    def __init__(
        self,
        setcard_dataset=None,
        batch_size: int = 32,
        num_workers: int = 4,
        image_embedder=None,
        data_dir: str = "./data",
        val_split: float = 0.2,
        test_split: float = 0.1,
        label_choice="pairwise_sim",
        balanced_sampling: bool = False,
    ):
        super().__init__(
            setcard_dataset=setcard_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            image_embedder=image_embedder,
            data_dir=data_dir,
            val_split=val_split,
            test_split=test_split,
        )
        self.label_choice = label_choice
        self.balanced_sampling = balanced_sampling

    def setup(self, stage=None):
        self._assure_existence_setcard_dataset()

        self.triples_dataset = SetPairsDataset(
            self.setcard_dataset, label_choice=self.label_choice
        )

        self._set_split_datasets(self.triples_dataset)

        if self.balanced_sampling:
            if self.train_split:
                self.train_sampler = self._get_balanced_sampler(self.train_dataset)
            if self.val_split:
                self.val_sampler = self._get_balanced_sampler(self.val_dataset)
            if self.test_split:
                self.test_sampler = self._get_balanced_sampler(self.test_dataset)

    def _get_balanced_sampler(self, dataset):
        """Creates a WeightedRandomSampler for balanced multilabel sampling."""
        labels = dataset.dataset.labels[dataset.indices]

        # Convert multi-hot encoding to tuples for counting
        if len(labels.shape) == 1:
            label_tuples = [label.tolist() for label in labels]
        else:
            label_tuples = [tuple(label.tolist()) for label in labels]
        class_counts = Counter(label_tuples)

        # Calculate weights based on inverse frequency
        weights = {label: 1.0 / count for label, count in class_counts.items()}
        sample_weights = [weights[label_tuple] for label_tuple in label_tuples]

        num_samples = len(dataset.indices)

        return WeightedRandomSampler(sample_weights, num_samples, replacement=True)
