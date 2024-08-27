import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from simple_abstractor import SimpleAbstractorEncoder
from set_models import (
    SetCNNEmbedder,
    SetSequenceModel,
    MaxPoolModule,
    FirstPoolModule,
    MeanPoolModule,
)
from set_data_lit import SetTriplesDataModule
from set_data import SetCardBaseDataset

debug = False
save = True

torch.set_float32_matmul_precision("medium")
pl.seed_everything(42, workers=True)

# parameters for both models and data
seq_len = 3
features_used = [0, 1, 2, 3]
feature_states_used = [0, 1, 2, 3]
n_feature_states = len(feature_states_used)

if n_feature_states > 3:
    balanced_subset = True
else:
    balanced_subset = False

use_official_cards = False
if use_official_cards:
    data_dir = "data"
    val_split = 0.01
    test_split = 0.01
else:
    data_dir = "data/custom_cards"
    val_split = 0.005
    test_split = 0.005


n_features = len(features_used)
label_choice = "features"

# prepare models
embedder_kwargs = dict()
abstractor_kwargs = {
    "num_layers": 2,
    "norm": None,  # Example normalization layer
    "use_pos_embedding": False,
    "use_learned_symbols": False,
    "learn_symbol_per_position": False,
    "use_symbolic_attention": True,
    "object_dim": 64,
    "symbol_dim": 32,  # Using a different symbol dimension
    "num_heads": 4,
    "ff_dim": 128,
    "dropout": 0.1,
    "MHA_kwargs": {
        "use_bias": False,
        "activation": nn.Identity(),  # Different activation function
        # "activation": nn.Softmax(-1),
        # "activation": nn.Sigmoid(),
        # "activation": sparsemax,
        "use_scaling": True,
        "shared_kv_proj": False,
    },
}

# seq_model_PT = torch.load("e2e_long_seq.pth")

cnn = SetCNNEmbedder()
# cnn = torch.load("./cnn_checkpoints/cnn.pt")


# classifier = seq_model_PT.final_layer
seq_model = SetSequenceModel(
    base_embedder=cnn,
    # base_embedder=None,
    contextual_embedder=None,
    seq_len=seq_len,
    # seq_len_final_layer=1,
    seq_len_final_layer=3,
    label_choice=label_choice,
    n_features=n_features,
    n_feature_states=n_feature_states,
    # aggregate_seq=MeanPoolModule(),
    aggregate_seq=nn.Flatten(),
    # classifier=classifier,
)

# prepare data
ds = SetCardBaseDataset(
    # image_embedder=cnn,
    features_used=features_used,
    feature_states_used=feature_states_used,
    data_dir=data_dir,
    use_official_cards=use_official_cards,
)
dm = SetTriplesDataModule(
    ds,
    batch_size=64,
    label_choice=label_choice,
    balanced_subset=balanced_subset,
    balanced_sampling=True,  # Enable balanced sampling
    val_split=val_split,
    test_split=test_split,
)
dm.setup()


# training
if not debug:
    logger = WandbLogger(
        project="first_project",
        name="PT_custom_0123",
        # name="NONPTcnn_asym_imbal_0123",
    )
else:
    logger = False

trainer_kwargs = dict(
    max_epochs=1,
    precision="16",
    logger=logger,
    callbacks=[
        EarlyStopping(monitor="val_loss", mode="min", patience=15),
    ],
    val_check_interval=25,
    deterministic=True,
)

trainer = pl.Trainer(**trainer_kwargs)
trainer.validate(seq_model, dm)
if not debug:
    trainer.fit(seq_model, dm)
    trainer.test(seq_model, dm)

    if save:
        torch.save(seq_model.base_embedder, "./cnn_checkpoints/cnn_custom_0123.pth")

    logger.experiment.finish()
