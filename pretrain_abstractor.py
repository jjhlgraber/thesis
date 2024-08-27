import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os

from simple_abstractor import SimpleAbstractorEncoder
from set_models import (
    SetCNNEmbedder,
    SetSequenceModel,
    SetClassifierLayer,
    SetVectorEmbedder,
)
from set_data_lit import SetTriplesDataModule
from set_data import SetCardBaseDataset


torch.set_float32_matmul_precision("medium")
pl.seed_everything(42, workers=True)

debug = False
save = False

use_official_cards = True
vectors_as_features = False


# parameters for both models and data
seq_len = 3
features_used = [0, 1, 2, 3]
feature_states_used = [0, 1, 2]

if len(feature_states_used) > 3:
    balanced_subset = True
else:
    balanced_subset = False


if use_official_cards:
    data_dir = "data"
    val_split = 0.01
    test_split = 0.01
else:
    data_dir = "data/custom_cards"
    val_split = 0.005
    test_split = 0.005
if vectors_as_features:
    val_split = 0.01
    test_split = 0.01


n_features = len(features_used)
label_choice = "is_set"

# prepare models
embedder_kwargs = dict()
abstractor_kwargs = {
    "num_layers": 1,
    "norm": None,  # Example normalization layer
    "use_pos_embedding": False,
    "use_learned_symbols": True,
    "learn_symbol_per_position": True,
    "use_symbolic_attention": True,
    "object_dim": 64,
    "symbol_dim": 64,  # Using a different symbol dimension
    "num_heads": 4,
    "ff_dim": 128,
    "dropout": 0.1,
    "norm_att": False,
    "norm_ff": False,
    "resid_att": False,
    "resid_ff": False,
    "MHA_kwargs": {
        "use_bias": False,
        "activation": nn.Identity(),  # Different activation function
        # "activation": nn.Softmax(-1),
        # "activation": nn.Sigmoid(),
        # "activation": sparsemax,
        "use_scaling": True,
        "shared_kv_proj": True,
    },
}

if vectors_as_features:
    card_embedder = SetVectorEmbedder(
        n_features=n_features, embed_dim=abstractor_kwargs["object_dim"]
    )
else:
    card_embedder = SetCNNEmbedder()

card_embedder = torch.load("./cnn_checkpoints/cnn.pt")

# abstractor_kwargs["use_pos_embedding"] = True
# abstractor_kwargs["use_learned_symbols"] = False
# abstractor_kwargs["learn_symbol_per_position"] = False
# abstractor_kwargs["use_symbolic_attention"] = False
AE1 = SimpleAbstractorEncoder(**abstractor_kwargs)
# abstractor_kwargs["object_dim"] = abstractor_kwargs["symbol_dim"]
# abstractor_kwargs["use_pos_embedding"] = False
# abstractor_kwargs["use_learned_symbols"] = False
# abstractor_kwargs["learn_symbol_per_position"] = False
# abstractor_kwargs["use_symbolic_attention"] = True

# AE2 = SimpleAbstractorEncoder(**abstractor_kwargs)
abstractor = nn.Sequential(
    AE1,
    #    AE2
)

# classifier = seq_model_PT.final_layer
seq_model = SetSequenceModel(
    base_embedder=card_embedder,
    # base_embedder=None,
    contextual_embedder=abstractor,
    seq_len=seq_len,
    # seq_len_final_layer=1,
    seq_len_final_layer=3,
    label_choice=label_choice,
    n_features=n_features,
    # aggregate_seq=MeanPoolModule(),
    aggregate_seq=nn.Flatten(),
    # classifier=classifier,
)

# prepare data
ds = SetCardBaseDataset(
    # image_embedder=cnn,
    # features_used=features_used,
    feature_states_used=feature_states_used,
    data_dir=data_dir,
    use_official_cards=use_official_cards,
)
if vectors_as_features:
    ds.set_card_representations_type("feature_vectors")
dm = SetTriplesDataModule(
    ds,
    batch_size=64,
    label_choice=label_choice,
    balanced_subset=balanced_subset,
    balanced_sampling=True,  # Enable balanced sampling
    balance_positions=True,
    val_split=val_split,
    test_split=test_split,
)
dm.setup()

name = "e2e_PT"
if vectors_as_features:
    name += "_vec"
elif use_official_cards:
    name += "_off"
else:
    name += "_custom"


if feature_states_used:
    name += "_"
    for f in feature_states_used:
        name += str(f)


# training
if not debug:
    logger = WandbLogger(
        project="first_project",
        name=name,
    )
else:
    logger = False

name += "posbal"

trainer_kwargs = dict(
    max_epochs=5,
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
        save_dir = "checkpoints"
        save_path = os.path.join(save_dir, name + ".pth")
        torch.save(seq_model, save_path)

    logger.experiment.finish()
