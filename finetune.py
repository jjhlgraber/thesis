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

PT_name = "e2e_PT_vec_012"


debug = False
save = False

use_official_cards = True
vectors_as_features = False

reset_CNN = True
reset_ABS1 = False
reset_ABS2 = False
reset_FL = False

freeze_ABS1 = False
freeze_ABS2 = True
freeze_FL = True


# parameters for both models and data
seq_len = 3
features_used = [0, 1, 2, 3]
feature_states_used = [0, 1, 2]

if len(feature_states_used) > 3:
    balanced_subset = True
else:
    balanced_subset = False

max_size_subset = 250
# max_size_subset = None


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

if max_size_subset:
    balanced_subset = True
    val_check_interval = 1
    val_split = 0.1
    test_split = 0.1
    max_epochs = 500
else:
    val_check_interval = 25
    max_epochs = 2

n_features = len(features_used)
label_choice = "is_set"

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

load_path = os.path.join("checkpoints", PT_name + ".pth")
seq_model = torch.load(load_path)


if reset_CNN:
    seq_model.base_embedder = SetCNNEmbedder()
if vectors_as_features:
    seq_model.base_embedder = SetVectorEmbedder(
        n_features=n_features, embed_dim=abstractor_kwargs["object_dim"]
    )
if reset_ABS1:
    abstractor_kwargs["use_pos_embedding"] = True
    abstractor_kwargs["use_learned_symbols"] = False
    abstractor_kwargs["learn_symbol_per_position"] = False
    abstractor_kwargs["use_symbolic_attention"] = False
    seq_model.contextual_embedder[0] = SimpleAbstractorEncoder(**abstractor_kwargs)
if reset_ABS2:
    abstractor_kwargs["object_dim"] = abstractor_kwargs["symbol_dim"]
    abstractor_kwargs["use_pos_embedding"] = False
    abstractor_kwargs["use_learned_symbols"] = False
    abstractor_kwargs["learn_symbol_per_position"] = False
    abstractor_kwargs["use_symbolic_attention"] = True
    seq_model.contextual_embedder[1] = SimpleAbstractorEncoder(**abstractor_kwargs)
if reset_FL:
    seq_model.final_layer = SetClassifierLayer(
        label_choice=seq_model.label_choice,
        embed_dim=seq_model.embed_dim,
        # seq_len=self.seq_len,
        seq_len=seq_model.seq_len_final_layer,
        n_features=seq_model.n_features,
        n_feature_states=seq_model.n_feature_states,
        hidden_sizes=seq_model.hidden_sizes_classifier,
    )

if freeze_ABS1:
    seq_model.contextual_embedder[0].requires_grad_(False)
if freeze_ABS2:
    seq_model.contextual_embedder[1].requires_grad_(False)
if freeze_FL:
    seq_model.final_layer.requires_grad_(False)

seq_model.__init__(
    base_embedder=seq_model.base_embedder,
    contextual_embedder=seq_model.contextual_embedder,
    aggregate_seq=seq_model.aggregate_seq,
    classifier=seq_model.final_layer,
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
    max_size_subset=max_size_subset,
    balanced_sampling=True,  # Enable balanced sampling
    val_split=val_split,
    test_split=test_split,
)
dm.setup()

name = "e2e_fine"
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

name += "_RS_"
if reset_CNN:
    name += "CNN"
if reset_ABS1:
    name += "A1"
if reset_ABS2:
    name += "A2"
if reset_FL:
    name += "FL"

name += "_FR_"
if freeze_ABS1:
    name += "A1"
if freeze_ABS2:
    name += "A2"
if freeze_FL:
    name += "FL"

if max_size_subset:
    name += "_" + str(max_size_subset)


name += PT_name[6:]

# training
if not debug:
    logger = WandbLogger(
        project="first_project",
        name=name,
        # name="NONPTcnn_asym_imbal_0123",
    )
else:
    logger = False

trainer_kwargs = dict(
    max_epochs=max_epochs,
    precision="16",
    logger=logger,
    callbacks=[
        EarlyStopping(monitor="val_loss", mode="min", patience=15),
    ],
    val_check_interval=val_check_interval,
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
