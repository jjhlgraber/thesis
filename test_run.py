import torch
import torch.nn as nn
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torchmetrics

from simple_abstractor import SimpleAbstractorEncoderLayer
from set_models import SetCNNEmbedder, SetClassifierLayer, SetSequenceModel
from set_data_lit import SetCardDataModule, SetTriplesDataModule
from set_data import SetCardBaseDataset, SetTriplesDataset
import os

abstractor_kwargs = dict(
    use_bias=False,
    activation=nn.Identity(),
    use_scaling=True,
    shared_kv_proj=True,
    embed_dim=64,
    num_heads=4,
    ff_dim=128,
)

label_choice = "features"

ds = SetCardBaseDataset()
dm = SetTriplesDataModule(ds, label_choice=label_choice)
dm.setup()
dm = SetCardDataModule(ds)
dm.setup()

cnn = SetCNNEmbedder()
AE = SimpleAbstractorEncoderLayer(**abstractor_kwargs)


seq_model = SetSequenceModel(
    base_embedder=cnn, contextual_embedder=AE, seq_len=1, label_choice=label_choice
)

logger = WandbLogger(project="first_project", name="test_snellius")

trainer_kwargs = dict(
    max_epochs=2,
    logger=logger,
    #   callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
    # val_check_interval=25,
    check_val_every_n_epoch=1,
)

trainer = pl.Trainer(**trainer_kwargs)
trainer.fit(seq_model, dm)

logger.experiment.finish()
