"""
ADMET prediction model (Chemprop-based MPNN)

This script trains a molecular property prediction model using
a Chemprop-based Message Passing Neural Network (MPNN).

Main components:
- Molecular graph representation from SMILES
- Message Passing Neural Network (MPNN)
- RDKit 2D descriptors as additional molecular features
- Murcko scaffold-based dataset splitting
- Focal loss for class imbalance handling
- MCC-based threshold optimization for classification evaluation

Dependencies:
- chemprop
- RDKit
- PyTorch
- PyTorch Lightning
- pandas
- scikit-learn
"""

import os

import torch
import torch.nn as tn
import torch.nn.functional as F
from torch import Tensor

import numpy as np
import pandas as pd
import random
import inspect

from torch import Tensor
from itertools import chain
from pathlib import Path

from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

from chemprop import data, featurizers, models, nn, utils
from chemprop.data import MoleculeDatapoint, make_split_indices, split_data_by_indices
from chemprop.nn.predictors import BinaryClassificationFFN
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.nn.metrics import BinaryAUROC, BCELoss
from chemprop.featurizers import MoleculeFeaturizerRegistry

print("GPU available", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("All packages loaded successfully.")

# load endpoints file
input_path = Path("path/to/endpoints.csv")
df_input = pd.read_csv(input_path)

num_workers = 6 # number of workers for dataloader. 0 means using main process for data loading
smiles_column = 'smiles' # name of the column containing SMILES strings
target_columns = ['label'] # list of names of the columns containing targets

# file check
smis = df_input.loc[:, smiles_column].values
ys = df_input.loc[:, target_columns].values


# Getting extra features and descriptors
mols = [utils.make_mol(smi, keep_h=False, add_h=False) for smi in smis]
for MoleculeFeaturizer in featurizers.MoleculeFeaturizerRegistry.keys():
    print(MoleculeFeaturizer)

featurizer = MoleculeFeaturizerRegistry["rdkit_2d"]()
extra_features = [featurizer(mol) for mol in mols]

df_input["X_d"] = extra_features

print(f"Total samples: {len(extra_features)}")
print(f"Valid feature vectors: {sum(x is not None for x in extra_features)}")
print(f"Feature vector length: {len([x for x in extra_features if x is not None][0])}")


# Get molecule datapoints
datapoints = [
    data.MoleculeDatapoint(mol, y, x_d=X_d)
    for mol, y, X_d in zip(
        mols,
        ys,
        extra_features
    )
]

# Dataset splitting (train / validation / test)
# Available Chemprop split types
list(data.SplitType.keys())
user_seed = 42

def get_murcko(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, isomericSmiles=False)
    except:
        return None

def chemprop_split(df_input, seed=user_seed):
    print("Fallback: using Chemprop's `make_split_indices()`")

    smis = df_input["smiles"].values
    ys = df_input["label"].values.reshape(-1, 1)
    X_d = df_input["X_d"].values if "X_d" in df_input.columns else [None] * len(df_input)
    mols = [utils.make_mol(smi, keep_h=False, add_h=False) for smi in smis]

    datapoints = [
        MoleculeDatapoint(
            mol=m,
            y=np.array([y], dtype=float),
            x_d=xd,
            name=s
        )
        for m, y, xd, s in zip(mols, ys, X_d, smis)
        if m is not None
    ]

    # Use chemprop default split to avoid scaffold matching failure
    split = make_split_indices(
        [dp.mol for dp in datapoints],
        split="SCAFFOLD_BALANCED",
        sizes=(0.8, 0.1, 0.1),
        seed=seed
    )
    train_idx, val_idx, test_idx = split

    train_data, val_data, test_data = split_data_by_indices(
        datapoints,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx
    )

    return train_data[0], val_data[0], test_data[0]


def try_murcko_split(df_input, seed=user_seed, max_deviation=0.03):
    df = df_input.copy()
    df["scaffold"] = df["smiles"].apply(get_murcko)

    # group molecules by scaffold
    scaffold_to_indices = {}
    for i, scaffold in enumerate(df["scaffold"]):
        if scaffold is None:
            continue
        scaffold_to_indices.setdefault(scaffold, []).append(i)

    # shuffle scaffold groups
    scaffold_sets = list(scaffold_to_indices.values())
    scaffold_sets.sort(key=len)
    random.seed(seed)
    random.shuffle(scaffold_sets)

    # target split sizes
    total = len(df)
    n_train = int(0.8 * total)
    n_val = int(0.1 * total)
    n_test = total - n_train - n_val

    train_idx, val_idx, test_idx = [], [], []

    for scaffold_set in scaffold_sets:
        if len(train_idx) + len(scaffold_set) <= n_train:
            train_idx.extend(scaffold_set)
        elif len(val_idx) + len(scaffold_set) <= n_val:
            val_idx.extend(scaffold_set)
        else:
            test_idx.extend(scaffold_set)

    # assign remaining scaffolds to the test set
    used = set(train_idx + val_idx + test_idx)
    remaining = set(chain.from_iterable(scaffold_sets)) - used
    test_idx.extend(remaining)

    # check split ratio
    r_train = len(train_idx) / total
    r_val = len(val_idx) / total
    r_test = len(test_idx) / total

    diffs = np.abs(np.array([r_train, r_val, r_test]) - np.array([0.8, 0.1, 0.1]))
    if diffs.max() > max_deviation:
        print(f"Murcko split failed: deviation too large (Train: {r_train:.3f}, Val: {r_val:.3f}, Test: {r_test:.3f})")
        return None

    print(f"Murcko split successful (Train: {r_train:.3f}, Val: {r_val:.3f}, Test: {r_test:.3f})")
    return (
        df.loc[train_idx].reset_index(drop=True),
        df.loc[val_idx].reset_index(drop=True),
        df.loc[test_idx].reset_index(drop=True),
    )

def scaffold_split_with_fallback(df_input, featurizer, seed=user_seed):
    result = try_murcko_split(df_input, seed=seed)
    
    if result is not None:
        train_df, val_df, test_df = result

        def df_to_datapoints(df, target_col="label"):
            datapoints = []
            for _, row in df.iterrows():
                smiles = row["smiles"]
                mol = Chem.MolFromSmiles(smiles)
                if mol is None or row["X_d"] is None:
                    continue

                dp = MoleculeDatapoint(
                    mol=mol,
                    y=np.array([row[target_col]], dtype=float),
                    x_d=row["X_d"],
                    name=smiles
                )
                datapoints.append(dp)
            return datapoints

        train_data = df_to_datapoints(train_df)
        val_data = df_to_datapoints(val_df)
        test_data = df_to_datapoints(test_df)

        return train_data, val_data, test_data

    else:
        return chemprop_split(df_input, seed=seed)

train_dp, val_dp, test_dp = scaffold_split_with_fallback(df_input, featurizer, seed=user_seed)

train_labels = np.array([dp.y[0] for dp in train_dp])
val_labels = np.array([dp.y[0] for dp in val_dp])
test_labels = np.array([dp.y[0] for dp in test_dp])

print(f"\n[Class Distribution Check]")
print(f"Train:      1 → {np.sum(train_labels == 1)}, 0 → {np.sum(train_labels == 0)}")
print(f"Validation: 1 → {np.sum(val_labels == 1)}, 0 → {np.sum(val_labels == 0)}")
print(f"Test:       1 → {np.sum(test_labels == 1)}, 0 → {np.sum(test_labels == 0)}")


# Descriptor scaling
train_X_d = np.array([dp.x_d for dp in train_dp if dp.x_d is not None])
scaler = MinMaxScaler()
scaler.fit(train_X_d)

for dp in train_dp:
    if dp.x_d is not None:
        dp.x_d = scaler.transform([dp.x_d])[0]
for dp in val_dp:
    if dp.x_d is not None:
        dp.x_d = scaler.transform([dp.x_d])[0]
for dp in test_dp:
    if dp.x_d is not None:
        dp.x_d = scaler.transform([dp.x_d])[0]

# Graph featurizer
featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
train_dset = data.MoleculeDataset(train_dp, featurizer)
val_dset = data.MoleculeDataset(val_dp, featurizer)
test_dset = data.MoleculeDataset(test_dp, featurizer)

X_d_transform = None
X_d_array = np.array(train_dset.X_d)


# Get DataLoader
# Featurize the train and val datasets to save computation time.
train_dset.cache = True
val_dset.cache = True

train_loader = data.build_dataloader(train_dset, batch_size=128, num_workers=num_workers)
val_loader = data.build_dataloader(val_dset, batch_size=128, num_workers=num_workers, shuffle=False)
test_loader = data.build_dataloader(test_dset, batch_size=128, num_workers=num_workers, shuffle=False)


### Set model ###
# All hyperparameters should be set your own values. (just example value)
# Hyperparameters should be tuned depending on the dataset

"""
## Message Passing
A `Message passing` constructs molecular graphs using message passing to learn node-level hidden representations.
Options are `mp = nn.BondMessagePassing()` or `mp = nn.AtomMessagePassing()`
"""

# print(inspect.getsource(nn.BondMessagePassing))

mp = nn.BondMessagePassing(depth=1,dropout=0.1) 
W_i, W_h, W_o, W_d = mp.setup(d_h=100) 
mp.W_i = W_i
mp.W_h = W_h
mp.W_o = W_o
mp.W_d = W_d


"""
## Aggregation
An `Aggregation` is responsible for constructing a graph-level representation from the set of node-level representations after message passing.
Available options can be found in ` nn.agg.AggregationRegistry`, including
- `agg = nn.MeanAggregation()`
- `agg = nn.SumAggregation()`
- `agg = nn.NormAggregation()`
"""

# print(nn.agg.AggregationRegistry)

agg = nn.MeanAggregation()

"""
## Feed-Forward Network (FFN)

A `FFN` takes the aggregated representations and make target predictions.

Available options can be found in `nn.PredictorRegistry`.

For regression:
- `ffn = nn.RegressionFFN()`
- `ffn = nn.MveFFN()`
- `ffn = nn.EvidentialFFN()`

For classification:
- `ffn = nn.BinaryClassificationFFN()`
- `ffn = nn.BinaryDirichletFFN()`
- `ffn = nn.MulticlassClassificationFFN()`
- `ffn = nn.MulticlassDirichletFFN()`

For spectral:
- `ffn = nn.SpectralFFN()` # will be available in future version
"""

# print(nn.PredictorRegistry)

has_xd = [dp.x_d is not None for dp in train_dset.data]
print(f"{sum(has_xd)} datapoints with x_d out of {len(has_xd)}")
# print(inspect.getsource(nn.predictors.BinaryClassificationFFN))

class CloneableBCELoss(tn.Module):
    def __init__(self, gamma=3.0, alpha=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, preds, targets, mask=None, weights=None, lt_mask=None, gt_mask=None):
        preds = preds.view(-1)
        targets = targets.view(-1)

        if mask is not None:
            mask = mask.view(-1).bool()
            preds = preds[mask]
            targets = targets[mask]

        # Focal loss
        BCE_loss = F.binary_cross_entropy_with_logits(preds, targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        return focal_loss.mean()

    def clone(self):
        return CloneableBCELoss(gamma=self.gamma, alpha=self.alpha)

class CustomBinaryClassificationFFN(BinaryClassificationFFN):
    def __init__(self, input_dim: int, hidden_size: int = 400, num_layers: int = 3, dropout: float = 0.1, pos_weight: float = 4.0):
        super().__init__()

        layers = []

        layers.append(tn.Linear(input_dim, hidden_size))
        layers.append(tn.ReLU())
        layers.append(tn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(tn.Linear(hidden_size, hidden_size))
            layers.append(tn.ReLU())
            layers.append(tn.Dropout(dropout))

        layers.append(tn.Linear(hidden_size, 1))

        self.ffn = tn.Sequential(*layers)
        self.output_transform = tn.Identity()
        self.criterion = CloneableBCELoss()
        self.metric = BinaryAUROC()

    def forward(self, Z: Tensor) -> Tensor:
        return self.output_transform(self.ffn(Z))

    def train_step(self, Z: Tensor) -> Tensor:
        return self.forward(Z)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

ffn_input_dim = mp.output_dim + len(train_dset.X_d[0])
ffn = CustomBinaryClassificationFFN(input_dim=ffn_input_dim)


"""
## Batch Norm
A `Batch Norm` normalizes the outputs of the aggregation by re-centering and re-scaling.

Whether to use batch norm
"""

batch_norm = True

"""
## Metrics
`Metrics` are the ways to evaluate the performance of model predictions.

Available options can be found in `metrics.MetricRegistry`, including
"""

print(nn.metrics.MetricRegistry)
metric_list = [nn.metrics.BinaryAUROC(), nn.metrics.BinaryAccuracy(), nn.metrics.BinaryMCCMetric(), nn.metrics.BinaryF1Score()] # Only the first metric is used for training and early stopping


# Constructs MPNN
mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list, X_d_transform=None)
mpnn


# Configure model checkpointing
checkpointing = ModelCheckpoint(
    "path/to/your/directory",  # Directory where model checkpoints will be saved
    "endpoints-{epoch}-{val_roc:.2f}",  # Filename format for checkpoints, including epoch and validation loss
    "val/roc",  # Metric used to select the best checkpoint (based on validation loss)
    mode="max",  # Save the checkpoint with the validation loss
    save_last=True,  # Always save the most recent checkpoint, even if it's not the best
)

# Set up trainer
trainer = pl.Trainer(
    logger=False,
    enable_checkpointing=True, # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
    enable_progress_bar=True,
    accelerator="auto",
    devices=1,
    max_epochs=200, # number of epochs to train for
    callbacks=[checkpointing], # Use the configured checkpoint callback
)


### Start training ###
trainer.fit(mpnn, train_loader, val_loader)

# Save state_dict
torch.save(mpnn.state_dict(), "model_checkpoint.pt")
# Save model
torch.save(mpnn, "model_full.pt")


### Performance ###
results_1 = trainer.test(mpnn, dataloaders=test_loader)
results_2 = trainer.validate(mpnn, dataloaders=val_loader)
results_3 = trainer.validate(mpnn, dataloaders=train_loader)
