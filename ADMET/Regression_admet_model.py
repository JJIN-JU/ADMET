"""
Molecular property regression model (Chemprop-based MPNN)

This script trains a molecular property regression model using
a Chemprop-based Message Passing Neural Network (MPNN).

Main components:
- Molecular graph representation from SMILES
- Message Passing Neural Network (MPNN)
- RDKit 2D descriptors as additional molecular features
- Murcko scaffold-based dataset splitting
- Descriptor normalization using MinMax scaling
- Target normalization for regression tasks
- Custom evidential feed-forward network for uncertainty-aware regression
- Regression evaluation with R2, RMSE, MAE, and MSE

Dependencies:
- chemprop
- RDKit
- PyTorch
- PyTorch Lightning
- pandas
- scikit-learn
- joblib
"""

import os

import torch
import torch.nn as tn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import random
import inspect

import joblib

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

from chemprop import data, featurizers, models, utils
from chemprop import nn as chemnn
from chemprop.data import MoleculeDatapoint, make_split_indices, split_data_by_indices
from chemprop.nn.predictors import BinaryClassificationFFN, RegressionFFN
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.nn.metrics import BinaryAUROC, BCELoss, EvidentialLoss
from chemprop.featurizers import MoleculeFeaturizerRegistry

print("GPU available", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("All packages loaded successfully.")

# load endpoints file
input_path = Path("path/to/endpoints.csv")
df_input = pd.read_csv(input_path)

num_workers = 6 # number of workers for dataloader. 0 means using main process for data loading
smiles_column = 'smiles' # name of the column containing SMILES strings
target_columns = ['values'] # list of names of the columns containing targets

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
    ys = df_input[target_columns].values.reshape(-1, 1)
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

train_dp, val_dp, test_dp = scaffold_split_with_fallback(df_input, featurizer, user_seed)

print(f"[Dataset split sizes]")
print(f"Train: {len(train_dp)}")
print(f"Val:   {len(val_dp)}")
print(f"Test:  {len(test_dp)}")


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
        
featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
train_dset = data.MoleculeDataset(train_dp, featurizer)
val_dset = data.MoleculeDataset(val_dp, featurizer)
test_dset = data.MoleculeDataset(test_dp, featurizer)

train_dset._Y = np.array(train_dset._Y).reshape(-1, 1)
val_dset._Y   = np.array(val_dset._Y).reshape(-1, 1)
test_dset._Y  = np.array(test_dset._Y).reshape(-1, 1)

output_scaler = train_dset.normalize_targets()
val_dset.normalize_targets(output_scaler)
test_dset.normalize_targets(output_scaler)

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

#output_transform = nn.UnscaleTransform.from_standard_scaler(output_scaler)

has_xd = [dp.x_d is not None for dp in train_dset.data]
print(f"{sum(has_xd)} datapoints with x_d out of {len(has_xd)}")

class CustomEvidentialFFN(tn.Module):
    n_targets = 4

    def __init__(self, input_dim: int, hidden_size: int = 300, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        layers = [tn.Linear(input_dim, hidden_size), tn.ReLU(), tn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [tn.Linear(hidden_size, hidden_size), tn.ReLU(), tn.Dropout(dropout)]
        self.backbone = tn.Sequential(*layers)
        self.head = tn.Linear(hidden_size, self.n_targets)  # outputs: (mu_raw, v_raw, alpha_raw, beta_raw)

        # Required attribute used internally by Chemprop
        self.output_transform = tn.Identity()

        # Loss function and evaluation metric
        self.criterion = EvidentialLoss()
        self.metric = nn.metrics.RMSE()

        # Hyperparameters stored for checkpointing
        self.hparams = {
            "predictor": "CustomEvidentialFFN",
            "input_dim": int(input_dim),
            "hidden_size": int(hidden_size),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "head": "evidential",
            "target_space": "standardized",
        }

    def forward(self, Z: Tensor) -> Tensor:
        Y = self.head(self.backbone(Z))          # (N, 4): [mu_raw, v_raw, alpha_raw, beta_raw]
        mu, v, alpha, beta = torch.chunk(Y, 4, dim=1)

        v     = F.softplus(v)
        alpha = F.softplus(alpha) + 1.0
        beta  = F.softplus(beta)

        mu = self.output_transform(mu)           # 현재는 Identity

        return torch.stack((mu, v, alpha, beta), dim=2)  # (N, 1, 4)

    train_step = forward

ffn_input_dim = mp.output_dim + train_dset.X_d.shape[1]
ffn = CustomEvidentialFFN(input_dim=ffn_input_dim)

X_d_transform=None # Descriptors are already scaled using MinMaxScaler


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
metric_list = [nn.metrics.R2Score(), nn.metrics.RMSE(), nn.metrics.MAE(), nn.metrics.MSE()] # Only the first metric is used for training and early stopping

# Constructs MPNN
mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list, X_d_transform=X_d_transform)
mpnn


# Configure model checkpointing
checkpointing = ModelCheckpoint(
    "path/to/your/directory",  # Directory where model checkpoints will be saved
    "endpoints-{epoch}-{val_loss:.2f}",  # Filename format for checkpoints, including epoch and validation loss
    "val_loss",  # Metric used to select the best checkpoint (based on validation loss)
    mode="min",  # Save the checkpoint with validation loss ("min" or "max")
    save_last=True,  # Always save the most recent checkpoint, even if it's not the best
)

# %%
""" trainer setting"""

# %%
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

# save only weight
torch.save(mpnn.state_dict(), "model_weight.pt")
# save model
torch.save(mpnn, "model_full.pt")
# save X_d scaler
joblib.dump(scaler, "model_Xd_scaler.pkl")
# save target scaler
joblib.dump(output_scaler, "model_scaler.pkl")

### Performance ###
results1 = trainer.test(mpnn, dataloaders=test_loader)
results2 = trainer.validate(mpnn, dataloaders=val_loader)
results3 = trainer.validate(mpnn, dataloaders=train_loader)

