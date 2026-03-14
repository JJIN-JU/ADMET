# -*- coding: utf-8 -*-
# %%
import os

import torch
import torch.nn as tn
import torch.nn.functional as F

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

print("GPU 사용 가능?", torch.cuda.is_available())
print("GPU 이름:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("All packages loaded successfully.")

# %%
input_path = Path("/home/yejin/ADMET/Tox21/data/sr-atad5.csv")
df_input = pd.read_csv(input_path)

num_workers = 6 # number of workers for dataloader. 0 means using main process for data loading (프로세서 병렬 작업)
smiles_column = 'smiles' # name of the column containing SMILES strings
target_columns = ['label'] # list of names of the columns containing targets

# %%
smis = df_input.loc[:, smiles_column].values
ys = df_input.loc[:, target_columns].values

# %%
smis[:5] # show first 5 SMILES strings

# %%
ys[:5] # show first 5 targets

# %%
"""## Getting extra features and descriptors"""

# %%
mols = [utils.make_mol(smi, keep_h=False, add_h=False) for smi in smis]

# %%
for MoleculeFeaturizer in featurizers.MoleculeFeaturizerRegistry.keys():
    print(MoleculeFeaturizer)

# %%
featurizer = MoleculeFeaturizerRegistry["rdkit_2d"]()
extra_features = [featurizer(mol) for mol in mols]

df_input["X_d"] = extra_features
# %%
print(f"총 개수: {len(extra_features)}")
print(f"None 아닌 벡터 수: {sum(x is not None for x in extra_features)}")
print(f"첫 번째 유효 벡터 길이: {len([x for x in extra_features if x is not None][0])}")

# %%
"""## Get molecule datapoints"""

# %%
datapoints = [
    data.MoleculeDatapoint(mol, y, x_d=X_d)
    for mol, y, X_d in zip(
        mols,
        ys,
        extra_features
    )
]

# %%
"""## Perform data splitting for training, validation, and testing"""

# %%
# available split types
list(data.SplitType.keys())

# %%
def get_murcko(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, isomericSmiles=False)
    except:
        return None

def fallback_to_chemprop_split(df_input, seed=42):
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

    # Use fallback chemprop default split to avoid scaffold matching failure
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


def try_murcko_split(df_input, seed=42, max_deviation=0.03):
    df = df_input.copy()
    df["scaffold"] = df["smiles"].apply(get_murcko)

    # scaffold 단위 그룹화
    scaffold_to_indices = {}
    for i, scaffold in enumerate(df["scaffold"]):
        if scaffold is None:
            continue
        scaffold_to_indices.setdefault(scaffold, []).append(i)

    # scaffold 셔플
    scaffold_sets = list(scaffold_to_indices.values())
    scaffold_sets.sort(key=len)
    random.seed(seed)
    random.shuffle(scaffold_sets)

    # 분할 목표 개수
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

    # 남은 scaffold가 있다면 test에 추가
    used = set(train_idx + val_idx + test_idx)
    remaining = set(chain.from_iterable(scaffold_sets)) - used
    test_idx.extend(remaining)

    # 비율 검사
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

def scaffold_split_with_fallback(df_input, featurizer, seed=42):
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
        return fallback_to_chemprop_split(df_input, seed=seed)

# %%
train_dp, val_dp, test_dp = scaffold_split_with_fallback(df_input, featurizer, seed=42)

# %%

train_labels = np.array([dp.y[0] for dp in train_dp])
val_labels = np.array([dp.y[0] for dp in val_dp])
test_labels = np.array([dp.y[0] for dp in test_dp])

print(f"\n[Class Distribution Check]")
print(f"Train:      1 → {np.sum(train_labels == 1)}, 0 → {np.sum(train_labels == 0)}")
print(f"Validation: 1 → {np.sum(val_labels == 1)}, 0 → {np.sum(val_labels == 0)}")
print(f"Test:       1 → {np.sum(test_labels == 1)}, 0 → {np.sum(test_labels == 0)}")


# %%
from sklearn.preprocessing import MinMaxScaler
import numpy as np

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

X_d_transform = None


# %%
""" scale"""

# %%

X_d_array = np.array(train_dset.X_d)

# %% [markdown]
# output_scaler = train_dset.normalize_targets() #회귀만 사용

# %%
"""## Get DataLoader"""

# %%
# Featurize the train and val datasets to save computation time. (한 번 featurize한 결과를 메모리에 캐시해놓고 재사용 - 메모리 여유없으면 False로)
train_dset.cache = True
val_dset.cache = True

# %%
train_loader = data.build_dataloader(train_dset, batch_size=128, num_workers=num_workers)
val_loader = data.build_dataloader(val_dset, batch_size=128, num_workers=num_workers, shuffle=False)
test_loader = data.build_dataloader(test_dset, batch_size=128, num_workers=num_workers, shuffle=False)

# %%
"""# Change Message-Passing Neural Network (MPNN) inputs here

## Message Passing
A `Message passing` constructs molecular graphs using message passing to learn node-level hidden representations.

Options are `mp = nn.BondMessagePassing()` or `mp = nn.AtomMessagePassing()`
"""

# %%
print(inspect.getsource(nn.BondMessagePassing))

# %%
mp = nn.BondMessagePassing(depth=6,dropout=0.25)
W_i, W_h, W_o, W_d = mp.setup(d_h=1200)
mp.W_i = W_i
mp.W_h = W_h
mp.W_o = W_o
mp.W_d = W_d

# %%
"""## Aggregation
An `Aggregation` is responsible for constructing a graph-level representation from the set of node-level representations after message passing.

Available options can be found in ` nn.agg.AggregationRegistry`, including
- `agg = nn.MeanAggregation()`
- `agg = nn.SumAggregation()`
- `agg = nn.NormAggregation()`
"""

# %%
print(nn.agg.AggregationRegistry)

# %%
agg = nn.MeanAggregation()

# %%
"""## Feed-Forward Network (FFN)

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

# %%
print(nn.PredictorRegistry)

# %% [markdown]
# output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

# %%
has_xd = [dp.x_d is not None for dp in train_dset.data]
print(f"{sum(has_xd)} datapoints with x_d out of {len(has_xd)}")


# %%
print(inspect.getsource(nn.predictors.BinaryClassificationFFN))

# %%
from chemprop.nn.predictors import BinaryClassificationFFN
from chemprop.nn.metrics import BinaryAUROC
from chemprop.nn.metrics import BCELoss
import torch.nn as tn
from torch import Tensor
import torch.nn.functional as F

# %%
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

        # Focal loss 계산
        BCE_loss = F.binary_cross_entropy_with_logits(preds, targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        return focal_loss.mean()

    def clone(self):
        return CloneableBCELoss(gamma=self.gamma, alpha=self.alpha)



# %%
class CustomBinaryClassificationFFN(BinaryClassificationFFN):
    def __init__(self, input_dim: int, hidden_size: int = 400, num_layers: int = 3, dropout: float = 0.1, pos_weight: float = 4.0):
        super().__init__()

        layers = []

        # 입력층
        layers.append(tn.Linear(input_dim, hidden_size))
        layers.append(tn.ReLU())
        layers.append(tn.Dropout(dropout))

        # 히든층
        for _ in range(num_layers - 1):
            layers.append(tn.Linear(hidden_size, hidden_size))
            layers.append(tn.ReLU())
            layers.append(tn.Dropout(dropout))

        # 출력층 (binary classification → 1 unit)
        layers.append(tn.Linear(hidden_size, 1))

        self.ffn = tn.Sequential(*layers)

        # Sigmoid 제거: raw logits 출력
        self.output_transform = tn.Identity()

        # pos_weight 반영된 loss 함수
        self.criterion = CloneableBCELoss(pos_weight)

        # AUC metric (raw logits로 계산 가능)
        self.metric = BinaryAUROC()

    def forward(self, Z: Tensor) -> Tensor:
        return self.output_transform(self.ffn(Z))

    def train_step(self, Z: Tensor) -> Tensor:
        return self.forward(Z)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


# %%
ffn_input_dim = mp.output_dim + len(train_dset.X_d[0])
labels = np.array([dp.y[0] for dp in train_dp])  # binary label 리스트로 추출
pos_weight = np.sum(labels == 0) / np.sum(labels == 1)


ffn = CustomBinaryClassificationFFN(input_dim=ffn_input_dim)

# %%
"""## Batch Norm
A `Batch Norm` normalizes the outputs of the aggregation by re-centering and re-scaling.

Whether to use batch norm
"""

# %%
batch_norm = True

# %%
"""## Metrics
`Metrics` are the ways to evaluate the performance of model predictions.

Available options can be found in `metrics.MetricRegistry`, including
"""

# %%
print(nn.metrics.MetricRegistry)

# %%
metric_list = [nn.metrics.BinaryAUROC(), nn.metrics.BinaryAccuracy(), nn.metrics.BinaryMCCMetric(), nn.metrics.BinaryF1Score()] # Only the first metric is used for training and early stopping

# %%
"""## Constructs MPNN"""

# %%
mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list, X_d_transform=None)
mpnn

# %%
"""# Set up trainer

모델저장
"""

# %%
# Configure model checkpointing
checkpointing = ModelCheckpoint(
    "/home/yejin/ADMET/Tox21/",  # Directory where model checkpoints will be saved
    "SR-ATAD5-{epoch}-{val_roc:.2f}",  # Filename format for checkpoints, including epoch and validation loss
    "val/roc",  # Metric used to select the best checkpoint (based on validation loss)
    mode="max",  # Save the checkpoint with the lowest validation loss (minimization objective)
    save_last=True,  # Always save the most recent checkpoint, even if it's not the best
)

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

# %%
"""# Start training"""

# %%
trainer.fit(mpnn, train_loader, val_loader)
# 모델 저장 (state_dict만 저장)
torch.save(mpnn.state_dict(), "model_checkpoint.pt")
# 모델 전체 저장 (구성 포함)
torch.save(mpnn, "model_full.pt")

# %%
"""# Test results"""

# %%
results_1 = trainer.test(mpnn, dataloaders=test_loader)
results_2 = trainer.validate(mpnn, dataloaders=val_loader)
results_3 = trainer.validate(mpnn, dataloaders=train_loader)

from scipy.special import expit
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix 
from sklearn.metrics import roc_auc_score

all_logits = []
all_targets = []

mpnn.eval()
with torch.no_grad():
    for batch in test_loader:
        logits = mpnn(batch.bmg, batch.V_d, batch.X_d)
        all_logits.append(logits.cpu().numpy())
        all_targets.append(batch.Y.cpu().numpy())

for batch in test_loader:
    print(f"len(batch): {len(batch)}")
    for i, item in enumerate(batch):
        print(f"batch[{i}] shape: {item.shape if hasattr(item, 'shape') else type(item)}")
    break
    
# Flatten arrays
all_logits = np.concatenate(all_logits).flatten()
all_targets = np.concatenate(all_targets).flatten()

# Apply sigmoid to get probabilities
probs = expit(all_logits)

# Find best threshold for MCC
def find_best_threshold(preds, targets):
    thresholds = np.linspace(0, 1, 101)
    best_mcc = -1
    best_threshold = 0.5
    for t in thresholds:
        binary_preds = (preds >= t).astype(int)
        mcc = matthews_corrcoef(targets, binary_preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = t
    return best_threshold, best_mcc

best_threshold, best_mcc = find_best_threshold(probs, all_targets)

# Apply best threshold and compute final metrics
final_preds = (probs >= best_threshold).astype(int)

conf_matrix = confusion_matrix(all_targets, final_preds)
acc = accuracy_score(all_targets, final_preds)
f1 = f1_score(all_targets, final_preds)
auc = roc_auc_score(all_targets, probs)

print(f"\n[Best Test MCC Evaluation]")
print(f"Best Threshold: {best_threshold:.3f}")
print(f"Best MCC:       {best_mcc:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy:        {acc:.4f}")
print(f"F1 Score:        {f1:.4f}")
print(f"AUC:             {auc:.4f}") 

all_logits = []
all_targets = []

mpnn.eval()
with torch.no_grad():
    for batch in val_loader:
        logits = mpnn(batch.bmg, batch.V_d, batch.X_d)
        all_logits.append(logits.cpu().numpy())
        all_targets.append(batch.Y.cpu().numpy())
    
# Flatten arrays
all_logits = np.concatenate(all_logits).flatten()
all_targets = np.concatenate(all_targets).flatten()

# Apply sigmoid to get probabilities
probs = expit(all_logits)

# Find best threshold for MCC
def find_best_threshold(preds, targets):
    thresholds = np.linspace(0, 1, 101)
    best_mcc = -1
    best_threshold = 0.5
    for t in thresholds:
        binary_preds = (preds >= t).astype(int)
        mcc = matthews_corrcoef(targets, binary_preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = t
    return best_threshold, best_mcc

best_threshold, best_mcc = find_best_threshold(probs, all_targets)

# Apply best threshold and compute final metrics
final_preds = (probs >= best_threshold).astype(int)

conf_matrix = confusion_matrix(all_targets, final_preds)
acc = accuracy_score(all_targets, final_preds)
f1 = f1_score(all_targets, final_preds)
auc = roc_auc_score(all_targets, probs)

print(f"\n[Best Val MCC Evaluation]")
print(f"Best Threshold: {best_threshold:.3f}")
print(f"Best MCC:       {best_mcc:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy:        {acc:.4f}")
print(f"F1 Score:        {f1:.4f}")
print(f"AUC:             {auc:.4f}") 


all_logits = []
all_targets = []

mpnn.eval()
with torch.no_grad():
    for batch in train_loader:
        logits = mpnn(batch.bmg, batch.V_d, batch.X_d)
        all_logits.append(logits.cpu().numpy())
        all_targets.append(batch.Y.cpu().numpy())
    
# Flatten arrays
all_logits = np.concatenate(all_logits).flatten()
all_targets = np.concatenate(all_targets).flatten()

# Apply sigmoid to get probabilities
probs = expit(all_logits)

# Find best threshold for MCC 
def find_best_threshold(preds, targets):
    thresholds = np.linspace(0, 1, 101)
    best_mcc = -1
    best_threshold = 0.5
    for t in thresholds:
        binary_preds = (preds >= t).astype(int)
        mcc = matthews_corrcoef(targets, binary_preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = t
    return best_threshold, best_mcc

best_threshold, best_mcc = find_best_threshold(probs, all_targets)

# Apply best threshold and compute final metrics
final_preds = (probs >= best_threshold).astype(int)

conf_matrix = confusion_matrix(all_targets, final_preds)
acc = accuracy_score(all_targets, final_preds)
f1 = f1_score(all_targets, final_preds)
auc = roc_auc_score(all_targets, probs)

print(f"\n[Best Train MCC Evaluation]")
print(f"Best Threshold: {best_threshold:.3f}")
print(f"Best MCC:       {best_mcc:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy:        {acc:.4f}")
print(f"F1 Score:        {f1:.4f}")
print(f"AUC:             {auc:.4f}") 
