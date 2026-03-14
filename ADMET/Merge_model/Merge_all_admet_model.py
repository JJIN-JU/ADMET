"""
ADMET multi-endpoint prediction pipeline

This script performs ADMET prediction using Chemprop-based models.

The input file must be an SDF file containing molecules.
SMILES are extracted from the SDF file and used to generate molecular
graphs and RDKit 2D descriptors.

Each molecule is then evaluated across multiple ADMET endpoints
including toxicity, absorption, metabolism, and physicochemical
properties.

Input
-----
SDF file containing compounds

Output
------
CSV file containing predicted ADMET properties for each compound
"""

import sys, torch, joblib
import numpy as np
import torch.nn as tn
import pandas as pd
import torch.nn.functional as F
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED
from chemprop import data, featurizers, utils
from chemprop.data import MoleculeDatapoint
from chemprop.featurizers import MoleculeFeaturizerRegistry

import final_1, final_2
sys.modules['__main__'].CustomBinaryClassificationFFN = final_1.CustomBinaryClassificationFFN
sys.modules['__main__'].CloneableBCELoss = final_1.CloneableBCELoss
sys.modules['__main__'].CustomEvidentialFFN = final_2.CustomEvidentialFFN
sys.modules['__main__'].CustomRegressionFFN = final_2.CustomRegressionFFN

import hashlib, numpy as np

def _fingerprint(x: np.ndarray) -> str:
    try:
        return hashlib.sha1(x.astype(np.float32).tobytes()).hexdigest()[:8]
    except Exception:
        return "na"

def xd_stats(dset, title: str):
    X = np.asarray(dset.X_d, dtype=np.float32)
    fin = np.isfinite(X)
    print(f"[{title}] X_d shape={X.shape} finite%={fin.mean()*100:.2f}% "
          f"mean={np.nanmean(X):.4g} std={np.nanstd(X):.4g} "
          f"min={np.nanmin(X):.4g} max={np.nanmax(X):.4g} fp={_fingerprint(X)}")
    if X.ndim == 2 and X.shape[0] > 0:
        print("  sample row[0][:8] =", np.round(X[0][:8], 4))

def _patch_predictor_for_evidential(m):
    try:
        pred = m.predictor
    except Exception:
        return m

    if hasattr(pred, "backbone") and hasattr(pred, "head") and not hasattr(pred, "ffn"):
        pred.ffn = tn.Sequential(pred.backbone, pred.head)

    if not hasattr(pred, "output_transform"):
        pred.output_transform = tn.Identity()

    return m
    
# featurizer
_RD2D = MoleculeFeaturizerRegistry["rdkit_2d"]()

def build_loader_from_smiles(smiles_list, names_list, batch_size=256, num_workers=0):
    dps, names = [], []
    for smi, nm in zip(smiles_list, names_list):
        mol = utils.make_mol(smi, keep_h=False, add_h=False)
        if mol is None:
            continue
        X_d = _RD2D(mol)
        X_d32 = np.asarray(X_d, dtype=np.float32)
        if not np.all(np.isfinite(X_d32)):
            continue
        dps.append(MoleculeDatapoint(mol=mol, y=None, x_d=X_d32, name=nm))
        names.append(nm)

    if not dps:
        return None, None, []

    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dset = data.MoleculeDataset(dps, feat)
    dset.X_d = np.asarray(dset.X_d, dtype=np.float32)
    loader = data.build_dataloader(dset, batch_size=batch_size,
                                   num_workers=num_workers, shuffle=False)
    return loader, dset, names

def _infer_probs_from_logits(logits: torch.Tensor) -> np.ndarray:
    if logits.ndim != 2:
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")
    if logits.shape[1] == 1:  # binary
        p1 = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        p0 = 1 - p1
        return np.stack([p0, p1], axis=1)
    else:
        return F.softmax(logits, dim=1).cpu().numpy()

from sklearn.preprocessing import MinMaxScaler

@torch.no_grad()
def predict_with_model(model_full_pt, scaler_pkl, loader, dset, names,
                       task_type="regression", threshold=None,
                       apply_y_scaler=False):
    model = torch.load(model_full_pt, map_location="cpu").eval()
    model = _patch_predictor_for_evidential(model)

    x_scaler, y_scaler = None, None
    if scaler_pkl is not None:
        if isinstance(scaler_pkl, (list, tuple)):
            x_scaler = joblib.load(scaler_pkl[0])
            if len(scaler_pkl) > 1:
                y_scaler = joblib.load(scaler_pkl[1])
        else:
            loaded = joblib.load(scaler_pkl)
            if isinstance(loaded, (tuple, list)):
                x_scaler, y_scaler = loaded[0], loaded[1]
            else:
                x_scaler = loaded

    X0 = np.asarray(dset.X_d, dtype=np.float32)
    X0 = np.nan_to_num(X0, nan=0.0, posinf=1e6, neginf=-1e6)

    X_used = x_scaler.transform(X0) if x_scaler is not None else X0
    X_used = X_used.astype(np.float32, copy=False)

    if isinstance(x_scaler, MinMaxScaler):
        X_used = np.clip(X_used, 0.0, 1.0)
    else:
        X_used = np.clip(X_used, -5.0, 5.0)

    tmp_dps = [
        MoleculeDatapoint(mol=dp.mol, y=None, x_d=x_row, name=dp.name)
        for dp, x_row in zip(dset.data, X_used)
    ]
    tmp_feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    tmp_dset = data.MoleculeDataset(tmp_dps, tmp_feat)
    bs = getattr(loader, "batch_size", 256)
    nw = getattr(loader, "num_workers", 0)
    tmp_loader = data.build_dataloader(tmp_dset, batch_size=bs, num_workers=nw, shuffle=False)

    outputs = [model(batch.bmg, batch.V_d, batch.X_d) for batch in tmp_loader]
    logits = torch.cat(outputs, dim=0)

    if task_type == "classification":
        probs = _infer_probs_from_logits(logits)
        labels = probs.argmax(axis=1).astype(int)
        return probs, labels

    elif task_type == "regression":
        logits2 = logits.squeeze(1) if (logits.ndim == 3 and logits.shape[1] == 1 and logits.shape[2] == 4) else logits
        if logits2.ndim == 2 and logits2.shape[1] == 4:
            mean, _, _, _ = torch.chunk(logits2, 4, dim=1)
            mean = mean.cpu().numpy().reshape(-1, 1)
        else:
            mean = logits2.cpu().numpy().reshape(-1, 1)

        if apply_y_scaler and (y_scaler is not None):
            try:
                mean = y_scaler.inverse_transform(mean)
            except Exception as e:
                print("[WARN] y_scaler.inverse_transform Fail:", repr(e))

        mean = mean.reshape(-1)

        try:
            print("raw logits sample:", logits[:5])
            print("pred sample (after optional inverse):", mean[:5])
        except Exception:
            pass

        return mean

    else:
        raise ValueError(f"Unknown task_type {task_type}")



def run_tasks(smiles, names, task_configs):
    task_to_output = {}
    for task, cfg in task_configs.items():
        loader, dset, names_used = build_loader_from_smiles(smiles, names)
        if loader is None:
            continue

        result = predict_with_model(
            model_full_pt=cfg["model"],
            scaler_pkl=cfg["scaler"],
            loader=loader,
            dset=dset,
            names=names_used,
            task_type=cfg.get("task_type", "classification"),
            threshold=cfg.get("threshold"),
            apply_y_scaler=cfg.get("apply_y_scaler", False),
        )

        if cfg.get("task_type") == "classification":
            probs, labels = result
            task_to_output[task] = {
                "type": "classification",
                "probs": probs,
                "labels": labels,
                "names": names_used,
            }
        else:
            task_to_output[task] = {
                "type": "regression",
                "mean": result,
                "names": names_used,
            }
    return task_to_output



def save_results_csv(df: pd.DataFrame, out_csv: str | Path):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[CSV saved] {out_csv.resolve()}")

def calc_simple_rules(mol):
    results = {}
    mw    = Descriptors.MolWt(mol)
    logp  = Crippen.MolLogP(mol)
    hba   = rdMolDescriptors.CalcNumHBA(mol)
    hbd   = rdMolDescriptors.CalcNumHBD(mol)
    tpsa  = rdMolDescriptors.CalcTPSA(mol)
    rotb  = rdMolDescriptors.CalcNumRotatableBonds(mol)
    nsp3  = sum(1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP3 and atom.GetAtomicNum()==6)
    nC = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum()==6)
    # Lipinski
    lipinski_viol = int(mw > 500) + int(logp > 5) + int(hba > 10) + int(hbd > 5)
    results["Lipinski_Viol"] = lipinski_viol
    # Pfizer
    results["Pfizer_Viol"] = int(logp > 3 and tpsa < 75)
    # GSK
    results["GSK_Viol"] = int(mw > 400) + int(logp > 4)
    # Golden Triangle
    results["GoldenTriangle_Viol"] = int(not (200 <= mw <= 500)) + int(not (-2 <= logp <= 5))
    results["QED"] = QED.qed(mol)
    results["Fsp3"] = (nsp3 / nC) if nC > 0 else 0.0
    return results

def add_rule_columns(df, smiles_list):
    rules = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        rules.append(calc_simple_rules(mol) if mol else {})
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rules)], axis=1)

def load_smiles_from_sdf(sdf_path, prop_name="compounds"):
    suppl = Chem.SDMolSupplier(sdf_path)
    smiles_list, names = [], []
    for mol in suppl:
        if mol is None: continue
        smi = Chem.MolToSmiles(mol)
        name = mol.GetProp(prop_name) if mol.HasProp(prop_name) else (mol.GetProp("_Name") if mol.HasProp("_Name") else smi)
        smiles_list.append(smi)
        names.append(name)
    return smiles_list, names


if __name__ == "__main__":
    sdf_file = "predict_admet.sdf"
    smiles, compounds = load_smiles_from_sdf(sdf_file, prop_name="compounds")

    loader, dset, names = build_loader_from_smiles(smiles, compounds)
    task_configs = {
      
        ###Classification###
        "Endpoints1": {"model": "path/to/your/model_full.pt", "scaler": "path/to/your/model_scaler.pkl", "threshold": 0.0, "task_type": "classification"},
        "Endpoints2": {"model": "path/to/your/model_full.pt", "scaler": "path/to/your/model_scaler.pkl", "threshold": 0.0, "task_type": "classification"}

        ###Regrassion###
        "Endpoints3": {"model":"path/to/your/model_full.pt", "scaler": ["path/to/your/model_Xd_scaler.pkl","path/to/your/model_scaler.pkl"], "task_type": "regression", "apply_y_scaler": True},
        "Endpoints4": {"model":"path/to/your/model_full.pt", "scaler": ["path/to/your/model_Xd_scaler.pkl","path/to/your/model_scaler.pkl"], "task_type": "regression", "apply_y_scaler": True}
                
    }

    from functools import reduce
    
    def make_results_dataframe(task_to_output, label_mapping=None):
        dfs = []
        for task, out in task_to_output.items():
            names_task = out["names"]
            rows = []
    
            if out["type"] == "classification":
                probs, labels = out["probs"], out["labels"]
                for i, name in enumerate(names_task):
                    if i >= len(probs) or i >= len(labels):
                        continue
                    p0, p1 = float(probs[i][0]), float(probs[i][1])
                    label = int(labels[i])
                    label_txt = label_mapping.get(label, str(label)) if label_mapping else str(label)
                    rows.append({
                        "compounds": name,
                        f"{task}_Label": label_txt,
                        f"{task}_Prob_non-toxic": f"{p0:.2f}",
                        f"{task}_Prob_toxic":     f"{p1:.2f}",
                    })
    
            else:  # regression
                mean = out["mean"]
                for i, name in enumerate(names_task):
                    if i >= len(mean):
                        continue
                    rows.append({
                        "compounds": name,
                        f"{task}_Pred": f"{float(mean[i]):.2f}",
                    })
    
            dfs.append(pd.DataFrame(rows))
    
        return reduce(lambda L, R: pd.merge(L, R, on="compounds", how="outer"), dfs) if dfs else pd.DataFrame(columns=["compounds"])


    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED
    def calc_simple_rules(mol):
        results = {}
        
        mw    = Descriptors.MolWt(mol)
        logp  = Crippen.MolLogP(mol)
        hba   = rdMolDescriptors.CalcNumHBA(mol)
        hbd   = rdMolDescriptors.CalcNumHBD(mol)
        tpsa  = rdMolDescriptors.CalcTPSA(mol)
        rotb  = rdMolDescriptors.CalcNumRotatableBonds(mol)
        ncarb = mol.GetNumAtoms(onlyExplicit=False)
        nsp3  = sum(1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP3 and atom.GetAtomicNum()==6)
        
        # Lipinski
        lipinski_viol = int(mw > 500) + int(logp > 5) + int(hba > 10) + int(hbd > 5)
        results["Lipinski_Rule_Violations"] = lipinski_viol
        # Pfizer Rule (logP > 3 and TPSA < 75 → 동시에 만족 시 위반)
        results["Pfizer_Rule_Violation"] = int(logp > 3 and tpsa < 75)
        
        # GSK Rule (MW ≤ 400, logP ≤ 4 → 위반 개수)
        gsk_viol = int(mw > 400) + int(logp > 4)
        results["GSK_Rule_Violations"] = gsk_viol

        # Golden Triangle (200 ≤ MW ≤ 500 and -2 ≤ logD ≤ 5)
        gt_viol = int(not (200 <= mw <= 500)) + int(not (-2 <= logp <= 5))
        results["GoldenTriangle_Violations"] = gt_viol
    
        return results

    def add_rule_columns(df, smiles_list):
        from rdkit import Chem
        all_rules = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                all_rules.append({})
                continue
            rules = calc_simple_rules(mol)
            all_rules.append(rules)

        rules_df = pd.DataFrame(all_rules)
        return pd.concat([df.reset_index(drop=True), rules_df.reset_index(drop=True)], axis=1)


    task_configs_cls = {k: v for k, v in task_configs.items() if v.get("task_type") == "classification"}
    task_configs_reg = {k: v for k, v in task_configs.items() if v.get("task_type") == "regression"}

    out_cls = run_tasks(smiles, compounds, task_configs_cls)
    df_cls  = make_results_dataframe(out_cls, label_mapping={0:"non-toxic", 1:"toxic"})

    out_reg = run_tasks(smiles, compounds, task_configs_reg)
    df_reg  = make_results_dataframe(out_reg)

    df = pd.merge(df_cls, df_reg, on="compounds", how="outer")


    save_results_csv(df, "admet_results.csv")

