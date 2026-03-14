"""
ADMET molecular preprocessing pipeline

This script performs molecular preprocessing including:
- SMILES column standardization
- salt/fragment removal
- molecular normalization and uncharging
- canonical SMILES generation
- duplicate removal

The pipeline is designed to standardize molecular representations
before training machine learning models for ADMET prediction.

Dependencies:
- RDKit
- pandas
"""

# Import required libraries
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

# Input dataset requirements:
# - The input file must be a CSV file containing a SMILES column.
# - The target property column (e.g., activity, toxicity, affinity)
#   should be curated and prepared by the user depending on the prediction task.
# - This preprocessing script only standardizes molecular structures
#   and does not modify target labels for classification or regression tasks.

# Load dataset
input_path = "path/to/your/endpoints.csv"
output_path = "path/to/your/endpoints_processed.csv"

df = pd.read_csv(input_path)

# Detect SMILES column
if 'smiles' in df.columns:
    smiles_col = 'smiles'
elif 'canonical_smiles' in df.columns:
    smiles_col = 'canonical_smiles'
elif 'SMILES' in df.columns:
    smiles_col = 'SMILES'
else:
    raise ValueError("SMILES column not found")

# RDKit standardization tools
normalizer = rdMolStandardize.Normalizer()
uncharger = rdMolStandardize.Uncharger()
fragment_remover = rdMolStandardize.FragmentRemover()

def preprocess_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None

        # Remove molecules containing metals
        metal_atomic_numbers = {
            3,4,11,12,13,19,20,21,22,23,24,25,
            26,27,28,29,30,31,37,38,39,40,41,42,
            43,44,45,46,47,48,49,50,55,56,57,72,
            73,74,75,76,77,78,79,80,81,82,83,87,
            88,89,90,91,92
        }

        if any(atom.GetAtomicNum() in metal_atomic_numbers for atom in mol.GetAtoms()):
            return None

        # Fragment removal
        mol = fragment_remover.remove(mol)

        # Normalization
        mol = normalizer.normalize(mol)

        # Uncharge
        mol = uncharger.uncharge(mol)

        # Canonical SMILES
        return Chem.MolToSmiles(mol, canonical=True)

    except Exception:
        return None

# Apply preprocessing
df["smiles"] = df[smiles_col].apply(preprocess_smiles)

# Remove invalid molecules
df = df[df["smiles"].notna()]

# Remove duplicates
df = df.drop_duplicates(subset=["smiles"])


# Save processed dataset
df.to_csv(output_path, index=False)

print(f"Processed molecules: {len(df)}")
