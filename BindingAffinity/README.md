# Ligand–Pocket Binding Affinity Prediction Model

This project implements a **protein–ligand binding affinity prediction model** using structural information from ligand molecules and protein binding pockets.

(This project implements a protein–ligand binding affinity prediction model developed as part of my independent research.)

## Overview

Development of a deep learning pipeline for predicting **protein–ligand binding affinity** using structural features extracted from ligand molecules and protein binding sites.

The model is designed to capture interaction patterns between ligands and protein pockets to estimate binding strength for small-molecule drug discovery applications.

## Model Architecture

The pipeline integrates ligand molecular representations with structural features from protein binding pockets.

Key components include:

- Ligand embedding generation from molecular structures
- Protein binding pocket feature extraction from PDB structures
- Residue-level and atom-level feature encoding
- Attention-based interaction modeling between ligand and pocket
- Binding affinity regression model

## Data Processing

- Ligand structures processed from **SDF files**
- Protein pocket structures extracted from **PDB structures**
- Binding affinity labels obtained from the **PDBbind refined set**

## Implementation

- Framework: **PyTorch**
- Molecular processing: **RDKit**
- Protein structure parsing: **Biopython**
- Data handling: **NumPy / Pandas**

## Additional Features

- Ligand and pocket embedding generation
- Interaction modeling between ligand atoms and pocket residues
- Binding affinity regression based on experimental binding data
