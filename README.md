# KRICT Research Projects

This repository contains personal projects related to computational drug discovery and cheminformatics developed while working at the Korea Research Institute of Chemical Technology (KRICT), Infectious Disease Therapeutics Research Center.
The work focuses on applying machine learning and structural analysis methods to small-molecule drug discovery problems, including ADMET prediction and protein–ligand binding affinity modeling.
Each project is organized in separate directories within this repository for clarity.

---

## Projects

### ADMET Prediction Model
Development of a machine learning pipeline for predicting pharmacokinetic and toxicity properties of small molecules.

Key features:
- Molecular graph representation
- RDKit molecular descriptors
- Chemprop-based MPNN architecture
- Handling class imbalance using focal loss
- Scaffold-based dataset splitting
- MCC-based threshold optimization

---

### Ligand–Pocket Binding Affinity Model
Development of a deep learning model to predict ligand–protein binding affinity using structural information.

Key features:
- Ligand embedding generation
- Protein pocket feature extraction from PDB structures
- Attention-based interaction modeling
- Binding affinity regression using the PDBbind dataset

