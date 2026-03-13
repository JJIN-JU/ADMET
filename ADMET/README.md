# ADMET Prediction Model

This directory contains code for building and training an ADMET prediction model based on molecular graph neural networks.

## Overview

Development of a machine learning pipeline for predicting pharmacokinetic and toxicity properties of small molecules.

The model is designed to support early-stage drug discovery by estimating key ADMET properties using graph-based molecular representations.

## Model Architecture

The pipeline is built on a Chemprop-based Message Passing Neural Network (MPNN) combined with additional molecular descriptors.

Key components include:

- Molecular graph representation
- RDKit molecular descriptors
- Chemprop-based MPNN architecture
- Feed-forward neural network prediction head
- Handling class imbalance using focal loss
- Scaffold-based dataset splitting
- MCC-based threshold optimization

## Implementation

- Framework: PyTorch / PyTorch Lightning
- Molecular processing: RDKit
- Graph neural network: Chemprop
- Feature engineering: RDKit descriptors

## Additional Features

- Custom AUROC metric (SafeBinaryAUROC)
- Manual descriptor normalization
- MCC-based decision threshold selection
