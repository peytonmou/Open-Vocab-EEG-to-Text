# EEG-to-Word Decoding

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Dataset-yellow.svg)](https://huggingface.co/datasets/peytonmou/EEG2Text)

Deep learning models for decoding single-word EEG signals into text. This repository provides complete pipeline from raw EEG conversion to model training and evaluation.


## 🚀 Quick Start

### Step 1: Download Dataset
```bash
# Install git lfs first
git lfs install
git clone https://huggingface.co/datasets/peytonmou/EEG2Text

### Step 2: Clone Repository
git clone https://github.com/peytonmou/EEG-to-Word_Decoding.git
cd EEG-to-Word_Decoding

### Step 3: Convert EDF to pickle
python edf2pickle.py

### Step 4: Train Model
Update file paths in train.ipynb:
Path to generated pickle files
Folder path for saving trained models
Run the training notebook or script

### Step 5: Evaluate Model
Update model loading path in eval.ipynb
Run evaluation notebook

### Requirements
pip install torch numpy scipy scikit-learn





