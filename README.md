# EEG-to-Word Decoding

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Dataset-yellow.svg)](https://huggingface.co/datasets/peytonmou/EEG2Text)

Deep learning models for decoding single-word EEG signals into text. This repository provides complete pipeline from raw EEG conversion to model training and evaluation.


## 📂 Project Structure

```
EEG-to-Word_Decoding/
├── data_singleword.py   # Processed EEG data
├── edf2pickle.ipynb     # Convert EDF to pickle
├── model_decoding.py    # T5Translator
├── train.ipynb          # Model training notebook
├── eval.ipynb           # Model evaluation notebook
└── README.md
```

---
## 🚀 Getting Started

### 1. Clone Dataset

```bash
# Install Git LFS (if not installed)
git lfs install

# Clone dataset
git clone https://huggingface.co/datasets/peytonmou/EEG2Text
```

---

### 2. Clone Repository

```bash
git clone https://github.com/peytonmou/Open-Vocab-EEG-to-Text.git
cd Open-Vocab-EEG-to-Text
```

---

### 3. Set Up Environment

```bash
python -m venv venv

# Activate environment
# Mac/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# Install dependencies
pip install torch numpy scipy scikit-learn
```

---

### 4. Data Preprocessing

Convert raw EDF files into pickle format:

```bash
python edf2pickle.py
```

---

### 5. Train Model

1. Open `train.ipynb`
2. Update the following paths:

   * Path to generated pickle files
   * Directory for saving trained models
3. Run the notebook (or convert to script if preferred)

---

### 6. Evaluate Model

1. Open `eval.ipynb`
2. Update the model loading path
3. Run the notebook

---

## ⚙️ Requirements

* Python 3.8+
* torch
* numpy
* scipy
* scikit-learn

Install all dependencies:

```bash
pip install torch numpy scipy scikit-learn
```

---

## 📊 Workflow

```
Raw EEG Data (EDF)
        ↓
Preprocessing (edf2pickle.py)
        ↓
Pickle Data
        ↓
Model Training (train.ipynb)
        ↓
Trained Model
        ↓
Evaluation (eval.ipynb)
```

---


