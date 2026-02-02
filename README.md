EEG-to-Word Decoding

This repository provides the implementation for EEG-to-Word decoding using deep learning models trained on single-word EEG data. It includes scripts for converting raw EEG files into pickle format, training decoding models, and evaluating performance.

Repository Structure
.
├── data_singleword.py     # Data loading and preprocessing
├── model_decoding.py      # Model architecture
├── edf2pickle.py          # Convert raw EEG (.edf) files to pickle format
├── train_config_t5.json   # Training configuration file
├── train.ipynb            # Training notebook 
├── eval.ipynb             # Evaluation notebook 
└── README.md

Step 1: Download the dataset from Hugging Face:
https://huggingface.co/datasets/peytonmou/EEG2Text

Using Git:
git lfs install
git clone https://huggingface.co/datasets/peytonmou/EEG2Text

Step 2: Clone the code repository:
git clone https://github.com/peytonmou/EEG-to-Word_Decoding.git
cd EEG-to-Word_Decoding
Place the dataset folder in the same directory as this repository:
├── EEG-to-Word_Decoding
│   ├── train.py
│   ├── eval.py
│   ├── edf2pickle.py
│   └── model_decoding.py
└── EEG2Text   ← dataset folder

Step 3: Convert EDF Files to Pickle
python edf2pickle.py
This will convert raw EEG files into serialized .pkl files for faster loading and training.

Step 4: Train the Model
Before training, update the file paths in train.py:

Path to the generated pickle files
Folder path to save trained models

Then run:
python train.py

Step 5: Evaluate the Model
Before evaluation, update the model loading path in eval.py to point to the saved model:
Then run:
python eval.py
The evaluation results can be reproduced using the provided dataset and trained model.

Recommended environment:
Python ≥ 3.8
PyTorch
NumPy
SciPy
scikit-learn

pip install torch numpy scipy scikit-learn
