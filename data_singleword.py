import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

# ----------------------------
# Normalization
# ----------------------------
def normalize_1d(x: torch.Tensor):
    mean = x.mean()
    std = x.std()
    if std < 1e-6:
        return x
    return (x - mean) / (std + 1e-8)

# ----------------------------
# EEG feature extraction
# ----------------------------
def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):
    """
    Extract word-level EEG features
    OUTPUT SHAPE: (channels × bands)
    """
    eeg_data = word_obj["word_level_EEG"]

    # fallback EEG type
    if eeg_type not in eeg_data:
        eeg_type = next(iter(eeg_data.keys()))

    type_data = eeg_data[eeg_type]
    band_features = []

    if eeg_type == "raw_eeg":
        # raw_eeg: (channels, time)
        x = torch.from_numpy(type_data).float()
        x = x.flatten()
        return normalize_1d(x)

    # GD / FFD / TRT
    for band in bands:
        band_key = "mean" + band
        if band_key not in type_data:
            raise ValueError(f"Missing band feature: {band_key}")

        feature = torch.from_numpy(type_data[band_key]).float()
        feature = feature.flatten()   # KEEP ORIGINAL CHANNEL COUNT
        band_features.append(feature)

    x = torch.cat(band_features, dim=0)
    return normalize_1d(x)

# ----------------------------
# Single-word sample builder
# ----------------------------
def get_input_sample(
    word_obj,
    tokenizer,
    eeg_type="GD",
    bands=["_t1","_t2","_a1","_a2","_b1","_b2","_g1","_g2"],
    max_len=2,
    add_CLS_token=False,
    test_input=None,
):
    if word_obj is None:
        return None

    # ----------------------------
    # Target token
    # ----------------------------
    target_string = word_obj["content"]
    target_tokenized = tokenizer(
        target_string,
        padding="max_length",
        max_length=max_len,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    # ----------------------------
    # EEG features
    # ----------------------------
    eeg_tensor = get_word_embedding_eeg_tensor(word_obj, eeg_type, bands)

    if torch.isnan(eeg_tensor).any():
        return None

    feature_dim = eeg_tensor.shape[0]

    # ----------------------------
    # Build token sequence
    # ----------------------------
    word_embeddings = []

    if add_CLS_token:
        word_embeddings.append(torch.zeros(feature_dim))

    if test_input == "noise":
        word_embeddings.append(torch.randn(feature_dim))
    else:
        word_embeddings.append(eeg_tensor)

    # pad tokens (NOT channels)
    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(feature_dim))

    input_embeddings = torch.stack(word_embeddings)  # (seq_len, feature_dim)

    # ----------------------------
    # Attention masks
    # ----------------------------
    input_attn_mask = torch.zeros(max_len)
    actual_tokens = 1 + int(add_CLS_token)
    input_attn_mask[:actual_tokens] = 1

    input_attn_mask_invert = 1 - input_attn_mask

    return {
        "input_embeddings": input_embeddings,
        "input_attn_mask": input_attn_mask,
        "input_attn_mask_invert": input_attn_mask_invert,
        "target_ids": target_tokenized["input_ids"][0],
        "target_mask": target_tokenized["attention_mask"][0],
        "seq_len": 1,
        "sentiment_label": torch.tensor(-100),
    }

# ----------------------------
# Dataset
# ----------------------------
class otago_dataset(Dataset):
    def __init__(
        self,
        input_dataset_dicts,
        phase,
        tokenizer,
        subject="ALL",
        eeg_type="GD",
        bands=["_t1","_t2","_a1","_a2","_b1","_b2","_g1","_g2"],
        setting="unique_sent",
        is_add_CLS_token=False,
        test_input=None,
    ):
        self.inputs = []
        self.tokenizer = tokenizer

        if not isinstance(input_dataset_dicts, list):
            input_dataset_dicts = [input_dataset_dicts]

        print(f"[INFO] Loading {len(input_dataset_dicts)} dataset(s)")

        for dataset in input_dataset_dicts:
            self._process_dataset(
                dataset,
                phase,
                subject,
                eeg_type,
                bands,
                setting,
                is_add_CLS_token,
                test_input,
            )

        if len(self.inputs) > 0:
            print("[INFO] Input tensor size:", self.inputs[0]["input_embeddings"].shape)
        print(f"[INFO] Total samples loaded: {len(self.inputs)}\n")

    def _process_dataset(
        self,
        dataset_dict,
        phase,
        subject,
        eeg_type,
        bands,
        setting,
        add_CLS_token,
        test_input,
    ):
        subjects = list(dataset_dict.keys()) if subject == "ALL" else [subject]
        total_num = len(dataset_dict[subjects[0]])

        train_div = int(0.8 * total_num)
        dev_div = int(0.9 * total_num)

        for subj in subjects:
            if phase == "train":
                indices = range(0, train_div)
            elif phase == "dev":
                indices = range(train_div, dev_div)
            else:
                indices = range(dev_div, total_num)

            for i in indices:
                sample = get_input_sample(
                    dataset_dict[subj][i],
                    self.tokenizer,
                    eeg_type,
                    bands,
                    add_CLS_token=add_CLS_token,
                    test_input=test_input,
                )
                if sample is not None:
                    self.inputs.append(sample)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        s = self.inputs[idx]
        return (
            s["input_embeddings"],
            s["seq_len"],
            s["input_attn_mask"],
            s["input_attn_mask_invert"],
            s["target_ids"],
            s["target_mask"],
            s["sentiment_label"],
        )
