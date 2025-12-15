# data_singleword.py
import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
import random

def normalize_1d(input_tensor):
    """Normalize a 1D tensor."""
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    return (input_tensor - mean) / (std + 1e-8)

def get_input_sample(word_obj, tokenizer, eeg_type='GD', 
                     bands=['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], 
                     max_len=56, add_CLS_token=False, test_input='noise'):
    """
    Adapted version of original ZuCo's get_input_sample for single-word data.
    word_obj: Dictionary containing the word and its EEG data.
    """
    if word_obj is None:
        return None
    
    def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):
        """Extract word-level EEG features."""
        frequency_features = []
        
        # Get EEG data
        eeg_data = word_obj['word_level_EEG']
        
        # Check if requested eeg_type exists, fall back to GD if not
        if eeg_type not in eeg_data:
            if 'GD' in eeg_data:
                eeg_type = 'GD'
            elif 'FFD' in eeg_data:
                eeg_type = 'FFD'
            elif 'TRT' in eeg_data:
                eeg_type = 'TRT'
            else:
                eeg_type = 'raw_eeg'
        
        type_data = eeg_data[eeg_type]
        
        # Handle raw_eeg differently
        if eeg_type == 'raw_eeg':
            # Flatten raw EEG and split into bands
            flattened = type_data.flatten()
            target_size = 105 * len(bands)
            
            if len(flattened) > target_size:
                flattened = flattened[:target_size]
            else:
                padding = target_size - len(flattened)
                flattened = np.pad(flattened, (0, padding), mode='constant')
            
            # Split into equal parts for each band
            for i in range(len(bands)):
                start = i * 105
                end = (i + 1) * 105
                frequency_features.append(flattened[start:end])
        else:
            # For GD/FFD/TRT - extract band features
            for band in bands:
                # Try to find mean feature for this band
                band_key = 'mean' + band
                
                if band_key in type_data:
                    feature = type_data[band_key]
                else:
                    # Try other features or use zeros
                    feature = np.zeros(65)  # Your features are (65,) per channel
                
                # Pad from 65 to 105 features
                feature = feature.flatten()
                if len(feature) < 105:
                    padding = 105 - len(feature)
                    feature = np.pad(feature, (0, padding), mode='constant')
                elif len(feature) > 105:
                    feature = feature[:105]
                
                frequency_features.append(feature)
        
        # Concatenate all band features
        word_eeg_embedding = np.concatenate(frequency_features)
        
        # Ensure correct dimension
        target_size = 105 * len(bands)
        if len(word_eeg_embedding) != target_size:
            if len(word_eeg_embedding) > target_size:
                word_eeg_embedding = word_eeg_embedding[:target_size]
            else:
                padding = target_size - len(word_eeg_embedding)
                word_eeg_embedding = np.pad(word_eeg_embedding, (0, padding), mode='constant')
        
        return_tensor = torch.from_numpy(word_eeg_embedding).float()
        return normalize_1d(return_tensor)
    
    def get_sent_eeg(word_obj, bands):
        """For single-word data, sentence EEG is same as word EEG."""
        return get_word_embedding_eeg_tensor(word_obj, eeg_type, bands)
    
    # Start creating input sample
    input_sample = {}
    
    # Get target word and tokenize
    target_string = word_obj['content']
    target_tokenized = tokenizer(
        target_string, 
        padding='max_length', 
        max_length=max_len, 
        truncation=True, 
        return_tensors='pt', 
        return_attention_mask=True
    )
    input_sample['target_ids'] = target_tokenized['input_ids'][0]
    
    # Get sentence level EEG features
    sent_level_eeg_tensor = get_sent_eeg(word_obj, bands)
    
    if torch.isnan(sent_level_eeg_tensor).any():
        return None
    
    input_sample['sent_level_EEG'] = sent_level_eeg_tensor
    
    # Get input embeddings (word-level EEG)
    word_embeddings = []
    
    # Add CLS token if requested
    if add_CLS_token:
        word_embeddings.append(torch.zeros(105 * len(bands)))
    
    # Get word EEG features
    word_level_eeg_tensor = get_word_embedding_eeg_tensor(word_obj, eeg_type, bands)
    
    if word_level_eeg_tensor is None or torch.isnan(word_level_eeg_tensor).any():
        return None
    
    word_embeddings.append(word_level_eeg_tensor)
    
    # Test input option (for debugging)
    if test_input == 'noise':
        word_embeddings = [torch.randn_like(emb) for emb in word_embeddings]
    
    # Pad to max_len
    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(105 * len(bands)))
    
    input_sample['input_embeddings'] = torch.stack(word_embeddings)
    
    # Create attention masks
    input_sample['input_attn_mask'] = torch.zeros(max_len)
    actual_words = 1  # Single word data
    
    if add_CLS_token:
        input_sample['input_attn_mask'][:actual_words + 1] = torch.ones(actual_words + 1)
    else:
        input_sample['input_attn_mask'][:actual_words] = torch.ones(actual_words)
    
    # Inverted attention mask (for transformers)
    input_sample['input_attn_mask_invert'] = torch.ones(max_len)
    
    if add_CLS_token:
        input_sample['input_attn_mask_invert'][:actual_words + 1] = torch.zeros(actual_words + 1)
    else:
        input_sample['input_attn_mask_invert'][:actual_words] = torch.zeros(actual_words)
    
    # Target mask
    input_sample['target_mask'] = target_tokenized['attention_mask'][0]
    input_sample['seq_len'] = actual_words
    
    # Sentiment label (dummy for compatibility)
    input_sample['sentiment_label'] = torch.tensor(-100)
    
    return input_sample

class ZuCo_dataset(Dataset):
    """
    Exact same class as original ZuCo_dataset.
    Compatible with your single-word EEG data.
    """
    
    def __init__(self, input_dataset_dicts, phase, tokenizer, subject='ALL', 
                 eeg_type='GD', bands=['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], 
                 setting='unique_sent', is_add_CLS_token=False, test_input='noise'):
        
        self.inputs = []
        self.tokenizer = tokenizer
        
        # Convert to list if single dataset
        if not isinstance(input_dataset_dicts, list):
            input_dataset_dicts = [input_dataset_dicts]
        
        print(f'[INFO] Loading {len(input_dataset_dicts)} task datasets')
        
        # Process each dataset
        for dataset in input_dataset_dicts:
            self._process_dataset(dataset, phase, subject, eeg_type, 
                                bands, setting, is_add_CLS_token, test_input)
        
        if len(self.inputs) > 0:
            print(f'[INFO] Input tensor size: {self.inputs[0]["input_embeddings"].size()}')
        else:
            print('[WARNING] No samples loaded!')
        print()
    
    def _process_dataset(self, dataset_dict, phase, subject, eeg_type, 
                        bands, setting, add_CLS_token, test_input):
        """Process dataset (original ZuCo format or your single-word format)."""
        # Get subjects
        if subject == 'ALL':
            subjects = list(dataset_dict.keys())
            print('[INFO] Using subjects: ', subjects)
        else:
            subjects = [subject]
            print('[INFO] Using subject: ', subject)
        
        # Count samples per subject
        total_num_samples = len(dataset_dict[subjects[0]])
        
        # Determine split points (80% train, 10% dev, 10% test)
        train_divider = int(0.8 * total_num_samples)
        dev_divider = train_divider + int(0.1 * total_num_samples)
        
        print(f'[INFO] Train divider = {train_divider}, Dev divider = {dev_divider}')
        
        if setting == 'unique_sent':
            # Split by samples within each subject
            if phase == 'train':
                print('[INFO] Initializing train set...')
                for key in subjects:
                    for i in range(train_divider):
                        sample = get_input_sample(
                            dataset_dict[key][i], self.tokenizer, eeg_type,
                            bands, add_CLS_token=add_CLS_token, test_input=test_input
                        )
                        if sample is not None:
                            self.inputs.append(sample)
            
            elif phase == 'dev':
                print('[INFO] Initializing dev set...')
                for key in subjects:
                    for i in range(train_divider, dev_divider):
                        sample = get_input_sample(
                            dataset_dict[key][i], self.tokenizer, eeg_type,
                            bands, add_CLS_token=add_CLS_token, test_input=test_input
                        )
                        if sample is not None:
                            self.inputs.append(sample)
            
            elif phase == 'test':
                print('[INFO] Initializing test set...')
                for key in subjects:
                    for i in range(dev_divider, total_num_samples):
                        sample = get_input_sample(
                            dataset_dict[key][i], self.tokenizer, eeg_type,
                            bands, add_CLS_token=add_CLS_token, test_input=test_input
                        )
                        if sample is not None:
                            self.inputs.append(sample)
        
        elif setting == 'unique_subj':
            # Split by subjects
            print('[INFO] Using unique_subj setting...')
            
            all_subjects = list(dataset_dict.keys())
            random.shuffle(all_subjects)
            
            train_subjs = all_subjects[:int(0.8 * len(all_subjects))]
            dev_subjs = all_subjects[int(0.8 * len(all_subjects)):int(0.9 * len(all_subjects))]
            test_subjs = all_subjects[int(0.9 * len(all_subjects)):]
            
            if phase == 'train':
                subjects_to_use = train_subjs
            elif phase == 'dev':
                subjects_to_use = dev_subjs
            elif phase == 'test':
                subjects_to_use = test_subjs
            else:
                return
            
            print(f'[INFO] Processing {phase} subjects: {subjects_to_use}')
            
            for key in subjects_to_use:
                for i in range(len(dataset_dict[key])):
                    sample = get_input_sample(
                        dataset_dict[key][i], self.tokenizer, eeg_type,
                        bands, add_CLS_token=add_CLS_token, test_input=test_input
                    )
                    if sample is not None:
                        self.inputs.append(sample)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        return (
            input_sample['input_embeddings'], 
            input_sample['seq_len'],
            input_sample['input_attn_mask'], 
            input_sample['input_attn_mask_invert'],
            input_sample['target_ids'], 
            input_sample['target_mask'], 
            input_sample['sentiment_label'],
            # input_sample['sent_level_EEG']  # Uncomment if needed
        )