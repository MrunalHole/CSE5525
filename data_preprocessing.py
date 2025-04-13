import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import json
import re

class PersonalityDisorderDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }

def load_custom_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        # Fallback for different encodings
        with open(filepath, 'r', encoding='latin1') as f:
            lines = f.readlines()
    
    data = []
    for line in lines:
        # Extract label and conversation
        match = re.match(r'^\[([01]), ([01]), ([01])\] (.*)', line.strip())
        if match:
            label = [int(match.group(1)), int(match.group(2)), int(match.group(3))]
            conversation = match.group(4)
            data.append({'text': conversation, 'label': label})
    
    df = pd.DataFrame(data)
    texts = df['text'].values
    labels = np.array(df['label'].tolist())
    
    return train_test_split(texts, labels, test_size=0.2, random_state=42)

def create_data_loaders(train_texts, val_texts, train_labels, val_labels, batch_size=16):
    """Create PyTorch data loaders"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_dataset = PersonalityDisorderDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_len=256
    )
    
    val_dataset = PersonalityDisorderDataset(
        texts=val_texts,
        labels=val_labels,
        tokenizer=tokenizer,
        max_len=256
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )
    
    return train_loader, val_loader