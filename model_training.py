import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, AdamW
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class PersonalityDisorderClassifier:
    def __init__(self, num_labels=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            self.optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            self.optimizer.step()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader, threshold=0.3):
        self.model.eval()
        val_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                loss = self.loss_fn(logits, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).int()  # Convert to 0/1
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
        
        return {
            'val_loss': val_loss / len(val_loader),
            'accuracy': accuracy,
            'f1_score': f1
        }
    
    def train(self, train_loader, val_loader, epochs=3):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val F1 Score: {val_metrics['f1_score']:.4f}")
            print("-" * 50)
        
        return self.model

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path, num_labels=3):
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    model.load_state_dict(torch.load(path))
    return model