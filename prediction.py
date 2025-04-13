import torch
from transformers import BertTokenizer
import numpy as np
from model_training import load_model
import pandas as pd
from tqdm import tqdm

class PersonalityDisorderPredictor:
    def __init__(self, model_path, threshold=0.3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.threshold = threshold
        self.disorders = ["Schizoid", "Narcissistic", "Avoidant"]
    
    def preprocess_text(self, text, max_len=256):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return encoding
    
    def predict_file(self, input_file, output_file):
        """Predict on an entire file and save results"""
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            conversations = [line.strip() for line in f if line.strip()]
        
        results = []
        for conv in tqdm(conversations, desc="Predicting"):
            if conv.startswith('['):  # Skip label lines if present
                continue
            result = self.predict(conv)
            results.append({
                'conversation': conv,
                'predictions': result['predictions'],
                'probabilities': result['probabilities'],
                'suggestion': result['suggestion']
            })
        
        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        return df
    
    def generate_suggestion(self, predictions, probs):
        """Generate human-readable suggestion based on predictions"""
        detected = [self.disorders[i] for i, pred in enumerate(predictions) if pred]
        
        if not detected:
            return "No significant personality disorder traits detected in this conversation."
        
        suggestion = "The conversation shows potential traits of "
        suggestion += ", ".join(detected) + " personality disorder(s). "
        suggestion += "Specific probabilities: "
        suggestion += "; ".join([f"{self.disorders[i]}: {probs[i]*100:.1f}%" for i in range(3)])
        suggestion += ". Consider professional evaluation if these patterns persist."
        
        return suggestion

# Example usage
if __name__ == "__main__":
    predictor = PersonalityDisorderPredictor("personality_disorder_model.pth")
    
    test_conversation = """
    A: I had to swing by the hospital today—nothing serious, just a quick check-up. 
    You should've seen how everyone in the waiting room was glaring at me. 
    They're all so jealous of how I carry myself, even there.
    B: Oh, glad it wasn't serious! I've been there a lot lately—my friend's recovering from surgery.
    A: Yeah, well, she's probably just soaking up all the pity she can get. 
    Meanwhile, I walked in and out like it was nothing—people can't stand how I don't need to lean on anyone.
    """
    
    result = predictor.predict(test_conversation)
    print("Prediction Results:")
    print(f"Probabilities: {result['probabilities']}")
    print(f"Predictions: {result['predictions']}")
    print(f"Suggestion: {result['suggestion']}")