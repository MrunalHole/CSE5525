from data_preprocessing import load_custom_data, create_data_loaders
from model_training import PersonalityDisorderClassifier, save_model
from prediction import PersonalityDisorderPredictor
import os

def main():
       
    # Step 1: Preprocess data
    print("\nPreprocessing data...")
    train_texts, val_texts, train_labels, val_labels = load_custom_data("data/gpt4-prompt-conversation-80.txt")
    train_loader, val_loader = create_data_loaders(train_texts, val_texts, train_labels, val_labels)
    
    # Step 2: Train model
    print("\nTraining model...")
    classifier = PersonalityDisorderClassifier()
    model = classifier.train(train_loader, val_loader, epochs=3)
    
    # Step 3: Save model
    model_path = "personality_disorder_model_1.pth"
    save_model(model, model_path)
    print("\nModel saved successfully!")
    
    # Step 4: Example prediction
    predictor = PersonalityDisorderPredictor(model_path)
    test_files = [
        "data/grok_ds_avoidant_4_light_100.txt",
        ]
    for test_file in test_files:
        if os.path.exists(test_file):
            output_file = test_file.replace('.txt', '_predictions.csv')
            print(f"\nProcessing {test_file}...")
            predictor.predict_file(test_file, output_file)
        else:
            print(f"\nTest file not found: {test_file}")
    # result = predictor.predict(test_files)
    
    # print("\nExample Prediction:")
    # print(f"Input: {test_files}")
    # print(f"Results: {result}")

if __name__ == "__main__":
    main()