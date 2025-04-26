from data_preprocessing import load_gemini_data, create_data_loaders
from model_training import PersonalityDisorderClassifier, save_model
from prediction import PersonalityDisorderPredictor

def main():
    # Step 1: Load and preprocess data
    print("Loading data...")
    train_texts, test_texts, train_labels, test_labels = load_gemini_data(
        "data/gemini_data.txt", 
        "data/gemini_prompt.txt"
    )
    
    # Step 2: Create data loaders
    train_loader, test_loader = create_data_loaders(
        train_texts, test_texts, train_labels, test_labels
    )
    
    # Step 3: Train model (unchanged)
    print("Training model...")
    classifier = PersonalityDisorderClassifier()
    model = classifier.train(train_loader, test_loader, epochs=10)  # Using test_loader for validation
    
    # Save model
    save_model(model, "personality_disorder_model.pth")

if __name__ == "__main__":
    main()





# from data_preprocessing import load_custom_data, create_data_loaders
# from model_training import PersonalityDisorderClassifier, save_model
# from prediction import PersonalityDisorderPredictor
# import os

# def main():
#     # Step 1: Preprocess data
#     print("\nPreprocessing data...")
#     train_texts, val_texts, train_labels, val_labels = load_custom_data("data/gemini.txt")
#     train_loader, val_loader = create_data_loaders(train_texts, val_texts, train_labels, val_labels)
    
#     # Step 2: Train model
#     print("\nTraining model...")
#     classifier = PersonalityDisorderClassifier()
#     model = classifier.train(train_loader, val_loader, epochs=3)
    
#     # Step 3: Save model
#     model_path = "personality_disorder_model.pth"
#     save_model(model, model_path)
#     print("\nModel saved successfully!")
    
#     # Step 4: Example prediction
#     predictor = PersonalityDisorderPredictor(model_path)
#     test_files = ["data/gemini.txt"]  # Using same file for testing
#     for test_file in test_files:
#         if os.path.exists(test_file):
#             output_file = test_file.replace('.txt', '_predictions.csv')
#             print(f"\nProcessing {test_file}...")
#             predictor.predict_file(test_file, output_file)
#         else:
#             print(f"\nTest file not found: {test_file}")

# if __name__ == "__main__":
#     main()