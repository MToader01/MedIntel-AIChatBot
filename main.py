import os
from datasets import  load_from_disk
from train_model import ModelTrainer
from data_processing import DataProcessor
from use_chatbot import Chatbot

if __name__ == "__main__":
    # Initialize the DataProcessor
    processor = DataProcessor()

    # Check if processed data exists
    processed_data_path = "./processed_data"

    if os.path.exists(processed_data_path):
        print(f"Processed data found at {processed_data_path}, loading it...")
        # Load the already processed data
        tokenized_data = load_from_disk(processed_data_path)  # Corrected line
    else:
        print("Processed data not found, processing and saving...")
        # Load the dataset
        dataset = processor.load_data()
        # Process and save the data
        tokenized_data = processor.process_data(dataset)

    # Initialize the ModelTrainer
    trainer = ModelTrainer(model_name="distilgpt2", output_dir="./model_output")

    # Train the model using the processed data
    trainer.train(dataset_path=processed_data_path)

    # Save the trained model
    trainer.save_model()

    # Load the model for further use
    model, tokenizer = trainer.load_model()

    # The model is now ready for use
    print("Model is ready for use!")

    chatbot = Chatbot(model_path="./model_output")

    # Start chatting
    chatbot.chat()