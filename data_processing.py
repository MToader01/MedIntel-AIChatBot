import os
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer


class DataProcessor:
    def __init__(self, tokenizer_name="distilgpt2"):
        print("Initializing DataProcessor...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Add a padding token if not present
        if self.tokenizer.pad_token is None:
            print("Tokenizer does not have a pad token. Adding a pad token...")
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # Set model max length to avoid too long inputs
            self.tokenizer.model_max_length = 512  # Optionally, set a max length
            print("Padding token added without resizing the embeddings!")
        print("Tokenizer loaded successfully!")

    def load_data(self, dataset_name="ruslanmv/ai-medical-chatbot"):
        print(f"Loading dataset: {dataset_name}...")
        dataset = load_dataset(dataset_name)
        print("Dataset loaded successfully!")
        return dataset

    def process_data(self, dataset):
        # Combine relevant columns for tokenization
        print("Combining columns...")
        dataset = dataset.map(self._combine_columns, batched=True)  # `batched=True` for efficiency
        print("Columns combined successfully!")

        # Tokenize the dataset
        print("Tokenizing dataset...")
        def tokenize_function(examples):
            tokenized = self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
            tokenized["labels"] = tokenized["input_ids"]  # Set labels as the same as input_ids
            return tokenized

        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["Description", "Patient", "Doctor"])
        print("Tokenization complete!")

        # Split into train and test sets (10% for testing)
        print("Splitting dataset into train and test...")
        tokenized_datasets = tokenized_datasets["train"].train_test_split(test_size=0.1)
        print("Dataset split into train and test!")

        # Save the processed data
        self.save_processed_data(tokenized_datasets)
        return tokenized_datasets

    def save_processed_data(self, tokenized_datasets, save_path="./processed_data"):
        # Check if the folder exists, if not, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print(f"Saving processed data to {save_path}...")
        # Save the dataset in parquet format
        tokenized_datasets.save_to_disk(save_path)
        print("Processed data saved successfully!")

    @staticmethod
    def _combine_columns(batch):
        # Combine `Description`, `Patient`, and `Doctor` into a single string for tokenization
        texts = [
            f"Description: {desc} Patient: {pat} Doctor: {doc}"
            for desc, pat, doc in zip(batch['Description'], batch['Patient'], batch['Doctor'])
        ]
        # Return as a dictionary
        return {"text": texts}
