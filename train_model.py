import os

from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_from_disk
from transformers import DataCollatorForSeq2Seq  # New import for the data collator

class ModelTrainer:
    def __init__(self, model_name="distilgpt2", output_dir="./model_output"):
        print("Initializing ModelTrainer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.output_dir = output_dir

        # Add padding token if not already present
        if self.tokenizer.pad_token is None:
            print("Tokenizer does not have a pad token. Adding a pad token...")
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))  # Resize model's token embeddings
        print("Model and tokenizer loaded successfully!")

    def train(self, dataset_path="./processed_data"):
        print("Loading the processed dataset...")
        dataset = load_from_disk(dataset_path)
        print("Dataset loaded successfully!")

        # Prepare the training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",  # Change to eval_strategy
            learning_rate=5e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            weight_decay=0.01,
            num_train_epochs=3,
            save_steps=10_000,
            save_total_limit=2,
            logging_dir="./logs",
            logging_steps=200,
        )

        # Initialize the data collator for Seq2Seq or CausalLM
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        # Initialize the Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],  # Assuming you have a test split
            data_collator=data_collator,  # Replacing the tokenizer argument
        )

        print("Starting the training process...")
        trainer.train()
        print("Training completed!")

        # Save the trained model
        self.save_model()

    def save_model(self):
        print(f"Saving the model to {self.output_dir}...")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print("Model and tokenizer saved successfully!")

    def load_model(self):
        print(f"Loading the model from {self.output_dir}...")
        model = AutoModelForCausalLM.from_pretrained(self.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
        print("Model and tokenizer loaded successfully!")
        return model, tokenizer
