import os
from transformers import AutoModelForCausalLM, AutoTokenizer

class Chatbot:
    def __init__(self, model_path="./model_output"):
        print("Initializing Chatbot...")

        # Load the trained model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Ensure the padding token is added if not present
        if self.tokenizer.pad_token is None:
            print("Tokenizer does not have a pad token. Adding a pad token...")
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))  # Resize model's token embeddings
        print("Model and tokenizer loaded successfully!")

    def chat(self):
        print("Chatbot is ready to chat! Type 'exit' to stop.")
        while True:
            # Get user input
            user_input = input("You: ")

            if user_input.lower() == "exit":
                print("Exiting the chatbot...")
                break

            # Tokenize the user input
            inputs = self.tokenizer.encode(user_input, return_tensors="pt")

            # Generate a response from the model
            output = self.model.generate(inputs, max_length=512, pad_token_id=self.tokenizer.pad_token_id)

            # Decode the response
            chatbot_response = self.tokenizer.decode(output[0], skip_special_tokens=True)

            # Print the response
            print("Chatbot:", chatbot_response)
