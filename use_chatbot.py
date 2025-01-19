import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_dir = os.path.join(os.getcwd(), "model")

os.makedirs(model_dir, exist_ok=True)

try:
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2-1.5B-Instruct")
    print("Tokenizer initialized successfully.")

    # Load the base model and apply the PEFT model
    base_model = AutoModelForCausalLM.from_pretrained("unsloth/Qwen2-1.5B-Instruct")
    model = PeftModel.from_pretrained(base_model, "Muhammad7865253/qwen-1.5B-medical-QA")
    print("Model initialized!")
except Exception as e:
    print("Error while loading tokenizer or model:", e)
    exit(1)


class Chatbot:
    def __init__(self, device="cpu"):
        print("Initializing Chatbot...")

        # Load the trained model and tokenizer
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.sys_message = ''' 
          You are an AI Medical Assistant trained on a vast dataset of health information. Please be thorough and
        provide an informative answer. If you don't know the answer to a specific medical inquiry, advise seeking professional help.
        Don't mention any institution and keep your answer short and simple, as you will be talking with patients, not medical staff.
        Don't mention any medicine to the patients, talk only about diagnostics and next steps for the patient to take regarding their issue.   
        Don't include the credentials of the doctor and the way to contact him.
        '''

        # Ensure the padding token is added if not present
        if self.tokenizer.pad_token is None:
            print("Tokenizer does not have a pad token. Adding a pad token...")
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))  # Resize model's token embeddings
        print("Model and tokenizer loaded successfully!")

    def format_prompt(self, question):
        messages = [
            {"role": "system", "content": self.sys_message},
            {"role": "user", "content": question}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def generate_response(self, question, max_new_tokens=100):
        # Format prompt and tokenize
        prompt = self.format_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
            self.device)

        # Generate response from the model with attention mask
        with torch.no_grad():
            output = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,  # Use max_new_tokens instead of max_length
                pad_token_id=self.tokenizer.pad_token_id,
                attention_mask=inputs["attention_mask"],
                use_cache=True
            )

        # Decode and return the response
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response.strip()

    def chat(self):
        print("Chatbot is ready to chat! Type 'exit' to stop.")
        while True:
            # Get user input
            user_input = input("You: ")

            if user_input.lower() == "exit":
                print("Exiting the chatbot...")
                break

            print("Generating response...")
            response = self.generate_response(user_input)
            print("Chatbot:", response)


if __name__ == "__main__":
    assistant = Chatbot()
    print("Initialized the chatbot assistant...")
    assistant.chat()
