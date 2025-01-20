import os
import PyPDF2
import docx
import pandas as pd
import spacy
import streamlit as st
from use_charbot import Chatbot  # Import the custom chatbot class

# Initialize Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the custom chatbot
custom_chatbot = Chatbot(model_path="./model_output")

# Load SciSpaCy model
nlp = spacy.load("en_core_sci_sm")

# Function to read PDF files
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to read DOCX files
def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to process text and extract medical terms
def process_text(text):
    doc = nlp(text)
    medical_terms = [ent.text for ent in doc.ents if ent.label_ in ["DISEASE", "TREATMENT", "TEST"]]
    return medical_terms

# Set Streamlit page layout
st.set_page_config(layout="wide")
st.title("Custom Chatbot & Medical Document Processor")

# Tabs for Chatbot and Document Processor
tab1, tab2 = st.tabs(["💬 Chatbot", "📄 Medical Document Processor"])

# Chatbot Tab
with tab1:
    st.subheader("Chatbot Conversation")
    
    # Display conversation history
    if st.session_state.messages:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Chatbot:** {message['content']}")
    
    # Input for user message
    user_input = st.text_input("Enter your message", key="user_input")
    
    # Handle user input and generate chatbot response
    if user_input:
        # Add the user's message to the conversation history
        st.session_state.messages.append({"role": "user", "content": user_input})
    
        # Generate a response using the custom chatbot
        inputs = custom_chatbot.tokenizer.encode(user_input, return_tensors="pt")
        output = custom_chatbot.model.generate(
            inputs,
            max_length=512,
            pad_token_id=custom_chatbot.tokenizer.pad_token_id
        )
        chatbot_response = custom_chatbot.tokenizer.decode(output[0], skip_special_tokens=True)
    
        # Add the chatbot's response to the conversation history
        st.session_state.messages.append({"role": "chatbot", "content": chatbot_response})
    
        # Display the chatbot's response
        st.markdown(f"**Chatbot:** {chatbot_response}")

# Document Processor Tab
with tab2:
    st.subheader("Medical Document Processor")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a document (PDF or DOCX)", type=["pdf", "docx"])
    
    if uploaded_file:
        # Extract text based on file type
        file_type = uploaded_file.name.split('.')[-1]
        if file_type == "pdf":
            text = read_pdf(uploaded_file)
        elif file_type == "docx":
            text = read_docx(uploaded_file)
        else:
            st.error("Unsupported file type!")
            st.stop()
        
        # Display extracted text
        st.subheader("Extracted Text")
        st.text_area("Document Content", text, height=300)
        
        # Process text to extract medical terms
        medical_terms = process_text(text)
        
        # Display extracted medical terms
        st.subheader("Extracted Medical Terms")
        st.write(medical_terms)
        
        # Export medical terms to CSV
        if st.button("Export Medical Terms to CSV"):
            csv_data = pd.DataFrame(medical_terms, columns=["Medical Terms"]).to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="medical_terms.csv",
                mime="text/csv"
            )
