import streamlit as st
from use_chatbot import Chatbot  # Import the custom chatbot class

# Initialize Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the custom chatbot
custom_chatbot = Chatbot()

st.set_page_config(layout="wide")
st.title("Custom Chatbot Demo")

# Display conversation history
st.subheader("Conversation History")
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