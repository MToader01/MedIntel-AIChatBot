import streamlit as st
from use_chatbot import Chatbot  # Import the custom chatbot class

# Initialize Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Check if the chatbot has already been initialized, if not, initialize it
if "custom_chatbot" not in st.session_state:
    st.session_state.custom_chatbot = Chatbot()  # Initialize only once

custom_chatbot = st.session_state.custom_chatbot

st.set_page_config(layout="wide")

# Center the title and change font size using custom CSS
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
        }
        .input-label {
            font-size: 24px;
        }
    </style>
""", unsafe_allow_html=True)

# Title with custom CSS
st.markdown('<h1 class="title">Anna - MedIntel AI Chatbot</h1>', unsafe_allow_html=True)

# Display conversation history
st.subheader("Conversation History")
if st.session_state.messages:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"You : {message['content']}")
        else:
            st.markdown(f"Anna : {message['content']}")

# Input for user message with larger label
st.markdown('<p class="input-label">Ask a question :</p>', unsafe_allow_html=True)
user_input = st.text_input("Enter your question...", key="user_input", label_visibility="hidden")

# Handle user input and generate chatbot response
if user_input:
    # Add the user's message to the conversation history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Create a placeholder for the "Generating..." message
    placeholder = st.empty()
    placeholder.markdown("**Anna : Thinking...**")

    # Generate a response using the custom chatbot
    chatbot_response = custom_chatbot.generate_response(user_input)

    # Remove the "Generating..." message and display the chatbot's response
    placeholder.empty()
    st.session_state.messages.append({"role": "chatbot", "content": chatbot_response})

    # Display the chatbot's response with larger font size
    st.markdown(f"<h3 style='font-size: 24px;'>Anna : {chatbot_response}</h3>", unsafe_allow_html=True)
