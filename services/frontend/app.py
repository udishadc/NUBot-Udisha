import streamlit as st
import requests
import os
from dotenv import load_dotenv
import uuid

load_dotenv(override=True)

API_URL = os.getenv('API_URL')

# Initialize session state for chat history
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.started = False
    st.session_state.chat_history = []

# If the user hasn't clicked Start
if not st.session_state.started:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>🤖 NuBot</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Your assistant for all things Northeastern!</p>", unsafe_allow_html=True)
        st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
        if st.button("Start", use_container_width=True):
            st.session_state.started = True
            st.rerun()  # Re-render to show chat interface
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()  # Prevent further execution until start is clicked

# Chat interface
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center;'>🤖 NuBot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Your assistant for all things Northeastern!</p>", unsafe_allow_html=True)

    # Input field and send button
    query = st.text_input("Type your message here")

    if st.button("Send"):
        if query:
            try:
                # Store the user query immediately
                st.session_state.chat_history.append(["user", query])
                
                # Make API request
                response = requests.post(API_URL, json={"query": query, "session_id": st.session_state.session_id})
                if response.status_code == 200:
                    # Try different possible formats for the response
                    try:
                        response_data = response.json()
                        # Check if response is directly a string
                        if isinstance(response_data, str):
                            bot_reply = response_data
                        # Check if response has a 'response' field
                        elif isinstance(response_data, dict) and "response" in response_data:
                            bot_reply = response_data["response"]
                        # Just take the raw JSON as a fallback
                        else:
                            bot_reply = str(response_data)
                    except Exception as e:
                        bot_reply = "Sorry, I received an invalid response format."
                        st.error(f"Failed to parse response: {str(e)}")
                    
                    st.session_state.chat_history.append(["bot", bot_reply])
                else:
                    error_msg = f"Error {response.status_code}: {response.text}"
                    st.session_state.chat_history.append(["bot", f"Sorry, I encountered an error: {error_msg}"])
                    st.error(error_msg)
            except requests.exceptions.RequestException as e:
                error_msg = f"Request failed: {e}"
                st.session_state.chat_history.append(["bot", f"Sorry, I encountered a connection error: {error_msg}"])
                st.error(error_msg)
        else:
            st.warning("Please enter a query")

    # Display chat history
    for sender, message in st.session_state.chat_history:
        if sender == "user":
            st.markdown(f"🧑‍💻 **You**: {message}")
        else:
            st.markdown(f"🤖 **NuBot**: {message}")