import streamlit as st
import requests
from jose import jwt
import os
import logging
from io import StringIO

# Set up logging
log_stream = StringIO()
logging.basicConfig(stream=log_stream, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Backend URL
BACKEND_URL = "http://localhost:8000"

# Secret key for decoding JWT tokens
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

# Login/Signup Page
def login_signup_page():
    st.title("Login / Signup")
    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        st.header("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            response = requests.post(
                f"{BACKEND_URL}/login",
                data={"username": username, "password": password},
            )
            if response.status_code == 200:
                st.session_state["token"] = response.json()["access_token"]
                st.session_state["username"] = username
                st.success("Logged in successfully!")
            else:
                st.error("Invalid username or password")

    with tab2:
        st.header("Signup")
        new_username = st.text_input("Username", key="signup_username")
        new_password = st.text_input("Password", type="password", key="signup_password")
        role = st.selectbox("Role", ["user", "admin"], key="signup_role")
        if st.button("Signup"):
            response = requests.post(
                f"{BACKEND_URL}/signup",
                json={"username": new_username, "password": new_password, "role": role},
            )
            if response.status_code == 200:
                st.success("User created successfully! Please log in.")
            else:
                st.error("Error creating user")

# Main App
def main_app():
    st.title("AI-Powered Chatbot")

    # List uploaded documents
    try:
        uploaded_files = os.listdir("uploads")
        if uploaded_files:
            st.header("Select Documents")
            selected_documents = st.multiselect("Choose documents", uploaded_files)
        else:
            st.warning("No documents have been uploaded yet.")
            selected_documents = []
    except FileNotFoundError:
        st.warning("No documents have been uploaded yet.")
        selected_documents = []

    # Chat section
    st.header("Chat with the Document")
    query = st.text_input("Ask a question about the document:")
    if query:
        try:
            # Send the query and selected documents as a JSON payload
            payload = {"query": query, "selected_documents": selected_documents}
            logger.info(f"Sending payload: {payload}")
            response = requests.post(
                f"{BACKEND_URL}/chat",
                json=payload,
                headers={"Authorization": f"Bearer {st.session_state['token']}"},
            )
            logger.info(f"Backend response: {response.status_code}, {response.text}")
            
            if response.status_code == 200:
                logger.info(f"Response generated for query: {query}")
                st.write("**Response:**")
                st.write(response.json()["response"])
            else:
                error_message = response.json().get("detail", "Unknown error")
                logger.error(f"Error generating response for query '{query}': {error_message}")
                st.error(f"Error: {error_message}")
        except Exception as e:
            logger.error(f"Failed to connect to backend: {str(e)}")
            st.error(f"Failed to connect to backend: {str(e)}")

    # Admin-only features
    if st.session_state.get("role") == "admin":
        st.header("Admin Features")

        # File upload
        st.subheader("Upload a Document")
        uploaded_file = st.file_uploader("Choose a file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
        if uploaded_file:
            try:
                response = requests.post(
                    f"{BACKEND_URL}/upload",
                    files={"file": uploaded_file},
                    headers={"Authorization": f"Bearer {st.session_state['token']}"},
                )
                if response.status_code == 200:
                    st.success("File uploaded and processed successfully!")
                else:
                    error_message = response.json().get("detail", "Unknown error")
                    st.error(f"Error: {error_message}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {str(e)}")

        # File deletion
        st.subheader("Delete a Document")
        try:
            uploaded_files = os.listdir("uploads")  # List files in the uploads directory
            if uploaded_files:
                for filename in uploaded_files:
                    if st.button(f"Delete {filename}"):
                        # Encode the filename for the URL
                        encoded_filename = requests.utils.quote(filename)
                        response = requests.delete(
                            f"{BACKEND_URL}/delete/{encoded_filename}",
                            headers={"Authorization": f"Bearer {st.session_state['token']}"},
                        )
                        if response.status_code == 200:
                            st.success(f"File '{filename}' and its embeddings deleted successfully!")
                        else:
                            error_message = response.json().get("detail", "Unknown error")
                            st.error(f"Error: {error_message}")
        except FileNotFoundError:
            st.warning("No files have been uploaded yet.")

# App Flow
if "token" not in st.session_state:
    login_signup_page()
else:
    # Decode the token to get the user's role
    try:
        payload = jwt.decode(st.session_state["token"], SECRET_KEY, algorithms=[ALGORITHM])
        st.session_state["role"] = payload.get("role")
        main_app()
    except jwt.JWTError:
        st.error("Invalid token. Please log in again.")
        st.session_state.clear()

# Display logs in the Streamlit app
st.header("Application Logs")
st.text(log_stream.getvalue())
