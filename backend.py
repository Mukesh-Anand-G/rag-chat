from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import Optional, List
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.http.models import Filter, PointIdsList  # Import for Qdrant delete
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import uuid
import logging
from urllib.parse import unquote

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Security settings
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Fake user database
fake_users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("adminpassword"),
        "role": "admin",
    },
    "user1": {
        "username": "user1",
        "hashed_password": pwd_context.hash("password1"),
        "role": "user",
    },
}

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize embedding model
embedding_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# Initialize Qdrant client
client = qdrant_client.QdrantClient("localhost", port=6333)

# Initialize Ollama LLM
llm = Ollama(model="gemma2:2b")

# Create a collection in Qdrant (if it doesn't exist)
COLLECTION_NAME = "documents"
try:
    client.get_collection(COLLECTION_NAME)
except:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "size": 1024,  # mxbai-embed-large embedding size
            "distance": "Cosine",  # Similarity metric
        },
    )

# Directory to store uploaded documents
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# User model
class User(BaseModel):
    username: str
    password: str
    role: str  # "admin" or "user"

# Define ChatRequest model
class ChatRequest(BaseModel):
    query: str
    selected_documents: List[str]  # List of selected document filenames

# Function to extract text from files
def extract_text(file: UploadFile):
    if file.filename.endswith(".pdf"):
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(file.file)
        text = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
    elif file.filename.endswith(".docx"):
        import docx
        doc = docx.Document(file.file)
        text = "\n".join(p.text for p in doc.paragraphs)
    elif file.filename.endswith(".txt"):
        text = file.file.read().decode("utf-8")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    return text

# Function to split text into chunks
def split_text(text: str, chunk_size: int = 500):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Authenticate user
def authenticate_user(username: str, password: str):
    user = fake_users_db.get(username)
    if not user or not pwd_context.verify(password, user["hashed_password"]):
        return False
    return user

# Create access token
def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Get current user
def get_current_user(credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        role = payload.get("role")
        if username is None or role is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"username": username, "role": role}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Require admin role
def require_admin(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

# Signup endpoint
@app.post("/signup")
async def signup(user: User):
    if user.username in fake_users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    fake_users_db[user.username] = {
        "username": user.username,
        "hashed_password": pwd_context.hash(user.password),
        "role": user.role,
    }
    return {"message": "User created successfully"}

# Login endpoint
@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Upload endpoint
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), current_user: dict = Depends(require_admin)):
    try:
        # Extract text from the file
        text = extract_text(file)
        chunks = split_text(text)

        # Generate embeddings for each chunk
        embeddings = embedding_model.encode(chunks)

        # Store embeddings in Qdrant
        points = [
            {
                "id": str(uuid.uuid4()),
                "vector": embedding.tolist(),
                "payload": {"text": chunk, "filename": file.filename},
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)

        # Save the original file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        logger.info(f"File '{file.filename}' uploaded and processed successfully.")
        return JSONResponse({"message": "File uploaded and processed successfully"})
    except Exception as e:
        logger.error(f"Error processing file '{file.filename}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        query = request.query
        selected_documents = request.selected_documents
        logger.info(f"Received query: {query} with selected documents: {selected_documents}")

        # Generate embedding for the query
        query_embedding = embedding_model.encode([query])[0]

        # Search Qdrant for relevant chunks within selected documents
        search_results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            limit=5,
            query_filter={
                "must": [
                    {"key": "filename", "match": {"any": selected_documents}}
                ]
            } if selected_documents else None
        )

        # Extract relevant text chunks
        relevant_chunks = [result.payload["text"] for result in search_results]

        if not relevant_chunks:
            return JSONResponse({"response": "I don't know."})

        # Generate response using Ollama
        prompt_template = PromptTemplate(
            input_variables=["context", "query"],
            template="Context: {context}\n\nQuestion: {query}\n\nAnswer:",
        )
        chain = LLMChain(llm=llm, prompt=prompt_template)
        response = chain.run(context="\n".join(relevant_chunks), query=query)

        logger.info(f"Response generated for query: {query}")
        return JSONResponse({"response": response})
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Delete endpoint
@app.delete("/delete/{filename}")
async def delete_file(filename: str, current_user: dict = Depends(require_admin)):
    try:
        # Decode the filename to handle URL encoding (e.g., %20 for spaces)
        decoded_filename = unquote(filename)
        file_path = os.path.join(UPLOAD_DIR, decoded_filename)
        logger.info(f"Attempting to delete file: {file_path}")

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise HTTPException(status_code=404, detail="File not found")

        # Delete the file from the filesystem
        os.remove(file_path)
        logger.info(f"File '{decoded_filename}' deleted from filesystem.")

        # Delete embeddings from Qdrant
        logger.info(f"Deleting embeddings for file '{decoded_filename}' from Qdrant.")

        # Create a filter to match the filename
        filename_filter = Filter(
            must=[
                {"key": "filename", "match": {"value": decoded_filename}}
            ]
        )

        # Search for points matching the filter
        search_results = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=filename_filter,
            with_vectors=False,
            with_payload=True,
            limit=1000  # Adjust limit as needed
        )

        # Extract point IDs from search results
        point_ids = [point.id for point in search_results[0]]

        if point_ids:
            # Delete points by their IDs
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=PointIdsList(points=point_ids)
            )
            logger.info(f"Deleted {len(point_ids)} embeddings for file '{decoded_filename}' from Qdrant.")
        else:
            logger.info(f"No embeddings found for file '{decoded_filename}' in Qdrant.")

        return {"message": f"File '{decoded_filename}' and its embeddings deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting file '{decoded_filename}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
