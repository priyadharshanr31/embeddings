import streamlit as st
import json
import requests     # to handle api calls in DeepSeek
from pypdf import PdfReader
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter  # breakdown texts into chunks
from openai import OpenAI
import datetime
from db_utils import log_data_to_arctic, calculate_token_count  # Import log function from db_utils

# Initialize ChromaDB in the root repository
chroma_client = chromadb.PersistentClient(path="chroma_db")  # loads a persistent vector database or creates one
collection = chroma_client.get_or_create_collection(name="pdf_embeddings")  # creates a collection to store embeddings

# Function to extract text from PDF
def pdf_process(file):
    pdf_reader = PdfReader(file)   # reads the uploaded file
    file_content = ""
    for page in pdf_reader.pages:
        text = page.extract_text()  # extracts text from each page
        if text:
            file_content += text + "\n"
    return file_content     # stores all the text in a single string

# Function to chunk text
def chunk_text(text, chunk_size=500, chunk_overlap=50):    
    # each chunk has 500 characters and overlaps 50 characters to preserve context across chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap) 
    # splits text into overlapping chunks for better processing
    return text_splitter.split_text(text)

# Function to generate embeddings using DeepSeek API
def generate_embeddings(text, api_key):
    url = "https://api.deepseek.com"  # sends the text to DeepSeek API
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": text,
        "model": "text-embedding-ada-002"  # OpenAI's embedding model for text (via DeepSeek API)
    }

    start_time = datetime.datetime.now(datetime.UTC)  # Updated for timezone compliance

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response_time = (datetime.datetime.now(datetime.UTC) - start_time).total_seconds() * 1000  # in ms

        print(f"DeepSeek API Response: {response.status_code}, {response.text}")

        if response.status_code == 200:
            embedding = response.json().get("data", [{}])[0].get("embedding", None)
            if embedding:
                log_data_to_arctic("DeepSeek Embedding", text, json.dumps(response.json()), response_time, calculate_token_count(text))
                return embedding
            else:
                print(f"Warning: No embedding returned for text: {text}")
                return None
        else:
            print(f"Error: DeepSeek API request failed with status {response.status_code}. Response: {response.text}")
            return None
    except Exception as e:
        print(f"Exception occurred while generating embeddings: {e}")
        return None

# Function to store vectorized chunks in ChromaDB
def store_chunks(chunks, api_key):
    for i, chunk in enumerate(chunks):   # enumerate() function is used to get both the index (i) and the chunk of text
        embedding = generate_embeddings(chunk, api_key)
        if embedding:
            chunk_id = str(len(collection.get()["ids"]) if collection.get() and "ids" in collection.get() else 0)
            collection.add(              # the chunk info is added to the db
                ids=[chunk_id],
                embeddings=[embedding],
                metadatas=[{"text": chunk}]
            )
            log_data_to_arctic("Vectorization", chunk, json.dumps({"embedding": embedding}), 0, calculate_token_count(chunk))
        else:
            print(f"Skipping chunk {i} due to missing embedding")

# Function to query stored embeddings
def query_embeddings(query_text, api_key, top_k=3):  # returns the top 3 results
    query_embedding = generate_embeddings(query_text, api_key)   # generates an embedding for the user query
    if not query_embedding:
        print("No embeddings generated for query text.")
        return None

    results = collection.query(   # This is a method from ChromaDB (or another vector database) that 
        # searches the database for the most similar embeddings to the provided query_embedding
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    print(f"Query results: {results}")
    return results

# Function to summarize PDF using DeepSeek
def summarize_pdf(text, api_key):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": f"Summarize the following document:\n\n{text}"}
    ]
    
    start_time = datetime.datetime.now(datetime.UTC)

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )
        response_time = (datetime.datetime.now(datetime.UTC) - start_time).total_seconds() * 1000  # in ms

        summary = response.choices[0].message.content
        log_data_to_arctic("DeepSeek Summary", text, summary, response_time, calculate_token_count(text))

        return summary
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Failed to generate summary."

# Function to chat with OpenAI/DeepSeek
def chat_with_openai(context, user_query, api_key):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_query}"}
    ]
    
    start_time = datetime.datetime.now(datetime.UTC)
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
    
    response_time = (datetime.datetime.now(datetime.UTC) - start_time).total_seconds() * 1000  # in milliseconds
    
    log_data_to_arctic("OpenAI Chat", user_query, response.choices[0].message.content, response_time, calculate_token_count(user_query))
    
    return response.choices[0].message.content

# Streamlit UI Setup
st.set_page_config(page_title="Vector Embeddings", layout="wide")
st.title("Vector Embeddings Conversion & ChatBot")

tabs = st.tabs(["Upload File", "Convert to Vector Embeddings", "View Data", "Summary", "ChatBot"])

with st.sidebar:
    chatbot_token_count_display = st.empty()
    api_key = ""

    if api_key:
        overall_tokens = 0
        if "chunks" in st.session_state:
            overall_tokens = sum(calculate_token_count(chunk) for chunk in st.session_state["chunks"])
        st.write(f"Overall Token Count: {overall_tokens}")

# Upload PDF
with tabs[0]:
    st.header("Upload a PDF File")
    file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if file is None:
        st.error("Please upload a valid PDF file.")

# Convert to Vector Embeddings
with tabs[1]:
    st.header("Convert to Vector Embeddings")
    
    if file:
        if st.button("Initialize Vectorization"):
            pdf_text = pdf_process(file)
            chunks = chunk_text(pdf_text)
            token_count = sum(calculate_token_count(chunk) for chunk in chunks)
            st.session_state["chunks"] = chunks
            st.session_state["vectorization_token_count"] = token_count
            st.success(f"Vectorization initialized. Total chunks: {len(chunks)}")
            st.info(f"Total tokens required for embedding: {token_count}")

        if "chunks" in st.session_state and "vectorization_token_count" in st.session_state:
            api_key = st.text_input("Enter your DeepSeek API Key:", type="password", key="api_key_vectorization")

            if api_key and st.button("Store Chunks in Vector DB"):
                store_chunks(st.session_state["chunks"], api_key)
                st.success("Chunks stored successfully!")

# View PDF Content
with tabs[2]:
    st.header("View PDF Content")
    if file:
        st.text_area("Extracted Text", value=pdf_process(file), height=300)

# Summarize PDF
with tabs[3]:
    st.header("Summarize PDF")
    if file:
        api_key = st.text_input("Enter your DeepSeek API Key:", type="password", key="api_key_summary")
        if api_key and st.button("Generate Summary"):
            st.text_area("Summary", value=summarize_pdf(pdf_process(file), api_key), height=300)

# Chat with PDF
with tabs[4]:
    st.header("Chat with PDF")
    user_query = st.text_input("Enter your question:")
    if user_query and api_key:
        retrieved_text = query_embeddings(user_query, api_key)
