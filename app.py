import streamlit as st
import json
import requests
from pypdf import PdfReader
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import datetime
from db_utils import log_data_to_arctic, calculate_token_count
from transformers import pipeline

# Load Hugging Face BART summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load local embedding model for vectorization
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB in a new database
chroma_client = chromadb.PersistentClient(path="new_db")
collection = chroma_client.get_or_create_collection(name="new_embeddings")

# Function to extract text from PDF
def pdf_process(file):
    pdf_reader = PdfReader(file)
    file_content = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return file_content

# Function to chunk text
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

# Function to generate embeddings using SentenceTransformers
def generate_embeddings(text):
    try:
        embedding = embedding_model.encode(text).tolist()
        return embedding
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return None

# Function to store vectorized chunks in ChromaDB
def store_chunks(chunks):
    for i, chunk in enumerate(chunks):
        embedding = generate_embeddings(chunk)
        if embedding:
            chunk_id = str(len(collection.get().get("ids", [])))  # Get stored IDs
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                metadatas=[{"text": chunk}]
            )
            log_data_to_arctic("Vectorization", chunk, json.dumps({"embedding": embedding}), 0, calculate_token_count(chunk))
            print(f"✅ Stored Chunk {i+1} with ID: {chunk_id}")
        else:
            print(f"⚠️ Skipping chunk {i} due to missing embedding")

def query_embeddings(query_text, top_k=3):
    query_embedding = generate_embeddings(query_text)
    if not query_embedding:
        print("❌ No embeddings generated for query text.")
        return None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    print(f"🔍 Query Results: {results}")  # Log results for debugging
    return results


# Function to summarize content using Hugging Face's BART model
def summarize_with_huggingface(text):
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Function to retrieve stored chunks from ChromaDB
def view_stored_chunks():
    results = collection.get()
    
    if not results.get("ids"):
        st.write("❌ No data found in the database.")
        return
    
    # Displaying Stored Chunks Count
    st.markdown(f"### 📊 **Stored Chunks Count**: {len(results['ids'])}")

    # Display each Chunk ID and its Text
    for i, chunk_id in enumerate(results["ids"]):
        st.markdown(f"#### 📌 **Chunk ID**: {chunk_id}")
        embeddings = results.get("embeddings")
        chunk_text = results.get("metadatas")[i].get("text", "No text available")
        st.markdown(f"🔢 **Chunk Text**: {chunk_text[:200]}...")  # Show only the first 200 characters of the chunk
        st.write("---")

def chatbot_with_deepseek(context, user_query, api_key):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Construct the full payload (context + question)
    full_prompt = f"Context: {context}\n\nQuestion: {user_query}"

    # Calculate token count for entire payload (context + question)
    tokens_used = calculate_token_count(full_prompt, model="deepseek-chat")
    print(f"🔢 Tokens Used (Full Payload): {tokens_used}")

    # Payload sent to the API
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": full_prompt}
        ]
    }

    print(f"📝 Full Payload Sent to API: {json.dumps(payload, indent=2)}")  # Print payload for debugging

    start_time = datetime.datetime.utcnow()
    response = requests.post(url, headers=headers, data=json.dumps(payload))  # Send correct payload
    response_time = (datetime.datetime.utcnow() - start_time).total_seconds() * 1000
    tokens_used_api = calculate_token_count(user_query, model="deepseek-chat")

    log_data_to_arctic("DeepSeek Chatbot", user_query, response.text, response_time, tokens_used_api)

    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "❌ Error: No response from DeepSeek API")
    else:
        return f"❌ Error: {response.text}"

# Streamlit UI Setup
st.set_page_config(page_title="Vector Embeddings & ChatBot", layout="wide")
st.title("Vector Embeddings Conversion, Summarization & ChatBot")

tabs = st.tabs(["Upload File", "Convert to Vector Embeddings", "View Data", "View Stored Chunks", "Summarization", "ChatBot"])

# Upload PDF
with tabs[0]:
    st.header("📂 Upload a PDF File")
    file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if file is None:
        st.error("❌ Please upload a valid PDF file.")

# Convert to Vector Embeddings
with tabs[1]:
    st.header("🔄 Convert to Vector Embeddings")
    
    if file:
        # Display the uploaded file name
        st.markdown(f"**Uploaded File Name**: {file.name}")
        
        if st.button("📊 Initialize Vectorization"):
            pdf_text = pdf_process(file)
            chunks = chunk_text(pdf_text)
            st.session_state["chunks"] = chunks
            st.success(f"✅ Vectorization initialized. Total chunks: {len(chunks)}")

    if "chunks" in st.session_state and st.button("💾 Store Chunks in Vector DB"):
        store_chunks(st.session_state["chunks"])
        st.success("✅ Chunks stored successfully!")

# View PDF Content
with tabs[2]:
    st.header("👀 View PDF Content")
    if file:
        st.text_area("Extracted Text", value=pdf_process(file), height=300)

# View Stored Chunks
with tabs[3]:
    st.header("📚 Stored Chunks in Vector DB")
    if st.button("🔍 View Stored Chunks"):
        view_stored_chunks()

# Summarization Tab
with tabs[4]:
    st.header("📝 Summarize PDF Content")
    if file:
        st.markdown("**Summarize your PDF content using Hugging Face's BART model!**")
        pdf_text = pdf_process(file)
        if st.button("Generate Summary"):
            # Use Hugging Face's BART for summarization
            summary = summarize_with_huggingface(pdf_text)
            st.text_area("Generated Summary", value=summary, height=300)

# ChatBot Tab
with tabs[5]:
    st.header("💬 Chat with PDF")
    user_query = st.text_input("Enter your question:")
    api_key = st.text_input("🔑 Enter DeepSeek API Key:", type="password", key="api_key_chat")
    
    if user_query:
        # Retrieve similar chunks from the database
        results = query_embeddings(user_query, top_k=3)

        if results and "metadatas" in results and results["metadatas"]:
            # Extract relevant context from metadata
            context = " ".join([meta["text"] for meta in results["metadatas"] if "text" in meta])
        else:
            context = "No relevant context found."
        
        # Calculate token usage for the full payload (context + question)
        full_prompt = f"Context: {context}\n\nQuestion: {user_query}"
        tokens_used = calculate_token_count(full_prompt, model="deepseek-chat")

        # Show token usage first
        st.write(f"📝 Total Tokens to be Used: {tokens_used}")

        # Show API key input only after showing token count
        if api_key and st.button("Get Response"):
            # Call the chatbot function
            response = chatbot_with_deepseek(context, user_query, api_key)
            st.text_area("AI Response", value=response, height=150)

            # Display the chunks and their IDs
            st.subheader("📚 Source Chunks:")
            for i, metadata in enumerate(results["metadatas"]):
                chunk_id = results["ids"][i]
                
                # Ensure metadata is a list and contains the "text" key
                chunk_text = metadata[0]["text"] if isinstance(metadata, list) and len(metadata) > 0 else "No text available"
                
                st.write(f"📌 **Chunk ID**: {chunk_id}")
                st.write(f"🔢 **Chunk**: {chunk_text[:200]}...")  # Display first 200 characters of the chunk
                st.write("---")
