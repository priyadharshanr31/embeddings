import streamlit as st
import json
import requests   # sending API requests to DeepSeek API
from pypdf import PdfReader
import chromadb   # storing and querying vector embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter  # for chunks
from sentence_transformers import SentenceTransformer  # vector embeddings using the pre trained model
import datetime
from db_utils import log_data_to_arctic, calculate_token_count
from transformers import pipeline   # summary part


# load Hugging Face BART summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# oad local embedding model for vectorization used via library
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# initialize ChromaDB in a new database
chroma_client = chromadb.PersistentClient(path="new_db")
collection = chroma_client.get_or_create_collection(name="new_embeddings")




# function to extract text from PDF
def pdf_process(file):    # used in tab 2 & 3 embeddings creation and view data
    pdf_reader = PdfReader(file)
    file_content = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return file_content   # extracts text from all the pages and returns as a single str
    # str - extracted text from the PDF is returned



# function to chunk text and used in Convert to Vector Embeddings tab 2
def chunk_text(text, chunk_size=500, chunk_overlap=50):   # overlap is 50 because to have meaningful chuks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text) 
    # list[str] - list of text chunks is returned


# function to generate embeddings using SentenceTransformers 
def generate_embeddings(text): # used inside store_chunks() when saving text to Chroma db
    try:
        embedding = embedding_model.encode(text).tolist()   # to get numerical vectors
        return embedding  
        # list[float]  - the vector data is returned as list
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return None



# function to store vectorized chunks in ChromaDB
def store_chunks(chunks):  # used in convert to vector embeddings tab 2
    for i, chunk in enumerate(chunks):      # loops through each chunk and stores it in Chroma Db
        embedding = generate_embeddings(chunk)
        if embedding:
            chunk_id = str(len(collection.get().get("ids", [])))  # get stored IDs
            collection.add(                # each chunk will have unique ID from 0 
                ids=[chunk_id],
                embeddings=[embedding],
                metadatas=[{"text": chunk}]
            )
            log_data_to_arctic("Vectorization", chunk, json.dumps({"embedding": embedding}), 0, calculate_token_count(chunk))
            print(f"✅ Stored Chunk {i+1} with ID: {chunk_id}")    # logs to the db
        else:
            print(f"⚠️ Skipping chunk {i} due to missing embedding")




def query_embeddings(query_text, top_k=3):   # converts user query into vector embeddings
    query_embedding = generate_embeddings(query_text)
    if not query_embedding:
        print("❌ No embeddings generated for query text.")
        return None

    results = collection.query(      # searches Chroma Db for top-3 most similar chunks
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    print(f"🔍 Query Results: {results}")  # log results for debugging (to check if it works fine)
    return results
    # dict - dictionary containing retrieved chunks and metadata is returned 
    # this dict. will have id, embeddings and metadata as key and their values will be
       # "id": ["chunk_1", "chunk_2", "chunk_3"],
       # "embeddings": [[0.12, -0.45, ...], [0.56, 0.78, ...]],
       # "metadatas": [
            # {"text": "This is chunk 1 text..."},
            # {"text": "This is chunk 2 text..."},
            # {"text": "This is chunk 3 text..."}
        # ]


# function to summarize content using Hugging Face's BART model
def summarize_with_huggingface(text):  # used in summary tab 5
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']    # limits summary between 50 to 150 words
    # str - is returned


# function to retrieve stored chunks from ChromaDB
def view_stored_chunks():  # used in view sstored chunks tab 4
    results = collection.get()  # fetches all stored chunks
    # ids - will be list 
    # embeddings - will be list eg. [[0.12, 0.34, ...], [0.23, 0.56, ...]]
    # metadatas - will be list of dict.
    
    if not results.get("ids"):     # if no chunk exists
        st.write("❌ No data found in the database.")
        return
    
    # displaying Stored Chunks Count
    st.markdown(f"### 📊 **Stored Chunks Count**: {len(results['ids'])}")  # shows how many chunks are stored overall

    # display each Chunk ID and its Text
    for i, chunk_id in enumerate(results["ids"]):  # loops through the list of chunk IDs
        st.markdown(f"#### 📌 **Chunk ID**: {chunk_id}")
        embeddings = results.get("embeddings")
        chunk_text = results.get("metadatas")[i].get("text", "No text available")
          #  ^ retrieves the metadata dictionary for the current chunk.
        st.markdown(f"🔢 **Chunk Text**: {chunk_text[:200]}...")  # show only the first 200 characters of the chunk
        st.write("---")



# used in chatbot tab 6 Uses Deepseek Api for AI response
def chatbot_with_deepseek(context, user_query, api_key):  # ses DeepSeek api to generate chatbot responses using retrieved chunks
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # construct the full payload   includes the vector data and user question
    full_prompt = f"Context: {context}\n\nQuestion: {user_query}"  # chatbot uses this generate relevent answer 

    # calculate token count for entire payload (full_prompt)
    tokens_used = calculate_token_count(full_prompt, model="deepseek-chat")
    print(f"🔢 Tokens Used (Full Payload): {tokens_used}")

    # payload sent to the API
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},  # this to for ai to know how to behave
            {"role": "user", "content": full_prompt}  # user_ques and the context is sent
        ]
    }

    print(f"📝 Full Payload Sent to API: {json.dumps(payload, indent=2)}")  # print payload for debugging

    start_time = datetime.datetime.utcnow()
    response = requests.post(url, headers=headers, data=json.dumps(payload))   # sending post request to deepseek
    
    response_time = (datetime.datetime.utcnow() - start_time).total_seconds() * 1000  # fpr api usage log
    tokens_used_api = calculate_token_count(user_query, model="deepseek-chat")


    log_data_to_arctic("DeepSeek Chatbot", user_query, response.text, response_time, tokens_used_api)

    if response.status_code == 200:  # this is if its success
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "❌ Error: No response from DeepSeek API")
        # extracts bot's response from API response JSON, uses .get() to avoid errors if data is missing
        # str - the ai generated response from deepseek is returned
    else:
        return f"❌ Error: {response.text}"
    




# streamlit UI Setup
st.set_page_config(page_title="Vector Embeddings & ChatBot", layout="wide")
st.title("Vector Embeddings Conversion, Summarization & ChatBot")

tabs = st.tabs(["Upload File", "Convert to Vector Embeddings", 
                "View Data", "View Stored Chunks", "Summarization", "ChatBot"])

# upload PDF
with tabs[0]:
    st.header("📂 Upload a PDF File")
    file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if file is None:
        st.error("❌ Please upload a valid PDF file.")


# convert to Vector Embeddings
with tabs[1]:
    st.header("🔄 Convert to Vector Embeddings")
    
    if file:
        # displays the uploaded file name
        st.markdown(f"**Uploaded File Name**: {file.name}")
        
        if st.button("📊 Initialize Vectorization"):
            pdf_text = pdf_process(file)  # pdf function is called
            chunks = chunk_text(pdf_text)  # chunk func is called
            st.session_state["chunks"] = chunks   # stores chunks in session state os streamlit
            st.success(f"✅ Vectorization initialized. Total chunks: {len(chunks)}")

    if "chunks" in st.session_state and st.button("💾 Store Chunks in Vector DB"):  # will appear only if chunk is present
        store_chunks(st.session_state["chunks"]) # store_chunk func is called here
        st.success("✅ Chunks stored successfully!")


# View PDF Content
with tabs[2]:
    st.header("👀 View PDF Content")
    if file:   # we get this from pdf_process fun. which is a single str
        st.text_area("Extracted Text", value=pdf_process(file), height=300)


# View Stored Chunks
with tabs[3]:
    st.header("📚 Stored Chunks in Vector DB")
    if st.button("🔍 View Stored Chunks"):
        view_stored_chunks()  # view_stored_chunks() is called to retrieve and display 
                              # stored text chunks from ChromaDB.

# Summarization Tab
with tabs[4]:
    st.header("📝 Summarize PDF Content")
    if file:
        st.markdown("**Summarize your PDF content using Hugging Face's BART model!**")
        pdf_text = pdf_process(file)  # pdf func. is called
        if st.button("Generate Summary"):
            # Uused Hugging Face's BART for summarization (zero cost)
            summary = summarize_with_huggingface(pdf_text)  # return the response from this func.
            st.text_area("Generated Summary", value=summary, height=300)



# ChatBot Tab
with tabs[5]:
    st.header("💬 Chat with PDF")
    user_query = st.text_input("Enter your question:")
    api_key = st.text_input("🔑 Enter DeepSeek API Key:", type="password", key="api_key_chat")
    
    if user_query:
        # retrieve similar chunks from the database
        results = query_embeddings(user_query, top_k=3)  


        if results and "metadatas" in results and results["metadatas"]: # ensures there are retrieved chunks
            # extracts text from the retrieved chunks and joins them into a single context string
            # this provides DeepSeek with useful background information before answering the user question
            context = " ".join([meta["text"] for meta in results["metadatas"] if "text" in meta])
        else:
            context = "No relevant context found."
        
        # calculate token usage for the full payload (context + question)
        full_prompt = f"Context: {context}\n\nQuestion: {user_query}"
        tokens_used = calculate_token_count(full_prompt, model="deepseek-chat")

        # shows token usage first
        st.write(f"📝 Total Tokens to be Used: {tokens_used}")

        # shows API key input only after showing token count to monitor the tokens used 
        if api_key and st.button("Get Response"):
            # call the chatbot function
            response = chatbot_with_deepseek(context, user_query, api_key)
            st.text_area("AI Response", value=response, height=150)

            # display the chunks and their IDs
            st.subheader("📚 Source Chunks:")
            for i, metadata in enumerate(results["metadatas"]):
                chunk_id = results["ids"][i]
                
                # ensure metadata is a list and contains the "text" key
                chunk_text = metadata[0]["text"] if isinstance(metadata, list) and len(metadata) > 0 else "No text available"
                
                st.write(f"📌 **Chunk ID**: {chunk_id}")
                st.write(f"🔢 **Chunk**: {chunk_text[:200]}...")  # displays only first 200 characters of the chunk
                st.write("---")
