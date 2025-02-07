# PDF Summarizer and Chatbot with Vector Embeddings

This project is a **Streamlit-based web app** that allows users to upload a **PDF file**, extract its content, generate **vector embeddings**, summarize the content using **Hugging Face's BART** model, and interact with the text using a **chatbot** powered by **DeepSeek API**. 

## Features:
- **Upload PDF**: Allows the user to upload a PDF file for processing.
- **Extract Text**: Extracts text from the PDF file for further processing.
- **Vectorization**: Converts text into **vector embeddings** using **Sentence Transformers**.
- **Summarization**: Summarizes the PDF content using **Hugging Face's BART** model.
- **Chatbot**: Allows interaction with the text using **DeepSeek's chatbot API**.
- **View Stored Chunks**: View and search through vectorized chunks of the document stored in **ChromaDB**.

## Requirements

Before running the app, make sure you have all the dependencies installed. You can use `pip` to install the required libraries from the `requirements.txt` file.
