import sqlite3
import datetime
import tiktoken
from transformers import DistilBertTokenizer

# Database setup (Creates a DB file in the same folder)
DB_FILE = "token_usage.db"

# Function to create table if not exists
def create_table():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            api_name TEXT,
            prompt TEXT,
            response TEXT,
            response_time_ms REAL,
            token_count INTEGER
        )
    """)
    conn.commit()
    conn.close()

# Ensure the table is created when the script runs
create_table()

# Function to calculate token count
def calculate_token_count(text, model="gpt-3.5-turbo"):
    if model in ["gpt-3.5-turbo", "deepseek-chat"]:
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-3.5 tokenizer
        tokens = encoding.encode(text)
        return len(tokens)
    
    elif model == "distilbert-base-cased":
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")
        tokens = tokenizer.encode(text)
        return len(tokens)
    
    else:
        raise ValueError("Unsupported model for token count calculation.")

# Function to log data into SQLite database
def log_data_to_arctic(api_name, prompt, response, response_time, token_count):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO api_logs (timestamp, api_name, prompt, response, response_time_ms, token_count) 
        VALUES (?, ?, ?, ?, ?, ?)
    """, (datetime.datetime.utcnow().isoformat(), api_name, prompt, response, response_time, token_count))
    
    conn.commit()
    conn.close()
