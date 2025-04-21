"""
Data processing utilities for the Boss Assistant application.
Contains functions for reading data files, creating vector databases, and keyword analysis.
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

def read_data_files():
    """
    Read HR and Finance data from corresponding files.
    Returns:
        tuple: (hr_data, finance_data)
    """
    # Read HR data
    print("Reading HR data...")
    try:
        with open('hr_data.txt', 'r', encoding='utf-8') as file:
            hr_data = file.read().splitlines()
    except Exception as e:
        print(f"Error reading HR data: {e}")
        hr_data = []

    # Read Finance data
    print("Reading Finance data...")
    try:
        with open('finance_data.txt', 'r', encoding='utf-8') as file:
            finance_data = file.read().splitlines()
    except Exception as e:
        print(f"Error reading Finance data: {e}")
        finance_data = []

    return hr_data, finance_data

def create_vector_db(texts, embedding_model):
    """
    Create FAISS vector database from texts.
    Args:
        texts (list): List of text documents
        embedding_model: Embedding model instance
    Returns:
        FAISS: Vector database
    """
    # Split texts into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )

    chunks = []
    for text in texts:
        chunks.extend(text_splitter.split_text(text))

    # Create vector database
    vector_db = FAISS.from_texts(chunks, embedding_model)
    return vector_db

def load_keywords_from_file(file_path):
    """
    Load keywords from file.
    Args:
        file_path (str): Path to the keyword file
    Returns:
        list: List of keywords
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return [line.strip().lower() for line in file if line.strip()]
    except Exception as e:
        print(f"Error loading keywords from {file_path}: {e}")
        return []

def calculate_keyword_confidence(question, keywords):
    """
    Calculate confidence score based on matching keywords.
    Args:
        question (str): Question to analyze
        keywords (list): List of keywords to check
    Returns:
        tuple: (confidence score, matched keywords)
    """
    question_lower = question.lower()
    matches = 0
    matched_keywords = []

    for keyword in keywords:
        if keyword in question_lower:
            matches += 1
            matched_keywords.append(keyword)

    # Calculate confidence score based on number of matching keywords
    # Use a simple formula to keep score between 0-1
    confidence = min(0.9, matches * 0.3)  # Each keyword adds 30% confidence, max 90%

    return confidence, matched_keywords
