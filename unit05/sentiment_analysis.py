import time
import os
import sys
import bz2
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import psutil
import pickle

# Download required NLTK data
nltk.download('stopwords')

# Định nghĩa đường dẫn lưu mô hình
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment_model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
DATA_TRAIN_PATH = '~/.cache/kagglehub/datasets/bittlingmayer/amazonreviews/versions/7/train.ft.txt.bz2'
DATA_TEST_PATH = '~/.cache/kagglehub/datasets/bittlingmayer/amazonreviews/versions/7/test.ft.txt.bz2'

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def parse_amazon_review(line):
    """Parse a single Amazon review line"""
    # Format: __label__X text
    # Where X is 1 or 2 for negative/positive
    parts = line.split(' ', 1)
    if len(parts) == 2:
        label = int(parts[0].replace('__label__', ''))
        text = parts[1].strip()
        return {'label': label, 'text': text}
    return None

def preprocess_text(text):
    """Preprocess text for sentiment analysis"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def save_model(model, vectorizer):
    """Lưu mô hình và vectorizer vào file"""
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Lưu mô hình
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    # Lưu vectorizer
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"\nĐã lưu mô hình tại: {MODEL_PATH}")
    print(f"Đã lưu vectorizer tại: {VECTORIZER_PATH}")

def load_model():
    """Tải mô hình và vectorizer từ file"""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None
    
    # Tải mô hình
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    # Tải vectorizer
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer

def predict_sentiment(text, model=None, vectorizer=None):
    """Dự đoán cảm xúc cho một đoạn text"""
    if model is None or vectorizer is None:
        model, vectorizer = load_model()
        if model is None or vectorizer is None:
            raise ValueError("Không tìm thấy mô hình đã lưu!")
    
    # Tiền xử lý văn bản
    processed_text = preprocess_text(text)
    
    # Chuyển đổi text thành vector
    X = vectorizer.transform([processed_text])
    
    # Dự đoán
    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X)[0][1] if prediction == 2 else model.predict_proba(X)[0][0]
    
    return "Positive" if prediction == 2 else "Negative", confidence

def main():
    print("Amazon Reviews Sentiment Analysis")
    start_time = time.time()
    initial_memory = get_memory_usage()
    
    # Kiểm tra xem đã có mô hình được lưu chưa
    model, vectorizer = load_model()
    if model is not None and vectorizer is not None:
        print("\nĐã tìm thấy mô hình đã lưu. Bạn có muốn huấn luyện lại không? (y/n)")
        choice = input().lower()
        if choice != 'y':
            print("\nSử dụng mô hình đã lưu...")
            # Test với các ví dụ
            example_reviews = [
                "This product is amazing! I love it so much.",
                "Terrible quality, broke after one use. Would not recommend.",
                "It's okay, nothing special but gets the job done.",
                "Waste of money, don't buy this garbage.",
                "Best purchase I've made all year, absolutely perfect!"
            ]
            
            print("\nKiểm tra với các đánh giá mẫu:")
            for review in example_reviews:
                sentiment, confidence = predict_sentiment(review, model, vectorizer)
                print(f"\nReview: {review}")
                print(f"Sentiment: {sentiment} (confidence: {confidence:.2f})")
            return

    # Read data
    print(f"Reading sample from {DATA_TRAIN_PATH}...")
    with bz2.open(DATA_TRAIN_PATH, 'rt') as f:
        train_lines = [line.strip() for line in f]
    print(f"Total lines in file: {len(train_lines)}")

    print(f"Reading sample from {DATA_TEST_PATH}...")
    with bz2.open(DATA_TEST_PATH, 'rt') as f:
        test_lines = [line.strip() for line in f]
    print(f"Total lines in file: {len(test_lines)}")

    # Parse data
    print("\nParsing training samples...")
    train_data = [parse_amazon_review(line) for line in train_lines[:5000]]
    train_df = pd.DataFrame(train_data)

    print("Parsing test samples...")
    test_data = [parse_amazon_review(line) for line in test_lines[:5000]]
    test_df = pd.DataFrame(test_data)

    # Preprocess text
    print("\nPreprocessing text...")
    train_df['processed_text'] = train_df['text'].apply(preprocess_text)
    test_df['processed_text'] = test_df['text'].apply(preprocess_text)

    # Extract features
    print("\nExtracting features...")
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train = vectorizer.fit_transform(train_df['processed_text'])
    X_test = vectorizer.transform(test_df['processed_text'])
    y_train = train_df['label']
    y_test = test_df['label']

    # Train model
    print("\nTraining model...")
    model_start_time = time.time()
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    training_time = time.time() - model_start_time

    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)

    # Calculate model size
    model_size = sys.getsizeof(pickle.dumps(model)) / 1024 / 1024  # Size in MB
    vectorizer_size = sys.getsizeof(pickle.dumps(vectorizer)) / 1024 / 1024  # Size in MB
    total_model_size = model_size + vectorizer_size

    # Print results
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Get feature importance
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]
    feature_importance = list(zip(feature_names, coef))
    feature_importance.sort(key=lambda x: x[1])

    print("\nMost important features:")
    print("\nTop 10 features for negative sentiment:")
    for word, score in feature_importance[:10]:
        print(f"{word}: {score:.4f}")

    print("\nTop 10 features for positive sentiment:")
    for word, score in feature_importance[-10:][::-1]:
        print(f"{word}: {score:.4f}")

    # Test with example reviews
    print("\nTesting with example reviews:")
    example_reviews = [
        "This product is amazing! I love it so much.",
        "Terrible quality, broke after one use. Would not recommend.",
        "It's okay, nothing special but gets the job done.",
        "Waste of money, don't buy this garbage.",
        "Best purchase I've made all year, absolutely perfect!"
    ]

    for review in example_reviews:
        sentiment, confidence = predict_sentiment(review, model, vectorizer)
        print(f"\nReview: {review}")
        print(f"Sentiment: {sentiment} (confidence: {confidence:.2f})")

    # Print timing and memory information
    total_time = time.time() - start_time
    final_memory = get_memory_usage()
    memory_used = final_memory - initial_memory

    print("\nPerformance Metrics:")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Model training time: {training_time:.2f} seconds")
    print(f"Memory usage: {memory_used:.2f} MB")
    print(f"Model size: {total_model_size:.2f} MB (Logistic Regression: {model_size:.2f} MB, Vectorizer: {vectorizer_size:.2f} MB)")

    # Lưu mô hình
    save_model(model, vectorizer)

if __name__ == "__main__":
    main()
