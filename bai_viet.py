import requests
import re
import pickle
import torch
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os
from tensorflow.keras.optimizers import Adam

# Disable TensorFlow GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

api_url = "http://api.rpa4edu.shop/api_bai_viet.php"

# Load and compile LSTM-CNN model
try:
    lstm_cnn_model = load_model("lstm_cnn_sentiment_model.h5")
    lstm_cnn_model.compile(optimizer=Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
except Exception as e:
    print(f"Error loading LSTM-CNN model: {e}")
    exit(1)

# Load PhoBERT
try:
    phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")  # Update to fine-tuned path
    phobert_model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)
    device = torch.device("cpu")
    phobert_model.to(device)
except Exception as e:
    print(f"Error loading PhoBERT model: {e}")
    exit(1)

# Load tokenizer
try:
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit(1)

def preprocess_text(text):
    try:
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return ""

def predict_sentiment_keras(model, text, max_length=70):
    try:
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=max_length, padding="post", truncating="post")
        prediction = model.predict(padded, verbose=0)
        return prediction[0].tolist()
    except Exception as e:
        print(f"Error in LSTM-CNN prediction: {e}")
        return [0.0, 0.0, 0.0, 0.0]

def predict_sentiment_phobert(text):
    try:
        inputs = phobert_tokenizer(
            text, truncation=True, padding=True, max_length=256, return_tensors="pt"
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = phobert_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs[0].tolist()
    except Exception as e:
        print(f"Error in PhoBERT prediction: {e}")
        return [0.0, 0.0, 0.0]

def classify_sentiment(probabilities, threshold=0.5):
    try:
        if len(probabilities) == 4:
            if probabilities[1] > threshold:
                return 1
            elif probabilities[3] > threshold:
                return -2
            elif probabilities[2] > threshold:
                return -1
            else:
                return 0
        elif len(probabilities) == 3:
            if probabilities[1] > threshold:
                return 1
            elif probabilities[2] > threshold:
                return -1
            else:
                return 0
        else:
            print(f"Unexpected probability length: {len(probabilities)}")
            return 0
    except Exception as e:
        print(f"Error classifying sentiment: {e}")
        return 0

def get_articles_from_api(page=1, limit=100):
    try:
        headers = {"Content-Type": "application/json"}
        params = {"page": page, "limit": limit}  # Adjust based on API
        response = requests.get(api_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        if response.text:
            data = response.json()
            print(f"Retrieved {len(data)} articles from API (page {page})")
            return data
        else:
            print("API returned empty response.")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching articles: {e}")
        return []

def update_article_to_api(article):
    try:
        payload = {
            "id_bai_viet": article["id_bai_viet"],
            "sentiment_result_lstm_cnn": article["sentiment_result_lstm_cnn"],
            "sentiment_result_phobert": article["sentiment_result_phobert"],
        }
        headers = {"Content-Type": "application/json"}
        response = requests.put(api_url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        print(f"Updated article ID {article['id_bai_viet']} successfully.")
    except Exception as e:
        print(f"Error updating article ID {article.get('id_bai_viet')}: {e}")

def main():
    page = 1
    all_articles = []
    while True:
        articles = get_articles_from_api(page=page)
        if not articles:
            break
        all_articles.extend(articles)
        print(f"Total articles retrieved so far: {len(all_articles)}")
        page += 1

    if not all_articles:
        print("No articles retrieved from API.")
        return

    print(f"Processing {len(all_articles)} articles.")
    for i, article in enumerate(all_articles, 1):
        try:
            text_vietnamese = article.get("noi_dung_bai_viet", "")
            if not text_vietnamese:
                print(f"Article ID {article.get('id_bai_viet')}: Empty content, skipping.")
                continue

            processed_text = preprocess_text(text_vietnamese)
            if not processed_text:
                print(f"Article ID {article.get('id_bai_viet')}: Empty after preprocessing, skipping.")
                continue

            lstm_cnn_probs = predict_sentiment_keras(lstm_cnn_model, processed_text)
            phobert_probs = predict_sentiment_phobert(processed_text)

            sentiment_lstm_cnn = classify_sentiment(lstm_cnn_probs, threshold=0.5)
            sentiment_phobert = classify_sentiment(phobert_probs, threshold=0.5)

            article["sentiment_result_lstm_cnn"] = sentiment_lstm_cnn
            article["sentiment_result_phobert"] = sentiment_phobert

            print(f"Article {i}/{len(all_articles)} - ID: {article['id_bai_viet']}")
            print(f"Text: {text_vietnamese[:100]}...")
            print(f"LSTM-CNN Probabilities: {lstm_cnn_probs}")
            print(f"LSTM-CNN Sentiment: {sentiment_lstm_cnn}")
            print(f"PhoBERT Probabilities: {phobert_probs}")
            print(f"PhoBERT Sentiment: {sentiment_phobert}")
            print("-" * 50)

            update_article_to_api(article)
        except Exception as e:
            print(f"Error processing article ID {article.get('id_bai_viet')}: {e}")
            continue

if __name__ == "__main__":
    main()