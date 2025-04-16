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
import json

# Disable TensorFlow GPU and warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

api_url = "http://api.rpa4edu.shop/api_bai_viet.php"

# Load and compile LSTM-CNN model
try:
    print("Loading LSTM-CNN model...")
    lstm_cnn_model = load_model("lstm_cnn_sentiment_model.h5")
    lstm_cnn_model.compile(optimizer=Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    print("LSTM-CNN model loaded and compiled.")
except Exception as e:
    print(f"Error loading LSTM-CNN model: {e}")
    exit(1)

# Load PhoBERT
try:
    print("Loading PhoBERT model...")
    phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")  # Update to "./phobert_fine_tuned" after fine-tuning
    phobert_model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)
    device = torch.device("cpu")
    phobert_model.to(device)
    print("PhoBERT model loaded.")
except Exception as e:
    print(f"Error loading PhoBERT model: {e}")
    exit(1)

# Load tokenizer
try:
    print("Loading Keras tokenizer...")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    print("Keras tokenizer loaded.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit(1)

# Load processed IDs
processed_ids = set()
try:
    with open("processed_ids.json", "r") as f:
        processed_ids = set(json.load(f))
    print(f"Loaded {len(processed_ids)} processed article IDs.")
except FileNotFoundError:
    print("No processed IDs file found, starting fresh.")
except Exception as e:
    print(f"Error loading processed IDs: {e}")

def preprocess_text(text):
    try:
        if not isinstance(text, str) or not text.strip():
            print("Invalid or empty text, skipping preprocessing.")
            return ""
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        if len(text.split()) < 3:  # Require at least 3 words
            print("Text too short after preprocessing, skipping.")
            return ""
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
        if len(probabilities) == 4:  # LSTM-CNN
            if probabilities[1] > threshold:
                return 1  # Positive
            elif probabilities[3] > threshold:
                return -2  # Strongly negative
            elif probabilities[2] > threshold:
                return -1  # Negative
            else:
                return 0  # Neutral
        elif len(probabilities) == 3:  # PhoBERT
            if probabilities[1] > threshold:
                return 1  # Positive
            elif probabilities[2] > threshold:
                return -1  # Negative
            else:
                return 0  # Neutral
        else:
            print(f"Unexpected probability length: {len(probabilities)}")
            return 0
    except Exception as e:
        print(f"Error classifying sentiment: {e}")
        return 0

def get_articles_from_api(page=1, limit=1029):
    try:
        headers = {"Content-Type": "application/json"}
        params = {"page": page, "limit": limit}  # Try {"offset": (page-1)*limit, "limit": limit} if needed
        print(f"Fetching articles from API (page {page}, limit {limit})...")
        response = requests.get(api_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        if response.text:
            data = response.json()
            print(f"Retrieved {len(data)} articles from API (page {page})")
            current_ids = {article.get("id_bai_viet", "unknown") for article in data}
            overlap = current_ids & processed_ids
            if overlap:
                print(f"Warning: Found {len(overlap)} duplicate IDs in page {page}: {overlap}")
                if len(overlap) / len(data) > 0.9:
                    print(f"Page {page} has {len(overlap)}/{len(data)} duplicate IDs. Stopping.")
                    return []
            return data
        else:
            print("API returned empty response.")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching articles (page {page}): {e}")
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
        print(f"Error updating article ID {article.get('id_bai_viet', 'unknown')}: {e}")

def main():
    page = 1
    batch_size = 1029
    global processed_ids
    max_pages = 10  # Limit to avoid infinite loop

    while page <= max_pages:
        articles = get_articles_from_api(page=page, limit=batch_size)
        if not articles:
            print("No more articles to process or too many duplicates.")
            break

        # Filter valid articles
        valid_articles = [a for a in articles if a.get("noi_dung_bai_viet", "").strip() and len(a.get("noi_dung_bai_viet", "").split()) >= 3]
        print(f"Filtered to {len(valid_articles)} valid articles out of {len(articles)}.")

        print(f"Processing batch of {len(valid_articles)} articles (page {page})...")
        new_articles_processed = 0
        skipped_empty = 0
        for i, article in enumerate(valid_articles, 1):
            try:
                article_id = article.get("id_bai_viet", "unknown")
                if article_id == "unknown":
                    print(f"Article {i}/{len(valid_articles)} (page {page}): Missing ID, skipping.")
                    continue
                if article_id in processed_ids:
                    print(f"Article ID {article_id}: Already processed, skipping.")
                    continue

                text_vietnamese = article.get("noi_dung_bai_viet", "")
                processed_text = preprocess_text(text_vietnamese)
                if not processed_text:
                    print(f"Article ID {article_id}: Empty after preprocessing, skipping.")
                    skipped_empty += 1
                    continue

                lstm_cnn_probs = predict_sentiment_keras(lstm_cnn_model, processed_text)
                phobert_probs = predict_sentiment_phobert(processed_text)

                sentiment_lstm_cnn = classify_sentiment(lstm_cnn_probs, threshold=0.5)
                sentiment_phobert = classify_sentiment(phobert_probs, threshold=0.5)

                article["sentiment_result_lstm_cnn"] = sentiment_lstm_cnn
                article["sentiment_result_phobert"] = sentiment_phobert

                print(f"Article {i}/{len(valid_articles)} (page {page}) - ID: {article_id}")
                print(f"Text: {text_vietnamese[:100]}...")
                print(f"LSTM-CNN Probabilities: {lstm_cnn_probs}")
                print(f"LSTM-CNN Sentiment: {sentiment_lstm_cnn}")
                print(f"PhoBERT Probabilities: {phobert_probs}")
                print(f"PhoBERT Sentiment: {sentiment_phobert}")
                print("-" * 50)

                update_article_to_api(article)
                processed_ids.add(article_id)
                new_articles_processed += 1

            except Exception as e:
                print(f"Error processing article ID {article.get('id_bai_viet', 'unknown')}: {e}")
                continue

        print(f"Finished processing page {page}. New articles processed: {new_articles_processed}, Skipped (duplicate): {len(valid_articles) - new_articles_processed - skipped_empty}, Skipped (empty): {skipped_empty}.")
        if new_articles_processed == 0:
            print("No new articles found in this page. Stopping.")
            break

        # Save processed IDs
        try:
            with open("processed_ids.json", "w") as f:
                json.dump(list(processed_ids), f)
            print(f"Saved {len(processed_ids)} processed IDs.")
        except Exception as e:
            print(f"Error saving processed IDs: {e}")

        page += 1

    print(f"Script completed. Total unique articles processed: {len(processed_ids)}.")

if __name__ == "__main__":
    print("Starting sentiment analysis script...")
    main()