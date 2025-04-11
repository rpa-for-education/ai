import requests
import re
import pickle
import torch
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

api_url = "http://api.rpa4edu.shop/api_bai_viet.php"

lstm_cnn_model = load_model("lstm_cnn_sentiment_model.h5")

phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
phobert_model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
phobert_model.to(device)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def preprocess_text(text):
    text = re.sub(r'<[^>]+>', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    return text

def predict_sentiment_keras(model, text, max_length=70):  
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length, padding="post", truncating="post")
    prediction = model.predict(padded, verbose=0)
    return prediction[0].tolist()

def predict_sentiment_phobert(text):
    inputs = phobert_tokenizer(
        text, truncation=True, padding=True, max_length=256, return_tensors="pt"
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = phobert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs[0].tolist()

def classify_sentiment(probabilities, threshold=0.4):
    """
    Phân loại cảm xúc: 1 (tích cực), 0 (trung tính), -1 (tiêu cực), -2 (cực kỳ tiêu cực).
    Dataset: 0 (trung tính), 1 (tích cực), 2 (tiêu cực), 3 (cực kỳ tiêu cực).
    """
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
        raise ValueError("Unexpected probability length")

def get_articles_from_api():
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        if response.text:
            return response.json()
        else:
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
        response = requests.put(api_url, json=payload, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"Error updating article ID {article.get('id_bai_viet')}: {e}")

def main():
    articles = get_articles_from_api()
    if not articles:
        print("No articles retrieved from API.")
        return

    for article in articles:
        text_vietnamese = article["noi_dung_bai_viet"]
        processed_text = preprocess_text(text_vietnamese)

        lstm_cnn_probs = predict_sentiment_keras(lstm_cnn_model, processed_text)
        phobert_probs = predict_sentiment_phobert(processed_text)

        sentiment_lstm_cnn = classify_sentiment(lstm_cnn_probs, threshold=0.5)
        sentiment_phobert = classify_sentiment(phobert_probs, threshold=0.5)

        article["sentiment_result_lstm_cnn"] = sentiment_lstm_cnn
        article["sentiment_result_phobert"] = sentiment_phobert

        print(f"Article ID: {article['id_bai_viet']}")
        print(f"Text: {text_vietnamese[:100]}...")  
        print(f"LSTM-CNN Probabilities: {lstm_cnn_probs}")
        print(f"LSTM-CNN Sentiment: {sentiment_lstm_cnn}")
        print(f"PhoBERT Probabilities: {phobert_probs}")
        print(f"PhoBERT Sentiment: {sentiment_phobert}")
        print("-" * 50)

        update_article_to_api(article)

if __name__ == "__main__":
    main()