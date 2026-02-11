import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "fake_news_model_latest.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer_latest.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print("Model loaded successfully!\n")

def predict_with_confidence(text: str):
    vec = vectorizer.transform([text])

    # decision_function gives confidence score
    score = model.decision_function(vec)[0]

    # convert score into probability-like value using sigmoid
    import math
    prob_real = 1 / (1 + math.exp(-score))

    if prob_real >= 0.85:
        label = "Verified Real / Highly Reliable"
    elif prob_real >= 0.60:
        label = "Likely Real"
    elif prob_real >= 0.40:
        label = "Mixed Signals"
    elif prob_real >= 0.15:
        label = "Likely Fake"
    else:
        label = "Highly Suspicious / Fake"

    return label, prob_real


while True:
    text = input("\nEnter headline/news (or type exit): ")

    if text.lower() == "exit":
        break

    label, confidence = predict_with_confidence(text)
    print(f"\nPrediction: {label}")
    print(f"Confidence (Real): {confidence:.4f}")