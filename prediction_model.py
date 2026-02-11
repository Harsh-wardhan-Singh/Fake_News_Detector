import joblib
import math
from pathlib import Path


BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "fake_news_model_latest.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer_latest.pkl"


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Latest model not found at: {MODEL_PATH}\n"
        "Run train_model.py first to generate the model."
    )

if not VECTORIZER_PATH.exists():
    raise FileNotFoundError(
        f"Latest vectorizer not found at: {VECTORIZER_PATH}\n"
        "Run train_model.py first to generate the vectorizer."
    )

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print("Latest model and vectorizer loaded successfully!")


def sigmoid(x):
    """Convert decision score into confidence-like probability"""
    return 1 / (1 + math.exp(-x))


def classify_confidence(confidence: float):
    """
    confidence is between 0 and 1
    """

    if confidence >= 0.85:
        return "Verified Real / Highly Reliable"

    elif 0.60 <= confidence < 0.85:
        return "Likely Real"

    elif 0.40 <= confidence < 0.60:
        return "Mixed Signals"

    elif 0.15 <= confidence < 0.40:
        return "Likely Fake"

    else:
        return "Highly Suspicious / Fake"


def predict_news(text: str):
    if text is None or text.strip() == "":
        return "ERROR: Empty input", 0.0

    text_vector = vectorizer.transform([text])

    # Model raw decision score
    decision_score = model.decision_function(text_vector)[0]

    # Convert to confidence score (0 to 1)
    confidence = sigmoid(decision_score)

    # Classify using confidence ranges
    result_label = classify_confidence(confidence)

    return result_label, confidence


if __name__ == "__main__":
    print("\nFake News Detection System")
    print("Using latest trained model.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter news headline/text: ")

        if user_input.lower() == "exit":
            print("Exiting...")
            break

        label, confidence = predict_news(user_input)

        print("\nPrediction:", label)
        print(f"Confidence Score: {round(confidence * 100, 2)}%")
        print("-" * 40)