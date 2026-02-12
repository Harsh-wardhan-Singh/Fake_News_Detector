import joblib
from pathlib import Path
from newsprocessor import clean_text


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
        return "ERROR: Empty input", "UNKNOWN", 0.0
    
    text = clean_text(text)

    text_vector = vectorizer.transform([text])

    # Probabilities from calibrated model
    probs = model.predict_proba(text_vector)[0]

    # model.classes_ tells order of labels
    classes = model.classes_

    # Find REAL probability safely
    if "REAL" in classes:
        real_index = list(classes).index("REAL")
        confidence_real = probs[real_index]
    else:
        confidence_real = max(probs)  # fallback

    predicted_label = model.predict(text_vector)[0]

    # Confidence label (Likely Real/Fake etc.)
    # confidence_label = classify_confidence(confidence_real)

    return predicted_label, confidence_real #confidence_label


if __name__ == "__main__":
    print("\nFake News Detection System")
    print("Using latest trained model.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter news headline/text: ")

        if user_input.lower() == "exit":
            print("Exiting...")
            break

        predicted_label, confidence = predict_news(user_input) #confidence label

        # print("\nPrediction:", confidence_label)
        print("Model Label:", predicted_label)
        print(f"Confidence: {round(confidence, 4)}")
        print("-" * 40)