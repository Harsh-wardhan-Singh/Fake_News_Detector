import joblib
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


def predict_news(text: str):
    if text is None or text.strip() == "":
        return "ERROR: Empty input"

    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)

    return prediction[0]


if __name__ == "__main__":
    print("\nFake News Detection System")
    print("Using latest trained model.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter news headline/text: ")

        if user_input.lower() == "exit":
            print("Exiting...")
            break

        result = predict_news(user_input)
        print("\nPrediction:", result)
        print("-" * 40)