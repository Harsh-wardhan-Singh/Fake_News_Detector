import os
import json
import joblib
import pandas as pd
from datetime import datetime
from pathlib import Path
from newsprocessor import preprocess_dataframe

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "datasets" / "training_data.csv"

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

LATEST_MODEL_PATH = MODELS_DIR / "fake_news_model_latest.pkl"
LATEST_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer_latest.pkl"

BACKUP_MODEL_PATH = MODELS_DIR / "fake_news_model_backup.pkl"
BACKUP_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer_backup.pkl"

METADATA_FILE = MODELS_DIR / "model_metadata.json"

TEST_SIZE = 0.2
RANDOM_STATE = 7
MIN_ACCURACY_TO_REPLACE = 0.0  


if not DATA_FILE.exists():
    print("ERROR: training_data.csv not found!")
    exit()

df = pd.read_csv(DATA_FILE, low_memory=False)
df = preprocess_dataframe(df)

required_cols = {"text", "label"}
if not required_cols.issubset(df.columns):
    print("ERROR: training_data.csv must contain columns: text, label")
    exit()

df = df.dropna(subset=["text", "label"]).reset_index(drop=True)

df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(str)

df = df[df["text"].str.strip() != ""]
df = df[df["label"].str.strip() != ""]

print("Dataset size after cleaning:", df.shape)


print("\nLabel distribution:")
print(df["label"].value_counts())


try:
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )
except ValueError:
    print("\nWARNING: Stratify failed (some class too small). Using normal split.")
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )


tfidf_vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    min_df=2,
    ngram_range=(1, 2)
)

tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)


model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)


y_pred = model.predict(tfidf_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n-------------------")
print(f"Accuracy: {round(accuracy * 100, 2)}%")
print("-------------------")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

versioned_model_path = MODELS_DIR / f"fake_news_model_{timestamp}.pkl"
versioned_vectorizer_path = MODELS_DIR / f"tfidf_vectorizer_{timestamp}.pkl"

joblib.dump(model, versioned_model_path)
joblib.dump(tfidf_vectorizer, versioned_vectorizer_path)

print(f"\nSaved versioned model: {versioned_model_path.name}")
print(f"Saved versioned vectorizer: {versioned_vectorizer_path.name}")


old_best_accuracy = 0.0

if METADATA_FILE.exists():
    try:
        with open(METADATA_FILE, "r") as f:
            old_data = json.load(f)
            old_best_accuracy = old_data.get("best_accuracy", 0.0)
    except:
        old_best_accuracy = 0.0


should_replace = False

if accuracy >= MIN_ACCURACY_TO_REPLACE and accuracy >= old_best_accuracy:
    should_replace = True

if should_replace:
    print("\nNew model is good. Updating latest model...")

    # Backup old latest model if it exists
    if LATEST_MODEL_PATH.exists():
        os.replace(LATEST_MODEL_PATH, BACKUP_MODEL_PATH)
        print("Old latest model moved to backup.")

    if LATEST_VECTORIZER_PATH.exists():
        os.replace(LATEST_VECTORIZER_PATH, BACKUP_VECTORIZER_PATH)
        print("Old latest vectorizer moved to backup.")

    # Save new latest model
    joblib.dump(model, LATEST_MODEL_PATH)
    joblib.dump(tfidf_vectorizer, LATEST_VECTORIZER_PATH)

    print("Latest model updated successfully.")

    # Update metadata
    metadata = {
        "best_accuracy": accuracy,
        "trained_on_rows": len(df),
        "timestamp": timestamp,
        "latest_model": str(LATEST_MODEL_PATH.name),
        "latest_vectorizer": str(LATEST_VECTORIZER_PATH.name)
    }

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

    print("Metadata updated.")

else:
    print("\nNew model did NOT beat the old model.")
    print("Latest model NOT replaced.")
    print(f"Best accuracy remains: {round(old_best_accuracy * 100, 2)}%")