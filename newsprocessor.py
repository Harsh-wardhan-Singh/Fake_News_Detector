import re
import pandas as pd
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Load spaCy model (install: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text)

    # 1) Lowercase (Case normalization)
    text = text.lower()

    # 2) Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # 3) Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # 4) Remove punctuation + special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # 5) Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # 6) Lemmatization + Stopword removal
    doc = nlp(text)

    cleaned_words = []
    for token in doc:
        lemma = token.lemma_.strip()

        if lemma and lemma not in ENGLISH_STOP_WORDS and len(lemma) > 2:
            cleaned_words.append(lemma)

    return " ".join(cleaned_words)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans text column and returns cleaned dataframe.
    """
    df = df.copy()

    # Drop missing
    df = df.dropna(subset=["text", "label"])

    # Apply cleaning
    df["text"] = df["text"].apply(clean_text)

    # Remove empty rows after cleaning
    df = df[df["text"].str.strip() != ""]

    # Reset index
    df = df.reset_index(drop=True)

    return df