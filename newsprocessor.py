import re
import pandas as pd

def clean_text(text: str) -> str:
    if text is None or pd.isna(text):
        return ""

    text = str(text)

    # 1) Lowercase
    text = text.lower()

    # 2) Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # 3) Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # 4) Remove punctuation + special characters (keep letters + spaces)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # 5) Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


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

