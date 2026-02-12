import pandas as pd
from pathlib import Path
import re

BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "datasets"

FAKE_FILE = DATASETS_DIR / "Fake.csv"
TRUE_FILE = DATASETS_DIR / "True.csv"
TRAINING_FILE = DATASETS_DIR / "training_data.csv"
REALFAKENEWS_FILE = DATASETS_DIR / "RealFakeNews.csv"

OUTPUT_FILE = DATASETS_DIR / "Big_training_data.csv"

# Headline-focused dataset limits
MIN_TEXT_LENGTH = 10
MAX_TEXT_LENGTH = 250


def clean_raw_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove strange encoding artifacts
    text = text.replace("Â", " ").replace("Ã", " ").replace("â", " ")

    # Collapse multiple spaces/newlines
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_labels(df):
    df["label"] = df["label"].astype(str).str.upper().str.strip()

    df["label"] = df["label"].replace({
        "0": "FAKE",
        "1": "REAL",
        "FALSE": "FAKE",
        "TRUE": "REAL"
    })

    return df


def filter_by_text_length(df):
    df["text_length"] = df["text"].apply(len)

    before = len(df)

    df = df[df["text_length"] >= MIN_TEXT_LENGTH]
    df = df[df["text_length"] <= MAX_TEXT_LENGTH]

    after = len(df)

    print("\nText Length Filtering Report:")
    print(f"Before: {before}")
    print(f"After: {after}")
    print(f"Removed: {before - after}")
    print(f"Min length allowed: {MIN_TEXT_LENGTH}")
    print(f"Max length allowed: {MAX_TEXT_LENGTH}")

    df = df.drop(columns=["text_length"])
    return df


def remove_duplicates(df):
    before = len(df)

    df = df.drop_duplicates()
    after_exact = len(df)

    df = df.drop_duplicates(subset=["text"])
    after_text = len(df)

    print("\nDuplicate Removal Report:")
    print(f"Rows before: {before}")
    print(f"After removing exact duplicates: {after_exact} (removed {before - after_exact})")
    print(f"After removing duplicate text: {after_text} (removed {after_exact - after_text})")

    return df


def load_fake_true_dataset(file_path, label_value):
    df = pd.read_csv(file_path, low_memory=False)

    if "title" not in df.columns:
        raise ValueError(f"{file_path.name} must contain 'title' column!")

    # USE ONLY HEADLINES
    df["text"] = df["title"].fillna("").astype(str).apply(clean_raw_text)

    df["label"] = label_value
    df = df[["text", "label"]]

    df = normalize_labels(df)
    df = df[df["text"].str.strip() != ""]

    return df


def load_training_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{file_path.name} must contain 'text' and 'label' columns!")

    df = df[["text", "label"]]
    df = df.dropna(subset=["text", "label"])

    df["text"] = df["text"].astype(str).apply(clean_raw_text)

    df = normalize_labels(df)
    df = df[df["text"].str.strip() != ""]
    df = df[df["label"].isin(["REAL", "FAKE"])]

    return df


def load_realfakenews(file_path):
    df = pd.read_csv(file_path, low_memory=False)

    possible_title_cols = ["title", "headline", "headlines","text"]
    possible_label_cols = ["label", "labels", "class"]

    title_col = None
    label_col = None

    for col in possible_title_cols:
        if col in df.columns:
            title_col = col
            break

    for col in possible_label_cols:
        if col in df.columns:
            label_col = col
            break

    if title_col is None or label_col is None:
        raise ValueError(
            f"{file_path.name} does not contain expected headline/label columns.\n"
            f"Found columns: {list(df.columns)}"
        )

    # ONLY USE HEADLINES
    df["text"] = df[title_col].fillna("").astype(str).apply(clean_raw_text)
    df["label"] = df[label_col].astype(str)

    df = df[["text", "label"]].copy()

    df = df.dropna(subset=["text", "label"])
    df = normalize_labels(df)

    df = df[df["text"].str.strip() != ""]
    df = df[df["label"].isin(["REAL", "FAKE"])]

    return df


def main():
    all_dfs = []

    print("Loading Fake.csv...")
    fake_df = load_fake_true_dataset(FAKE_FILE, "FAKE")
    print("Fake.csv loaded:", fake_df.shape)

    # print("Loading True.csv...")
    # true_df = load_fake_true_dataset(TRUE_FILE, "REAL")
    # print("True.csv loaded:", true_df.shape)

    print("Loading training_data.csv...")
    training_df = load_training_data(TRAINING_FILE)
    print("training_data.csv loaded:", training_df.shape)

    print("Loading RealFakeNews.csv (HEADLINES ONLY)...")
    realfake_df = load_realfakenews(REALFAKENEWS_FILE)
    print("RealFakeNews.csv loaded:", realfake_df.shape)

    all_dfs.extend([fake_df, training_df, realfake_df]) #true_df

    print("\nMerging all datasets...")
    combined = pd.concat(all_dfs, ignore_index=True)

    combined = combined.dropna(subset=["text", "label"])
    combined = normalize_labels(combined)

    combined = combined[combined["label"].isin(["REAL", "FAKE"])]
    combined = combined[combined["text"].str.strip() != ""]

    combined = filter_by_text_length(combined)
    combined = remove_duplicates(combined)

    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nFinal dataset size:", combined.shape)
    print("\nFinal label distribution:")
    print(combined["label"].value_counts())

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved merged dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()