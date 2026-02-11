import pandas as pd
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "datasets" / "Big_training_data.csv"   # change name if needed

print("Loading dataset...")
df = pd.read_csv(DATA_FILE, low_memory=False)

print("\n==============================")
print("DATASET BASIC INFO")
print("==============================")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Ensure columns exist
if "text" not in df.columns or "label" not in df.columns:
    print("\nERROR: Dataset must contain 'text' and 'label' columns.")
    exit()

df["text"] = df["text"].astype(str).fillna("")
df["label"] = df["label"].astype(str).fillna("")

df["text"] = df["text"].str.strip()
df["label"] = df["label"].str.strip().str.upper()

print("\n==============================")
print("LABEL CHECK")
print("==============================")
print(df["label"].value_counts())
print("\nUnique labels:", df["label"].unique())

print("\n==============================")
print("MISSING / EMPTY CHECK")
print("==============================")
empty_text = (df["text"].str.strip() == "").sum()
empty_label = (df["label"].str.strip() == "").sum()
print("Empty text rows:", empty_text)
print("Empty label rows:", empty_label)

print("\n==============================")
print("TEXT LENGTH CHECK")
print("==============================")
df["text_length"] = df["text"].str.len()

print(df["text_length"].describe())

print("\nShort text count:")
print("Text length < 30:", (df["text_length"] < 30).sum())
print("Text length < 50:", (df["text_length"] < 50).sum())
print("Text length < 100:", (df["text_length"] < 100).sum())

print("\n==============================")
print("DUPLICATE CHECK")
print("==============================")

exact_duplicates = df.duplicated(subset=["text", "label"]).sum()
text_only_duplicates = df.duplicated(subset=["text"]).sum()

print("Exact duplicates (same text + label):", exact_duplicates)
print("Text-only duplicates (same text repeated):", text_only_duplicates)

print("\nTop 10 most repeated texts:")
top_dup = df["text"].value_counts().head(10)
print(top_dup)

print("\n==============================")
print("CONFLICTING LABEL CHECK")
print("==============================")
# Same text appears with different labels
conflicts = df.groupby("text")["label"].nunique()
conflicting_texts = conflicts[conflicts > 1]

print("Texts appearing with BOTH REAL and FAKE labels:", len(conflicting_texts))

if len(conflicting_texts) > 0:
    print("\nExample conflicting texts:")
    sample_conflicts = conflicting_texts.head(5).index.tolist()
    for i, t in enumerate(sample_conflicts, start=1):
        labels = df[df["text"] == t]["label"].unique()
        print(f"\nConflict {i}: Labels = {labels}")
        print("Text preview:", t[:200], "...")

print("\n==============================")
print("URL / HTML / SYMBOL CHECK")
print("==============================")

url_count = df["text"].str.contains(r"http[s]?://", regex=True).sum()
html_count = df["text"].str.contains(r"<.*?>", regex=True).sum()
many_symbols_count = df["text"].str.contains(r"[^a-zA-Z0-9\s]{15,}", regex=True).sum()

print("Rows containing URLs:", url_count)
print("Rows containing HTML tags:", html_count)
print("Rows with heavy symbols/noise:", many_symbols_count)

print("\n==============================")
print("DATASET HEALTH SUMMARY")
print("==============================")
print("Total rows:", len(df))
print("REAL %:", round((df["label"] == "REAL").mean() * 100, 2))
print("FAKE %:", round((df["label"] == "FAKE").mean() * 100, 2))
print("Avg text length:", round(df["text_length"].mean(), 2))
print("Median text length:", round(df["text_length"].median(), 2))

print("\nDone.")
