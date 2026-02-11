import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "datasets" / "Big_training_data.csv"   # change if name differs

df = pd.read_csv(DATA_FILE, low_memory=False)

print("Dataset shape:", df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nUnique labels:")
print(df["label"].unique())

