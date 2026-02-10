import numpy as np
import pandas as pd
import itertools
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report


def News_Detection_Module():
    file_path = Path(__file__).parent / "training_data.csv"

    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print("Error: training_data.csv not found in project folder.")
        exit()

    df = df[['title', 'text', 'label']]
    df = df.dropna(subset=['text', 'label']).reset_index(drop=True)

    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(str)

    df = df[df['text'].str.strip() != ""]
    df = df[df['label'].str.strip() != ""]

    x_train, x_test, y_train, y_test = train_test_split(
    df['text'], df['label'],
    test_size=0.2,
    random_state=7,
    )

    tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    min_df=2,
    ngram_range=(1,2)
    )

    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)

    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)

    y_pred = pac.predict(tfidf_test)

    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {round(score*100,2)}%')