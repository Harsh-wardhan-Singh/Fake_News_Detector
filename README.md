#  Fake News Detection System (AI-Powered Web App)

 **Live Demo:** https://fake-news-detector-bg89.onrender.com/  
*(Click above to try the system in real time!)*

---

##  Overview

The **Fake News Detection System** is a full-stack machine learning web application designed to classify news headlines/text as **REAL** or **FAKE** using modern Natural Language Processing (NLP) and Machine Learning (ML) techniques.

This project demonstrates a complete **end-to-end pipeline**:

 Data preprocessing → ML training → Model deployment → Interactive web interface

It combines:
- **Machine Learning** (TF-IDF + LinearSVC)
- **Text preprocessing**
- **Flask backend**
- **HTML/CSS/JavaScript frontend**
- **Cloud deployment (Render.com)**

---

##  Motivation

In today’s digital age, misinformation spreads rapidly through social platforms, blogs, and messaging channels, leading to confusion, fear, and societal discord. This project aims to:

- Help users quickly evaluate the reliability of a piece of news  
- Provide transparent AI confidence scores  
- Promote awareness and critical thinking  
- Serve as an educational example of ML-based web systems

---

##  Architecture 

                   ┌────────────────┐
                   │   Web Browser  │
                   │ (User Input)   │
                   └───────▲────────┘
                           │
                    POST text via Form
                           │
                   ┌───────▼────────┐
                   │   Flask API    │
                   │ process_input()│
                   └───────▲────────┘
                           │
                 TF-IDF Vectorizer transforms text
                           │
                    Model predicts label
                           │
               Confidence calculation & formatting
                           │
                   ┌───────▼────────┐
                   │ Frontend Result│
                   │ (REAL/FAKE UI) │
                   └────────────────┘
---

##  Core Components

###  1. Data Collection & Cleaning

Multiple CSV datasets were merged and preprocessed:
- Fake vs Real datasets
- RealFakeNews data
- Custom aggregated dataset

Text was cleaned by:
- Lowercasing  
- HTML stripping  
- URL removal  
- Punctuation removal  
- Stop words removal  

The final merged dataset was filtered to remove:
- Too short/long texts  
- Duplicate texts  

---

###  2. Feature Extraction (TF-IDF)
We use **Term Frequency–Inverse Document Frequency (TF-IDF)** to convert raw text into meaningful numeric vectors.

Key settings:
```python
TfidfVectorizer(
    stop_words="english",
    max_df=0.9,
    min_df=2,
    ngram_range=(1,2),
    max_features=50000,
    sublinear_tf=True
)
```
This allows the model to capture both:

- single words

- two-word combinations (bigrams)

### 3. Machine Learning Model

We used:

- Linear Support Vector Classifier (LinearSVC)
wrapped with

- CalibratedClassifierCV
→ to provide predict_proba() confidence scores

This combo gave us:

- High accuracy

- Fast inference

- Confidence probabilities

Typical performance:
```mathamatica
Accuracy ~94%
F1-Score ~0.94
Recall ~0.92
```

### 4. Model Deployment (Render.com)

The trained model and vectorizer are saved as:
```bash
models/fake_news_model_latest.pkl  
models/tfidf_vectorizer_latest.pkl
```
These are loaded inside:
```python
prediction_model.py
```
Deployment manages:
- Dependency installation
- Flask app hosting
- Gunicorn web server

## How It Works (User Flow)

- User enters headline/text in search bar

- Flask receives the text

- Input is cleaned using text preprocessing

- Vectorizer transforms it

- Model predicts:

     - Label (REAL or FAKE)

     - Confidence score

     - Reliability category

- Results displayed in UI

  
## Confidence Interpretation

Predictions include a confidence score between **0 and 1**, representing how confident the model is about its classification.

We categorize confidence into the following ranges:

| Confidence Score | Interpretation |
|-----------------|----------------|
| **≥ 0.85**      | Highly Reliable / Highly Suspicious |
| **0.60 – 0.85** | Likely Real / Likely Fake |
| **0.40 – 0.60** | Mixed Signals (Uncertain) |
| **< 0.40**      | Very Uncertain / Weak Prediction |

---

##  Example

**Input:**  
`"NASA confirms discovery of alien life"`

**Prediction:** Fake  
**Confidence:** 87.43%  
**Category:** Likely Fake   

---

 **Open the live site to try more!**  
 https://fake-news-detector-bg89.onrender.com/

---

## Full Code Structure

```text
project_root
├── app.py                  # Flask app
├── main.py                 # Inference logic
├── newsprocessor.py        # Text cleaning utilities
├── train_model.py          # ML training script
├── prediction_model_SVC.py # Model loading + prediction
├── requirements.txt        # Python dependencies
├── models/
│   ├── fake_news_model_latest.pkl
│   └── tfidf_vectorizer_latest.pkl
├── datasets/               # Source CSVs
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── README.md
```
## Installation (Local Development)
Clone Repository
```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

Install Dependencies
```bash
pip install -r requirements.txt
```

Run Locally
```bash
python app.py
```

Once running, open:
```text
http://127.0.0.1:5000
```
## Deployment

This project is hosted using Render.com with the following configuration:

**Build Command**
```bash
pip install -r requirements.txt
```

**Start Command**
```bash
gunicorn app:app
```
 The trained models are already included inside the repository, so no retraining is required after deployment.

## Future Improvements

- Expand dataset diversity for better real-world accuracy
- Add lazy loading for large models
- Add REST API for external integration
- Add caching for repeated queries
- Add user login / history tracking

## Contact

Created by Harsh Wardhan Singh.
Reach out on socials linked on the website.

## Live Demo

Visit  https://fake-news-detector-bg89.onrender.com/

Try your own headlines!

Thank you for using this project — Feedback and stars on GitHub are appreciated!
