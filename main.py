from newsprocessor import clean_text
from prediction_model_SVC import predict_news


def format_result(label, confidence):
    confidence_percent = round(confidence * 100, 2)

    
    if confidence >= 0.85:
            verdict = "Highly Reliable (Most likely REAL)"
    elif confidence >= 0.60 and confidence < 0.85:
            verdict = "Likely Real "
    elif confidence >= 0.50 and confidence < 0.60:
            verdict = "Uncertain (but leaning towards REAL)"
    elif confidence >= 0.40 and confidence < 0.50:
            verdict = "Uncertain (but leaning towards FAKE)"
    elif confidence >= 0.15 and confidence < 0.40:
            verdict = "Likely Fake"
    elif confidence >= 0.0 and confidence < 0.15:
            verdict = "Highly Unreliable(Most likely FAKE)"

    return {
        "label": label,
        "confidence": confidence_percent,
        "verdict": verdict
    }


def process_input(user_input: str):
    """
    This function is called by Flask.
    It preprocesses user input and returns prediction results.
    """

    cleaned_text = clean_text(user_input)

    if cleaned_text == "":
        return {
            "label": "ERROR",
            "confidence": 0,
            "verdict": "No valid input provided."
        }

    label, confidence = predict_news(cleaned_text)

    return format_result(label, confidence)
