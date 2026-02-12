from newsprocessor import clean_text
from prediction_model_SVC import predict_news


def format_result(label, confidence):
    confidence_percent = round(confidence * 100, 2)

    if label == "REAL":
        if confidence >= 0.85:
            verdict = "Highly Reliable"
        elif confidence >= 0.60:
            verdict = "Likely Real "
        else:
            verdict = "Uncertain (but leaning towards REAL)"

    else:  # FAKE
        if confidence >= 0.85:
            verdict = "Highly Suspicious"
        elif confidence >= 0.60:
            verdict = "Likely Fake"
        else:
            verdict = "Uncertain (but leaning towards FAKE)"

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