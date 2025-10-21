from transformers import pipeline

# Using DistilBERT for general sentiment (would swap for ClinicalBERT in production)
sentiment_classifier = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Mental health intent model works pretty well for medical conversations
intent_classifier = pipeline(
    "text-classification", 
    model="mindpadi/intent_classifier"
)

def map_sentiment(hf_result):
    """Convert model output (POSITIVE/NEGATIVE) to medical context labels."""
    label = hf_result[0]['label']
    score = hf_result[0]['score']
    
    if label == "NEGATIVE" and score > 0.7:
        return "Anxious"
    elif label == "POSITIVE":
        return "Reassured"
    else:
        return "Neutral"

def map_intent(hf_result):
    """Map mental health intent labels to medical consultation intents."""
    label = hf_result[0]['label']
    
    if label in ["help_request", "crisis", "vent"]:
        return "Expressing concern / Seeking reassurance"
    if label in ["request_info", "journal_analysis"]:
        return "Reporting symptoms"
    else:
        return "General statement"

def analyze_patient_utterance(text):
    """Run sentiment and intent classification on patient text."""
    sentiment_result = sentiment_classifier(text)
    intent_result = intent_classifier(text)
    
    output = {
        "Sentiment": map_sentiment(sentiment_result),
        "Intent": map_intent(intent_result)
    }
    
    return output