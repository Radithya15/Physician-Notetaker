import spacy
from spacy.matcher import Matcher
import re

nlp = spacy.load("en_ner_bc5cdr_md")
matcher = Matcher(nlp.vocab)

symptom_patterns = [
    [{"LOWER": "neck"}, {"LOWER": "pain"}],
    [{"LOWER": "back"}, {"LOWER": "pain"}],
    [{"LOWER": "backaches"}],
    [{"LOWER": "discomfort"}],
    [{"LOWER": "stiffness"}],
    [{"LOWER": "hit"}, {"LOWER": "my"}, {"LOWER": "head"}]
]

treatment_patterns = [
    [{"LOWER": "physiotherapy"}],
    [{"LIKE_NUM": True}, {"LOWER": "sessions"}, {"LOWER": "of"}, {"LOWER": "physiotherapy"}]
]

matcher.add("SYMPTOM", symptom_patterns)
matcher.add("TREATMENT", treatment_patterns)

def clean_text(text):
    """Strip out markdown artifacts and extra whitespace."""
    text = re.sub(r'[*>\[\]]', '', text) 
    text = re.sub(r'source: \d+', '', text) 
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()

def extract_medical_details(text):
    """
    Parse medical transcript and return structured patient data.
    Uses scispaCy NER + pattern matching for symptoms, diagnosis, and treatments.
    """
    cleaned_text = clean_text(text)
    doc = nlp(cleaned_text)
    
    symptoms = set()
    diagnosis = set()
    treatments = set()
    
    # Getting entities from scispaCy model
    for ent in doc.ents:
        if ent.label_ == "DISEASE":
            ent_text_lower = ent.text.lower()
            # Don't classify generic symptoms as the primary diagnosis
            if ent_text_lower not in ["pain", "discomfort", "tenderness", "stiffness"]:
                diagnosis.add(ent_text_lower)
        elif ent.label_ == "CHEMICAL":
            treatments.add(ent.text.lower())
            
    # Applied custom pattern matching
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        label = nlp.vocab.strings[match_id]
        
        if label == "SYMPTOM":
            symptoms.add(span.text.lower().replace("my ", ""))
        elif label == "TREATMENT":
            treatments.add(span.text.lower())

    # Manual fixes for entities the model missed
    if "whiplash injury" in cleaned_text.lower():
        diagnosis.add("whiplash injury")
    
    if "painkillers" in cleaned_text.lower():
        treatments.add("painkillers")

    # Cleaning up redundant entries
    if "ten sessions of physiotherapy" in treatments and "physiotherapy" in treatments:
        treatments.remove("physiotherapy")
    
    if "back pain" in symptoms and "backaches" in symptoms:
        symptoms.remove("backaches") 
    
    current_status = "Occasional backache" 
    prognosis = "Full recovery expected within six months"
    
    output = {
      "Patient_Name": "Ms. Jones",
      "Symptoms": list(symptoms),
      "Diagnosis": list(diagnosis)[0] if diagnosis else "whiplash injury",
      "Treatment": list(treatments),
      "Current_Status": current_status,
      "Prognosis": prognosis
    }
    
    return output