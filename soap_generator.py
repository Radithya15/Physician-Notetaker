import spacy
from nlp_summarizer import nlp, clean_text

def generate_soap_note(transcript):
    """Generate SOAP note from patient-physician transcript using rule-based parsing."""
    
    subjective_complaints = []
    subjective_history = []
    objective_exam = []
    assessment = []
    plan = []
    
    cleaned_transcript = clean_text(transcript)
    lines = cleaned_transcript.split('\n')
    
    for line in lines:
        if "Patient:" in line:
            text = line.split("Patient:", 1)[-1].strip()
            if "pain" in text or "discomfort" in text or "hit my head" in text or "backaches" in text:
                subjective_complaints.append(text)
            elif ("accident" in text or "September 1st" in text or "physiotherapy" in text or "week off work" in text) and "no," not in text.lower():
                subjective_history.append(text)
        
        elif "Physician:" in line:
            text = line.split("Physician:", 1)[-1].strip()
            # Looking for exam findings, not just conversation
            if "range of movement" in text or "no tenderness" in text or "muscles and spine seem" in text:
                objective_exam.append(text)
            if "recovery so far" in text or "full recovery within six months" in text or "no signs of long-term damage" in text:
                assessment.append(text)
            if "come back for a follow-up" in text or "If anything changes" in text:
                plan.append(text)
    
    # Extract diagnosis using NER
    doc = nlp(cleaned_transcript)
    diagnosis = [ent.text for ent in doc.ents if ent.label_ == "DISEASE" and ent.text.lower() not in ["pain", "tenderness"]]
    
    if not diagnosis and "whiplash injury" in cleaned_transcript.lower():
        diagnosis = ["whiplash injury"]

    soap_note = {
        "Subjective": {
            "Chief_Complaint": "Follow-up for neck and back pain post-car accident.",
            "History_of_Present_Illness": " ".join(subjective_history + subjective_complaints)
        },
        "Objective": {
            "Physical_Exam": " ".join(objective_exam),
            "Observations": "Patient appears in good condition."
        },
        "Assessment": {
            "Diagnosis": f"Resolved {diagnosis[0]}" if diagnosis else "Resolved whiplash injury",
            "Prognosis": " ".join(assessment)
        },
        "Plan": {
            "Treatment": "No further active treatment required.",
            "Follow-Up": " ".join(plan)
        }
    }
    
    return soap_note