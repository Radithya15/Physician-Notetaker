import json
from nlp_summarizer import extract_medical_details
from sentiment_analyzer import analyze_patient_utterance
from soap_generator import generate_soap_note

with open('transcript.txt', 'r', encoding='utf-8') as f:
    transcript = f.read()

print("1. Medical NLP Summarization")
medical_summary = extract_medical_details(transcript)
print(json.dumps(medical_summary, indent=2))
print("\n" + "="*40 + "\n")


print("2. Sentiment & Intent Analysis")
utterance_1 = "I'm doing better, but I still have some discomfort now and then."
utterance_2 = "That’s great to hear. So, I don’t need to worry about this affecting me in the future?"

print(f"Utterance: '{utterance_1}'")
analysis_1 = analyze_patient_utterance(utterance_1)
print(json.dumps(analysis_1, indent=2))

print(f"\nUtterance: '{utterance_2}'")
analysis_2 = analyze_patient_utterance(utterance_2)
print(json.dumps(analysis_2, indent=2))
print("\n" + "="*40 + "\n")


print("3. SOAP Note Generation (Bonus)")
soap_note = generate_soap_note(transcript)
print(json.dumps(soap_note, indent=2))
print("\n" + "="*40 + "\n")