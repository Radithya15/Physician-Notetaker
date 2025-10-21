import pandas as pd
from datasets import load_dataset

def extract_patient_lines():
    print("Loading dataset 'har1/MTS_Dialogue-Clinical_Note'...")
    # Loads the clinical dialogue dataset
    dataset = load_dataset("har1/MTS_Dialogue-Clinical_Note", "default")['train']
    patient_lines = []
    for item in dataset:
        dialogue = item['dialogue']
        lines = dialogue.split('\n')
        for line in lines:
            # Only keeps the lines spoken by the patient
            if line.startswith('Patient:'):
                # Remove the speaker label and extra whitespace
                clean_line = line.replace('Patient:', '').strip()
                if clean_line:
                    patient_lines.append(clean_line)
    print(f"Found {len(patient_lines)} patient utterances.")
    # Save all patient utterances to a CSV file
    df = pd.DataFrame(patient_lines, columns=['text'])
    df.to_csv('patient_utterances.csv', index=False)
    print("Successfully saved all patient lines to 'patient_utterances.csv'")

if __name__ == "__main__":
    extract_patient_lines()