import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split

# Automatically label patient utterances and split into train/validation sets

def auto_label_and_split():
    analyzer = SentimentIntensityAnalyzer()

    try:
        df = pd.read_csv("patient_utterances.csv")
    except FileNotFoundError:
        print("Error: 'patient_utterances.csv' not found.")
        print("Please run '1_extract_patient_lines.py' first.")
        return

    def get_vader_label(sentence):
        score = analyzer.polarity_scores(sentence)['compound']
        # Use standard VADER thresholds
        if score >= 0.05:
            return "RELIEVED/REASSURED"
        elif score <= -0.05:
            return "ANXIOUS/CONCERNED"
        else:
            return "NEUTRAL/REPORTING"

    print("Applying automated VADER sentiment labels...")
    df['final_label'] = df['text'].apply(get_vader_label)
    df.to_csv("patient_labels_FINAL.csv", index=False)
    print("Saved all auto-labeled data to 'patient_labels_FINAL.csv'")

    print("Splitting data into train and validation sets...")
    label_counts = df['final_label'].value_counts()
    if (label_counts < 2).any():
        print("Error: The automated labeling resulted in less than 2 examples for a label class.")
        print(label_counts)
        return

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['final_label'],
        random_state=42
    )

    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("validation.csv", index=False)

    print("\nSUCCESS!")
    print(f"Split data into {len(train_df)} training rows and {len(val_df)} validation rows.")
    print("Files 'train.csv' and 'validation.csv' are ready for training.")
    print("You can now run '3_train_model.py'.")

if __name__ == "__main__":
    auto_label_and_split()