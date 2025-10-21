import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def run_training():
    print("Loading train.csv and validation.csv...")
    try:
        dataset = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'validation.csv'})
    except FileNotFoundError:
        print("Error: 'train.csv' or 'validation.csv' not found.")
        print("Please run '2_split_data.py' first.")
        return

    # Using BioBERT since it's pre-trained on medical text
    model_name = "monologg/biobert_v1.1_pubmed"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create label mappings
    labels_list = pd.read_csv("train.csv")['final_label'].unique().tolist()
    labels_list.sort()
    
    label2id = {label: i for i, label in enumerate(labels_list)}
    id2label = {i: label for i, label in enumerate(labels_list)}
    
    print(f"Label mapping created: {label2id}")

    def preprocess_function(examples):
        tokenized_inputs = tokenizer(examples['text'], truncation=True, padding=True)
        tokenized_inputs['label'] = [label2id[label] for label in examples['final_label']]
        return tokenized_inputs

    print("Tokenizing the dataset...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        f1 = f1_score(labels, predictions, average='weighted')
        acc = accuracy_score(labels, predictions)
        
        return {'accuracy': acc, 'f1': f1}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="./medical_sentiment_model",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # Effective batch size of 16
        num_train_epochs=3, 
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")

    trainer.save_model("./medical_sentiment_model/final_model")
    tokenizer.save_pretrained("./medical_sentiment_model/final_model")
    print("Final model saved to './medical_sentiment_model/final_model'")

if __name__ == "__main__":
    run_training()