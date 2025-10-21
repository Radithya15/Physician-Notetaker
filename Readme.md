# Physician Notetaker NLP Pipeline

## Introduction

This project addresses the AI Engineer task from Emitree. It's a Python-based system designed to process transcripts of doctor-patient conversations. The pipeline extracts important medical details, analyzes the patient's sentiment during the talk, and generates a structured SOAP note summarizing the encounter.

The main prototype uses established NLP libraries like `spaCy` (specifically `scispaCy` for medical text) and `Hugging Face Transformers`, combined with some custom logic to ensure accuracy and reliability.

---

## Core Features

* **Medical Information Extraction:** Identifies key items like symptoms, the diagnosis, and treatments mentioned in the conversation.
* **Sentiment Analysis:** Assesses the patient's likely emotional state (Anxious, Neutral, Reassured) based on their statements.
* **SOAP Note Generation:** Automatically structures the conversation into the standard Subjective, Objective, Assessment, and Plan format used in clinical notes.

---

## Running the Prototype

Follow these steps to set up and run the main pipeline which fulfills the original assignment requirements.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Radithya15/Physician-Notetaker.git
    cd Physician-Notetaker
    ```

2.  **Set Up a Virtual Environment (Crucial Step):**
    Using a virtual environment is **strongly recommended**. Certain dependencies, especially the `scispaCy` medical model, have compatibility issues with the latest Python versions (e.g., 3.12+).
    Creating an environment with an older Python version (like 3.10 or 3.11) ensures all packages install correctly without affecting your main system setup.

    ```bash
    # Example using Python 3.10 (adjust path if needed)
    python3.10 -m venv physician
    # On Windows:
    .\physician\Scripts\activate
    # On macOS/Linux:
    # source physician/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install the scispaCy Model:**
    This specialized medical NER model needs to be installed separately.
    ```bash
    pip install [https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz)
    ```

5.  **Run the Main Script:**
    ```bash
    python main.py
    ```
    The script will process the `transcript.txt` file and print the JSON results for summarization, sentiment analysis, and the SOAP note to your console.

---

## Technical Details and Assignment Questions

This section provides answers to the questions posed in the assignment brief and explains some of the technical choices.

### Medical NLP Summarization

* **Q: What pre-trained NLP models would you use for medical summarization?**
    * **A:** The prototype uses `scispaCy`'s `en_ner_bc5cdr_md`. This model is specifically trained on biomedical literature, making it highly effective at recognizing `DISEASE` and `CHEMICAL` entities compared to general-purpose models. For specific phrases not covered by the NER model (like "ten sessions of physiotherapy"), `spaCy`'s rule-based `Matcher` was used for reliable extraction.

* **Q: How would you handle ambiguous or missing medical data in the transcript?**
    * **A:** If data is **missing**, the system outputs `null` or uses a reasonable default derived from the conversation context. For **ambiguous** patient statements, the standard practice in a production system is to implement a Human-in-the-Loop (HITL) workflow. The AI would flag uncertain extractions, allowing a human expert to quickly review and confirm or correct them, ensuring clinical safety.

### Sentiment & Intent Analysis

* **Q: How would you fine-tune BERT for medical sentiment detection?**
    * **A:** Fine-tuning is essential because general sentiment models don't grasp medical context well. The best method involves adapting a domain-specific model using relevant data.
    * **Demonstration:** To illustrate this process thoroughly, I developed a complete fine-tuning pipeline for this project (see "Bonus: Sentiment Model Training Pipeline" below). The core steps are:
        1.  **Obtain Data:** Use a suitable public dataset like `har1/MTS_Dialogue-Clinical_Note`.
        2.  **Generate Labels:** Since this dataset lacks sentiment labels, a labeling strategy is needed. Two primary methods were considered:
            * **Method 1 (BART Zero-Shot + Manual Review):** Use a powerful zero-shot model (like BART) to generate initial label guesses, followed by essential manual review and correction. This offers higher potential quality due to BART's contextual understanding but requires significant human effort.
            * **Method 2 (VADER Automated Labeling):** Use a simpler, rule-based tool like VADER to automatically assign labels based on sentiment scores. This is much faster and requires no manual work, providing a good baseline quality, though it might miss nuances.
            * **Chosen Method:** For this project, **VADER was used** (`2_split_data`) because manual review wasn't feasible. This provided a fully automated way to create the necessary training data quickly.
        3.  **Select a Base Model:** Start with a medically-aware model like `monologg/biobert_v1.1_pubmed`.
        4.  **Train:** Use a script (`3_train_model.py`) with the Hugging Face `Trainer` API to fine-tune the base model on the VADER-labeled data.
    * **Outcome:** This process yielded a custom sentiment model (included in `/final_model`) that achieved **89.8% accuracy** on validation data, proving the effectiveness of the automated labeling and fine-tuning strategy.

* **Q: What datasets would you use for training a healthcare-specific sentiment model?**
    * **A:** Finding high-quality, publicly available datasets specifically labeled for sentiment in general *medical dialogues* is challenging. However, several good options and strategies exist:
        * **Domain-Specific Dialogue Datasets (requiring labeling):** Datasets like `MTS-Dialog` (available on GitHub and Hugging Face) contain patient-doctor conversations. While excellent for understanding medical context, they typically lack pre-existing sentiment labels. These would need to be added, either manually or programmatically (as demonstrated in the bonus training pipeline of this project).
        * **Related Domain Datasets:** Datasets focused on emotional expression in adjacent fields can be very useful. For instance, the `Amod/mental_health_counseling_conversations` dataset on Hugging Face contains dialogues with clear emotional content that can help train a model to recognize patient sentiment.
        * **Social Media and Forum Data:** Curated datasets from health-related social media (like specialized subreddits or Twitter datasets focused on health discussions) or medical forums can be gathered. These often require significant cleaning and manual labeling but can provide real-world examples of patient sentiment.
        * **Patient Review Data:** Datasets consisting of patient reviews for doctors or treatments often contain strong sentiment expressions, though the format differs from conversational dialogue.
        
### SOAP Note Generation

* **Q: How would you train an NLP model to map medical transcripts into SOAP format?**
    * **A:** A common deep learning approach is to train a sequence-to-sequence model (like T5 or BART). Using a dataset like `MTS-Dialog` which pairs dialogues with SOAP notes, the model learns to transform the conversation into the structured note format.
    * **Prototype Method:** This project implements a **hybrid NLP model** for SOAP generation. The `soap_generator.py` script uses:
        1.  **Rules:** It analyzes the speaker and keywords in each line to classify sentences into Subjective, Objective, Assessment, or Plan sections.
        2.  **NER:** It leverages the `scispaCy` model to extract specific medical terms within those classified sentences. This approach is reliable, easier to interpret, and effective for this specific task.
* **Q: What rule-based or deep-learning techniques would improve the accuracy of SOAP note generation?**
    * **A:** Several techniques can enhance SOAP note accuracy:
        * **Rule-Based Methods:**
            * **Enhanced Keyword/Pattern Matching:** Developing more comprehensive lists of keywords, phrases, and regular expressions specific to S/O/A/P sections can increase coverage.
            * **Speaker Diarization:** Accurately identifying who is speaking (Patient vs. Physician) allows for stronger rules (e.g., patient statements are usually Subjective).
            * **Dependency Parsing:** Analyzing the grammatical structure of sentences can help identify relationships (e.g., linking a medication to a dosage mentioned in the Plan).
        * **Deep Learning Methods:**
            * **Sentence Classification:** Training a transformer model (like `ClinicalBERT` or `BioBERT`) on labeled transcript sentences can classify each sentence as S, O, A, P, or irrelevant, capturing context better than simple keywords.
            * **Sequence-to-Sequence (Seq2Seq) Models:** Fine-tuning models like T5 or BART on paired (transcript, SOAP note) data enables them to learn the direct transformation, potentially generating more fluent and comprehensive notes.
            * **Named Entity Recognition (NER) + Relation Extraction:** Using advanced NER models to identify all medical entities and then training a relation extraction model to link them (e.g., connecting a diagnosis in Assessment to a treatment in Plan) improves coherence.
        * **Hybrid Approach:** The most robust method often combines deep learning for understanding context and extracting information (NER, sentence classification) with rule-based systems for reliably structuring the final SOAP note according to required formats and clinical logic.
---

## Bonus: Sentiment Model Training Pipeline

As a practical answer to the fine-tuning question, I've included the scripts for getting a resulting model from a complete sentiment analysis training pipeline. This demonstrates the process of creating a high-performance, domain-specific model using automated labeling.

*(The main `main.py` script uses a standard pre-trained model as requested in the initial task. This pipeline represents the next step towards a production-grade system.)*

**Note on Execution:** Due to local GPU memory limitations (requiring >2GB VRAM), the `3_train_model.py` script was successfully run and validated using Google Colab's free GPU resources. The script includes memory-optimization settings (`batch_size=1`, `gradient_accumulation_steps=16`) designed for low-VRAM environments.

### Training Files Included:

* **`1_extract_patient_lines.py`**: Extracts patient dialogue lines from the `MTS-Dialog` dataset.
* **`2_split_data`**: Automatically labels the extracted lines using VADER and splits the data into `train.csv` and `validation.csv`.

* **`3_train_model.py`**: Fine-tunes the `monologg/biobert_v1.1_pubmed` model using the prepared data.Running this script (python 3_train_model.py) will generate the final, trained model.This model achieved 89.8% accuracy on the validation set during testing.
