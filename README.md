# DSA4213 Assignment 3 — Full Fine-Tuning vs LoRA on TweetEval Sentiment

This repository contains the implementation and experiments comparing **Full Fine-Tuning** and **LoRA (Low-Rank Adaptation)** on the **TweetEval Sentiment** dataset using the **DistilBERT** model.

---

## 1. Environment Setup

```bash
# (optional) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Mac/Linux
# .venv\Scripts\activate           # Windows PowerShell

# install dependencies
pip install -r requirements.txt
```
## 2. How to Run the Experiments
```bash
python main.py
```
This command will:
Download the TweetEval sentiment dataset automatically

Train two models:

1. Full Fine-Tuning

2. LoRA (parameter-efficient fine-tuning)

Evaluate on validation and test splits

Generate results (metrics & plots)

Save outputs to the outputs/ directory

## 3. Dataset Handling
No dataset upload is required.

main.py automatically handles loading:

python
```bash
from datasets import load_dataset
load_dataset("tweet_eval", "sentiment")
```
## 4. Output Files
After running main.py, the following will be created in outputs/:

results.json — evaluation metrics

Training/validation loss curves

Accuracy, precision, recall, and F1 score bar charts

(Optional) Confusion matrices

Saved model folders:

distilbert_full/

distilbert_lora/

Directories are created automatically if missing.

## 5. Entry Point
To reproduce all experiments end-to-end:
```bash
python main.py
```
This will:

Tokenize the dataset

Train both strategies

Evaluate on validation and test sets

Save results and figures

##6. Repository Structure

Assignment3/
├── main.py                 # Entry script to run everything
├── requirements.txt        # Dependency list
├── README.md               # This file
├── outputs/                # Generated after running main.py
