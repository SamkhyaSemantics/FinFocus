# FinFocus
Knowledge Distillation: FinBERT to Tiny BERT using finance-alpaca. This repository presents a knowledge distillation pipeline to train a compact BERT model (student) for finance domain tasks, using a large, domain-specialized FinBERT model (teacher). The process uses the open gbharti/finance-alpaca dataset and Hugging Face Transformers library.

## Overview
- Teacher Model: ProsusAI/finbert
- Student Model: bert-base-uncased (with explicit classification head)
- Dataset: gbharti/finance-alpaca
- Distillation Loss: Mix of hard (true label) and soft (teacher output) supervision
- Frameworks: PyTorch, Hugging Face Transformers/Datasets

# Main Features
- Loads FinBERT and a student BERT model with a custom classification head without initialization warnings
- Uses the teacher’s predictions to create "soft labels" as distillation targets
- Distillation loss combines: Hard label cross-entropy (student <-> teacher's argmax) and Soft label KL-divergence (student <-> teacher's output probabilities)
- Fully scriptable and easy to adapt to new financial datasets or models
- Trains a lightweight model that closely matches FinBERT’s behavior on finance prompts

## Quickstart
1. Install dependencies
pip install torch transformers datasets

3. Run the script
python succinctBERT.py

5. Result
The distilled student model and tokenizer are saved in ./finbert-distilled-student

## Code Structure
- Teacher + Student: Loads both as Hugging Face models. The student model is defined with a custom linear head to avoid weight mismatch warnings.
- Data Preparation: Loads the finance-alpaca dataset, tokenizes it, and uses the teacher to generate both hard and soft label targets for each example.
- Distillation Training: A custom Trainer runs, using a custom loss that blends hard (classification) and soft (distribution) supervision.
- Saving: Trained model artifacts are saved for downstream use.

## How It Works
- Initialize Models
- Teacher: Full FinBERT loaded from Hugging Face (sequence classification)
- Student: bert-base-uncased + custom classification head
- Tokenize Dataset
- The "input" column of finance-alpaca is tokenized to produce inputs for classification.
- Teacher Labeling
- For each example, teacher FinBERT outputs both softmaxed probabilities (soft labels) and predicted class (hard label).

## Training
- The student is trained with a custom loss:
- Hard loss = cross-entropy with teacher's predicted class
- Soft loss = KL-divergence from student to teacher probabilities (temperature-scaled)
- Both losses are mixed for best transfer.

## Evaluation & Saving
- The student model is evaluated during training and saved at the end.

## Customization
- Change Student Model: Swap out for any small Transformer architecture.
- Change Dataset: Replace "gbharti/finance-alpaca" with another Hugging Face dataset (ensure input column name matches).
- Tune Distillation: Adjust alpha (mix factor) or temperature for different knowledge blending.

## Citation
If you use this template or workflow in research, please cite the underlying models and datasets (FinBERT, finance-alpaca, Hugging Face Transformers).
