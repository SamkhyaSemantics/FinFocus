import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertModel, BertConfig, Trainer, TrainingArguments
from datasets import load_dataset

# Load teacher: full FinBERT for classification
teacher_model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_name)
teacher_model.eval()

# Define student model with explicit head
student_model_name = "bert-base-uncased"
student_config = BertConfig.from_pretrained(student_model_name, num_labels=teacher_model.config.num_labels)
student_base = BertModel.from_pretrained(student_model_name, config=student_config)

class StudentModel(nn.Module):
    def __init__(self, bert_model, config):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.classifier.out_features), labels.view(-1))
        return (loss, logits) if loss is not None else logits

student_model = StudentModel(student_base, student_config)

# Load gbharti/finance-alpaca dataset
dataset = load_dataset("gbharti/finance-alpaca")

# Inspect columns to decide which field to use as input and label.
# print(dataset.column_names)
# Typically, 'input' and 'output' columns.

def tokenize_fn(examples):
    return tokenizer(
        examples["input"],  # prompt column
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize_fn, batched=True)

# Since there is no hard label column for classification,
# We use teacher to generate soft labels for distillation.
def create_soft_labels(batch):
    input_ids = torch.tensor(batch["input_ids"])
    attention_mask = torch.tensor(batch["attention_mask"])
    with torch.no_grad():
        logits = teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits
        soft_labels = F.softmax(logits / 2.0, dim=-1)  # temperature=2.0
        hard_labels = torch.argmax(logits, dim=-1)     # for hard supervision
    batch["soft_labels"] = soft_labels.cpu().numpy()
    batch["labels"] = hard_labels.cpu().numpy()
    return batch

tokenized_dataset = tokenized_dataset.map(create_soft_labels, batched=True)

# Prepare columns for trainer
columns_to_keep = ["input_ids", "attention_mask", "labels", "soft_labels"]
tokenized_dataset = tokenized_dataset.remove_columns(
    [col for col in tokenized_dataset["train"].column_names if col not in columns_to_keep]
)

def distillation_loss(outputs, labels, soft_labels, temperature=2.0, alpha=0.5):
    logits = outputs[1] if isinstance(outputs, tuple) else outputs.logits
    hard_loss = F.cross_entropy(logits, labels)
    soft_loss = F.kl_div(
        F.log_softmax(logits / temperature, dim=-1),
        torch.tensor(soft_labels).to(logits.device),
        reduction="batchmean",
    ) * (temperature ** 2)
    return alpha * hard_loss + (1 - alpha) * soft_loss

class DistillTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = torch.tensor(inputs.pop("labels")).to(model.classifier.weight.device)
        soft_labels = inputs.pop("soft_labels")
        outputs = model(**inputs)
        loss = distillation_loss(outputs, labels, soft_labels)
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="./finbert-distilled",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
)

trainer = DistillTrainer(
    model=student_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"] if "test" in tokenized_dataset else tokenized_dataset["train"],
    tokenizer=tokenizer,
)

trainer.train()

# Save distilled student model and tokenizer
student_model.save_pretrained("./finbert-distilled-student")
tokenizer.save_pretrained("./finbert-distilled-student")
