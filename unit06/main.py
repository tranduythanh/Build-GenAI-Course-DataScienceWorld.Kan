from datasets import load_dataset
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np, evaluate

# Dictionary to store metrics
training_metrics = defaultdict(list)
training_losses = []

# Load metric
metric = evaluate.load("seqeval")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

raw_datasets = load_dataset("conll2003")        # ba split: train/validation/test
print(raw_datasets["train"][0])


from transformers import AutoTokenizer
label_list = raw_datasets["train"].features["ner_tags"].feature.names
id2label = {i: l for i, l in enumerate(label_list)}
label2id = {l: i for i, l in enumerate(label_list)}

model_checkpoint = "bert-base-cased"           # BERT gốc, cased giữ nguyên HOA/thường
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


from functools import partial

def tokenize_and_align(examples, label_all_tokens=False):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding=True,
        return_offsets_mapping=False,
    )
    labels = []
    for i in range(len(examples["tokens"])):
        word_ids = tokenized.word_ids(batch_index=i)
        word_labels = examples["ner_tags"][i]
        label_ids, prev = [], None
        for word_id in word_ids:
            if word_id is None:                 # token đặc biệt [CLS]/[SEP]
                label_ids.append(-100)          # -100 = Trainer sẽ bỏ qua khi tính loss
            elif word_id != prev:               # token đầu của word
                label_ids.append(word_labels[word_id])
            else:                               # sub-token tiếp theo
                label_ids.append(
                    word_labels[word_id] if label_all_tokens else -100
                )
            prev = word_id
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized

tokenized_datasets = raw_datasets.map(
    tokenize_and_align,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
    desc="Tokenizing + aligning labels",
)



from transformers import (AutoModelForTokenClassification,
                          DataCollatorForTokenClassification,
                          TrainingArguments, Trainer)

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=-1)
    true_preds, true_labels = [], []
    for pred, lab in zip(preds, labels):
        for p_i, l_i in zip(pred, lab):
            if l_i != -100:
                true_preds.append(label_list[p_i])
                true_labels.append(label_list[l_i])
    results = metric.compute(predictions=[true_preds], references=[true_labels])
    metrics = {k: results[f"overall_{k}"] for k in ["precision", "recall", "f1"]}
    
    # Store metrics
    for k, v in metrics.items():
        training_metrics[k].append(v)
    
    return metrics

args = TrainingArguments(
    output_dir="ner-bert-conll03",
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    push_to_hub=False,
    no_cuda=True if device.type == "mps" else False,  # Disable CUDA if using MPS
    report_to="none",  # Disable wandb/tensorboard logging
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        if self.state.global_step % self.args.logging_steps == 0:
            training_losses.append(loss.item())
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)



trainer.train()
trainer.evaluate()      # In ra Precision / Recall / F1
trainer.save_model()    # Lưu weights + cấu hình
tokenizer.save_pretrained("ner-bert-conll03")

# Plot training metrics
plt.figure(figsize=(15, 10))

# Plot metrics
plt.subplot(2, 1, 1)
for metric_name, values in training_metrics.items():
    plt.plot(values, label=f'{metric_name.capitalize()}: {values[-1]:.4f}')

plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Training Metrics')
plt.legend(loc='center right', bbox_to_anchor=(1, 0.5))  # Move legend outside the plot
plt.grid(True)

# Plot loss
plt.subplot(2, 1, 2)
plt.plot(training_losses, label=f'Training Loss: {training_losses[-1]:.4f}', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend(loc='center right', bbox_to_anchor=(1, 0.5))  # Move legend outside the plot
plt.grid(True)

plt.tight_layout()
plt.savefig('training_metrics_and_loss.png', bbox_inches='tight')  # Save with extra space for legend
plt.close()


from transformers import pipeline

ner = pipeline(
    "token-classification",
    model="ner-bert-conll03",
    tokenizer="ner-bert-conll03",
    aggregation_strategy="simple"     # gộp sub-token cùng nhãn
)

sentence = "Barack Obama visited Hà Nội in 2016."
print(ner(sentence))