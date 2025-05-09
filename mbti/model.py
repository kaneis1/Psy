import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer, TrainingArguments, Trainer

# ========== 1. Load and Prepare Data ==========
df = pd.read_csv('mbti/mbti_1.csv')

label2id = {label: idx for idx, label in enumerate(sorted(df['type'].unique()))}
id2label = {v: k for k, v in label2id.items()}
df['label'] = df['type'].map(label2id)

# Preprocess text
df['clean_posts'] = df['posts'].str.replace(r'\|\|\|', ' ', regex=True)
print(df.head())

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['clean_posts'], df['label'], test_size=0.2, random_state=42
)

# Tokenize text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512, return_tensors='pt')
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=512, return_tensors='pt')

# Convert train_labels to a plain list of ints
train_labels = list(train_labels.values)
val_labels = list(val_labels.values)

# ========== 2. Create Dataset Class ==========
class MBTIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = MBTIDataset(train_encodings, train_labels)
val_dataset = MBTIDataset(val_encodings, val_labels)

# ========== 3. Define Positional Encoding ==========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# ========== 4. Define Custom Attention Model with Class Weights ==========
class CustomAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=4, num_classes=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        class_counts = torch.bincount(torch.tensor(train_labels))
        weights = 1.0 / class_counts.float()
        weights = weights / weights.sum() * len(class_counts)
        self.loss_fct = nn.CrossEntropyLoss(weight=weights)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / Q.size(-1) ** 0.5
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        pooled = context.mean(dim=1)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)
        return {'logits': logits, 'loss': loss} if loss is not None else {'logits': logits}

model = CustomAttentionClassifier(vocab_size=tokenizer.vocab_size).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# ========== 5. Training Configuration ==========
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=100,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
    logging_steps=20,
    save_total_limit=2,
    save_strategy='no',
    report_to="none"
)

# ========== 6. Trainer Wrapper ==========
from transformers import Trainer

class WrappedModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        output = self.model(input_ids, attention_mask, labels)
        return output

trainer = Trainer(
    model=WrappedModel(model),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# ========== 7. Train ==========
trainer.train()

# ========== 8. Save Tokenizer Only ==========
tokenizer.save_pretrained('./mbti_model')
print("Model trained and tokenizer saved to ./mbti_model")

# ========== 9. Evaluate ==========
print("Evaluating...")
preds = trainer.predict(val_dataset)
y_pred = np.argmax(preds.predictions, axis=1)
y_true = preds.label_ids

print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1 Score (weighted):", f1_score(y_true, y_pred, average='weighted'))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(16)]))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=id2label.values(), yticklabels=id2label.values(), cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
