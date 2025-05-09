# Psy
# MBTI Personality Type Classification with Custom Attention Model

This project aims to classify MBTI personality types from text data using a custom attention-based neural network, implemented in PyTorch and Hugging Face Transformers.

## 📂 Dataset

* **Source**: Kaggle MBTI dataset (8,675 rows)
* **Content**:

  * `type`: One of 16 MBTI personality types (e.g., INFP, ESTJ)
  * `posts`: Concatenation of a user's 50 social media posts
* **Preprocessing**:

  * Replaced `|||` delimiters with spaces
  * Label encoded MBTI types to integers

## 🧠 Model Architecture

* **Tokenizer**: `bert-base-uncased` (used only for tokenization, not weights)
* **Custom Model Components**:

  * `Embedding` layer with learned token vectors
  * `PositionalEncoding` layer to encode word order
  * Single-head self-attention:

    * Query, Key, Value projection layers
    * Softmax-scaled dot-product attention
  * Output classifier:

    * Linear → ReLU → Linear (16 classes)
* **Loss Function**: Weighted `CrossEntropyLoss` to address class imbalance

## ⚙️ Training Configuration

* **Framework**: Hugging Face `Trainer`
* **Device**: GPU or CPU depending on availability
* **Epochs**: 10
* **Batch Size**: 8
* **Save Strategy**: Manual save of tokenizer

## 🧪 Evaluation

* Evaluated on 20% validation split
* Reported:

  * Accuracy
  * Weighted F1 score
  * Full classification report
  * Confusion matrix heatmap

## 🔍 Results Summary (Initial Run)

* Model initially overfit to INFP due to imbalance
* After applying weighted loss and increasing depth:

  * Accuracy improved moderately
  * Class-wise F1 scores distributed more evenly

## 🛠 Future Improvements

* Add Multi-Head Attention
* Incorporate LayerNorm and Residual Connections
* Train a custom tokenizer to replace BERT vocabulary
* Use data augmentation or sampling to reduce label imbalance

## 📌 Requirements

```bash
pip install torch transformers scikit-learn matplotlib seaborn
```

---

Created as part of an NLP experiment to combine attention-based architectures with psychological profiling from natural language.
