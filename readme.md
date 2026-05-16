# People Slurs Dataset: Text Classification Pipeline

```python
# ============================================================
# People Slurs Dataset - Emotion/Sentiment Classification
# Methodology:
# 1. TF-IDF Features + ML Models
# 2. POS Tagging Features + ML Models
# Models:
# SVM, DT, RF, KNN, GB, XGB, AdaBoost, LightGBM
# Evaluation:
# Accuracy, Precision, Recall, F1-score, ROC-AUC
# Combined ROC-AUC Plot
# ============================================================

# =========================
# INSTALL REQUIRED PACKAGES
# =========================
!pip install lightgbm xgboost nltk scikit-learn -q

# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import re
import nltk
import warnings
warnings.filterwarnings('ignore')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    roc_curve,
    auc
)

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# =========================
# LOAD DATASET
# =========================

from google.colab import files
uploaded = files.upload()

# Replace filename if needed
file_name = list(uploaded.keys())[0]

df = pd.read_csv(file_name)

print(df.head())
print(df.shape)

# =========================
# SELECT TARGET COLUMN
# =========================
# Change target column if required
# Example: f_anger, f_fear, f_panic etc.

TARGET_COLUMN = 'f_anger'

# =========================
# TEXT PREPROCESSING
# =========================

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)

    cleaned_tokens = []

    for word in tokens:
        if word not in stop_words:
            lemma = lemmatizer.lemmatize(word)
            cleaned_tokens.append(lemma)

    return ' '.join(cleaned_tokens)


df['clean_text'] = df['text'].apply(preprocess_text)

# =========================
# HANDLE MISSING VALUES
# =========================

# Fill numerical missing values
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# =========================
# LABEL CREATION
# =========================
# Convert continuous emotion values into classes
# Binary Classification:
# 0 = Low Emotion
# 1 = High Emotion

median_value = df[TARGET_COLUMN].median()

df['label'] = df[TARGET_COLUMN].apply(lambda x: 1 if x >= median_value else 0)

print(df['label'].value_counts())

# =========================
# TF-IDF FEATURE EXTRACTION
# =========================

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

X_tfidf = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# =========================
# POS FEATURE EXTRACTION
# =========================


def extract_pos_tags(text):
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    pos_sequence = ' '.join([tag for word, tag in tags])
    return pos_sequence


df['pos_text'] = df['clean_text'].apply(extract_pos_tags)

pos_vectorizer = TfidfVectorizer(max_features=3000)
X_pos = pos_vectorizer.fit_transform(df['pos_text'])

# =========================
# TRAIN TEST SPLIT
# =========================

X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_pos, X_test_pos, _, _ = train_test_split(
    X_pos,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# DEFINE MODELS
# =========================

models = {
    'SVM': SVC(probability=True, random_state=42),

    'Decision Tree': DecisionTreeClassifier(random_state=42),

    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    'KNN': KNeighborsClassifier(n_neighbors=5),

    'Gradient Boosting': GradientBoostingClassifier(random_state=42),

    'XGBoost': XGBClassifier(
        eval_metric='logloss',
        random_state=42
    ),

    'AdaBoost': AdaBoostClassifier(random_state=42),

    'LightGBM': LGBMClassifier(random_state=42)
}

# =========================
# TRAINING + EVALUATION
# =========================

results_tfidf = []
results_pos = []

roc_data_tfidf = {}
roc_data_pos = {}

# ============================================================
# TF-IDF MODELS
# ============================================================

print('='*70)
print('TF-IDF FEATURE RESULTS')
print('='*70)

for name, model in models.items():

    print(f'\nTraining {name}...')

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    y_prob = model.predict_proba(X_test_tfidf)[:,1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    results_tfidf.append([
        name,
        acc,
        prec,
        rec,
        f1,
        roc_auc
    ])

    print('\nClassification Report')
    print(classification_report(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data_tfidf[name] = (fpr, tpr, roc_auc)

# ============================================================
# POS MODELS
# ============================================================

print('='*70)
print('POS FEATURE RESULTS')
print('='*70)

for name, model in models.items():

    print(f'\nTraining {name}...')

    model.fit(X_train_pos, y_train)

    y_pred = model.predict(X_test_pos)
    y_prob = model.predict_proba(X_test_pos)[:,1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    results_pos.append([
        name,
        acc,
        prec,
        rec,
        f1,
        roc_auc
    ])

    print('\nClassification Report')
    print(classification_report(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data_pos[name] = (fpr, tpr, roc_auc)

# =========================
# RESULTS TABLES
# =========================

results_tfidf_df = pd.DataFrame(
    results_tfidf,
    columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
)

results_pos_df = pd.DataFrame(
    results_pos,
    columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
)

print('\nTF-IDF RESULTS')
print(results_tfidf_df)

print('\nPOS RESULTS')
print(results_pos_df)

# =========================
# SAVE RESULTS
# =========================

results_tfidf_df.to_csv('tfidf_results.csv', index=False)
results_pos_df.to_csv('pos_results.csv', index=False)

# =========================
# COMBINED ROC CURVE - TFIDF
# =========================

plt.figure(figsize=(10,8))

for name, (fpr, tpr, roc_auc) in roc_data_tfidf.items():
    plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.2f})')

plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined ROC-AUC Curve (TF-IDF Features)')
plt.legend()
plt.grid(True)
plt.show()

# =========================
# COMBINED ROC CURVE - POS
# =========================

plt.figure(figsize=(10,8))

for name, (fpr, tpr, roc_auc) in roc_data_pos.items():
    plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.2f})')

plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined ROC-AUC Curve (POS Features)')
plt.legend()
plt.grid(True)
plt.show()

# =========================
# BEST MODEL DISPLAY
# =========================

best_tfidf = results_tfidf_df.sort_values(
    by='ROC-AUC',
    ascending=False
).iloc[0]

best_pos = results_pos_df.sort_values(
    by='ROC-AUC',
    ascending=False
).iloc[0]

print('\nBest TF-IDF Model')
print(best_tfidf)

print('\nBest POS Model')
print(best_pos)

# =========================
# OPTIONAL: DOWNLOAD RESULTS
# =========================

from google.colab import files

files.download('tfidf_results.csv')
files.download('pos_results.csv')

print('\nPipeline Execution Completed Successfully!')
```

# Notes

- Change `TARGET_COLUMN` to any emotion label:
  - `f_anger`
  - `f_fear`
  - `f_panic`
  - `f_guilt`
  - `f_humiliation`
  - `f_pain`

- The code performs:
  - Text preprocessing
  - TF-IDF feature extraction
  - POS-tag feature extraction
  - Multi-model classification
  - Classification reports
  - Combined ROC-AUC visualization
  - CSV result export

- Dataset upload is automatic through Google Colab.


---

# README.md

```markdown
# People Slurs Dataset: Emotion and Sentiment Classification Framework

## Overview

This repository presents a lightweight and resource-efficient machine learning framework for contextual emotion and sentiment classification using the People Slurs Dataset. The framework integrates linguistic text preprocessing, TF-IDF feature engineering, POS-based textual representation, and multiple machine learning classifiers for emotion prediction and contextual sentiment analysis.

The study focuses on computationally efficient and reproducible NLP experimentation suitable for lightweight deployment environments and practical text analytics scenarios.

---

# Dataset Information

Dataset Name: People Slurs Dataset

Total Samples: 504

License: MIT

Dataset Type:
- Text Classification
- NLP
- Psychology-Based Emotion Analysis
- Sentiment Analysis

Attributes:
- text
- condition
- recalled
- slur_source
- slur_gender
- subj_anger
- f_pain
- f_fear
- f_panic
- f_anger
- f_guilt
- f_humiliation

---

# Proposed Methodology

The proposed framework consists of the following stages:

1. Dataset Loading
2. Text Cleaning and Preprocessing
3. Tokenization and Lemmatization
4. Stopword Removal
5. TF-IDF Feature Extraction
6. POS-Based Feature Representation
7. Multi-Model Machine Learning Classification
8. Performance Evaluation and ROC-AUC Analysis

---

# Experimental Environment

## Hardware Environment

| Component | Specification |
|---|---|
| Platform | Google Colab |
| CPU | Intel Xeon Processor |
| RAM | 12 GB |
| GPU | NVIDIA Tesla T4 (Optional) |
| Storage | Google Drive / Local Runtime |

---

## Software Environment

| Software | Version |
|---|---|
| Python | 3.10+ |
| Google Colab | Latest |
| Scikit-learn | Latest |
| NLTK | Latest |
| XGBoost | Latest |
| LightGBM | Latest |
| Pandas | Latest |
| NumPy | Latest |
| Matplotlib | Latest |

---

# Required Libraries

Install all required libraries using:

```python
!pip install lightgbm xgboost nltk scikit-learn
```

Main Libraries:

- pandas
- numpy
- nltk
- scikit-learn
- xgboost
- lightgbm
- matplotlib
- re
- warnings

---

# Text Preprocessing Pipeline

The preprocessing stage includes:

- Lowercase conversion
- URL removal
- Special character removal
- Tokenization
- Stopword removal
- Lemmatization
- POS tagging

---

# Feature Engineering

## 1. TF-IDF Features

TF-IDF vectorization is applied using unigram and bigram representations.

Parameters:

| Parameter | Value |
|---|---|
| max_features | 5000 |
| ngram_range | (1,2) |

---

## 2. POS-Based Features

Part-of-Speech (POS) tagging is used to generate syntactic textual representations for contextual emotion analysis.

Parameters:

| Parameter | Value |
|---|---|
| max_features | 3000 |

---

# Machine Learning Models

The following machine learning classifiers are implemented:

- Support Vector Machine (SVM)
- Decision Tree (DT)
- Random Forest (RF)
- K-Nearest Neighbor (KNN)
- Gradient Boosting (GB)
- XGBoost (XGB)
- AdaBoost
- LightGBM

---

# Hyperparameter Configuration

| Model | Hyperparameters |
|---|---|
| SVM | probability=True, random_state=42 |
| Decision Tree | random_state=42 |
| Random Forest | n_estimators=200, random_state=42 |
| KNN | n_neighbors=5 |
| Gradient Boosting | random_state=42 |
| XGBoost | eval_metric='logloss', random_state=42 |
| AdaBoost | random_state=42 |
| LightGBM | random_state=42 |

---

# Training Configuration

| Parameter | Value |
|---|---|
| Train-Test Split | 80:20 |
| Random State | 42 |
| Stratification | Enabled |
| Classification Type | Binary Classification |

---

# Evaluation Metrics

The framework evaluates model performance using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Classification Report
- Combined ROC Curves

---

# Deployment Scenario

The proposed framework is designed for lightweight and resource-efficient deployment scenarios where rapid text classification and contextual emotion analysis are required. The framework can support:

- Social media sentiment monitoring
- Psychological emotion assessment
- Context-aware text analytics
- Educational emotional feedback systems
- Human behavior analysis
- Lightweight NLP applications in resource-constrained environments

The use of conventional machine learning models combined with TF-IDF and POS-based feature representations reduces computational complexity and enables efficient inference on standard computing environments without requiring high-end GPU infrastructure.

---

# Reproducibility

To ensure reproducibility:

- Public implementation code is provided
- Hyperparameter configurations are documented
- Experimental environment details are included
- Random seed initialization is fixed
- Dataset preprocessing steps are explicitly defined

---

# Outputs Generated

The framework generates:

- Classification Reports
- ROC-AUC Curves
- Model Comparison Tables
- CSV Result Files
- Best Model Identification

---

# Execution

Run the notebook sequentially in Google Colab:

1. Upload dataset
2. Execute preprocessing cells
3. Run feature extraction
4. Train all models
5. Evaluate results
6. Export outputs

---

# Repository Structure

```text
project/
│
├── slurs-dataset.csv
├── main_notebook.ipynb
├── tfidf_results.csv
├── pos_results.csv
├── README.md
└── output_figures/
```

---

# Citation

If you use this framework in research, please cite the corresponding conference paper.

```

