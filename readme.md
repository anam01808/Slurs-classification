

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
Title: Gender-based Negative Emotion Detection from Online Textual Content Using Machine Learning
```

---

