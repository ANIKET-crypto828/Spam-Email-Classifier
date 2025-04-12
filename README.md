# 📧 Spam Email Classifier using Logistic Regression

This project builds a **machine learning model** using **Logistic Regression** to classify emails as **spam** or **not spam (ham)**. It uses natural language processing (NLP) techniques for preprocessing and vectorization of text data and evaluates the model using **Precision**, **Recall**, and **F1 Score**.

---

## 🧠 Problem Statement

Spam emails are unsolicited messages that often clutter inboxes and sometimes carry malicious content. This project aims to create a binary classification model that detects spam emails based on their content.

---

## 📁 Dataset

The project uses the [UCI SMS Spam Collection Dataset](https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv), which contains over 5,000 labeled messages as either `ham` (non-spam) or `spam`.

---

## 🛠️ Tools & Technologies

- **Python 3**
- **Pandas** – Data handling
- **NumPy** – Numerical computation
- **Scikit-learn** – Machine learning and model evaluation
---

## ⚙️ Project Pipeline

### 1. Data Preprocessing
- Load and inspect dataset
- Clean and normalize text (lowercasing, removing stopwords, punctuation, lemmatization)
- Encode labels (ham = 0, spam = 1)

### 2. Feature Extraction
- Use `TfidfVectorizer` to convert text into numerical features

### 3. Model Training
- Train a **Logistic Regression** model using Scikit-learn

### 4. Model Evaluation
- Evaluate performance using:
  - **Precision**
  - **Recall**
  - **F1 Score**
- Display classification report and confusion matrix

---

## 📊 Evaluation Metrics

These metrics help evaluate how well the classifier performs:
- **Precision**: How many predicted spams were actually spam
- **Recall**: How many actual spam emails were correctly identified
- **F1 Score**: Harmonic mean of Precision and Recall

---

## 🚀 Installation & Running

### 1. Clone the Repository
```bash
git clone https://github.com/ANIKET-crypto828/Spam-Email-Classifier.git
cd Spam-Email-Classifier
