import numpy as np
import pandas as pd
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score

def load_dataset(file_path):
    data = pd.read_csv(file_path) #Insert actual dataset
    return data['text'], data.iloc[:, 1:]

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compute_tfidf(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix, vectorizer

def compute_LIR(labels):
    label_freqs = labels.sum(axis=0)
    LIR = label_freqs.min() / label_freqs.max()
    return LIR, label_freqs
    
def assign_bins(labels, bins):
    bin_assignments = np.digitize(labels.sum(axis=1), bins)
    return bin_assignments

def apply_smote(X, y):
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled
    
def train_classifiers(X_train, y_train):
    classifiers = {
        "SVM": MultiOutputClassifier(SVC(kernel='linear', probability=True)),
        "DecisionTree": MultiOutputClassifier(DecisionTreeClassifier()),
        "RandomForest": MultiOutputClassifier(RandomForestClassifier())
    }
    
    trained_models = {}
    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

def predict_labels(models, X_test):
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(X_test)
    return predictions

def train_meta_classifier(predictions, y_train):
    meta_features = np.hstack([pred for pred in predictions.values()])
    meta_classifier = RandomForestClassifier()
    meta_classifier.fit(meta_features, y_train)
    return meta_classifier

def mithem_pipeline(file_path):
    # Load and preprocess dataset
    texts, labels = load_dataset(file_path)
    texts = texts.apply(preprocess_text)

    X, vectorizer = compute_tfidf(texts)
    LIR, label_freqs = compute_LIR(labels)
    bins = np.linspace(label_freqs.min(), label_freqs.max(), num=5)
    bin_assignments = assign_bins(labels, bins)
    X_resampled, y_resampled = apply_smote(X, labels)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    models = train_classifiers(X_train, y_train)
    predictions = predict_labels(models, X_test)
    meta_classifier = train_meta_classifier(predictions, y_train)
    meta_features_test = np.hstack([pred for pred in predictions.values()])
    final_predictions = meta_classifier.predict(meta_features_test)

    # You can add more evalution metrics
    accuracy = accuracy_score(y_test, final_predictions)
    print(f"Final Model Accuracy: {accuracy:.4f}")

    return final_predictions

# Run the algorithm
final_labels = mithem_pipeline("biomedical_text_data.csv")
