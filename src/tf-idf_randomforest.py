import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import re
import warnings
warnings.filterwarnings('ignore')

# Global variables
models = {}
label_encoders = {}
vectorizer = None
hierarchy_levels = ['segment', 'family', 'class', 'brick']

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = ' '.join(text.split())
    return text

def fit_vectorizer(texts):
    global vectorizer
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        stop_words='english',
        sublinear_tf=True,
        norm='l2'
    )
    return vectorizer.fit_transform(texts)

def train_models(X_train, y_train):
    global models, label_encoders
    print("Training Random Forest models...")
    for level in hierarchy_levels:
        print(f"Training {level} model...")
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_train[level])
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        rf.fit(X_train, y_encoded)
        models[level] = rf
        label_encoders[level] = le
        print(f"{level} model trained with {len(np.unique(y_encoded))} classes")

def predict(X_test):
    predictions = {}
    for level in hierarchy_levels:
        y_pred_encoded = models[level].predict(X_test)
        predictions[level] = label_encoders[level].inverse_transform(y_pred_encoded)
    return pd.DataFrame(predictions)

def evaluate_model(X_test, y_test, dataset_name):
    print(f"\nEvaluating on {dataset_name}...")
    predictions = predict(X_test)
    f1_scores = {}
    for level in hierarchy_levels:
        y_true = y_test[level]
        y_pred = predictions[level]
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_scores[level] = f1_weighted
        print(f"{level.upper()}: F1-Weighted: {f1_weighted:.4f}")
    overall_f1 = np.mean(list(f1_scores.values()))
    print(f"Overall F1: {overall_f1:.4f}")
    return overall_f1, f1_scores

def load_and_prepare_data():
    print("Loading datasets...")
    train_df = pd.read_csv('data/correctly_matched_mapped_gpc.csv')
    test1_df = pd.read_csv('data/product_gpc_mapping.csv')
    test2_df = pd.read_csv('data/validated_actually_labeled_test_dataset.csv')
    print(f"Training data loaded: {len(train_df)} samples")
    print(f"Test dataset 1 loaded: {len(test1_df)} samples")
    print(f"Test dataset 2 loaded: {len(test2_df)} samples")
    return train_df, test1_df, test2_df

def prepare_training_data(train_df):
    train_df = train_df.copy()
    train_df['processed_name'] = train_df['product_name'].apply(preprocess_text)

    train_split, val_split = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df['segment']
    )
    print(f"Training split: {len(train_split)} samples")
    print(f"Validation split: {len(val_split)} samples")

    X_train = fit_vectorizer(train_split['processed_name'])
    X_val = vectorizer.transform(val_split['processed_name'])

    y_train = train_split[hierarchy_levels].copy()
    y_val = val_split[hierarchy_levels].copy()
    return X_train, X_val, y_train, y_val

def prepare_test_data(test_df, text_col, target_cols):
    test_df = test_df.copy()
    test_df['processed_name'] = test_df[text_col].apply(preprocess_text)
    X_test = vectorizer.transform(test_df['processed_name'])
    y_test = test_df[target_cols].copy()
    y_test.columns = hierarchy_levels
    return X_test, y_test

def run_complete_pipeline():
    print("=== WINDOWS GPC CLASSIFIER PIPELINE (TF-IDF only) ===")
    train_df, test1_df, test2_df = load_and_prepare_data()
    X_train, X_val, y_train, y_val = prepare_training_data(train_df)
    print(f"Training features shape: {X_train.shape}")
    print(f"Validation features shape: {X_val.shape}")

    train_models(X_train, y_train)
    val_f1, _ = evaluate_model(X_val, y_val, "Validation Set")

    X_test1, y_test1 = prepare_test_data(
        test1_df, 'Name', ['SegmentTitle', 'FamilyTitle', 'ClassTitle', 'BrickTitle']
    )
    test1_f1, _ = evaluate_model(X_test1, y_test1, "Test Set 1 (product_gpc_mapping)")

    X_test2, y_test2 = prepare_test_data(
        test2_df, 'translated_name',
        ['predicted_segment', 'predicted_family', 'predicted_class', 'predicted_brick']
    )
    test2_f1, _ = evaluate_model(X_test2, y_test2, "Test Set 2 (validated_actually_labeled)")

    avg_f1 = (test1_f1 + test2_f1) / 2
    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print(f"Validation F1 Score: {val_f1:.4f}")
    print(f"Test Set 1 F1 Score: {test1_f1:.4f}")
    print(f"Test Set 2 F1 Score: {test2_f1:.4f}")
    print(f"Average Test F1 Score: {avg_f1:.4f}")
    print("="*60)

    return {
        'validation_f1': val_f1,
        'test1_f1': test1_f1,
        'test2_f1': test2_f1,
        'average_f1': avg_f1
    }

# Run the pipeline
results = run_complete_pipeline()
print("\n=== TRAINING COMPLETE ===")
print("Ready to run on Windows machine!")