import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def load_and_clean_data(file_path):
    """Loads, cleans, and validates the raw SMS data."""
    # Note: Using 'latin-1' encoding is common for this dataset
    df = pd.read_csv(file_path, sep='\t', header=None, names=['v1', 'v2'])
    
    # Rename columns for clarity
    df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
    
    # Convert label to binary target (ham=0, spam=1)
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})
    df.drop('label', axis=1, inplace=True)
    
    # --- MLE Data Validation Checks (Your robust assertions) ---
    assert df['message'].dtype == object, "Message column should be of type object" 
    assert df['target'].dtype == np.int64, "Target column should be of type int64"
    assert df['target'].nunique() == 2, "Target column should have exactly 2 unique values"
    assert df.isnull().sum().sum() == 0, "Dataframe should not contain any null values"
    
    return df[['message', 'target']]

def build_vectorizer(X_train):
    """Initializes, fits, and returns the TfidfVectorizer."""
    # Parameters can be logged to MLflow later
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, max_features=5000)
    vectorizer.fit(X_train)
    return vectorizer