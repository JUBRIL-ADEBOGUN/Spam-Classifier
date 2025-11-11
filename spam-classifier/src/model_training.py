import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
from data_pipeline import load_and_clean_data, build_vectorizer

def train_and_save_model(data_path, C_param=1.0):
    """Loads data, trains a Logistic Regression model, and tracks with MLflow."""
    
    # --- MLflow Setup ---
    mlflow.set_experiment("Spam_Classifier_Training")
    with mlflow.start_run():
        
        # 1. Load and Clean Data
        df = load_and_clean_data(data_path)
        X, y = df['message'], df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 2. Build and Save Feature Pipeline (Vectorizer)
        vectorizer = build_vectorizer(X_train)
        X_train_features = vectorizer.transform(X_train)
        X_test_features = vectorizer.transform(X_test)

        # 3. Model Training
        model = LogisticRegression(C=C_param, solver='liblinear')
        model.fit(X_train_features, y_train)
        y_pred = model.predict(X_test_features)

        # 4. Evaluation and Tracking
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log Hyperparameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C_parameter", C_param)
        mlflow.log_param("vectorizer_max_features", vectorizer.max_features)
        
        # Log Metrics (Your successfully implemented code)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        # 5. Save Artifacts (For API Deployment)
        os.makedirs("spam-classifier/models", exist_ok=True)
        joblib.dump(model, "spam-classifier/models/classifier.pkl")
        joblib.dump(vectorizer, "spam-classifier/models/vectorizer.pkl")
        
        # Log artifacts to MLflow as well
        mlflow.log_artifact("spam-classifier/models/classifier.pkl", "model_artifact")
        mlflow.log_artifact("spam-classifier/models/vectorizer.pkl", "model_artifact")

if __name__ == "__main__":
    # Example execution with a parameter
    DATA_FILE = "spam-classifier/data/raw/SMSSpamCollection" 
    train_and_save_model(DATA_FILE, C_param=0.8)