import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for scripts
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from data_pipeline import load_and_clean_data, build_vectorizer

def generate_training_report(model, vectorizer, X_test_features, y_test, y_pred):
    """Generates a visual and textual report of the model's performance."""
    
    print("INFO: Generating training report...")
    
    
    # --- 2. Generate Confusion Matrix Plot ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['HAM', 'SPAM'], yticklabels=['HAM', 'SPAM'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    confusion_matrix_path = "spam-classifier/src/report/confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    plt.close() # Close the plot to free up memory

    # --- 3. Generate Feature Importance Plot ---
    # For Logistic Regression, coefficients are the "feature importances"
    feature_names = vectorizer.get_feature_names_out()

    # try to get coefficients, if model has no coef_ attribute, use feature_importances_
    try:
        coefficients = model.coef_[0]
    except AttributeError:
        coefficients = model.feature_importances_
    
    # Get top 15 SPAM (positive) and HAM (negative) features
    top_spam_indices = coefficients.argsort()[-15:]
    top_ham_indices = coefficients.argsort()[:15]
    
    top_spam_features = feature_names[top_spam_indices]
    top_spam_coefs = coefficients[top_spam_indices]
    
    top_ham_features = feature_names[top_ham_indices]
    top_ham_coefs = coefficients[top_ham_indices]

    plt.figure(figsize=(12, 8))
    plt.barh(top_spam_features, top_spam_coefs, color='red', label='Spam Words')
    plt.barh(top_ham_features, top_ham_coefs, color='green', label='Ham Words')
    plt.xlabel('Coefficient Value (Importance)')
    plt.title('Top 15 Words Influencing Prediction')
    plt.legend()
    plt.tight_layout()
    feature_importance_path = "spam-classifier/src/report/feature_importance.png"
    plt.savefig(feature_importance_path)
    plt.close()

    # --- 4. Generate and Save the Markdown Report ---
    report_path = "spam-classifier/src/report/training_report.md"
    
    # Get detailed classification report as text
    class_report = classification_report(y_test, y_pred, target_names=['HAM', 'SPAM'])
    
    with open(report_path, "w") as f:
        f.write("# Model Training Report\n\n")
        f.write(f"**Timestamp:** `{pd.Timestamp.now()}`\n\n")
        f.write("## 1. Performance Metrics\n\n")
        f.write("```\n")
        f.write(class_report)
        f.write("\n```\n\n")
        f.write("## 2. Confusion Matrix\n\n")
        f.write("![Confusion Matrix](confusion_matrix.png)\n\n")
        f.write("## 3. Feature Importance\n\n")
        f.write("This plot shows the top words that push the prediction towards SPAM (red) or HAM (green).\n\n")
        f.write("![Feature Importance](feature_importance.png)\n")
        
    print(f"INFO: Report saved to {report_path}")

def train_and_save_model(data_path, max_depth=7):
    """Loads data, trains a Random Forest model, and tracks with MLflow."""
    
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
        model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
        model.fit(X_train_features, y_train)
        y_pred = model.predict(X_test_features)

        # 4. Evaluation and Tracking
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log Hyperparameters
        mlflow.log_param("model_type", "Random Forest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 7)
        mlflow.log_param("vectorizer_max_features", vectorizer.max_features)
        
        # Log Metrics (Your successfully implemented code)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        # 5. Save Artifacts (For API Deployment)
        # os.makedirs("spam-classifier/models", exist_ok=True)
        joblib.dump(model, "spam-classifier/models/classifier.pkl")
        joblib.dump(vectorizer, "spam-classifier/models/vectorizer.pkl")
        
        # Log artifacts to MLflow as well
        # mlflow.log_artifact("spam-classifier/models/classifier.pkl", "model_artifact")
        # mlflow.log_artifact("spam-classifier/models/vectorizer.pkl", "model_artifact")
        
        # --- NEW: Call the report generation function ---
        generate_training_report(model, vectorizer, X_test_features, y_test, y_pred)

if __name__ == "__main__":
    # Example execution with a parameter
    DATA_FILE = "spam-classifier/data/raw/SMSSpamCollection" 
    train_and_save_model(DATA_FILE, max_depth=7)
