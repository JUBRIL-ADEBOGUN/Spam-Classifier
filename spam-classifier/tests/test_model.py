import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Load the test dataset
def load_test_data():
    # Replace with the actual path to your test dataset
    test_data_path = "spam-classifier/data/raw/SMSSpamCollection"
    df = pd.read_csv(test_data_path, sep='\t', names=['target', 'message'])
    df['target'] = df['target'].map({'ham': 0, 'spam': 1})
    return df

# Load the trained model and pipeline
def load_model_pipeline():
    model_pipeline_path = "final_model_pipeline.pkl"
    model, pipeline = joblib.load(model_pipeline_path)
    return model, pipeline

# Evaluate the model
def evaluate_model(model, pipeline, X_test, y_test):
    X_test_transformed = pipeline.transform(X_test)
    y_pred = model.predict(X_test_transformed)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    return accuracy, precision, recall, f1

if __name__ == "__main__":
    # Load test data
    df = load_test_data()
    X_test = df['message']
    y_test = df['target']

    # Load model and pipeline
    model, pipeline = load_model_pipeline()

    # Evaluate the model
    evaluate_model(model, pipeline, X_test, y_test)