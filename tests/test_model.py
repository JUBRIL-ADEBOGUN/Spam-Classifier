import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Load the test dataset
def load_test_data():
    # Replace with the actual path to your test dataset
    test_data_path = "data/raw/SMSSpamCollection"
    df = pd.read_csv(test_data_path, sep='\t', names=['target', 'message'])
    df['target'] = df['target'].map({'ham': 0, 'spam': 1})
    return df

# Load the trained model and pipeline
def load_model_pipeline():
    model_path = "models/classifier.pkl"
    vectorizer_path = "models/vectorizer.pkl"
    
    model = joblib.load(model_path)
    pipeline = joblib.load(vectorizer_path)
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

# Visualize model performance
def visualize_performance(y_test, y_pred):
    # Generate and display a classification report
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Generate and display a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    # Load test data
    df = load_test_data()
    X_test = df['message']
    y_test = df['target']

    # Load model and pipeline
    model, pipeline = load_model_pipeline()

    # Evaluate the model
    accuracy, precision, recall, f1 = evaluate_model(model, pipeline, X_test, y_test)

    # Visualize performance
    y_pred = model.predict(pipeline.transform(X_test))
    visualize_performance(y_test, y_pred)
    # save the visual metrics to a file
    plt.savefig("visuals/confusion_matrix.png")