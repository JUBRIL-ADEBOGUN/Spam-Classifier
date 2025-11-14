# tests/test_model.py
import pytest
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# --- Test Fixture: A reusable setup for loading the model ---
@pytest.fixture(scope="session")
def loaded_artifacts():
    """
    Loads the model and vectorizer once per test session.
    This is efficient as we don't reload for every single test.
    """
    try:
        model = joblib.load("models/classifier.pkl")
        vectorizer = joblib.load("models/vectorizer.pkl")
        return {"model": model, "vectorizer": vectorizer}
    except FileNotFoundError:
        # Fail the test session if artifacts aren't found
        pytest.fail("Model or vectorizer artifacts not found in 'models/' directory.")

# --- Test Case 1: Smoke Test (Does it run without errors?) ---
def test_model_predict(loaded_artifacts):
    """
    Tests if the model can make a prediction without crashing.
    This is a basic "smoke test".
    """
    model = loaded_artifacts["model"]
    vectorizer = loaded_artifacts["vectorizer"]
    
    sample_input = ["Congratulations! You've won a free ticket."]
    
    # Transform and predict
    features = vectorizer.transform(sample_input)
    prediction = model.predict(features)
    
    # Assert that the output is what we expect (a numpy array of length 1)
    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1] # Ensure the output is a valid class

# --- Test Case 2: Performance Threshold Test (Is the model good enough?) ---
def test_model_performance(loaded_artifacts):
    """
    Tests the model's performance against a known, small "golden dataset".
    This prevents deploying a model that is catastrophically bad.
    """
    model = loaded_artifacts["model"]
    vectorizer = loaded_artifacts["vectorizer"]
    
    # A small, representative dataset that should never be changed.
    # This acts as a contract for minimum performance.
    golden_data = {
        'message': [
            "URGENT! Your account has been compromised. Click here.", # SPAM
            "Hey, are we still on for dinner tonight?",              # HAM
            "WIN a free vacation to the Bahamas now!",               # SPAM
            "See you at the meeting at 4pm."                         # HAM
        ],
        'target': [1, 0, 1, 0]
    }
    df = pd.DataFrame(golden_data)
    
    # Transform and predict
    features = vectorizer.transform(df['message'])
    predictions = model.predict(features)
    
    # Calculate accuracy on this known set
    accuracy = accuracy_score(df['target'], predictions)
    
    # Define a minimum performance threshold
    MINIMUM_ACCURACY = 0.9 # We expect 100% on this tiny set
    
    print(f"Model accuracy on golden dataset: {accuracy:.2f}")
    assert accuracy >= MINIMUM_ACCURACY, f"Model performance ({accuracy:.2f}) is below the threshold of {MINIMUM_ACCURACY}."
