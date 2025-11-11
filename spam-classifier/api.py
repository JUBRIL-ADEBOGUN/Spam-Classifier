from fastapi import FastAPI
from pydantic import BaseModel
import joblib # Used for loading scikit-learn artifacts
import os
from contextlib import asynccontextmanager

# --- 1. Global Variables (Loaded ONCE at server startup) ---
VECTORIZER = None
MODEL = None

# Define the expected JSON body for the POST request
class PredictionRequest(BaseModel):
    message: str # e.g., "Free entry to a contest! Text WIN to 888"
# Initialize the FastAPI app with a lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the fitted feature pipeline and model from the models/ folder at startup."""
    global VECTORIZER
    global MODEL

    # Paths relative to the root of the project (or where Docker copies them)
    vectorizer_path = os.path.join("spam-classifier/models", "vectorizer.pkl")
    model_path = os.path.join("spam-classifier/models", "classifier.pkl")
    
    try:
        VECTORIZER = joblib.load(vectorizer_path)
        MODEL = joblib.load(model_path)
        print("INFO: Model and Vectorizer artifacts loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Could not load artifacts. Check 'models/' directory. Error: {e}")
        # In a production environment, you might stop the application here (raise e)

    # Yield control to start the application; code after yield runs on shutdown.
    yield

app = FastAPI(title="Spam Classifier API", lifespan=lifespan)
        # In a production environment, you might stop the application here (raise e)

# --- 3. Prediction Endpoint ---
@app.post("/predict")
def predict(request: PredictionRequest):
    """Accepts a message and returns a SPAM/HAM prediction."""
    
    if VECTORIZER is None or MODEL is None:
        return {"error": "Model not loaded. Check startup logs."}, 500
        
    message_text = [request.message] # Vectorizer expects a list of strings

    # 1. Transformation (The MLE step)
    features = VECTORIZER.transform(message_text)
    
    # 2. Prediction
    prediction = MODEL.predict(features)[0]
    probability = MODEL.predict_proba(features)[0].max()

    return {
        "message_in": message_text[0],
        "prediction_label": "SPAM" if prediction == 1 else "HAM",
        "confidence": round(float(probability), 4)
    }

# --- 4. Health Check (Crucial for Orchestration systems like Kubernetes) ---
@app.get("/health")
def health_check():
    """Returns a status check for the load balancer."""
    return {"status": "ok", "model_ready": MODEL is not None}