
# Real-Time SMS Spam Classifier API

![CI/CD Pipeline Status](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY/actions/workflows/cicd.yml/badge.svg)

This project is an end-to-end demonstration of a production-ready Machine Learning system. It builds, tests, and deploys a real-time SMS Spam Classifier as a containerized microservice, following modern MLOps best practices.

The core of the project is a REST API built with FastAPI and containerized with Docker. The entire process, from model training to deployment on Docker Hub, is automated using a CI/CD pipeline in GitHub Actions.

## ✨ Features

*   **Real-Time Prediction API**: A high-performance API endpoint (`/predict`) to classify messages as SPAM or HAM.
*   **Containerized Service**: The entire application is packaged with Docker for consistent, portable, and scalable deployments.
*   **Automated CI/CD Pipeline**: Every push to the `main` branch automatically triggers a pipeline that trains, tests, and deploys the model.
*   **Experiment Tracking**: Model training runs, parameters, and metrics are tracked using MLflow.
*   **Automated Model Testing**: A dedicated testing suite (`pytest`) validates model performance and behavior before deployment, acting as a quality gate.
*   **Modular & Production-Ready Code**: The codebase is structured for maintainability, scalability, and readability.

## 🛠️ Tech Stack

| Category | Technology |
| :--- | :--- |
| **Backend & Serving** | FastAPI, Uvicorn |
| **MLOps & Deployment** | Docker, GitHub Actions, MLflow |
| **Data & Modeling** | Scikit-learn, Pandas, NumPy |
| **Testing** | Pytest |
| **Language** | Python 3.9+ |

## 📂 Project Structure

```
spam-classifier/
├── .github/workflows/      # CI/CD pipeline definitions
│   └── cicd.yml
├── data/                   # Raw datasets
│   └── sms_spam.csv
├── models/                 # Saved model and vectorizer artifacts
│   ├── classifier.pkl
│   └── vectorizer.pkl
├── src/                    # Source code for the application
│   ├── __init__.py
│   ├── api.py              # FastAPI application logic
│   ├── data_pipeline.py    # Data loading and cleaning
│   └── model_training.py   # Model training and MLflow tracking
├── tests/                  # Automated tests
│   └── test_model.py
├── Dockerfile              # Instructions for building the service container
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🚀 Getting Started

### Prerequisites

*   Git
*   Python 3.9+
*   Docker Desktop

### 1. Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
    cd YOUR_REPOSITORY
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 2. Usage

#### A. Running the Training Pipeline

This script will train the model, evaluate it, and save the final artifacts (`classifier.pkl` and `vectorizer.pkl`) to the `models/` directory.

```bash
python src/model_training.py
```

To view the experiment logs, run the MLflow UI:
```bash
mlflow ui
```
Navigate to `http://127.0.0.1:5000` in your browser.

#### B. Running the API Service

You can run the API locally for development or as a Docker container for a production-like environment.

**Option 1: Run Locally with Uvicorn**
```bash
uvicorn src.api:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

**Option 2: Run with Docker (Recommended)**
1.  **Build the Docker image:**
    ```bash
    docker build -t spam-classifier:latest .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -d -p 8080:8000 --name spam_api spam-classifier:latest
    ```
    The API will be available at `http://localhost:8080`.

### 3. Testing the API

Once the service is running, you can send a `POST` request to the `/predict` endpoint.

```bash
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{"message": "URGENT! You have won a 1 week FREE membership in our prize draw. Txt the word CLAIM to No: 81010."}'
```

**Expected Response:**
```json
{
  "message_in": "URGENT! You have won a 1 week FREE membership in our prize draw. Txt the word CLAIM to No: 81010.",
  "prediction_label": "SPAM",
  "confidence": 0.9985
}
```

## 🤖 MLOps Pipeline (CI/CD)

The CI/CD pipeline is defined in `.github/workflows/cicd.yml` and automates the entire MLOps lifecycle.

**Trigger:** A push to the `main` branch.

1.  **`train` Job**:
    *   Checks out the code.
    *   Installs dependencies.
    *   Runs the `model_training.py` script to produce new model artifacts.
    *   Uploads the `models/` directory as an artifact for the next job.

2.  **`test` Job (Quality Gate)**:
    *   Depends on the success of the `train` job.
    *   Downloads the trained model artifacts.
    *   Runs the `pytest` suite in `tests/test_model.py` to validate model behavior and performance against a baseline.
    *   **If tests fail, the pipeline stops here.**

3.  **`deploy` Job**:
    *   Depends on the success of the `test` job.
    *   Logs in to Docker Hub using repository secrets.
    *   Builds the Docker image, which includes the newly trained and tested model artifacts.
    *   Pushes the image to Docker Hub with two tags: `latest` and the unique run number.

## 📖 API Endpoints

| Endpoint | Method | Request Body | Success Response |
| :--- | :--- | :--- | :--- |
| `/predict` | `POST` | `{"message": "string"}` | `{"prediction_label": "string", "confidence": float}` |
| `/health` | `GET` | (None) | `{"status": "ok", "model_ready": true}` |

## 📈 Future Improvements

*   **Data & Model Drift Monitoring**: Implement a system to monitor production data and model predictions to detect drift.
*   **Cloud Deployment**: Deploy the container to a cloud service like AWS ECS, Google Cloud Run, or Azure Container Apps.
*   **Feature Store Integration**: Use a feature store like Feast for more robust feature management.
*   **A/B Testing Framework**: Implement a framework to safely roll out new models and compare their performance against the current production model.
