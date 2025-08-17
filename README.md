# Movie Review Sentiment Analyzer

This project implements a multi-container MLOps system that:

* Serves sentiment predictions, either **positive** or **negative**, via a FastAPI service
* Monitors model behavior and data drift in real-time using a Streamlit dashboard
* Shares prediction logs using a Docker volume, accessible to both containers
* Includes a script that uses a test file to systematically evaluate model performance via the API

## Prerequisites

Before running this app, make sure you have the following installed:

### 1. Python 3.13+

You can check your version with:

```bash
python --version
```

### 2. pip (Python package manager)

```bash
pip --version
```

### 3. Git

```bash
git --version
```

### 4. WSL 2 Backend (Windows users only)

```bash
wsl --version
```

### 5. Docker Installed (required)

```bash
docker --version
```

## How to Run

Follow these steps to run the app locally (using Docker):

1. Clone the repo using bash (if on Windows, use WSL that allows for ssh cloning):

```bash
bash

git clone git@github.com:alantico98/Assignment5_ModelMonitoring.git

cd Assignment5_ModelMonitoring
```

2. Build the image (using WSL or Mac)

```bash    
make build
```

4. Run the Container (using WSL or Mac)

```bash    
make run
```

This will:
* Start the FastAPI app on http://localhost:8000
* Start the Streamlit Dashboard on http://localhost:8501.
* Create a named Docker volume sentiment-logs

If using Postman Desktop:
* Send a request to http://0.0.0.0:8000/health to check that the server is running and healthy

To use **curl** to interact with the API:
* Health Check:

```bash
curl -X GET http://localhost:8000/health
```

* Predict Sentiment"

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"text": "I loved this movie!", "true_label": "positive"}'
```

* Example response: {"sentiment": "positive"}

5. (Optional) Run the evaluation script

From ./Assignment5_ModelMonitoring:

```bash
python evaluate.py
```

This will request responses from the API server via JSON body objects, using the test.json file

6. Clean Up (using WSL or Mac)

```bash    
make clean
```