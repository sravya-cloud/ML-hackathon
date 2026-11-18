# PathSnap: Distributed Map-Matching & Route Reconstruction

**PES University – UE24CS352A Machine Learning Hackathon**

**Team:** Sravya, Mohit, Monica, Adil

## Project Overview

PathSnap is a machine-learning-based map-matching and route-reconstruction system. It processes GPS trajectories, generates candidate road segments, extracts features, predicts matching road segments and reconstructs connected routes.

## Project Structure

- `src/` – Data loading, candidate generation, feature extraction, ML models, Bayesian optimisation, HMM/Viterbi and evaluation.
- `api/` – FastAPI backend for route prediction.
- `ui/` – Streamlit evaluator interface and web UI.
- `artifacts/` – Saved models and configurations.
- `results/` – Evaluation results, plots and maps.
- `report/` – Project report.
- `config.yaml` – Project configuration.

## Technologies Used

- Python
- Scikit-learn
- FastAPI
- Streamlit
- Pandas and NumPy
- Folium
- HMM and Viterbi algorithm
- Bayesian optimisation

## Installation

Clone the repository:

```bash
git clone https://github.com/sravya-cloud/ML-hackathon.git
cd ML-hackathon
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install streamlit
```

## Dataset

The dataset is not included in this repository.

Place the dataset in the following structure:

```text
data/
└── Student folder/
    ├── train/
    └── student_test/
```

## Build Dataset

```bash
cd src
python build_dataset.py --data-root "../data/Student folder" --config ../config.yaml --out ../results
cd ..
```

## Train and Compare Models

```bash
python src/train_compare.py --data-root "data/Student folder" --models KNN --tag yourname
```

Available models include Logistic Regression, MLP, KNN, Random Forest and HistGradientBoosting.

## Run the FastAPI Backend

From the project root:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The API documentation is available at:

`http://localhost:8000/docs`

## Run the Streamlit Evaluator UI

Open a second terminal from the project root:

```bash
python -m streamlit run ui/app.py
```

Before running the UI, ensure its API URL in `ui/app.py` points to the machine running the FastAPI backend.

## Evaluation

The evaluator interface displays:

- Top-1 Accuracy
- Macro-F1 Score
- Route Continuity
- Inference Time
- Actual and predicted route visualisations

## Team Contributions

The project covers data preparation, feature engineering, model development, hyperparameter optimisation, route reconstruction, backend integration and evaluator UI development.

## Repository

https://github.com/sravya-cloud/ML-hackathon