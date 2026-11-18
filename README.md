# PathSnap: Distributed Map-Matching & Route Reconstruction

PES University, UE24CS352A Machine Learning hackathon.
Team: Sravya, Mohit, Monica, Adil.

## Structure
- `src/`      data loading, candidates, labeling, features, models, evaluation, training
- `api/`      FastAPI backend (Laptop 1)
- `ui/`       evaluator interface (Laptop 2)
- `artifacts/` saved model and config
- `results/`  CSV results, plots, Folium maps
- `report/`   technical report
- `config.yaml` all settings

## Setup
```
pip install -r requirements.txt
```

## Data
Place the dataset in `data/Student folder/` (train/ and student_test/). It is not committed.

## Build datasets
```
cd src
python build_dataset.py --data-root "../data/Student folder" --config ../config.yaml --out ../results
```

## Train and compare models (run only your own models)
```
python train_compare.py --data-root "../data/Student folder" --models KNN --tag yourname
```
Models: LogisticRegression, MLP, KNN, RandomForest, HistGradientBoosting.

## Run the API / UI
To be added.
