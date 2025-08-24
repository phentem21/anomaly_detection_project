# Anomaly Detection Project

This project provides a pipeline for detecting anomalies in tabular datasets using machine learning. It includes data loading, preprocessing, model training, evaluation, and attribution of anomalies.

## Features

- **Data Loading & Preprocessing:** Cleans and scales your dataset for modeling.
- **Anomaly Detection Model:** Uses scikit-learn models (e.g., Isolation Forest).
- **Attribution:** Explains detected anomalies.

## Project Structure

```
anomaly_detection_project/
├── data/                  # Place your CSV dataset here (e.g., my_dataset.csv)
├── main.py                # Entry point for running the pipeline
└── src/
    ├── __init__.py
    ├── data_loader.py     # Data loading and preprocessing
    ├── models.py          # Model definition and training
    └── attribution.py     # Attribution/explanation of anomalies
```

## Getting Started

### 1. Install Dependencies

```bash
pip install pandas scikit-learn numpy
```

### 2. Prepare Your Data

- Place your CSV file in the `data/` folder.
- Ensure your file has a `label` column (optional) and only numeric features.

### 3. Run the Pipeline

```bash
python3 main.py --epochs 10
```

### 4. Output

- The script prints progress and results to the console.
- Anomalies are attributed and explained at the end.

## Customization

- **Change dataset path:** Edit the path in `main.py` or pass a different path to `load_data`.
- **Model:** Modify `src/models.py` to use different algorithms or parameters.
- **Attribution:** Customize explanations in `src/attribution.py`.

## License

MIT