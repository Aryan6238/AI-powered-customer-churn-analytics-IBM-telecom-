---
description: How to run the database, ETL, and training pipeline
---

# Run Churn Prediction Pipeline

## 1. Install Dependencies
Ensure you are in the root directory (`Data`).
```powershell
pip install -r requirements.txt
```

## 2. Initialize Database (SQLite)
This creates the `churn.db` file and all tables locally.
```powershell
python -m src.database.init_db
```

## 3. Run Data Ingestion (ETL)
This reads `data/raw/telco.csv`, cleans it, and loads it into MySQL.
```powershell
python -m src.etl.ingest
```

## 4. Train Model
This fetches data from MySQL, preprocesses it, trains XGBoost/LightGBM/LogReg, and saves the best model.
```powershell
python -m src.ml.train
```

## 5. View Results
The best model will be saved to `src/ml/model_registry/best_model.pkl`.
Classification metrics will be printed to the console.
