
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.utils.logger import setup_logger
from src.ml.preprocessing import get_data_split
from src.utils.config import settings

logger = setup_logger("ML_Trainer")

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Trains a model and returns performance metrics.
    """
    logger.info(f"Training {model_name}...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Metrics
    # Focus on Class 1 (Churn)
    recall = recall_score(y_test, y_pred, pos_label=1)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
    
    logger.info(f"--- {model_name} Results ---")
    logger.info(f"Recall (Churn Capture): {recall:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    
    return {
        "model": model,
        "name": model_name,
        "recall": recall,
        "precision": precision,
        "roc_auc": roc_auc
    }

def main():
    try:
        # 1. Get Preprocessed Data
        # We need the preprocessor fitted on training data to transform test data correctly
        # The get_data_split function returns X_train, X_test as DataFrames, and a preprocessor pipeline
        X_train_raw, X_test_raw, y_train, y_test, preprocessor = get_data_split()
        
        # Fit-Transform Training Data
        logger.info("Transforming training data...")
        X_train = preprocessor.fit_transform(X_train_raw)
        
        # Transform Test Data (Do NOT refit)
        logger.info("Transforming test data...")
        X_test = preprocessor.transform(X_test_raw)
        
        # 2. Define Models
        # Using class_weight='balanced' or scale_pos_weight for handling imbalance
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        models = [
            (LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42), "Logistic Regression"),
            (XGBClassifier(scale_pos_weight=pos_weight, eval_metric='logloss', random_state=42), "XGBoost"),
            (LGBMClassifier(class_weight='balanced', verbosity=-1, random_state=42), "LightGBM")
        ]
        
        results = []
        for model, name in models:
            metrics = train_and_evaluate(model, X_train, y_train, X_test, y_test, name)
            results.append(metrics)
            
        # 3. Model Selection
        # Business Reasoning: Churn is costly. False Negatives (missing a churner) > False Positives.
        # We prioritize RECALL while maintaining reasonable Precision.
        
        results.sort(key=lambda x: x['recall'], reverse=True)
        best_model_info = results[0]
        
        logger.info(f"\n🏆 Best Model selected by Recall: {best_model_info['name']}")
        logger.info(f"Metrics: Recall={best_model_info['recall']:.4f}, AUC={best_model_info['roc_auc']:.4f}")
        
        # 4. Save Best Model
        # We save a Pipeline that includes the preprocessor to ensure raw data can be passed in production
        final_pipeline = settings.PROJECT_NAME 
        # Actually we need to reconstruct the pipeline: Preprocessor -> Model
        # But sklearn Pipeline expects steps.
        
        from sklearn.pipeline import Pipeline as SklearnPipeline
        full_pipeline = SklearnPipeline([
            ('preprocessor', preprocessor),
            ('classifier', best_model_info['model'])
        ])
        
        import os
        model_path = os.path.join(settings.DATA_DIR, '..', 'src', 'ml', 'model_registry', 'best_model.pkl')
        # Ensure dir exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        joblib.dump(full_pipeline, model_path)
        logger.info(f"Model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
