
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from src.utils.config import settings
from src.utils.logger import setup_logger
from src.ml.preprocessing import get_data_split

logger = setup_logger("ML_Explainer")

class ChurnExplainer:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(settings.DATA_DIR, '..', 'src', 'ml', 'model_registry', 'best_model.pkl')
        
        logger.info(f"Loading model from {model_path}")
        self.pipeline = joblib.load(model_path)
        self.model = self.pipeline.named_steps['classifier']
        self.preprocessor = self.pipeline.named_steps['preprocessor']
        self.explainer = None
        self.feature_names = None

    def fit_explainer(self, X_sample):
        """
        Fits the SHAP explainer on a sample of data.
        """
        logger.info("Initializing SHAP explainer...")
        
        # We need feature names from the preprocessor
        # OneHotEncoder names + Numeric names
        # This can be tricky with ColumnTransformer, let's try to extract them
        try:
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                self.feature_names = self.preprocessor.get_feature_names_out()
            else:
                # Fallback or manual extraction if sklearn version is old (unlikely given reqs)
                logger.warning("Could not extract feature names automatically.")
                self.feature_names = [f"feature_{i}" for i in range(X_sample.shape[1])]
        except Exception as e:
             logger.warning(f"Feature name extraction error: {e}")
             self.feature_names = [f"feature_{i}" for i in range(X_sample.shape[1])]

        # Use TreeExplainer for Tree models (XGB/LGBM), Linear for LogReg
        # We can check model type
        model_type = type(self.model).__name__
        logger.info(f"Detected model type: {model_type}")

        if 'XGB' in model_type or 'LGBM' in model_type:
            self.explainer = shap.TreeExplainer(self.model)
        elif 'Logistic' in model_type:
            # LinearExplainer requires background data
            self.explainer = shap.LinearExplainer(self.model, X_sample)
        else:
            # KernelExplainer as fallback (slow)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, X_sample)

    def explain_global(self, X_processed, save_path="global_importance.png"):
        """
        Generates and saves a global feature importance plot.
        """
        logger.info("Generating global feature importance...")
        shap_values = self.explainer.shap_values(X_processed)
        
        # Robust handling for binary classification shape
        # XGBoost/LGBM might return (N, Features)
        # sklearn might return list [(N, Features), (N, Features)] for classes
        if isinstance(shap_values, list):
             # Usually index 1 is the positive class
             shap_vals = shap_values[1]
        else:
             shap_vals = shap_values

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals, X_processed, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Global importance plot saved to {save_path}")

    def explain_local(self, X_single_row, customer_id="Unknown", save_path="local_waterfall.png"):
        """
        Generates a waterfall plot for a single customer.
        """
        # SHAP values for single instance
        shap_values = self.explainer(X_single_row)
        
        # Waterfall plot
        plt.figure()
        # shap.plots.waterfall is for the new API (Explanation object)
        # If explainer returns numpy, we might need to adjust or use force_plot
        # TreeExplainer usually returns Explanation object in newer SHAP versions if called directly?
        # Actually explainer.shap_values() returns arrays. explainer() returns Explanation.
        
        # Let's ensure we have valid feature names in the explanation object
        shap_values.feature_names = list(self.feature_names)
        
        # For binary classification, we care about the positive class
        # If shape is (Features,), it's fine. If (2, Features), we take [:, 1]??
        # TreeExplainer on binary XGB usually returns just the margin unless specified.
        # Let's assume standard behavior for now and handle potential errors.
        
        # Pick the first (and only) row
        sv = shap_values[0]
        
        shap.plots.waterfall(sv, show=False)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Local explanation saved to {save_path}")
        
    def interpret_prediction(self, X_single_row):
        """
        Returns a text explanation of the prediction.
        """
        shap_values = self.explainer.shap_values(X_single_row)
        
        if isinstance(shap_values, list):
             vals = shap_values[1][0]
        else:
             vals = shap_values[0]
             
        # Sort by absolute impact
        indices = np.argsort(np.abs(vals))[::-1]
        
        top_factors = []
        for i in indices[:3]:
            fname = self.feature_names[i]
            impact = vals[i]
            direction = "INCREASES" if impact > 0 else "DECREASES"
            top_factors.append((fname, impact, direction))
            
        return top_factors

def run_explanation():
    try:
        # 1. Load Data & Preprocess
        X_train_raw, X_test_raw, y_train, y_test, _ = get_data_split()
        
        explainer_sys = ChurnExplainer()
        
        # Transform data
        X_train_processed = explainer_sys.preprocessor.transform(X_train_raw)
        X_test_processed = explainer_sys.preprocessor.transform(X_test_raw)
        
        # 2. Fit Explainer
        # We use a subsample of training data for background if needed, or just to init
        # For TreeExplainer, passing model is mostly enough, but feature names matter
        explainer_sys.fit_explainer(X_train_processed)
        
        # 3. Global Explainability
        explainer_sys.explain_global(X_test_processed, save_path="shap_global.png")
        
        # 4. Local Explainability (First customer in test set)
        sample_idx = 0
        customer_row = X_test_processed[sample_idx:sample_idx+1] # Keep 2D
        explainer_sys.explain_local(customer_row, save_path="shap_local.png")
        
        # 5. Text Explanation
        factors = explainer_sys.interpret_prediction(customer_row)
        print("\n--- Business Explanation (Top Drivers) ---")
        for feature, imp, direction in factors:
            print(f"Feature '{feature}' {direction} churn risk (Impact: {imp:.4f})")
            
    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        raise

if __name__ == "__main__":
    run_explanation()
