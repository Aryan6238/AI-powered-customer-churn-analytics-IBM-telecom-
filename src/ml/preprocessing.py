
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.utils.config import settings
from src.utils.logger import setup_logger
from src.ml.features import FeatureConfig

logger = setup_logger("ML_Preprocessing")

def fetch_churn_data() -> pd.DataFrame:
    """
    Fetches and joins data from MySQL tables to create a flat dataset.
    """
    engine = create_engine(settings.DATABASE_URL)
    
    # We join:
    # 1. Subscriptions (base)
    # 2. Customers (demographics)
    # 3. Service Details (products)
    # 4. Churn Labels (target)
    
    query = """
    SELECT 
        s.customer_id,
        c.gender, c.senior_citizen, c.partner, c.dependents,
        s.contract_type AS contract, s.paperless_billing, s.payment_method, 
        s.monthly_charges, s.total_charges, s.tenure_months AS tenure_in_months,
        sd.phone_service, sd.multiple_lines, sd.internet_service, 
        sd.online_security, sd.online_backup, sd.device_protection, 
        sd.tech_support, sd.streaming_tv, 
        sd.streaming_movies,
        cl.churn_value
    FROM subscriptions s
    JOIN customers c ON s.customer_id = c.customer_id
    JOIN service_details sd ON s.subscription_id = sd.subscriptions_id
    JOIN churn_labels cl ON s.subscription_id = cl.subscriptions_id
    """
    
    try:
        logger.info("Fetching training data from database...")
        df = pd.read_sql(query, engine)
        logger.info(f"Fetched {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        raise

def build_pipeline(numeric_features, categorical_features):
    """
    Builds a Scikit-Learn preprocessing pipeline.
    """
    logger.info("Building preprocessing pipeline...")
    
    # 1. Numeric Transformer: Scale features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # 2. Categorical Transformer: One-Hot Encode
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Column Transformer: Apply to respective columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop unnecessary columns like ID
    )
    
    return preprocessor

def clean_and_prepare(df: pd.DataFrame):
    """
    Cleaning logic specific to training readiness.
    E.g. converting 'Yes'/'No' target to 1/0.
    """
    # Convert Target 'Yes'/'No' to 1/0 if not already
    # The ETL might have stored it as string 'Yes'/'No'
    if df[FeatureConfig.TARGET].dtype == 'object':
        df[FeatureConfig.TARGET] = df[FeatureConfig.TARGET].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
        
    return df

def get_data_split(handle_imbalance: bool = False):
    """
    Main entry point. Fetches data, preprocesses, and returns splits.
    
    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    # 1. Load
    df = fetch_churn_data()
    df = clean_and_prepare(df)
    
    # 2. Separate Features and Target
    target_col = FeatureConfig.TARGET
    X = df.drop(columns=[target_col, FeatureConfig.IDENTIFIER], errors='ignore')
    y = df[target_col]
    
    # 3. Identify Types based on config, but intersected with actual DB columns
    # (In case DB schema differs slightly from updated feature list)
    available_cols = set(X.columns)
    
    # Get config lists
    cat_cols = [c for c in FeatureConfig.get_categorical_features() if c in available_cols]
    
    # Numeric is everything else that is not excluded
    exclude = set(FeatureConfig.EXCLUDE_COLUMNS)
    num_cols = [c for c in X.columns if c not in cat_cols and c not in exclude]

    logger.info(f"Numeric features: {len(num_cols)}")
    logger.info(f"Categorical features: {len(cat_cols)}")
    
    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Build Pipeline
    preprocessor = build_pipeline(num_cols, cat_cols)
    
    # 6. Fit-Transform Training Data
    # Note: We return the preprocessor so it can be used in the inference pipeline
    # For training return raw X_train/X_test and let the pipeline model handle it, 
    # OR transform here. 
    # Best practice: Return the pipeline object to be part of the final model pipeline.
    
    return X_train, X_test, y_train, y_test, preprocessor
