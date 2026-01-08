
from typing import List, Dict

class FeatureConfig:
    """
    Configuration for Feature Selection and Grouping.
    Acts as the single source of truth for model features.
    """

    # --- 1. Target Variable ---
    TARGET: str = "churn_value"  # Mapped from 'Churn Label' in ETL

    # --- 2. Identity Column (Exclude from training) ---
    IDENTIFIER: str = "customer_id"

    # --- 3. Data Leakage & Irrelevant Columns ---
    # These must be EXCLUDED from training.
    EXCLUDE_COLUMNS: List[str] = [
        "customer_id",
        "churn_label",      # Duplicate of target
        "churn_score",      # LEAKAGE: The probability output of another model (or ground truth proxy)
        "churn_category",   # LEAKAGE: Only known after the customer churns (e.g., 'Competitor')
        "churn_reason",     # LEAKAGE: Post-hoc explanation (e.g., 'Competitor made better offer')
        "customer_status",  # LEAKAGE: Direct proxy for target (e.g., 'Churned', 'Stayed')
        "count",            # Metadata
        "country",          # Single value (US) in this dataset
        "state",            # Single value (CA) in this dataset
        "quarter",          # Data collection artifact
        "lat_long",         # Redundant if Lat/Long exist
        "latitude",         # Spatial (often excluded for base models to avoid overfitting location)
        "longitude",        # Spatial
        "zip_code",         # High cardinality ID-like feature
        "city"              # High cardinality
    ]

    # --- 4. Feature Groups ---
    
    DEMOGRAPHIC_FEATURES: List[str] = [
        "gender",
        "age",
        "under_30",
        "senior_citizen",
        "partner",
        "dependents",
        "number_of_dependents"
    ]

    TENURE_ACCOUNT_FEATURES: List[str] = [
        "tenure_in_months",
        "contract",         # e.g., Month-to-month
        "paperless_billing",
        "payment_method",   # e.g., Bank transfer
        "offer"             # Marketing offer accepted
    ]

    SERVICE_USAGE_FEATURES: List[str] = [
        "phone_service",
        "multiple_lines",
        "internet_service", # DSL, Fiber, No
        "internet_type",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
        "streaming_music",
        "unlimited_data",
        "avg_monthly_gb_download",
        "avg_monthly_long_distance_charges"
    ]

    FINANCIAL_FEATURES: List[str] = [
        "monthly_charge",
        "total_charges",
        "total_refunds",
        "total_extra_data_charges",
        "total_long_distance_charges",
        "total_revenue"
    ]

    SATISFACTION_LOYALTY_FEATURES: List[str] = [
        "satisfaction_score", # 1-5 rating
        "cltv",               # Customer Lifetime Value
        "referred_a_friend",
        "number_of_referrals"
    ]

    @classmethod
    def get_all_features(cls) -> List[str]:
        """Returns flattened list of all input features."""
        return (
            cls.DEMOGRAPHIC_FEATURES +
            cls.TENURE_ACCOUNT_FEATURES +
            cls.SERVICE_USAGE_FEATURES +
            cls.FINANCIAL_FEATURES +
            cls.SATISFACTION_LOYALTY_FEATURES
        )

    @classmethod
    def get_categorical_features(cls) -> List[str]:
        """
        Manually defined list of categorical features for encoding.
        In production, this might be inferred, but explicit is better for pipelines.
        """
        return [
            "gender", "under_30", "partner", "dependents", "offer",
            "phone_service", "multiple_lines", "internet_service", "internet_type",
            "online_security", "online_backup", "device_protection",
            "tech_support", "streaming_tv", "streaming_movies", 
            "streaming_music", "unlimited_data", "contract", 
            "paperless_billing", "payment_method", "referred_a_friend"
        ]
