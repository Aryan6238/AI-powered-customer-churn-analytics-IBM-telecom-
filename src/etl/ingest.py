
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from src.utils.config import settings
from src.utils.logger import setup_logger

logger = setup_logger("ETL_Ingest")

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Reads the raw CSV file."""
    try:
        logger.info(f"Reading raw data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Successfully read {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Performs basic data cleaning and type conversion."""
    logger.info("Starting data cleaning...")
    
    # Copy to avoid SettingWithCopy warnings
    df = df.copy()
    
    # 1. Handle Total Charges (force numeric, coerce errors to NaN)
    # The dataset often has ' ' for new customers with 0 tenure
    if 'Total Charges' in df.columns:
        df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
        df['Total Charges'] = df['Total Charges'].fillna(0.0)

    # 2. Standardize Column Names (snake_case for easier mapping)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # 3. Handle Binary/Categorical mapping
    # Rename columns to match schema expectations
    df.rename(columns={
        'married': 'partner',
        'device_protection_plan': 'device_protection',
        'premium_tech_support': 'tech_support',
        'streaming_music': 'streaming_music', 
        'unlimited_data': 'unlimited_data'
    }, inplace=True)

    if 'senior_citizen' in df.columns:
        df['senior_citizen'] = df['senior_citizen'].apply(lambda x: True if x == 'Yes' else False)

    logger.info("Data cleaning completed.")
    return df

def push_to_db(df: pd.DataFrame):
    """Normalizes and pushes data to MySQL tables."""
    engine = create_engine(settings.DATABASE_URL)
    
    try:
        with engine.connect() as conn:
            logger.info("Connected to database.")
            
            # --- 1. CUSTOMERS TABLE ---
            logger.info("Processing 'customers' table...")
            customers_df = df[['customer_id', 'gender', 'senior_citizen', 'partner', 'dependents']].copy()
            # Drop duplicates if any
            customers_df.drop_duplicates(subset=['customer_id'], inplace=True)
            
            # Write to DB
            customers_df.to_sql('customers', con=conn, if_exists='append', index=False)
            logger.info(f"Loaded {len(customers_df)} customers.")

            # --- 2. SUBSCRIPTIONS TABLE ---
            logger.info("Processing 'subscriptions' table...")
            subs_df = df[['customer_id', 'contract', 'paperless_billing', 'payment_method', 
                          'monthly_charge', 'total_charges', 'tenure_in_months']].copy()
            
            # Rename columns to match schema
            subs_df.rename(columns={
                'contract': 'contract_type',
                'monthly_charge': 'monthly_charges',
                'tenure_in_months': 'tenure_months'
            }, inplace=True)
            
            subs_df.to_sql('subscriptions', con=conn, if_exists='append', index=False)
            logger.info(f"Loaded {len(subs_df)} subscriptions.")
            
            # --- 3. FETCH SUBSCRIPTION IDs FOR RELATIONSHIPS ---
            # We need the auto-incremented subscription_ids to link service_details and churn_labels
            logger.info("Fetching generated subscription IDs...")
            map_query = text("SELECT subscription_id, customer_id FROM subscriptions")
            mapping_df = pd.read_sql(map_query, conn)
            
            # Merge raw df with mapping to get subscription_id attached to the data
            merged_df = df.merge(mapping_df, on='customer_id', how='inner')
            
            # --- 4. SERVICE_DETAILS TABLE ---
            logger.info("Processing 'service_details' table...")
            services_cols = [
                'subscription_id', 'phone_service', 'multiple_lines', 'internet_service',
                'online_security', 'online_backup', 'device_protection_plan', 
                'premium_tech_support', 'streaming_tv', 'streaming_movies', 'unlimited_data'
            ]
            
            # Cleaned DF already has 'device_protection' and 'tech_support' from clean_data rename
            srv_df = merged_df.copy()
            srv_df['subscriptions_id'] = srv_df['subscription_id']
            
            # reliable list of schema cols
            schema_cols = [
                'subscriptions_id', 'phone_service', 'multiple_lines', 'internet_service', 
                'online_security', 'online_backup', 'device_protection', 
                'tech_support', 'streaming_tv', 'streaming_movies', 'streaming_music', 'unlimited_data'
            ]
            
            # Ensure only existing cols are selected
            srv_to_load = srv_df[schema_cols].copy()
            
            # Boolean conversions for services often "Yes"/"No"/"No internet service"
            # Schema defined them as VARCHAR(50) mostly, except PhoneService boolean.
            # Let's trust the VARCHAR definition for flexibility, as requested ("Generic").
            
            srv_to_load.to_sql('service_details', con=conn, if_exists='append', index=False)
            logger.info(f"Loaded {len(srv_to_load)} service details.")

            # --- 5. CHURN_LABELS TABLE ---
            logger.info("Processing 'churn_labels' table...")
            churn_df = merged_df[['subscription_id', 'churn_label', 'churn_score']].copy()
            
            churn_df.rename(columns={
                'subscription_id': 'subscriptions_id',
                'churn_label': 'churn_value',
                'churn_score': 'churn_score'
            }, inplace=True)
            
            churn_df.to_sql('churn_labels', con=conn, if_exists='append', index=False)
            logger.info(f"Loaded {len(churn_df)} churn labels.")
            
            conn.commit()
            logger.info("ETL Process Completed Successfully.")

    except Exception as e:
        logger.error(f"Database transaction failed: {e}")
        raise

if __name__ == "__main__":
    try:
        data = load_raw_data(settings.RAW_DATA_PATH)
        cleaned_data = clean_data(data)
        push_to_db(cleaned_data)
        logger.info("Data ingestion complete.")
    except Exception as e:
        logger.error(f"ETL Failed: {e}")
