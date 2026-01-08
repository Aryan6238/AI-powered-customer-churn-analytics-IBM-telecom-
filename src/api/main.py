
from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager
import pandas as pd
import joblib
import os
from sqlalchemy import create_engine, text

from src.utils.config import settings
from src.utils.logger import setup_logger
from src.ml.explain import ChurnExplainer
from src.ml.recommendations import RetentionRecommender
from src.api.schemas import CustomerInput, PredictionResponse

logger = setup_logger("API")

# Global variables for model and artifacts
ml_artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Model
    model_path = os.path.join(settings.DATA_DIR, '..', 'src', 'ml', 'model_registry', 'best_model.pkl')
    try:
        logger.info(f"Loading model from {model_path}...")
        pipeline = joblib.load(model_path)
        
        # Initialize Explainer
        explainer = ChurnExplainer(model_path=model_path)
        # We need to fit explainer on some dummy data or training data 
        # For simplicity in this demo, we assume TreeExplainer handles lazy loading or we fit on request (slow)
        # Ideally, we load a pre-fitted explainer. 
        # For now, we will lazily init the explainer on first request or strictly use the model.
        # But `explain.py` does `fit_explainer` which is fast for TreeExplainer.
        
        # We also need the Recommender
        recommender = RetentionRecommender()
        
        ml_artifacts['pipeline'] = pipeline
        ml_artifacts['explainer'] = explainer
        ml_artifacts['recommender'] = recommender
        logger.info("ML Artifacts loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load ML artifacts: {e}")
        raise
    
    yield
    
    # Clean up
    ml_artifacts.clear()

app = FastAPI(title="Churn Prediction AI", lifespan=lifespan)

def get_db_connection():
    engine = create_engine(settings.DATABASE_URL)
    try:
        conn = engine.connect()
        return conn
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": 'pipeline' in ml_artifacts}

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(input_data: CustomerInput):
    """
    Predict churn from raw JSON input.
    """
    try:
        # Convert input to DataFrame
        data_dict = input_data.model_dump()
        df = pd.DataFrame([data_dict])
        
        # 1. Feature Engineering / Rename if needed
        # Our Pydantic model uses snake_case which matches the training expectations mostly
        # ensuring `partner` maps to `partner` etc.
        
        # 2. Predict
        pipeline = ml_artifacts['pipeline']
        prob = pipeline.predict_proba(df)[0][1]
        pred = int(prob > 0.5)
        
        # 3. Explain
        explainer = ml_artifacts['explainer']
        # We need transformed features for SHAP
        preprocessor = pipeline.named_steps['preprocessor']
        X_processed = preprocessor.transform(df)
        
        # Init explainer if needed (first run)
        if explainer.explainer is None:
             explainer.fit_explainer(X_processed) 
        
        # Get factors
        drivers_raw = explainer.interpret_prediction(X_processed)
        drivers = [
            {"feature": f, "impact": i, "direction": d} 
            for f, i, d in drivers_raw
        ]
        
        # 4. Recommend
        recommender = ml_artifacts['recommender']
        plan = recommender.generate_plan(data_dict, prob, drivers_raw)
        
        return {
            "customer_id": input_data.customer_id,
            "churn_probability": float(prob),
            "churn_prediction": pred,
            "drivers": drivers,
            "retention_plan": plan
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict_customer/{customer_id}", response_model=PredictionResponse)
def predict_customer_from_db(customer_id: str):
    """
    Fetch customer latest state from DB and predict.
    """
    conn = get_db_connection()
    try:
        # Fetch joined flattened data similar to ML training query
        # We need a reusable way to fetch single row. 
        # For now, let's just run the big query with a WHERE clause.
        query = text("""
        SELECT 
            s.customer_id,
            c.gender, c.senior_citizen, c.partner, c.dependents,
            s.contract_type AS contract, s.paperless_billing, s.payment_method, 
            s.monthly_charges, s.total_charges, s.tenure_months,
            sd.phone_service, sd.multiple_lines, sd.internet_service, 
            sd.internet_type,
            sd.online_security, sd.online_backup, sd.device_protection, 
            sd.tech_support, sd.streaming_tv, 
            sd.streaming_movies, sd.streaming_music, sd.unlimited_data
        FROM subscriptions s
        JOIN customers c ON s.customer_id = c.customer_id
        JOIN service_details sd ON s.subscription_id = sd.subscriptions_id
        WHERE s.customer_id = :cid
        LIMIT 1
        """)
        
        result = conn.execute(query, {"cid": customer_id}).mappings().first()
        if not result:
            raise HTTPException(status_code=404, detail="Customer not found in database")
            
        data_dict = dict(result)
        
        # Convert to CustomerInput structure to ensure type safety & defaults
        # Handle some SQL -> Pydantic naming mismatches if any
        # 'tenure_months' -> 'tenure_months' (Schema has 'tenure_months', Pydantic has 'tenure_months')
        # 'contract_type' -> 'contract' (Schema query aliased?, wait. query says `s.contract_type AS contract`)
        # Booleans in DB (0/1) need to be handled if Pydantic expects Str 'Yes'/'No'
        # My Pydantic schema expects STR 'Yes'/'No' for many fields because model was trained on Mapped data?
        # WAIT. `ingest.py` mapped 'Yes'/'No' -> Boolean/0/1 in database.
        # But `train.py` fetches from DB (0/1/Boolean) and trains.
        # So the Model expects 0/1/Boolean or Scaled values?
        # `preprocessing.py`:
        #   fetch_churn_data -> pd.read_sql -> returns Booleans/Ints usually.
        #   build_pipeline -> OneHotEncoder.
        # OHE can handle 0/1 integers.
        # BUT Pydantic `CustomerInput` defined fields as `str` (max_length=5) e.g. "Yes"/"No".
        # If I pass this to the model, OHE might fail if training saw 0/1 and now sees "Yes".
        # CHECK `train.py` -> `get_data_split` -> `fetch_churn_data`:
        # Query: `c.partner` (BOOLEAN in Table).
        # So Model was trained on [0, 1] (False/True).
        # My Pydantic Input expects strings "Yes"/"No".
        # I need to align them.
        
        # FIX: The Pydantic model should ideally match the DB/Training format (Booleans/Ints).
        # However, typically JSON APIs use strings or explicit booleans.
        # If input is JSON "Yes", I must convert to 1/0 before DataFrame.
        # If input is DB 1/0, it's already ready.
        
        # Let's adjust the logic in `predict_churn` to handle mapping if needed.
        # OR update Pydantic schema to be stricter.
        # Given trained model uses database Types (0/1 for booleans), 
        # API payloads usually send readable "Yes"/"No".
        # I will map input strings to 1/0 in `predict_churn` before creating DataFrame.
        
        # For this `predict_customer_from_db` function, data comes from DB as 0/1/Boolean.
        # So it's already correct format for Model? Yes.
        # But `predict_churn` (JSON) needs mapping.
        
        # Let's proceed with creating the DataFrame directly from DB dict.
        conn.close()
        
        # We invoke the logic of predict_churn BUT skipping Pydantic validation 
        # or we construct Pydantic object (which might fail if types differ).
        # Let's just refactor the core logic to a helper function.
        
        return _run_prediction_logic(data_dict, customer_id)

    except Exception as e:
        logger.error(f"DB Lookup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

def _run_prediction_logic(data_dict: Dict, customer_id: str):
    # Preprocessing to match Training Data Types
    # Map "Yes"/"No" to 1/0 if strings exist (from JSON input)
    for k, v in data_dict.items():
        if v == "Yes": data_dict[k] = 1
        if v == "No": data_dict[k] = 0
        if v == "True": data_dict[k] = 1
        if v == "False": data_dict[k] = 0
            
    df = pd.DataFrame([data_dict])
    
    # Fill defaults for missing columns (e.g. from Pydantic optionals)
    # The pipeline handles scaling/encoding.
    
    pipeline = ml_artifacts['pipeline']
    try:
        prob = pipeline.predict_proba(df)[0][1]
    except ValueError as ve:
        # Likely column mismatch
        logger.error(f"Model mismatch: {ve}")
        # Try to fix columns?
        raise HTTPException(status_code=400, detail=f"Data format error: {ve}")

    pred = int(prob > 0.5)
    
    explainer = ml_artifacts['explainer']
    preprocessor = pipeline.named_steps['preprocessor']
    X_processed = preprocessor.transform(df)
    
    if explainer.explainer is None:
         explainer.fit_explainer(X_processed) 
    
    drivers_raw = explainer.interpret_prediction(X_processed)
    drivers = [{"feature": f, "impact": i, "direction": d} for f, i, d in drivers_raw]
    
    recommender = ml_artifacts['recommender']
    plan = recommender.generate_plan(data_dict, prob, drivers_raw)
    
    return {
        "customer_id": customer_id,
        "churn_probability": float(prob),
        "churn_prediction": pred,
        "drivers": drivers,
        "retention_plan": plan
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
