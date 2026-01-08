
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class CustomerInput(BaseModel):
    """
    Input schema matching the training features.
    """
    # Identifiers
    customer_id: Optional[str] = "input_user"
    
    # Demographics
    gender: str = Field(..., max_length=20)
    senior_citizen: int = Field(..., ge=0, le=1) # 0 or 1
    partner: str = Field(..., max_length=5) # Yes/No
    dependents: str = Field(..., max_length=5)
    
    # Account
    tenure_months: int = Field(..., ge=0)
    phone_service: str = Field(..., max_length=5)
    multiple_lines: str = Field(..., max_length=50)
    internet_service: str = Field(..., max_length=50)
    internet_type: Optional[str] = "None"
    online_security: str = Field(..., max_length=50)
    online_backup: str = Field(..., max_length=50)
    device_protection: str = Field(..., max_length=50)
    tech_support: str = Field(..., max_length=50)
    streaming_tv: str = Field(..., max_length=50)
    streaming_movies: str = Field(..., max_length=50)
    streaming_music: Optional[str] = "No"
    unlimited_data: Optional[str] = "No"
    
    # Contract
    contract: str = Field(..., max_length=50)
    paperless_billing: str = Field(..., max_length=5)
    payment_method: str = Field(..., max_length=50)
    
    # Financials
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    
    # Other
    offer: Optional[str] = "None"
    avg_monthly_gb_download: Optional[float] = 0.0
    
    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "senior_citizen": 0,
                "partner": "Yes",
                "dependents": "No",
                "tenure_months": 12,
                "phone_service": "Yes",
                "multiple_lines": "No",
                "internet_service": "Fiber Optic",
                "online_security": "No",
                "online_backup": "Yes",
                "device_protection": "No",
                "tech_support": "No",
                "streaming_tv": "Yes",
                "streaming_movies": "No",
                "contract": "Month-to-month",
                "paperless_billing": "Yes",
                "payment_method": "Electronic check",
                "monthly_charges": 89.5,
                "total_charges": 1074.0
            }
        }

class ExplanationPoint(BaseModel):
    feature: str
    impact: float
    direction: str

class RecommendationPlan(BaseModel):
    risk_level: str
    actions: List[str]
    reasoning: List[str]

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: int
    drivers: List[ExplanationPoint]
    retention_plan: RecommendationPlan
