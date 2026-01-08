
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Churn Prediction AI"
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    RAW_DATA_PATH: str = os.path.join(DATA_DIR, "raw", "telco.csv")
    
    @property
    def DATABASE_URL(self):
        # Creates churn.db in the data folder
        db_path = os.path.join(self.DATA_DIR, "churn.db")
        return f"sqlite:///{db_path}"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )

settings = Settings()
