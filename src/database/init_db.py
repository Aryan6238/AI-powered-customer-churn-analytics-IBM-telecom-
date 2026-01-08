
from sqlalchemy import create_engine
from src.utils.config import settings
from src.database.models import Base
from src.utils.logger import setup_logger

logger = setup_logger("DB_Init")

def init_db():
    logger.info(f"Initializing database at: {settings.DATABASE_URL}")
    engine = create_engine(settings.DATABASE_URL)
    
    logger.info("Creating tables...")
    Base.metadata.create_all(engine)
    logger.info("Tables created successfully.")

if __name__ == "__main__":
    init_db()
