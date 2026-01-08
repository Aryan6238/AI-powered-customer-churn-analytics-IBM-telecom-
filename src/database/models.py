
from sqlalchemy import Column, Integer, String, Boolean, Float, ForeignKey, DateTime, func
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Customer(Base):
    __tablename__ = 'customers'
    
    customer_id = Column(String(64), primary_key=True)
    gender = Column(String(20))
    senior_citizen = Column(Boolean)
    partner = Column(Boolean)
    dependents = Column(Boolean)
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    subscriptions = relationship("Subscription", back_populates="customer")

class Subscription(Base):
    __tablename__ = 'subscriptions'
    
    subscription_id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(64), ForeignKey('customers.customer_id'))
    contract_type = Column(String(50))
    payment_method = Column(String(50))
    paperless_billing = Column(Boolean)
    monthly_charges = Column(Float)
    total_charges = Column(Float)
    tenure_months = Column(Integer)
    
    customer = relationship("Customer", back_populates="subscriptions")
    service_details = relationship("ServiceDetail", back_populates="subscription", uselist=False)
    churn_label = relationship("ChurnLabel", back_populates="subscription", uselist=False)

class ServiceDetail(Base):
    __tablename__ = 'service_details'
    
    service_id = Column(Integer, primary_key=True, autoincrement=True)
    subscriptions_id = Column(Integer, ForeignKey('subscriptions.subscription_id'))
    
    phone_service = Column(Boolean)
    multiple_lines = Column(String(50))
    internet_service = Column(String(50))
    internet_type = Column(String(50))
    online_security = Column(String(50))
    online_backup = Column(String(50))
    device_protection = Column(String(50))
    tech_support = Column(String(50))
    streaming_tv = Column(String(50))
    streaming_movies = Column(String(50))
    streaming_music = Column(String(50))
    unlimited_data = Column(String(50))
    
    subscription = relationship("Subscription", back_populates="service_details")

class ChurnLabel(Base):
    __tablename__ = 'churn_labels'
    
    label_id = Column(Integer, primary_key=True, autoincrement=True)
    subscriptions_id = Column(Integer, ForeignKey('subscriptions.subscription_id'))
    churn_value = Column(Integer) # 0 or 1
    churn_score = Column(Float)
    observation_date = Column(DateTime, server_default=func.now())
    
    subscription = relationship("Subscription", back_populates="churn_label")
