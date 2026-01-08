
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import shap
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import sys

# Add project root to path for imports to work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.config import settings
from src.ml.explain import ChurnExplainer
from src.ml.recommendations import RetentionRecommender
from src.ml.preprocessing import fetch_churn_data

# Page Config
st.set_page_config(
    page_title="AI Churn Analytics",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- CACHED RESOURCES ---
@st.cache_resource
def load_artifacts():
    model_path = os.path.join(settings.DATA_DIR, '..', 'src', 'ml', 'model_registry', 'best_model.pkl')
    pipeline = joblib.load(model_path)
    
    explainer = ChurnExplainer(model_path=model_path)
    recommender = RetentionRecommender()
    
    return pipeline, explainer, recommender

@st.cache_data
def load_data():
    # Fetch data using existing ETL/Preprocessing function
    df = fetch_churn_data()
    return df

# --- MAIN LOGIC ---
try:
    pipeline, explainer_sys, recommender = load_artifacts()
    df_raw = load_data()
    
    # Generate Predictions for the whole dataset (for the dashboard view)
    # We need to preprocess first
    preprocessor = pipeline.named_steps['preprocessor']
    X_raw = df_raw.drop(columns=['churn_value', 'customer_id'], errors='ignore')
    
    # Prediction
    probs = pipeline.predict_proba(X_raw)[:, 1]
    df_raw['Churn Probability'] = probs
    df_raw['Predicted Churn'] = (probs > 0.5).astype(int)
    
    # Init Explainer once
    X_processed = preprocessor.transform(X_raw)
    if explainer_sys.explainer is None:
        explainer_sys.fit_explainer(X_processed)

except Exception as e:
    st.error(f"Error loading system: {e}")
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("🔍 Filters")
prob_threshold = st.sidebar.slider("High Risk Threshold", 0.0, 1.0, 0.7, 0.05)
tenure_filter = st.sidebar.slider("Min Tenure (Months)", 0, 72, 0)

# Filter Data
df_filtered = df_raw[
    (df_raw['Churn Probability'] >= 0.0) & 
    (df_raw['tenure_in_months'] >= tenure_filter)
]
high_risk_customers = df_filtered[df_filtered['Churn Probability'] > prob_threshold]

# --- DASHBOARD LAYOUT ---

st.title("📉 AI-Powered Customer Retention Platform")

# 1. KPIs
col1, col2, col3, col4 = st.columns(4)

total_customers = len(df_filtered)
avg_churn_prob = df_filtered['Churn Probability'].mean()
high_risk_count = len(high_risk_customers)
revenue_at_risk = high_risk_customers['monthly_charges'].sum()

with col1:
    st.markdown(f"""<div class='metric-card'>Total Customers<br><span class='metric-value'>{total_customers}</span></div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class='metric-card'>Avg Churn Probability<br><span class='metric-value'>{avg_churn_prob:.1%}</span></div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class='metric-card'>High Risk Customers<br><span class='metric-value'>{high_risk_count}</span></div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class='metric-card'>Revenue at Risk (Monthly)<br><span class='metric-value'>${revenue_at_risk:,.0f}</span></div>""", unsafe_allow_html=True)

st.markdown("---")

# 2. Charts
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("📊 Churn Risk Distribution")
    fig_hist = px.histogram(df_filtered, x="Churn Probability", nbins=50, 
                            color="Predicted Churn", title="Probability Distribution",
                            color_discrete_map={0: "green", 1: "red"})
    st.plotly_chart(fig_hist, use_container_width=True)

with c2:
    st.subheader("💰 Revenue vs Risk")
    fig_scatter = px.scatter(df_filtered, x="tenure_in_months", y="monthly_charges", 
                             color="Churn Probability", size="total_charges",
                             color_continuous_scale="RdYlGn_r", title="Tenure vs Charges (Size=LTV)")
    st.plotly_chart(fig_scatter, use_container_width=True)

# 3. Customer Drill Down
st.markdown("---")
st.subheader("🕵️ Individual Customer Analysis")

# Select Customer
selected_customer_id = st.selectbox("Select High Risk Customer to Analyze", high_risk_customers['customer_id'].head(50))

if selected_customer_id:
    # Get Data
    cust_row = df_raw[df_raw['customer_id'] == selected_customer_id].iloc[0]
    
    # Layout using columns
    det1, det2 = st.columns([1, 2])
    
    with det1:
        st.info(f"**Customer ID:** {selected_customer_id}")
        st.write(f"**Churn Probability:** `{cust_row['Churn Probability']:.2%}`")
        st.write(f"**Tenure:** {cust_row['tenure_in_months']} months")
        st.write(f"**Contract:** {cust_row['contract']}")
        st.write(f"**Monthly Charges:** ${cust_row['monthly_charges']}")
        
        # Prepare for explanation
        X_single = X_raw[df_raw['customer_id'] == selected_customer_id]
        X_single_processed = preprocessor.transform(X_single)
        
        # Recommendation
        prob = cust_row['Churn Probability']
        drivers_raw = explainer_sys.interpret_prediction(X_single_processed)
        # Convert df row to dict
        cust_dict = cust_row.to_dict()
        
        plan = recommender.generate_plan(cust_dict, prob, drivers_raw)
        
        st.markdown("### 💡 Recommended Actions")
        st.error(f"**Risk Level:** {plan['risk_level']}")
        for action in plan['actions']:
             st.success(f"👉 {action}")
             
        with st.expander("Why these actions?"):
            for reason in plan['reasoning']:
                st.write(f"- {reason}")
    
    with det2:
        st.markdown("### 🔍 Why this prediction?")
        # Waterfall plot
        shap_values = explainer_sys.explainer(X_single_processed)
        shap_values.feature_names = list(explainer_sys.feature_names)
        
        # Create figure explicitly
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
        
        st.caption("Red bars push risk HIGHER. Blue bars push risk LOWER.")

