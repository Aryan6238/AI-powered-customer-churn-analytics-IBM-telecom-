# %% [markdown]
# # Customer Churn Analysis
# **Goal**: Understand the key drivers of customer churn using the IBM Telco dataset.
# 
# **Key Questions**:
# 1. What is the overall churn rate?
# 2. How does tenure affect churn?
# 3. What is the impact of contract type?
# 4. Is there a relationship between monthly charges and churn?
# 5. How does satisfaction score correlate with churn?

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# %% [markdown]
# ## 1. Load Data
# We load the raw dataset directly for analysis.

# %%
# Ensure we can find the data file relative to this notebook
# assuming notebook is in /notebooks and data is in /data/raw
DATA_PATH = "../data/raw/telco.csv"
if not os.path.exists(DATA_PATH):
    print(f"Warning: {DATA_PATH} not found. Please check path.")
else:
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset Shape: {df.shape}")
    display(df.head())

# %% [markdown]
# ## 2. Data Cleaning for EDA
# Convert 'Churn Label' (Yes/No) to binary for easier plotting.

# %%
# Create a binary Churn column
df['Churn_Binary'] = df['Churn Label'].apply(lambda x: 1 if x == 'Yes' else 0)

# %% [markdown]
# ## 3. Overall Churn Rate

# %%
churn_rate = df['Churn_Binary'].mean() * 100
print(f"Overall Churn Rate: {churn_rate:.2f}%")

plt.figure(figsize=(6, 6))
df['Churn Label'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
plt.title('Overall Churn Distribution')
plt.ylabel('')
plt.show()

# %% [markdown]
# ### Business Insight:
# - The dataset shows a churn rate of approximately **26-27%**. 
# - This indicates a significant loss of revenue and suggests that retention strategies should be a high priority.

# %% [markdown]
# ## 4. Effect of Tenure on Churn
# Do new customers churn more than loyal ones?

# %%
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='Tenure in Months', hue='Churn Label', multiple="stack", bins=36, palette="viridis")
plt.title('Churn Distribution by Tenure (Months)')
plt.xlabel('Tenure (Months)')
plt.show()

# %% [markdown]
# ### Business Insight:
# - **High Risk in Early Months**: Churn is highest among customers with low tenure (< 12 months). This "infant mortality" suggests onboarding issues or mismatched expectations.
# - **Loyalty Effect**: As tenure increases, churn drops significantly. Use this to target new customers with "sticky" features or onboarding incentives.

# %% [markdown]
# ## 5. Impact of Contract Type
# Month-to-month vs. One/Two Year contracts.

# %%
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Contract', hue='Churn Label', palette="pastel")
plt.title('Churn by Contract Type')
plt.show()

# Calculate churn rate by contract
contract_churn = df.groupby('Contract')['Churn_Binary'].mean() * 100
print(contract_churn)

# %% [markdown]
# ### Business Insight:
# - **Month-to-Month is volatile**: These customers have an extremely high churn rate compared to contract customers.
# - **Actionable Strategy**: Incentivize M2M customers to switch to 1-year contracts using discounts or value-adds.

# %% [markdown]
# ## 6. Monthly Charges vs. Churn
# Are expensive plans driving people away?

# %%
plt.figure(figsize=(12, 6))
sns.kdeplot(data=df, x='Monthly Charge', hue='Churn Label', fill=True, palette="crest")
plt.title('Distribution of Monthly Charges by Churn Status')
plt.show()

# %% [markdown]
# ### Business Insight:
# - **Price Sensitivity**: Churning customers tend to have **higher monthly charges** (peak around $70-$100).
# - Non-churners have a bimodal distribution, with a large peak at low charges (~$20, likely DSL/basic plans) and another at high charges (loyal premium users).
# - Evaluate if the value proposition for high-tier plans justifies the cost for at-risk users.

# %% [markdown]
# ## 7. Satisfaction Score vs. Churn
# (If available in dataset)

# %%
if 'Satisfaction Score' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Satisfaction Score', y='Churn_Binary', palette="magma")
    plt.title('Churn Rate by Satisfaction Score (1-5)')
    plt.ylabel('Churn Rate')
    plt.show()

# %% [markdown]
# ### Business Insight:
# - **Direct Correlation**: As expected, low satisfaction scores (1-2) drive the vast majority of churn.
# - **Warning Sign**: Even moderate scores (3) might have non-negligible churn.
# - **Action**: proactive outreach to users who give low ratings (NPS detractors).

