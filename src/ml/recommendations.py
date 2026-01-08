
from typing import List, Dict, Any

class RetentionRecommender:
    """
    Generates actionable retention recommendations based on Churn Probability and SHAP explanations.
    
    Why Rule-Based?
    1. Deterministic: Marketing teams need consistent, approved messaging.
    2. Compliance: Ensures offers (e.g., discounts) are only given when strict criteria are met.
    3. Performance: Microsecond latency compared to LLM calls.
    """
    
    def __init__(self):
        # Define Offer Catalog
        self.OFFERS = {
            "DISCOUNT_10": "Offer 10% loyalty discount for 1 year commitment.",
            "TECH_SUPPORT_PROMO": "Free upgrade to Premium Tech Support for 3 months.",
            "ONBOARDING_CALL": "Schedule a 'success check-in' call with a specialist.",
            "FAMILY_PLAN": "Suggest merging lines into a Family Plan for savings.",
            "DATA_UPGRADE": "Double data limit for free for 6 months.",
            "CONTRACT_RENEWAL": "Waive activation fee for switching to 1-year contract."
        }

    def generate_plan(self, customer_data: Dict[str, Any], churn_prob: float, top_drivers: List[tuple]) -> Dict[str, Any]:
        """
        Creates a personalized retention plan.
        
        Args:
            customer_data: Raw customer dict (e.g., {'tenure': 2, 'monthly_charges': 100...})
            churn_prob: Float 0-1
            top_drivers: List of (feature_name, shap_value, direction)
        
        Returns:
            Dict containing 'risk_level', 'primary_reason', 'recommended_actions', 'message_template'
        """
        plan = {
            "customer_id": customer_data.get('customer_id', 'Unknown'),
            "risk_level": "Low",
            "actions": [],
            "reasoning": []
        }
        
        # 1. Determine Risk Level
        if churn_prob > 0.7:
            plan['risk_level'] = "Critical"
        elif churn_prob > 0.4:
            plan['risk_level'] = "High"
        elif churn_prob > 0.2:
            plan['risk_level'] = "Medium"
        
        # If Low risk, no aggressive action needed
        if plan['risk_level'] == "Low":
            plan['actions'].append("Standard Newsletter")
            return plan

        # 2. Analyze Drivers to Pick Actions
        # We look at the top 3 drivers that INCREASE churn (positive SHAP)
        risk_drivers = [d for d in top_drivers if d[2] == "INCREASES"]
        
        for feature, impact, _ in risk_drivers:
            feature_base = feature.replace('num__', '').replace('cat__', '')
            
            # --- Rule Set ---
            
            # Scenario A: High Prices / Pricing Issue
            if 'monthly_charges' in feature_base or 'total_charges' in feature_base:
                plan['actions'].append(self.OFFERS["DISCOUNT_10"])
                plan['reasoning'].append("Customer is price sensitive.")
            
            # Scenario B: Low Tenure / Early Churn Risk
            elif 'tenure' in feature_base:
                # Check raw value if available to be specific
                tenure = customer_data.get('tenure_months', 0)
                if tenure < 6:
                    plan['actions'].append(self.OFFERS["ONBOARDING_CALL"])
                    plan['reasoning'].append("Early lifecycle churn risk - needs engagement.")
                else:
                    plan['actions'].append(self.OFFERS["CONTRACT_RENEWAL"])
                    plan['reasoning'].append("Long-term customer at risk - lock in renewal.")

            # Scenario C: Tech Support / Service Quality
            elif 'tech_support' in feature_base or 'online_security' in feature_base:
                plan['actions'].append(self.OFFERS["TECH_SUPPORT_PROMO"])
                plan['reasoning'].append("Service gap detected in support/security features.")
                
            # Scenario D: Contract Type (Month-to-month)
            elif 'contract' in feature_base:
                 if 'Month' in str(customer_data.get('contract', '')):
                     plan['actions'].append(self.OFFERS["CONTRACT_RENEWAL"])
                     plan['reasoning'].append("Move from volatile Month-to-Month to stable contract.")

        # 3. Fallback Action
        if not plan['actions']:
            plan['actions'].append("General Retention Survey")
        
        # Deduplicate
        plan['actions'] = list(set(plan['actions']))
        
        return plan

    def generate_llm_prompt(self, plan: Dict) -> str:
        """
        Extensibility: Generates a prompt for an LLM to write the actual email/script.
        """
        return f"""
        Write a polite, empathetic retention email for a customer with {plan['risk_level']} churn risk.
        Main pain points: {', '.join(plan['reasoning'])}.
        Offer them: {', '.join(plan['actions'])}.
        Tone: Professional but warm.
        """
