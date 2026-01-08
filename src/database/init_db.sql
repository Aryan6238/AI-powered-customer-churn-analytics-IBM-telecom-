
-- Disable foreign key checks to allow creating tables in any order (if needed later)
SET FOREIGN_KEY_CHECKS = 0;

-- 1. Customers
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(64) PRIMARY KEY,
    gender VARCHAR(20),
    senior_citizen BOOLEAN,
    partner BOOLEAN,
    dependents BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Subscriptions
CREATE TABLE IF NOT EXISTS subscriptions (
    subscription_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id VARCHAR(64) NOT NULL,
    contract_type VARCHAR(50),
    payment_method VARCHAR(50),
    paperless_billing BOOLEAN,
    monthly_charges DECIMAL(10, 2),
    total_charges DECIMAL(10, 2),
    tenure_months INT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE,
    INDEX idx_tenure (tenure_months),
    INDEX idx_charges (monthly_charges)
);

-- 3. Service Details
CREATE TABLE IF NOT EXISTS service_details (
    service_id INT AUTO_INCREMENT PRIMARY KEY,
    subscriptions_id INT NOT NULL,
    phone_service BOOLEAN,
    multiple_lines VARCHAR(50),
    internet_service VARCHAR(50),
    internet_type VARCHAR(50), -- Added this based on column in CSV/Feature config if needed, usually mapped to internet_service
    online_security VARCHAR(50),
    online_backup VARCHAR(50),
    device_protection VARCHAR(50),
    tech_support VARCHAR(50),
    streaming_tv VARCHAR(50),
    streaming_movies VARCHAR(50),
    streaming_music VARCHAR(50), -- Found in feature list
    unlimited_data VARCHAR(50), -- Found in feature list
    FOREIGN KEY (subscriptions_id) REFERENCES subscriptions(subscription_id) ON DELETE CASCADE
);

-- 4. Churn Labels
CREATE TABLE IF NOT EXISTS churn_labels (
    label_id INT AUTO_INCREMENT PRIMARY KEY,
    subscriptions_id INT NOT NULL,
    churn_value INT, -- 1/0
    churn_score FLOAT, 
    observation_date DATE DEFAULT (CURRENT_DATE),
    FOREIGN KEY (subscriptions_id) REFERENCES subscriptions(subscription_id) ON DELETE CASCADE,
    INDEX idx_churn (churn_value)
);

SET FOREIGN_KEY_CHECKS = 1;
