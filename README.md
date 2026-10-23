# Bank Customer Churn Prediction Project

## 📌 Project Overview
This project focuses on predicting bank customer churn using machine learning. By identifying customers who are likely to leave the bank in advance, proactive retention campaigns can be implemented to reduce churn rates.

## 📂 Repository Contents
- **`Bank Customer Churn Prediction.pptx`**: The complete presentation slides covering Milestones 1, 2, and 3 (Problem Definition, Data Cleaning, EDA, Feature Engineering, and Preprocessing Pipelines).
- **`Bank cutomer.py`**: The Python script used for loading, exploring, and cleaning the dataset.
- **`Churn_Modelling_Uncleaned.csv`**: The raw, uncleaned dataset containing 10,018 customer records before preprocessing.
- **`train_model_ready.csv`**: Preprocessed training dataset ready for machine learning (8,000 rows).
- **`test_model_ready.csv`**: Preprocessed testing dataset for evaluation (2,000 rows).
- **`churn_preprocessing_pipeline.joblib`**: Saved, reusable preprocessing pipeline artifact.

## 📊 Dataset & Cleaning Summary (Milestone 2)
- **Initial Rows**: 10,018 rows (Reduced to 10,000 clean records).
- **Missing Values**: 550 missing records successfully handled and resolved.
- **Duplicates**: 18 duplicate rows removed.
- **Outliers**: Account balance extreme anomalies were capped and handled properly.

## 📌 Milestone 3: Exploratory Data Analysis & Preprocessing Pipeline

In this milestone, we conducted an in-depth Exploratory Data Analysis (EDA) to uncover key churn patterns, performed feature engineering, and constructed a robust, leak-free preprocessing pipeline to prepare the dataset for machine learning models.

---

### 🔍 Key Business & Data Insights
* **Geographic Variations:** Customers in **Germany** exhibited the highest observed churn rate (**32.44%**), significantly higher than Spain (**16.67%**) and France (**16.15%**).
* **Demographic Patterns:** 
  * Churn peaked drastically among customers aged **50–59** (**56.37%**), compared to only **7.45%** in the 18–29 age group (a difference of **48.92 percentage points**).
  * Female customers showed a higher observed churn rate (**25.07%**) compared to males (**16.46%**).
* **Account Activity & Balance:**
  * Inactive members recorded higher churn (**26.85%**) than active members (**14.27%**).
  * Positive-balance customers had a higher churn rate (**24.08%**) than zero-balance customers (**13.82%**).
* **Linear Correlations:** `Age` showed the strongest positive linear correlation with churn (`+0.287`), while `IsActiveMember` had a negative correlation (`-0.156`).

---

### 🛠️ Feature Engineering
Two domain-specific features were created to enhance model usability:
1. **`AgeGroup`**: Binned continuous age values into 5 categorical bands (`18–29`, `30–39`, `40–49`, `50–59`, `60+`) to capture life-stage behaviors.
2. **`HasBalance`**: Converted account balance into a binary indicator (`0 = zero balance`, `1 = positive balance`).

---

### ⚙️ Preprocessing Pipeline & Data Leakage Prevention
To prevent data leakage, the dataset was split prior to any fitting operations:
* **Train / Test Split:** **80% Training** (8,000 rows) and **20% Testing** (2,000 rows).
* **Numeric Transformation:** Imputed missing values using the **Training Median** $\rightarrow$ Scaled using **`StandardScaler`**.
* **Categorical Transformation:** Imputed missing values using the **Most Frequent** strategy $\rightarrow$ Encoded using **`OneHotEncoder`**.
* **Strict Leakage Prevention:** Pipelines were fitted strictly on the training set using `fit_transform()`, while the test set was only transformed using `transform()`.

---

### ✅ Readiness & Output Deliverables
* **Final Input Features:** **19 numeric features** post-encoding and scaling.
* **Missing Values:** **0** missing values remaining across train/test sets.
* **Row Overlap:** Verified **0 overlap** between training and testing rows.
* **Saved Artifacts:**
  * `train_model_ready.csv` (8,000 rows)
  * `test_model_ready.csv` (2,000 rows)
  * `churn_preprocessing_pipeline.joblib` (Reusable pipeline artifact)

---

## 🎯 Project Success Criteria
- **Target Variable**: `Exited` (Binary classification: 0 for retention, 1 for churn)
- **Goal**: Isolate and target the top 15% high-risk churn customers

## 👤 Author
- **Hiba Mukaram Mahmoud** (Team 6)
