import numpy as np
import pandas as pd

df = pd.read_csv("Churn_Modelling_Uncleaned.csv")

print("--- [Milestone 1] Dataset Overview ---")
print("Data Shape:", df.shape)
print(df.head())

print("\n--- Summary Info ---")
print(df.info())

print("\n--- [Milestone 2] Data Quality Check ---")
print("Duplicate Rows:", df.duplicated().sum())
print("Missing Values:\n", df.isnull().sum())

df_clean = df.copy()

df_clean.drop(
    columns=["RowNumber", "CustomerId", "Surname"], inplace=True, errors="ignore"
)

for col in df_clean.columns:
  if df_clean[col].isnull().sum() > 0:
    if df_clean[col].dtype in ["int64", "float64"]:
      fill_value = df_clean[col].median()
    else:
      fill_value = df_clean[col].mode()[0]
    
    df_clean.fillna({col: fill_value}, inplace=True)

df_clean.drop_duplicates(inplace=True)

df_clean.to_csv("Churn_Modelling_Cleaned.csv", index=False)

print("\n--- Final Result ---")
print("Shape Before:", df.shape)
print("Shape After:", df_clean.shape)
print("Cleaned file saved: Churn_Modelling_Cleaned.csv")