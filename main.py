import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("loan_credit_risk.csv")

# First 5 Records
print("\n===== FIRST 5 RECORDS =====")
print(df.head())

# Last 5 Records
print("\n===== LAST 5 RECORDS =====")
print(df.tail())

# Dataset Shape
print("\n===== DATASET SHAPE =====")
print(df.shape)

# Column Names
print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

# ==============================
# DATA EXPLORATION & CLEANING
# ==============================

# Data Types
print("\n===== DATA TYPES =====")
print(df.dtypes)

# Statistical Information
print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

# Missing Values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# Duplicate Records
print("\n===== DUPLICATE RECORDS =====")
print("Duplicates:", df.duplicated().sum())

# Remove Duplicates
df = df.drop_duplicates()

# Unique Values
print("\n===== UNIQUE VALUES =====")

for col in ["Gender",
            "Education",
            "Employment_Status",
            "Loan_Type",
            "Region",
            "Default_Status"]:
    print(f"\n{col}")
    print(df[col].unique())

# Missing Values After Cleaning
print("\n===== MISSING VALUES AFTER CLEANING =====")
print(df.isnull().sum())

# ==============================
# OUTLIER CHECK
# ==============================

numerical_columns = [
    "Annual_Income",
    "Credit_Score",
    "Loan_Amount",
    "Debt_to_Income_Ratio"
]

for col in numerical_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(f"\n{col}")
    print("Outliers:", len(outliers))

print("\n===== CLEAN DATASET SHAPE =====")
print(df.shape)

print("\n===== CLEAN DATASET SAMPLE =====")
print(df.head())
print(df.shape)

# ==============================
# DATA ANALYSIS
# ==============================

print("\n===== DATA ANALYSIS =====")

# Total Customers
total_customers = df["Customer_ID"].nunique()
print("Total Customers:", total_customers)

# Total Loans
total_loans = df["Loan_ID"].nunique()
print("Total Loans:", total_loans)

# Defaulted Loans
defaulted_loans = (df["Default_Status"] == "Default").sum()
print("Defaulted Loans:", defaulted_loans)

# Non-Defaulted Loans
non_defaulted_loans = (df["Default_Status"] == "Non-Default").sum()
print("Non-Defaulted Loans:", non_defaulted_loans)

# Default Rate
default_rate = (defaulted_loans / total_loans) * 100
print("Default Rate (%):", round(default_rate, 2))

# Average Annual Income
avg_income = df["Annual_Income"].mean()
print("Average Annual Income:", round(avg_income, 2))

# Average Credit Score
avg_credit_score = df["Credit_Score"].mean()
print("Average Credit Score:", round(avg_credit_score, 2))

# Average Loan Amount
avg_loan_amount = df["Loan_Amount"].mean()
print("Average Loan Amount:", round(avg_loan_amount, 2))

# Average Interest Rate
avg_interest_rate = df["Interest_Rate"].mean()
print("Average Interest Rate:", round(avg_interest_rate, 2))

# Average Debt-to-Income Ratio
avg_dti = df["Debt_to_Income_Ratio"].mean()
print("Average Debt-to-Income Ratio:", round(avg_dti, 2))

# ==============================
# LOANS BY LOAN TYPE
# ==============================

print("\n===== LOANS BY LOAN TYPE =====")
print(df["Loan_Type"].value_counts())

# ==============================
# LOANS BY REGION
# ==============================

print("\n===== LOANS BY REGION =====")
print(df["Region"].value_counts())

# ==============================
# DEFAULT RATE BY LOAN TYPE
# ==============================

print("\n===== DEFAULT RATE BY LOAN TYPE =====")

loan_default_rate = (
    df.groupby("Loan_Type")["Default_Status"]
    .apply(lambda x: (x == "Default").mean() * 100)
)

print(loan_default_rate.round(2))

# ==============================
# DEFAULT RATE BY EMPLOYMENT STATUS
# ==============================

print("\n===== DEFAULT RATE BY EMPLOYMENT STATUS =====")

employment_default_rate = (
    df.groupby("Employment_Status")["Default_Status"]
    .apply(lambda x: (x == "Default").mean() * 100)
)

print(employment_default_rate.round(2))

# ==============================
# DEFAULT RATE BY EDUCATION
# ==============================

print("\n===== DEFAULT RATE BY EDUCATION =====")

education_default_rate = (
    df.groupby("Education")["Default_Status"]
    .apply(lambda x: (x == "Default").mean() * 100)
)

print(education_default_rate.round(2))

# ==============================
# DEFAULT RATE BY PROPERTY OWNERSHIP
# ==============================

print("\n===== DEFAULT RATE BY PROPERTY OWNERSHIP =====")

property_default_rate = (
    df.groupby("Property_Ownership")["Default_Status"]
    .apply(lambda x: (x == "Default").mean() * 100)
)

print(property_default_rate.round(2))

# ==============================
# CREDIT SCORE GROUP
# ==============================

df["Credit_Score_Group"] = pd.cut(
    df["Credit_Score"],
    bins=[300, 580, 670, 740, 850],
    labels=["Poor", "Fair", "Good", "Excellent"]
)

print("\n===== DEFAULT RATE BY CREDIT SCORE GROUP =====")

credit_default_rate = (
    df.groupby("Credit_Score_Group")["Default_Status"]
    .apply(lambda x: (x == "Default").mean() * 100)
)

print(credit_default_rate.round(2))

# ==============================
# INCOME GROUP
# ==============================

df["Income_Group"] = pd.qcut(
    df["Annual_Income"],
    q=4,
    labels=["Low", "Medium", "High", "Very High"]
)

print("\n===== DEFAULT RATE BY INCOME GROUP =====")

income_default_rate = (
    df.groupby("Income_Group")["Default_Status"]
    .apply(lambda x: (x == "Default").mean() * 100)
)

print(income_default_rate.round(2))

print("\n===== TOP 10 CUSTOMERS BY LOAN AMOUNT =====")

top10 = df.nlargest(
    10,
    "Loan_Amount"
)[["Customer_ID", "Loan_Amount"]]

print(top10)

print("\n===== HIGHEST DEFAULT RATE LOAN TYPE =====")
print(loan_default_rate.idxmax())

print("\n===== HIGHEST DEFAULT RATE REGION =====")

region_default_rate = (
    df.groupby("Region")["Default_Status"]
    .apply(lambda x: (x == "Default").mean() * 100)
)

print(region_default_rate.round(2))

print("\nRegion With Highest Default Rate:")
print(region_default_rate.idxmax())

#Loan Default Distribution Chart

plt.figure(figsize=(6,6))

df["Default_Status"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title("Loan Default Distribution")
plt.ylabel("")
plt.show()

#Loan Count by Loan Type Chart

plt.figure(figsize=(8,5))

sns.countplot(
    x="Loan_Type",
    data=df
)

plt.title("Loan Count by Loan Type")
plt.show()

#Default Rate by Loan Type chart

plt.figure(figsize=(8,5))

loan_default_rate.sort_values().plot(
    kind="bar"
)

plt.title("Default Rate by Loan Type")
plt.ylabel("Default Rate (%)")
plt.show()

#Default Rate by Employment Status Chart

plt.figure(figsize=(8,5))

employment_default_rate.sort_values().plot(
    kind="bar"
)

plt.title("Default Rate by Employment Status")
plt.ylabel("Default Rate (%)")
plt.show()

#Default Rate by Education Chart

plt.figure(figsize=(8,5))

education_default_rate.sort_values().plot(
    kind="bar"
)

plt.title("Default Rate by Education")
plt.ylabel("Default Rate (%)")
plt.show()

#Credit Score Distribution Chart

plt.figure(figsize=(8,5))

sns.histplot(
    df["Credit_Score"],
    bins=20,
    kde=True
)

plt.title("Credit Score Distribution")
plt.show()

#Loan Amount Distribution

plt.figure(figsize=(8,5))

sns.histplot(
    df["Loan_Amount"],
    bins=20,
    kde=True
)

plt.title("Loan Amount Distribution")
plt.show()

#Annual Income Distribution

plt.figure(figsize=(8,5))

sns.histplot(
    df["Annual_Income"],
    bins=20,
    kde=True
)

plt.title("Annual Income Distribution")
plt.show()

#Credit Score vs Loan Amount

plt.figure(figsize=(8,5))

sns.scatterplot(
    x="Credit_Score",
    y="Loan_Amount",
    data=df
)

plt.title("Credit Score vs Loan Amount")
plt.show()

#Income vs Loan Amount

plt.figure(figsize=(8,5))

sns.scatterplot(
    x="Annual_Income",
    y="Loan_Amount",
    data=df
)

plt.title("Income vs Loan Amount")
plt.show()

#Loan Amount by Default Status CHart

plt.figure(figsize=(8,5))

sns.boxplot(
    x="Default_Status",
    y="Loan_Amount",
    data=df
)

plt.title("Loan Amount by Default Status")
plt.show()

#Credit Score by Default Status

plt.figure(figsize=(8,5))

sns.boxplot(
    x="Default_Status",
    y="Credit_Score",
    data=df
)

plt.title("Credit Score by Default Status")
plt.show()

#Debt-to-Income Ratio by Default Status

plt.figure(figsize=(8,5))

sns.boxplot(
    x="Default_Status",
    y="Debt_to_Income_Ratio",
    data=df
)

plt.title("Debt-to-Income Ratio by Default Status")
plt.show()

#Default Rate by Income Group

plt.figure(figsize=(8,5))

income_default_rate.sort_values().plot(
    kind="bar"
)

plt.title("Default Rate by Income Group")
plt.ylabel("Default Rate (%)")
plt.show()

#Correlation Heatmap

plt.figure(figsize=(10,8))

numeric_df = df.select_dtypes(include=["int64", "float64"])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()