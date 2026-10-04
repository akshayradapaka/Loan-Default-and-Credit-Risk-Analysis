import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

data = {
    "Customer_ID": [f"CUST{i+1}" for i in range(n)],
    "Gender": np.random.choice(["Male", "Female"], n),
    "Age": np.random.randint(21, 65, n),
    "Education": np.random.choice(
        ["High School", "Bachelor", "Master", "PhD"], n
    ),
    "Employment_Status": np.random.choice(
        ["Employed", "Self-Employed", "Unemployed"], n
    ),
    "Marital_Status": np.random.choice(
        ["Single", "Married"], n
    ),
    "Dependents": np.random.randint(0, 5, n),
    "Annual_Income": np.random.randint(200000, 1500000, n),
    "Credit_Score": np.random.randint(300, 900, n),
    "Existing_Loans": np.random.randint(0, 5, n),
    "Loan_ID": [f"LN{i+1}" for i in range(n)],
    "Loan_Type": np.random.choice(
        ["Personal", "Home", "Auto", "Education"], n
    ),
    "Loan_Amount": np.random.randint(50000, 2000000, n),
    "Loan_Term_Months": np.random.choice(
        [12,24,36,48,60,84,120], n
    ),
    "Interest_Rate": np.round(
        np.random.uniform(6,18,n),2
    ),
    "Monthly_Installment": np.random.randint(
        2000,50000,n
    ),
    "Debt_to_Income_Ratio": np.round(
        np.random.uniform(0.1,0.8,n),2
    ),
    "Employment_Years": np.random.randint(0,30,n),
    "Previous_Defaults": np.random.randint(0,4,n),
    "Credit_History": np.random.choice(
        ["Good","Average","Poor"], n
    ),
    "Property_Ownership": np.random.choice(
        ["Owned","Rented"], n
    ),
    "Region": np.random.choice(
        ["North","South","East","West"], n
    ),
    "Loan_Status": np.random.choice(
        ["Approved","Rejected"], n
    ),
    "Default_Status": np.random.choice(
        ["Default","Non-Default"],
        n,
        p=[0.25,0.75]
    )
}

df = pd.DataFrame(data)

df.to_csv("loan_credit_risk.csv", index=False)

print("Dataset Created Successfully")
print(df.head())