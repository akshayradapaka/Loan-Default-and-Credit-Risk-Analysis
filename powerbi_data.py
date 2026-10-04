import pandas as pd

df = pd.read_csv("loan_credit_risk.csv")

# Credit Score Group
df["Credit_Score_Group"] = pd.cut(
    df["Credit_Score"],
    bins=[300,580,670,740,850],
    labels=["Poor","Fair","Good","Excellent"]
)

# Income Group
df["Income_Group"] = pd.qcut(
    df["Annual_Income"],
    q=4,
    labels=["Low","Medium","High","Very High"]
)

df.to_csv(
    "loan_credit_risk_powerbi.csv",
    index=False
)

print("Power BI dataset exported successfully")