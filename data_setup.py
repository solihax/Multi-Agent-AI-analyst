"""
data_setup.py
Downloads the telco churn dataset, loads it into SQLite, and generates
the churn analysis report used by the retriever agent.
"""

import os
import sqlite3
import pandas as pd

CSV_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DB_PATH = "data/company.db"
REPORT_PATH = "data/docs/churn_analysis_report.txt"


def load_dataframe() -> pd.DataFrame:
    os.makedirs("data", exist_ok=True)
    df = pd.read_csv(CSV_URL)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    print("Shape:", df.shape)
    print(df["Churn"].value_counts())
    return df


def build_sqlite_db(df: pd.DataFrame, db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    df.to_sql("customers", conn, if_exists="replace", index=False)
    conn.close()


def build_churn_report(df: pd.DataFrame, report_path: str = REPORT_PATH) -> str:
    os.makedirs("data/docs", exist_ok=True)

    churn_by_contract = df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).round(1)
    churn_by_internet = df.groupby("InternetService")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).round(1)
    churn_by_payment = df.groupby("PaymentMethod")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).round(1)

    report = f"""Customer Churn Analysis Report (derived from real customer data)

Churn rate by contract type:
{churn_by_contract.to_string()}

Churn rate by internet service type:
{churn_by_internet.to_string()}

Churn rate by payment method:
{churn_by_payment.to_string()}

Key finding: Month-to-month contracts show substantially higher churn than one-year
or two-year contracts, suggesting contract length is a strong retention lever.
Electronic check payment method also correlates with higher churn compared to
automatic payment methods, which may indicate lower engagement or payment friction.
"""

    with open(report_path, "w") as f:
        f.write(report)

    print(report)
    return report


def setup_data():
    """Runs the full data pipeline and returns the dataframe."""
    df = load_dataframe()
    build_sqlite_db(df)
    build_churn_report(df)
    return df