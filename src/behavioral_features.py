from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "Online Retail.xlsx"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

OUT_PATH = OUT_DIR / "customers_behavioral.csv"


df = pd.read_excel(RAW_PATH)
df.columns = [c.strip() for c in df.columns]


df = df.dropna(subset=["CustomerID"])
df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]


orders = (
    df.groupby(["CustomerID", "InvoiceNo"])
      .agg(
          order_value=("TotalPrice", "sum"),
          items=("Quantity", "sum"),
          order_date=("InvoiceDate", "max"),
          distinct_products=("StockCode", "nunique"),
      )
      .reset_index()
)


cust = orders.groupby("CustomerID").agg(
    avg_order_value=("order_value", "mean"),
    avg_items_per_order=("items", "mean"),
    total_orders=("InvoiceNo", "nunique"),
    total_items=("items", "sum"),
    distinct_products=("distinct_products", "sum"),
    active_days=("order_date", lambda x: x.dt.date.nunique()),
    customer_lifetime_days=("order_date", lambda x: (x.max() - x.min()).days),
)


orders_sorted = orders.sort_values(["CustomerID", "order_date"])
orders_sorted["days_between"] = (
    orders_sorted.groupby("CustomerID")["order_date"]
    .diff()
    .dt.days
)

gap_features = orders_sorted.groupby("CustomerID").agg(
    avg_days_between_orders=("days_between", "mean"),
    order_std_days=("days_between", "std"),
)

features = cust.join(gap_features)


features = features.fillna(0)
features = features.round(2)

features.to_csv(OUT_PATH)
print("Saved:", OUT_PATH)
print("Shape:", features.shape)
print(features.head(10))
