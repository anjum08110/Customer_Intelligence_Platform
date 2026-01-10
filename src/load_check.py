import pandas as pd

path = "data/raw/Online Retail.xlsx"

df = pd.read_excel(path)

print("Shape:",df.shape)
print("\nColoums\n",df.columns)
print("\nMissing Values\n",df.isna().sum())
print("\nSample Data\n",df.head())