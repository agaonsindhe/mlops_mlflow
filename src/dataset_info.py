"""
This file provides info about dataset.
"""
import pandas as pd

# Load the dataset
df = pd.read_csv("data/stocks_df.csv")

# Total rows
total_rows = len(df)

# Unique values
unique_stocks = df["Stock"].nunique()
unique_dates = df["Date"].nunique()

# Check for duplicates (Date + Stock should be unique)
duplicates = df.duplicated(subset=["Date", "Stock"]).sum()

# Check for missing values
missing_values = df.isnull().sum()

# Summary statistics for numeric columns (Open, High, Low, Close, Volume, Change Pct)
numeric_stats = df[["Open", "High", "Low", "Close", "Volume", "Change Pct"]].describe()

# Check for unique combinations (Date + Stock + Close)
unique_combinations = df[["Date", "Stock", "Close"]].drop_duplicates()
unique_combinations_count = len(unique_combinations)

# Print the metrics
print(f"Total Rows: {total_rows}")
print(f"Unique Stocks: {unique_stocks}")
print(f"Unique Dates: {unique_dates}")
print(f"Duplicate Rows (Date + Stock): {duplicates}")
print("\nMissing Values Per Column:")
print(missing_values)
print("\nSummary Statistics for Numeric Columns:")
print(numeric_stats)
print(f"\nTotal Unique Combinations (Date + Stock + Close): {unique_combinations_count}")
