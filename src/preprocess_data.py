"""
This module preprocess the data.
"""
import pandas as pd

# Load the dataset
def load_dataset(file_path):
    """Loads the stock market dataset."""
    return pd.read_csv(file_path)

# Handle outliers
def remove_outliers(df, column, lower_percentile=0.01, upper_percentile=0.99):
    """Removes outliers in a specified column based on percentile values."""
    lower_limit = df[column].quantile(lower_percentile)
    upper_limit = df[column].quantile(upper_percentile)
    return df[(df[column] >= lower_limit) & (df[column] <= upper_limit)]

# Preprocess the data
def preprocess_data(df):
    """Cleans and preprocesses the stock market data."""
    # Remove duplicates
    df = df.drop_duplicates(subset=["Date", "Stock"])

    # Handle missing values (if any)
    df = df.dropna()

    # Remove outliers in numeric columns
    numeric_columns = ["Open", "High", "Low", "Close", "Volume", "Change Pct"]
    for column in numeric_columns:
        df = remove_outliers(df, column)

    # Sort by date for each stock
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["Stock", "Date"])

    return df

# Feature engineering
def add_features(df):
    """Adds features to the dataset for stock price prediction."""
    # Calculate daily price range
    df["Price Range"] = df["High"] - df["Low"]

    # Calculate moving averages (7-day and 30-day)
    df["7-Day MA"] = df.groupby("Stock")["Close"].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df["30-Day MA"] = df.groupby("Stock")["Close"].transform(lambda x: x.rolling(window=30, min_periods=1).mean())

    # Calculate daily percentage change
    df["Daily Change"] = df.groupby("Stock")["Close"].pct_change()

    # Drop rows with NaN values introduced by feature engineering
    df = df.dropna()

    return df

# Save preprocessed data
def save_preprocessed_data(df, output_path):
    """Saves the preprocessed dataset to a CSV file."""
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # File paths
    INPUT_FILE = "data/stocks_df.csv"
    OUTPUT_FILE = "preprocessed_stock_data.csv"

    # Load the dataset
    data = load_dataset(INPUT_FILE)

    # Preprocess the data
    data = preprocess_data(data)

    # Add features
    data = add_features(data)

    # Save the preprocessed dataset
    save_preprocessed_data(data, OUTPUT_FILE)

    print("Preprocessing complete. Preprocessed data saved to", OUTPUT_FILE)
