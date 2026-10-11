"""Script to explore twcs.csv schema and analyze AmazonHelp conversation pairs."""
import re
from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path(r"C:\Users\kesha\Downloads\archive\twcs\twcs.csv")

def explore():
    print(f"Reading sample from {RAW_DATA_PATH}...")
    # Read fast chunk of 50,000 rows
    df = pd.read_csv(RAW_DATA_PATH, nrows=50000)
    print("Schema:", df.dtypes.to_dict())
    print("Total rows loaded:", len(df))

    # Top brands
    brands = df[~df['inbound']]['author_id'].value_counts()
    print("\nTop 5 brands:")
    print(brands.head(5))

    # Filter AmazonHelp
    brand_df = df[(df['author_id'] == 'AmazonHelp') & (~df['inbound']) & (df['in_response_to_tweet_id'].notna())].copy()
    brand_df['in_response_to_tweet_id'] = brand_df['in_response_to_tweet_id'].astype(int)

    cust_df = df[df['inbound']].copy()
    pairs = brand_df.merge(
        cust_df,
        left_on='in_response_to_tweet_id',
        right_on='tweet_id',
        suffixes=('_brand', '_customer')
    )

    print(f"\nPairs extracted for AmazonHelp in 500k rows: {len(pairs)}")

    # Simple ASCII/English check
    def is_english(text):
        if not isinstance(text, str):
            return False
        # Ratio of ascii printable chars
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        return (ascii_chars / max(1, len(text))) > 0.85

    pairs = pairs[pairs['text_customer'].apply(is_english) & pairs['text_brand'].apply(is_english)].copy()
    print(f"English pairs count: {len(pairs)}")

    print("\nSample 3 pairs:")
    for idx, row in pairs.head(3).iterrows():
        print(f"[{row['tweet_id_customer']}] Customer: {row['text_customer']}")
        print(f"[{row['tweet_id_brand']}] Brand:    {row['text_brand']}")
        print("-" * 50)

if __name__ == "__main__":
    explore()
