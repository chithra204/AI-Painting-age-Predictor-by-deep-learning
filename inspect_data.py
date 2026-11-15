import pandas as pd
import os

CSV_FILE = "./data/MetObjects.csv.txt"

if not os.path.exists(CSV_FILE):
    print("CSV file not found!")
else:
    print("Reading dataset...")

    df = pd.read_csv(CSV_FILE, nrows=5, low_memory=False)

    print("\nDataset columns:")
    print(df.columns.tolist())

    print("\nFirst 5 records:")
    print(df.head())

    print("\nDataset shape (sample):")
    print(df.shape)