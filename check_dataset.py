import os
import pandas as pd

IMG_FOLDER = "./data/images"
CSV_FILE = "./data/painting_labels.csv"

df = pd.read_csv(CSV_FILE)

images = os.listdir(IMG_FOLDER)

print("Total labels:", len(df))
print("Total images:", len(images))

print("\nFirst 10 records:")
print(df.head(10))

print("\nAge statistics:")
print(df["age"].describe())

print("\nDataset verification completed!")