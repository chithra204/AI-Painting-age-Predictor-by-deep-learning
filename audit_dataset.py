
import pandas as pd

metadata = pd.read_csv("./data/painting_metadata.csv")
labels = pd.read_csv("./data/painting_labels.csv")

print("Object Name distribution:")
print(metadata["Object Name"].value_counts().head(20))

print("\nClassification distribution:")
print(metadata["Classification"].value_counts().head(20))

print("\nNumber of labeled images:", len(labels))
print("Unique artwork IDs:", labels["Object ID"].nunique())
print("Duplicate filenames:", labels["filename"].duplicated().sum())
