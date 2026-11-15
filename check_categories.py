
import pandas as pd

metadata = pd.read_csv("./data/painting_metadata.csv")
labels = pd.read_csv("./data/painting_labels.csv")

data = labels.merge(
    metadata[["Object ID", "Object Name", "Classification"]],
    on="Object ID",
    how="left"
)

print("Downloaded image categories:")
print(data["Object Name"].value_counts())

print("\nMissing categories:")
print(data["Object Name"].isna().sum())

print("\nSample records:")
print(data[["filename", "Object Name", "age"]].head(15))
