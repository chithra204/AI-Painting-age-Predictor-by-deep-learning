
import os
import pandas as pd
from PIL import Image

CSV_FILE = "./data/painting_labels.csv"
IMG_FOLDER = "./data/images"
OUTPUT_FILE = "./data/final_dataset.csv"

df = pd.read_csv(CSV_FILE)

metadata = pd.read_csv("./data/painting_metadata.csv")

df = df.merge(
    metadata[["Object ID", "Object Name"]],
    on="Object ID",
    how="left"
)

allowed_categories = [
    "Painting",
    "Painting, miniature"
]

df = df[df["Object Name"].isin(allowed_categories)].copy()

valid_records = []

for _, row in df.iterrows():
    image_path = os.path.join(
        IMG_FOLDER, row["filename"]
    )

    try:
        with Image.open(image_path) as img:
            img.verify()

        if pd.notna(row["age"]) and row["age"] > 0:
            valid_records.append(row)

    except Exception:
        print("Invalid image:", row["filename"])

final_df = pd.DataFrame(valid_records)

final_df.to_csv(OUTPUT_FILE, index=False)

print("\nDataset preparation completed!")
print("Valid images:", len(final_df))
print("Saved to:", OUTPUT_FILE)
