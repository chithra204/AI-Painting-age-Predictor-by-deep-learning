
import os
import time
import requests
import pandas as pd
from PIL import Image
from io import BytesIO

CSV_FILE = "./data/painting_metadata.csv"
IMG_FOLDER = "./data/images"
OUTPUT_CSV = "./data/painting_labels.csv"

MAX_IMAGES = 1000

os.makedirs(IMG_FOLDER, exist_ok=True)

df = pd.read_csv(CSV_FILE)

session = requests.Session()

if os.path.exists(OUTPUT_CSV):
    existing = pd.read_csv(OUTPUT_CSV)
    completed = set(existing["Object ID"].astype(str))
else:
    existing = pd.DataFrame()
    completed = set()

downloaded = 0
skipped = 0

for _, row in df.iterrows():

    object_id = str(row["Object ID"])

    if object_id in completed:
        continue

    if downloaded >= MAX_IMAGES:
        break

    api_url = (
        "https://collectionapi.metmuseum.org/public/"
        "collection/v1/objects/" + object_id
    )

    try:
        response = session.get(api_url, timeout=20)
        response.raise_for_status()
        artwork = response.json()

        image_url = artwork.get("primaryImage")

        if not image_url:
            skipped += 1
            continue

        begin = pd.to_numeric(
            row["Object Begin Date"], errors="coerce"
        )
        end = pd.to_numeric(
            row["Object End Date"], errors="coerce"
        )

        if pd.isna(begin) or pd.isna(end):
            skipped += 1
            continue

        creation_year = (begin + end) / 2
        age = 2026 - creation_year

        if age <= 0:
            skipped += 1
            continue

        image_response = session.get(
            image_url, timeout=30
        )
        image_response.raise_for_status()

        image = Image.open(
            BytesIO(image_response.content)
        ).convert("RGB")

        image_path = os.path.join(
            IMG_FOLDER, object_id + ".jpg"
        )

        image.save(image_path, "JPEG")

        record = pd.DataFrame([{
            "Object ID": object_id,
            "filename": object_id + ".jpg",
            "title": row["Title"],
            "creation_year": creation_year,
            "age": age
        }])

        record.to_csv(
            OUTPUT_CSV,
            mode="a",
            index=False,
            header=not os.path.exists(OUTPUT_CSV)
        )

        completed.add(object_id)
        downloaded += 1

        print(
            f"Downloaded {downloaded}: "
            f"{row['Title']} | Age: {age:.1f}"
        )

        time.sleep(0.25)

    except Exception as e:
        skipped += 1
        print("Skipped", object_id, ":", e)
        time.sleep(1)

print("\nDownload process completed!")
print("Images downloaded:", downloaded)
print("Records skipped:", skipped)
print("Images folder:", IMG_FOLDER)
print("Labels file:", OUTPUT_CSV)
