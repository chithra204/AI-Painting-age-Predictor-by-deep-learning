
import pandas as pd
import os

CSV_FILE = "./data/MetObjects.csv.txt"
OUTPUT_FILE = "./data/painting_metadata.csv"

print("Scanning the museum dataset...")

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

total = 0
painting_count = 0
valid_count = 0

columns = [
    "Object ID",
    "Is Public Domain",
    "Object Name",
    "Title",
    "Object Date",
    "Object Begin Date",
    "Object End Date",
    "Classification",
    "Link Resource"
]

for chunk in pd.read_csv(
    CSV_FILE,
    usecols=columns,
    chunksize=50000,
    low_memory=False
):
    total += len(chunk)

    object_name = chunk["Object Name"].fillna("").astype(str)
    classification = chunk["Classification"].fillna("").astype(str)

    is_painting = (
        object_name.str.contains("painting", case=False) |
        classification.str.contains("painting", case=False)
    )

    paintings = chunk[is_painting].copy()
    painting_count += len(paintings)

    public_domain = (
        paintings["Is Public Domain"]
        .astype(str).str.lower().isin(["true", "1"])
    )

    begin = pd.to_numeric(
        paintings["Object Begin Date"], errors="coerce"
    )
    end = pd.to_numeric(
        paintings["Object End Date"], errors="coerce"
    )

    valid_dates = begin.notna() & end.notna()

    valid = paintings[
        public_domain & valid_dates &
        paintings["Object ID"].notna()
    ].copy()

    valid_count += len(valid)

    if not valid.empty:
        valid.to_csv(
            OUTPUT_FILE,
            mode="a",
            index=False,
            header=not os.path.exists(OUTPUT_FILE)
            or os.path.getsize(OUTPUT_FILE) == 0
        )

    print("Records scanned:", total)

print("\nScanning completed!")
print("Total records:", total)
print("Painting records:", painting_count)
print("Public-domain paintings with dates:", valid_count)
print("Output saved to:", OUTPUT_FILE)
