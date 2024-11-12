import os
import json
import gzip
from datasets import load_dataset

# Load the dataset in streaming mode
dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

# Define the output directory structure
base_output_dir = "test_fixtures"
json_folder = os.path.join(base_output_dir, "fineweb")
os.makedirs(json_folder, exist_ok=True)

# Number of records to download
num_records_to_download = 1000

# Process only the first 1000 rows
records = []
for i, record in enumerate(dataset):
    # Stop after 1000 records
    if i >= num_records_to_download:
        break

    # Only keep the 'text' field in the desired format
    text_data = {"text": record["text"]}
    records.append(text_data)

# Save the records in a JSONL file and compress it as a .gz file
file_path = os.path.join("test_fineweb_edu.json.gz")
with gzip.open(file_path, "wt", encoding="utf-8") as gz_file:
    for entry in records:
        json.dump(entry, gz_file)
        gz_file.write("\n")

print(f"Compressed JSON saved to: {file_path}")
