import os
import json
import tarfile
from datasets import load_dataset

# Load the dataset in streaming mode
dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

# Define the output directory structure
base_output_dir = "test_fixtures"
json_folder = os.path.join(base_output_dir, "filename.json")
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

# Save the records in a JSONL file within the filename.json directory
file_path = os.path.join(json_folder, "c4-sample.01.json")
with open(file_path, "w") as f:
    for entry in records:
        json.dump(entry, f)
        f.write("\n")

# Create a .tar.gz file named filename.json.gz containing the filename.json directory
gz_filename = os.path.join(base_output_dir, "filename.json.gz")
with tarfile.open(gz_filename, "w:gz") as tar:
    tar.add(json_folder, arcname="filename.json")  # Set arcname to ensure folder structure in the archive

print(f"Data has been processed and saved as '{gz_filename}'.")
