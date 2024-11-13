import os
import json
import gzip
from datasets import load_dataset
from tqdm import tqdm  # Import tqdm for the progress bar

# Load the dataset in streaming mode
dataset = load_dataset("Zyphra/Zyda-2", split="train", streaming=True)

# Define the output directory structure
output_dir = "test_fixtures/Zyphra"
os.makedirs(output_dir, exist_ok=True)

# Output file path
output_file = os.path.join(output_dir, "Zyda_2.json.gz")

# Process records with a progress bar and write directly to the .gz file
with gzip.open(output_file, "wt", encoding="utf-8") as gz_file:
    # Use tqdm for progress tracking, estimating total as a large number (e.g., 1000)
    for record in tqdm(dataset, desc="Processing Records", unit="record"):
        # Write the 'text' field in JSONL format directly to the .gz file
        json.dump({"text": record["text"]}, gz_file)
        gz_file.write("\n")

print(f"Compressed JSON saved to: {output_file}")
