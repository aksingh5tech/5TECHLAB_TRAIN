import os
import json
import gzip
from datasets import load_dataset
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Function to process a chunk of records
def process_chunk(records):
    return [{"text": record["text"]} for record in records]

# Main function
def main():
    # Load the dataset in streaming mode
    dataset = load_dataset("Zyphra/Zyda-2", split="train", streaming=True)

    # Convert the streaming dataset to a list for easier chunking
    records = list(dataset)

    # Define the output directory structure
    output_dir = "test_fixtures/Zyphra"
    os.makedirs(output_dir, exist_ok=True)

    # Output file path
    output_file = os.path.join(output_dir, "Zyda_2.json.gz")

    # Determine the number of processes to use (64 or available CPUs)
    num_processes = min(64, cpu_count())

    # Split the data into chunks for multiprocessing
    chunk_size = len(records) // num_processes + 1
    chunks = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]

    # Use multiprocessing to process the dataset
    with Pool(num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_chunk, chunks),
                desc="Processing Records",
                unit="chunk",
                total=len(chunks),
            )
        )

    # Flatten the results
    processed_records = [item for sublist in results for item in sublist]

    # Write the processed records to a compressed .gz file
    with gzip.open(output_file, "wt", encoding="utf-8") as gz_file:
        for record in processed_records:
            json.dump(record, gz_file)
            gz_file.write("\n")

    print(f"Compressed JSON saved to: {output_file}")

if __name__ == "__main__":
    main()
