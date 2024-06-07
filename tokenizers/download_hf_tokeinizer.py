from transformers import AutoTokenizer
import os
tokenizer = AutoTokenizer.from_pretrained(
   "meta-llama/Llama-2-7b-hf",
)

save_directory = "./"

# Create the directory if it does not exist
os.makedirs(save_directory, exist_ok=True)

# Save the tokenizer
tokenizer.save_pretrained(save_directory)

print(f"Tokenizer saved in {save_directory}")
