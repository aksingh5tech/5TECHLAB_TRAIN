from transformers import AutoTokenizer

BASE_MODEL = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained("./gemma/")

if __name__ == '__main__':
    pass