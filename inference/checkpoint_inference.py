from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class CustomOlmo:
    def __init__(self, model_path, tokenizer_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def generate_text(self, input_text, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95):
        inputs = self.tokenizer([input_text], return_tensors='pt', return_token_type_ids=False)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k,
                                      top_p=top_p)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


if __name__ == '__main__':
    model_path = "no_exist/checkpoints/OLMo-gemma-1.2b/step63-unsharded"
    tokenizer_path = "google/gemma-2b-it"

    lm = CustomOlmo(model_path, tokenizer_path)

    while True:
        input_text = input("Enter a prompt (type 'exit' to quit): ")
        if input_text.lower() == 'exit':
            print("Exiting...")
            break
        generated_text = lm.generate_text(input_text)
        print(generated_text)
