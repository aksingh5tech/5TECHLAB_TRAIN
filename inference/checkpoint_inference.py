from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class LanguageModel:
    def __init__(self, model_path, tokenizer_path):
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model.cuda()

    def generate_text(self, input_text, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95):
        # Prepare input tokens
        inputs = self.tokenizer([input_text], return_tensors='pt', return_token_type_ids=False)

        # Move input tensors to the same device as model
        inputs = {key: val.to(self.model.device) for key, val in inputs.items()}

        # Generate output
        output_sequences = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                               top_k=top_k, top_p=top_p, use_cache=True)

        # Streaming output
        for output in output_sequences:
            yield self.tokenizer.decode(output, skip_special_tokens=True)


if __name__ == '__main__':
    model_path = "no_exist/checkpoints/OLMo-gemma-1.2b/step63-unsharded"
    tokenizer_path = "google/gemma-2b-it"

    lm = LanguageModel(model_path, tokenizer_path)

    while True:
        input_text = input("Enter a prompt (type 'exit' to quit): ")
        if input_text.lower() == 'exit':
            print("Exiting...")
            break
        print("Generating text:")
        for text in lm.generate_text(input_text):
            print(text, end='', flush=True)
        print()  # Ensure newline after generation

 # python inference/checkpoint_inference.py