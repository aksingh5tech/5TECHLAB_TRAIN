import argparse
from transformers import OlmoForCausalLM, AutoTokenizer
import torch

class LanguageModel:
    def __init__(self, model_path, tokenizer_path):
        self.model = OlmoForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if torch.cuda.is_available():
            self.model.cuda()

    def generate_text(self, input_text, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95):
        inputs = self.tokenizer([input_text], return_tensors='pt', return_token_type_ids=False)
        inputs = {key: val.to(self.model.device) for key, val in inputs.items()}
        output_sequences = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                               top_k=top_k, top_p=top_p, use_cache=True)
        for output in output_sequences:
            yield self.tokenizer.decode(output, skip_special_tokens=True)

def main(model_path, tokenizer_path):
    lm = LanguageModel(model_path, tokenizer_path)
    while True:
        input_text = input("Enter a prompt (type 'exit' to quit): ")
        if input_text.lower() == 'exit':
            print("Exiting...")
            break
        print("Generating text:")
        for text in lm.generate_text(input_text):
            print(text, end='', flush=True)
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a pretrained OLMo language model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer.')
    args = parser.parse_args()

    main(args.model_path, args.tokenizer_path)


#python inference/checkpoint_inference.py --model_path no_exist/checkpoints/OLMo-gemma-1.2b/hf/latest-unsharded --tokenizer_path google/gemma-2b-it
#python inference/checkpoint_inference.py --model_path no_exist/checkpoints/OLMo-gemma-1.2b/latest-unsharded --tokenizer_path google/gemma-2b-it