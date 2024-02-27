from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    # Set file names
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    outdir = Path("/mimer/NOBACKUP/groups/ci-nlp-alvis/")
    model_path = outdir / Path("models/Mistral-7B-Instruct-v0.2")
    tokenizer_path = outdir / Path("tokenizers/Mistral-7B-Instruct-v0.2")
    # Download models
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    # Save models
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(tokenizer_path)
