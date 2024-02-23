import argparse
from enum import Enum

from transformers import AutoModelForCausalLM, AutoTokenizer


class RunMode(Enum):
    ALVIS = "alvis"
    OLLAMA = "ollama"


def run_alvis():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    device = "cuda"
    model_filepath = (
        "/mimer/NOBACKUP/groups/ci-nlp-alvis/models/Mistral-7B-Instruct-v0.2"
    )

    model = AutoModelForCausalLM.from_pretrained(model_filepath)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    encodends = tokenizer("The capital of France is", return_tensors="pt")

    model.to(device)
    model_inputs = encodends.to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=200)
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    print(decoded[0])


def run_ollama():
    print("Running model using Ollama")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill text with an LLM")
    parser.add_argument(
        "runmode",
        choices=[m.value for m in RunMode],
        help="How to run the model",
    )
    args = parser.parse_args()

    if args.runmode == RunMode.ALVIS.value:
        run_alvis()
    else:
        run_ollama()
