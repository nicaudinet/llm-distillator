import argparse
from enum import Enum
from pathlib import Path

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer


class RunMode(Enum):
    ALVIS = "alvis"
    OLLAMA = "ollama"


class Sentiment(Enum):
    POSITIVE = "pos"
    NEGATIVE = "neg"


class Topic(Enum):
    MUSIC = "music"
    BOOKS = "books"
    DVD = "dvd"
    CAMERA = "camera"
    HEALTH = "health"
    SOFTWARE = "software"


def parse_amazon_reviews(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    data = []
    for line in lines:
        split_line = line.split(" ")
        data.append(
            {
                "topic": Topic(split_line[0]),
                "sentiment": Sentiment(split_line[1]),
                "review_id": int(split_line[2].split(".")[0]),
                "review": " ".join(split_line[3:]),
            }
        )
    return pd.DataFrame(data)


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


def run_ollama(data):
    print(data.head(10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill text with an LLM")
    parser.add_argument(
        "runmode",
        choices=[m.value for m in RunMode],
        help="How to run the model",
    )
    parser.add_argument(
        "-d",
        "--datapath",
        type=Path,
        default=Path("data/dredze_amazon_reviews.txt"),
        help="Path to the Amazon reviews corpus",
    )
    args = parser.parse_args()

    data = parse_amazon_reviews(args.datapath)
    if args.runmode == RunMode.ALVIS.value:
        run_alvis()
    else:
        run_ollama(data)
