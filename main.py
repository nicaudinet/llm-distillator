import argparse
from enum import Enum
from pathlib import Path

import ollama
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Conversation,
    ConversationalPipeline,
)


class RunMode(Enum):
    ALVIS = "alvis"
    OLLAMA = "ollama"


def parse_amazon_review(line: dict) -> dict:
    split_line = line["text"].split(" ")
    return {"text": " ".join(split_line[3:])}


def make_amazon_prompt(text: str) -> str:
    return f"""
        Rewrite the review such that the sentiment is completely neutral. It is
        very important that one cannot tell whether the review is positive or
        negative at all. Try and keep all other information in the review.
        
        Here's the review:

        {text}
        """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill text with an LLM")
    parser.add_argument(
        "runmode",
        choices=[m.value for m in RunMode],
        help="How to run the model",
    )
    parser.add_argument(
        "-d",
        "--data_path",
        type=Path,
        default="data/dredze_amazon_reviews.txt",
        help="Path to the Amazon review data file",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=Path,
        default="results",
        help="The directory to save responses to. If not provided, responses are printed on stdout instead",
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=None,
        help="The number of samples to use (default: all)",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1,
        help="The batch size to use for the pipeline",
    )
    args = parser.parse_args()

    data = load_dataset("text", data_files=args.data_path)
    data = data.map(parse_amazon_review)

    if args.num_samples is not None:
        data = data.select(range(args.num_samples))

    prompts = data.map(make_amazon_prompt)

    if args.runmode == RunMode.OLLAMA:
        responses = []
        for prompt in prompts:
            response = ollama.chat(
                model="mistral",
                messages=[{"role": "user", "content": prompt}],
            )
            print(f"Prompt:\n{prompt}")
            print(f"Response:\n{response['message']['content']}")

    elif args.runmode == RunMode.ALVIS:
        in_dir = Path("/mimer/NOBACKUP/groups/ci-nlp-alvis/")
        model_path = in_dir / Path("models/Mistral-7B-Instruct-v0.2")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        tokenizer_path = in_dir / Path("tokenizers/Mistral-7B-Instruct-v0.2")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        distillator = ConversationalPipeline(
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            framework="pt",
        )
        conversations = prompts.map(Conversation)
        conversations = conversations.map(distillator)
        conversations = conversations.map(lambda c: c.messages[-1]["content"])
        for i, content in enumerate(conversations):
            with open(args.out_dir / Path(f"{str(i).zfill(3)}.txt"), "w") as f:
                f.write(content)
