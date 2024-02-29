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


def parse_amazon_review(line: dict) -> dict:
    split_line = line["text"].split(" ")
    return {"text": " ".join(split_line[3:])}


def make_amazon_prompt(text: str) -> str:
    prompt = f"""
Rewrite the review such that the sentiment is completely neutral. It is
very important that one cannot tell whether the review is positive or
negative at all. Try and keep all other information in the review.

Here's the review:

{text}
        """
    return {"text": prompt}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill text with an LLM")
    parser.add_argument(
        "runmode",
        choices=["ollama", "alvis"],
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
        help="Directory to save responses to (default: ./results)",
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

    # Load the data
    data = load_dataset(
            "text", # Type of data to load (text file)
            data_files=str(args.data_path), # Filepath for the data
            split="train" # Return a Dataset rather than a DatasetDict
        )

    # Preprocess
    data = data.map(parse_amazon_review)
    if args.num_samples is not None:
        data = data.select(range(args.num_samples))
    prompts = data.map(make_amazon_prompt)

    if args.runmode == "ollama":
        print("Running with Ollama")
        responses = []
        for prompt in prompts:
            response = ollama.chat(
                model="mistral",
                messages=[{"role": "user", "content": prompt}],
            )
            print(f"Prompt:\n{prompt}")
            print(f"Response:\n{response['message']['content']}")

    elif args.runmode == "alvis":
        print("Running on Alvis")
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

        for i, prompt in enumerate(prompts):
            conversation = Conversation(prompt["text"])
            response = distillator(conversation)
            answer = response.messages[-1]["content"]
            with open(args.out_dir / Path(f"{str(i).zfill(3)}.txt"), "w") as f:
                f.write(answer)

    else:
        raise ValueError(f"Invalid runmode argument {runmode}")
