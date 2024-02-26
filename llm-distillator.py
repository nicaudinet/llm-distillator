import argparse
from enum import Enum
from pathlib import Path

import ollama
import pandas as pd
from bs4 import BeautifulSoup
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


def parse_synthetic_data(dirpath, N=100):
    """
    Parse the documents in text_full and return them as a pandas dataframe
    """

    documents = []
    sources = []
    values = []
    prompts = []
    responses = []

    def get_response(tag):
        return tag.find_next_sibling(string=True).get_text(strip=True)

    for i in range(1, N + 1):
        doc_name = str(i) + ".txt"
        filepath = Path(dirpath) / Path(doc_name)
        with open(filepath, "r", encoding="utf-8") as file:
            contents = file.read()

        soup = BeautifulSoup(contents, "lxml")
        prompt_tags = soup.find_all("prompt")

        documents += [doc_name for _ in prompt_tags]
        sources += [tag["source"] for tag in prompt_tags]
        values += [tag["value"] for tag in prompt_tags]
        prompts += [tag.get_text(strip=True) for tag in prompt_tags]
        responses += [get_response(tag) for tag in prompt_tags]

    return pd.DataFrame(
        {
            "Document": documents,
            "Source": sources,
            "Value": values,
            "Prompt": prompts,
            "Response": responses,
        }
    )


def group_paragraphs(data):
    """
    Group paragraphs from the same document into a single string
    """
    data = data.copy()
    # NOTE: it's crucial to use sort=False in groupby, otherwise pandas will
    # sort the rows by document name which messes things up
    documents = data.groupby("Document", sort=False).agg({"Response": "\n".join})
    return list(documents["Response"])


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


def make_amazon_prompt(sentiment, review):
    return f"""
        The sentiment of the following review is {sentiment.value}.

        Rewrite the review such that the sentiment is completely neutral. It is
        very important that one cannot tell whether the review is positive or
        negative at all. Try and keep all other information in the review.
        
        Here's the review:

        {review}
        """


def make_synthetic_prompt(W):
    return f"""
        Rewrite the text such that any information about whether the country
        joined an IMF program or not is completely removed, while keeping any
        other information about the country intact.
        
        Here's the text:

        {W}
        """


def make_synthetic_prompt_dgp(W):
    return f"""
        The following text was created by an GPT 2 model. Each paragraph was
        generated independently from a prompt that either:

        1). asked the model to write a generic paragraph about the country

            Example: "Antigua and Barbuda."

        2). asked the model to write a paragraph about how the country asked for
          an IMF program 

            Example: "Antigua and Barbuda's government has asked the IMF for a program."

        3). asked the model to write a paragraph about the demands of the IMF on
        the country

            Example: "International Monetary Fund: No labor policy liberalization in Antigua and Barbuda."
       
        Given how the data was generated, remove paragraphs that were generated
        from a prompt of type 3 completely, while keeping the other paragraphs
        exactly the same.

        Here's the text:

        {W}
        """


def run_ollama(message):
    stream = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": message}],
        stream=True,
    )
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)


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

    # data = parse_amazon_reviews(args.datapath)
    # row = 0
    # sentiment = data["sentiment"][row]
    # review = data["review"][row]
    # print("Original review:")
    # print(review)
    # print("Distilled review:")
    # message = make_amazon_prompt(sentiment, review)

    data = parse_synthetic_data("data/treatment_leakage/text_full")
    documents = group_paragraphs(data)
    text = documents[0]
    print("Original text:")
    print(text)
    print("\n\n")
    print("Prompt:")
    message = make_synthetic_prompt_dgp(text)
    print(message)
    print("\n\n")
    print("Distilled text:")

    if args.runmode == RunMode.ALVIS.value:
        run_alvis()
    else:
        run_ollama(message)
