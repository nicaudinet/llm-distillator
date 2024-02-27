import argparse
from enum import Enum
from pathlib import Path

from distillators import (
    AmazonReviewDistillator,
    SyntheticDataDistillator,
    SyntheticDataDistillatorDGP,
)
from parsers import AmazonReviewParser, SyntheticDataParser
from runner import RunMode


class Experiment(Enum):
    AMAZON_REVIEWS = "amazon_reviews"
    SYNTHETIC = "synthetic"
    SYNTHETIC_DGP = "dgp"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill text with an LLM")
    parser.add_argument(
        "runmode",
        choices=[m.value for m in RunMode],
        help="How to run the model",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        choices=[e.value for e in Experiment],
        default=Experiment.AMAZON_REVIEWS.value,
        help="The experiment to run",
    )
    args = parser.parse_args()

    if args.experiment == Experiment.AMAZON_REVIEWS.value:
        datapath = Path("data/dredze_amazon_reviews.txt")
        parser = AmazonReviewParser()
        distillator = AmazonReviewDistillator(RunMode(args.runmode))
    elif args.experiment == Experiment.SYNTHETIC.value:
        datapath = Path("data/treatment_leakage/text_full")
        parser = SyntheticDataParser()
        distillator = SyntheticDataDistillator(RunMode(args.runmode))
    elif args.experiment == Experiment.SYNTHETIC_DGP.value:
        datapath = Path("data/treatment_leakage/text_full")
        parser = SyntheticDataParser()
        distillator = SyntheticDataDistillatorDGP(RunMode(args.runmode))
    else:
        assert False, "Invalid input"

    data = parser.parse(datapath)
    text = data["text"][0]
    assert type(text) == str
    prompt, response = distillator.distill(text)

    print(f"Original text:\n{text}")
    print(f"\n\nPrompt:\n{prompt}")
    print(f"\n\nDistilled text:\n{response}")
