from enum import Enum
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


class Parser:
    def __init__(self):
        pass

    def parse(self, datapath: Path) -> pd.DataFrame:
        assert False, "Not implemented"


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


class AmazonReviewParser(Parser):
    def __init__(self):
        pass

    def parse(self, datapath: Path) -> pd.DataFrame:
        with open(datapath, "r") as f:
            lines = f.readlines()
        data = []
        for line in lines:
            split_line = line.split(" ")
            data.append(
                {
                    "topic": Topic(split_line[0]),
                    "sentiment": Sentiment(split_line[1]),
                    "review_id": int(split_line[2].split(".")[0]),
                    "text": " ".join(split_line[3:]),
                }
            )
        return pd.DataFrame(data)


class SyntheticDataParser(Parser):
    def __init__(self):
        pass

    def parse(self, datapath: Path, N=100) -> pd.DataFrame:
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
            filepath = Path(datapath) / Path(doc_name)
            with open(filepath, "r", encoding="utf-8") as file:
                contents = file.read()

            soup = BeautifulSoup(contents, "lxml")
            prompt_tags = soup.find_all("prompt")

            documents += [doc_name for _ in prompt_tags]
            sources += [tag["source"] for tag in prompt_tags]
            values += [tag["value"] for tag in prompt_tags]
            prompts += [tag.get_text(strip=True) for tag in prompt_tags]
            responses += [get_response(tag) for tag in prompt_tags]

        data = pd.DataFrame(
            {
                "Document": documents,
                "Source": sources,
                "Value": values,
                "Prompt": prompts,
                "text": responses,
            }
        )

        documents = data.groupby("Document", sort=False).agg({"Response": "\n".join})
        return pd.DataFrame(documents)
