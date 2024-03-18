import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch.utils.data import DataLoader
from transformers import DistilBertModel, DistilBertTokenizer

sentiment_to_int = {"pos": 1, "neg": 0}

topic_to_int = {
    "books": 0,
    "dvd": 1,
    "music": 2,
    "health": 3,
    "software": 4,
    "camera": 5,
}


def load_reviews(original_path, distilled_path):
    """
    Load the original reviews, distilled reviews and the labels and return them
    as a dataframe
    """
    # Load the original Amazon reviews and their labels
    data = []
    with open(original_path, "r") as file:
        for line in file.readlines():
            words = line.split(" ")
            data.append(
                {
                    "topic": words[0],
                    "sentiment": words[1],
                    "review_id": words[2],
                    "original": " ".join(words[3:]),
                }
            )
    data = pd.DataFrame(data)
    # Load the distilled Amazon reviews
    with open(distilled_path, "r") as file:
        distilled = file.readlines()
    # Keep the same number of reviews
    num_reviews = min(len(data), len(distilled))
    data = data[:num_reviews]
    distilled = distilled[:num_reviews]
    # Add distilled reviews to the dataframe
    data["distilled"] = distilled
    return data


def load_reviews_and_log(original_path, distilled_path):
    with wandb.init(project=WANDB_PROJECT, job_type="load-reviews") as run:
        reviews = load_reviews(original_path, distilled_path)
        table = wandb.Table(dataframe=reviews)
        artifact = wandb.Artifact(
            "raw-reviews",
            type="dataset",
            description="The original and distilled Amazon reviews with labels",
            metadata={
                "num_reviews": len(reviews),
            },
        )
        artifact.add(table, "reviews-table")
        run.log_artifact(artifact)
    return reviews


def embed_tfidf(data, max_features: int) -> torch.Tensor:
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(list(data)).toarray()
    return torch.from_numpy(tfidf_matrix)


def embed_bert(data, batch_size: int) -> torch.Tensor:
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data_loader = DataLoader(data, batch_size=batch_size)

    embeddings = []
    for batch in data_loader:
        batch_tokens = tokenizer(
            batch, padding="max_length", truncation="longest_first", return_tensors="pt"
        )
        batch_tokens.to(device)
        with torch.no_grad():
            outputs = model(**batch_tokens)
        embeddings.append(outputs.last_hidden_state)

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings.mean(dim=1)


def classify(embeddings, labels, test_size, B, CL):
    n_samples = len(embeddings)
    Ys = labels["sentiment"].apply(lambda x: sentiment_to_int[x])
    Yt = labels["topic"].apply(lambda x: topic_to_int[x])
    acc_s = np.zeros(B)
    acc_t = np.zeros(B)
    for b in range(B):
        # Sentiment classifier
        Xb, Yb = resample(embeddings, Ys, n_samples=n_samples)
        Xb_tr, Xb_te, Yb_tr, Yb_te = train_test_split(Xb, Yb, test_size=test_size)
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(Xb_tr, Yb_tr)
        preds = logreg.predict(Xb_te)
        acc_s[b] = accuracy_score(Yb_te, preds)
        # Topic classifier
        Xb, Yb = resample(embeddings, Yt, n_samples=n_samples)
        Xb_tr, Xb_te, Yb_tr, Yb_te = train_test_split(Xb, Yb, test_size=test_size)
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(Xb_tr, Yb_tr)
        preds = logreg.predict(Xb_te)
        acc_t[b] = accuracy_score(Yb_te, preds)
    lower_idx = int(np.floor(B * (1 - CL) / 2))
    upper_idx = min(int(np.ceil(B * (1 + CL) / 2)), n_samples - 1)
    acc_s.sort()
    mean_s = acc_s.mean()
    lower_s = acc_s[lower_idx]
    upper_s = acc_s[upper_idx]
    acc_t.sort()
    mean_t = acc_t.mean()
    lower_t = acc_t[lower_idx]
    upper_t = acc_t[upper_idx]
    return (mean_s, lower_s, upper_s), (mean_t, lower_t, upper_t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill text with an LLM")
    parser.add_argument(
        "--embedding_method",
        choices=["tfidf", "bert"],
        help="Type of method to use for embedding the text (ignored if embeddings are given)",
    )
    parser.add_argument(
        "--embedding_in",
        type=Path,
        help="Path to folder with pre-trained embeddings",
    )
    parser.add_argument(
        "--embedding_out",
        type=Path,
        help="Path to folder to which to save generated embeddings",
    )
    parser.add_argument(
        "--original_reviews",
        type=Path,
        help="Path to the Amazon reviews BEFORE distillation",
    )
    parser.add_argument(
        "--distilled_reviews",
        type=Path,
        help="Path to the Amazon reviews AFTER distillation",
    )
    parser.add_argument(
        "--out_file",
        type=Path,
        help="Path to save classification output to",
    )
    parser.add_argument(
        "--num_reviews",
        type=int,
        help="The number of reviews to use (default: all)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        help="Fraction of samples in test set",
    )
    parser.add_argument(
        "--num_bootstrap",
        type=int,
        help="Number of bootstrap samples",
    )
    parser.add_argument(
        "--confidence_level",
        type=float,
        help="Confidence level for the bootstrap interval",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        help="The maximum number of features to use for the Tfidf embedding",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="The batch size to use for the pipeline (bert)",
    )
    args = parser.parse_args()

    WANDB_PROJECT = "llm-distillation"
    with open(".wandb_api_key", "r") as file:
        WANDB_API_KEY = file.read().strip()
    wandb.login(key=WANDB_API_KEY)

    reviews = load_reviews_and_log(args.original_reviews, args.distilled_reviews)

    # Vectorize the texts
    if args.embedding_in:
        embedding_orig = torch.load(args.embedding_in / "embedding_orig.pt")
        embedding_dist = torch.load(args.embedding_in / "embedding_dist.pt")
        num_reviews = min(len(embedding_orig), len(embedding_dist), args.num_reviews)
        embedding_orig = embedding_orig[:num_reviews]
        embedding_dist = embedding_dist[:num_reviews]
    else:
        if args.embedding_method == "tfidf":
            embedding_orig = embed_tfidf(reviews["original"], args.max_features)
            embedding_dist = embed_tfidf(reviews["distilled"], args.max_features)
        elif args.embedding_method == "bert":
            embedding_orig = embed_bert(reviews["original"], args.batch_size)
            embedding_dist = embed_bert(reviews["distilled"], args.batch_size)
        else:
            raise ValueError("Invalid embedding method")

    embedding_orig = embedding_orig.detach().cpu()
    embedding_dist = embedding_dist.detach().cpu()

    if args.embedding_out:
        torch.save(embedding_orig, args.embedding_out / "embedding_orig.pt")
        torch.save(embedding_dist, args.embedding_out / "embedding_dist.pt")

    labels = reviews[["sentiment", "topic"]]

    # Classify before distillation
    acc_s, acc_t = classify(
        embedding_orig,
        labels,
        args.test_size,
        args.num_bootstrap,
        args.confidence_level,
    )
    results = "\n".join(
        [
            "Before distillation:",
            f"\tSentiment Classifier Accuracy: {acc_s[0]:.3f} ({acc_s[1]:.3f}, {acc_s[2]:.3f})",
            f"\tTopic Classifier Accuracy: {acc_t[0]:.3f} ({acc_t[1]:.3f}, {acc_t[2]:.3f})",
        ]
    )
    if args.out_file:
        with open(args.out_file, "w") as out:
            out.write(results)
    else:
        print(results)

    # Classify after distillation
    acc_s, acc_t = classify(
        embedding_dist,
        labels,
        args.test_size,
        args.num_bootstrap,
        args.confidence_level,
    )
    results = "\n".join(
        [
            "After distillation:",
            f"\tSentiment Classifier Accuracy: {acc_s[0]:.3f} ({acc_s[1]:.3f}, {acc_s[2]:.3f})",
            f"\tTopic Classifier Accuracy: {acc_t[0]:.3f} ({acc_t[1]:.3f}, {acc_t[2]:.3f})",
        ]
    )
    if args.out_file:
        with open(args.out_file, "a") as out:
            out.write(results)
