import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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


def embed_tfidf(data: list[str], max_features: int) -> torch.Tensor:
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(data).toarray()
    return torch.from_numpy(tfidf_matrix)


def embed_bert(data: list[str], batch_size: int) -> torch.Tensor:
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
        logreg = LogisticRegression()
        logreg.fit(Xb_tr, Yb_tr)
        preds = logreg.predict(Xb_te)
        acc_s[b] = accuracy_score(Yb_te, preds)
        # Topic classifier
        Xb, Yb = resample(embeddings, Yt, n_samples=n_samples)
        Xb_tr, Xb_te, Yb_tr, Yb_te = train_test_split(Xb, Yb, test_size=test_size)
        logreg = LogisticRegression()
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
        help="Type of method to use for embedding the text",
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

    # Load the distilled Amazon review data
    with open(args.distilled_reviews, "r") as f:
        distilled = f.readlines()

    if args.num_reviews:
        num_reviews = len(distilled)
    distilled = distilled[: args.num_reviews]

    # Load the original Amazon review data
    labels = []
    original = []
    with open(args.original_reviews, "r") as f:
        lines = f.readlines()
    for line in lines[: args.num_reviews]:
        split_line = line.split(" ")
        labels.append(
            {
                "topic": split_line[0],
                "sentiment": split_line[1],
            }
        )
        original.append(" ".join(split_line[3:]))
    labels = pd.DataFrame(labels)

    # Vectorize the texts
    if args.embedding_method == "tfidf":
        embedding_orig = embed_tfidf(original, args.max_features)
        embedding_dist = embed_tfidf(distilled, args.max_features)
    elif args.embedding_method == "bert":
        embedding_orig = embed_bert(original, args.batch_size)
        embedding_dist = embed_bert(distilled, args.batch_size)
    else:
        raise ValueError("Invalid embedding method")

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
