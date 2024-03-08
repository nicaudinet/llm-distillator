import numpy as np
import pandas as pd
import torch
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


def embed(data: list[str]):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = 8
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
    num_reviews = 20  # Set to None to use all reviews
    test_size = 0.2
    num_bootstrap = 20
    CL = 0.95

    # Load the distilled Amazon review data
    with open("data/amazon_reviews/distilled/few-shot.txt", "r") as f:
        distilled = f.readlines()

    if num_reviews == None:
        num_reviews = len(distilled)
    distilled = distilled[:num_reviews]

    # Load the original Amazon review data
    labels = []
    original = []
    with open("data/amazon_reviews/original.txt", "r") as f:
        lines = f.readlines()
    for line in lines[:num_reviews]:
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
    embedding_orig = embed(original)
    embedding_dist = embed(distilled)

    # Classify before distillation
    acc_s, acc_t = classify(embedding_orig, labels, test_size, num_bootstrap, CL)
    with open("/cephyr/users/audinet/Alvis/llm-distillator/results/few-shot/classify_bert.txt", "w") as out:
        out.writelines([
            "Before distillation:",
            f"\tSentiment Classifier Accuracy: {acc_s[0]:.3f} ({acc_s[1]:.3f}, {acc_s[2]:.3f})",
            f"\tTopic Classifier Accuracy: {acc_t[0]:.3f} ({acc_t[1]:.3f}, {acc_t[2]:.3f})",
        ])

    # Classify after distillation
    acc_s, acc_t = classify(embedding_dist, labels, test_size, num_bootstrap, CL)
    with open("/cephyr/users/audinet/Alvis/llm-distillator/results/few-shot/classify_bert.txt", "w") as out:
        out.writelines([
            "After distillation:",
            f"\tSentiment Classifier Accuracy: {acc_s[0]:.3f} ({acc_s[1]:.3f}, {acc_s[2]:.3f})",
            f"\tTopic Classifier Accuracy: {acc_t[0]:.3f} ({acc_t[1]:.3f}, {acc_t[2]:.3f})",
            ])
