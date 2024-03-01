from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

sentiment_to_int = {"pos": 1, "neg": 0}
topic_to_int = {
    "books": 0,
    "dvd": 1,
    "music": 2,
    "health": 3,
    "software": 4,
    "camera": 5,
}


def vectorize(data: pd.DataFrame, max_vocab_size: int):
    vectorizer = TfidfVectorizer(max_features=max_vocab_size)
    tfidf_matrix = vectorizer.fit_transform(data["text"]).toarray()
    return pd.DataFrame(tfidf_matrix, columns=vectorizer.get_feature_names_out())


def classify(embeddings, labels, test_size, B, CL):
    n_samples = len(labels)
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
    upper_idx = int(np.ceil(B * (1 + CL) / 2))
    acc_s.sort()
    mean_s = acc_s.mean()
    lower_s = acc_s[lower_idx]
    upper_s = acc_s[upper_idx]
    acc_t.sort()
    mean_t = acc_t.mean()
    lower_t = acc_t[lower_idx]
    upper_t = acc_t[upper_idx]
    return (mean_s, lower_s, upper_s), (mean_t, lower_t, upper_t)


# Load the distilled Amazon review data
with open("data/amazon_reviews/distilled.txt", "r") as f:
    distilled_lines = f.readlines()
num_reviews = len(distilled_lines)

# Load the original Amazon review data
original = []
with open("data/amazon_reviews/original.txt", "r") as f:
    lines = f.readlines()
for line in lines[:num_reviews]:
    split_line = line.split(" ")
    original.append(
        {
            "topic": split_line[0],
            "sentiment": split_line[1],
            "review_id": split_line[2],
            "text": " ".join(split_line[3:]),
        }
    )
original = pd.DataFrame(original)

# Make distilled dataframe
distilled = original.copy()
distilled["text"] = distilled_lines

# Vectorize the texts
max_vocab_size = 256
embedding_orig = vectorize(original, 256)
embedding_dist = vectorize(distilled, 256)

# Classify and compare
test_size = 0.2
num_bootstrap = 500
CL = 0.95
acc_s, acc_t = classify(embedding_orig, original, test_size, num_bootstrap, CL)
print("Before distillation:")
print(
    f"\tSentiment Classifier Accuracy: {acc_s[0]:.3f} ({acc_s[1]:.3f}, {acc_s[2]:.3f})"
)
print(f"\tTopic Classifier Accuracy: {acc_t[0]:.3f} ({acc_t[1]:.3f}, {acc_t[2]:.3f})")

acc_s, acc_t = classify(embedding_dist, distilled, test_size, num_bootstrap, CL)
print("After distillation:")
print(
    f"\tSentiment Classifier Accuracy: {acc_s[0]:.3f} ({acc_s[1]:.3f}, {acc_s[2]:.3f})"
)
print(f"\tTopic Classifier Accuracy: {acc_t[0]:.3f} ({acc_t[1]:.3f}, {acc_t[2]:.3f})")
