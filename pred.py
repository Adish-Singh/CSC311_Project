
"""
This Python file is our `pred.py` script. It contains a function `predict_all`
that takes in the name of a CSV file, and returns a list of predictions
("Baseball", "Basketball" or "Soccer", one per row).

The test CSV file has the same columns as the training file
"csc311_project_training_202609.csv", except that it may not have
the `label` column (we never read it when predicting).

Here's an example of how the script is used in the test file:

    from pred import predict_all
    predict_all("example_test_set.csv")

How it works: the preprocessing (column parsing, Likert / price / text
features) is implemented below with only numpy, pandas and the standard
library. Everything fitted during training (medians, means, vocabularies,
model weights) is read from `model.json`, which must sit in the same folder
as this file.
"""
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# Columns (exact headers of the course CSV, copied from the data exploration)
# ----------------------------------------------------------------------------
ID_COL = "id"
LABEL_COL = "label"

EASE_COL = ("How easy do you think it would be for someone with no prior experience "
            "to start playing this sport recreationally? Rate it on a scale from 0 to 10, where "
            "0 means extremely difficult and 10 means extremely easy.")
LIKERT_COLS = ["Success in this sport depends heavily on strategy and planning.",
               "Players must be comfortable with a high risk of injury.",
               "Individual brilliance can outweigh teamwork in this sport."]
WHERE_COL = "Where might you play this sport?"
SEASON_COL = "During which seasons would you be most likely to play this sport recreationally in Toronto?"
MUSIC_COL = "What types of music do you most associate with this sport?"
FOOD_COL = ("What foods and drinks do you associate with watching this sport "
            "in the place or culture you know best?")
COLOUR_COL = "What colour do you associate with this sport?"
SONG_COL = "Which song do you associate with this sport?"
FEELING_COL = "How do you feel when you watch this sport?"
PRICE_COL = ("How much would you be willing to pay for a decent seat at a "
             "professional competition of this sport?")

MULTI_COLS = {"where": WHERE_COL, "season": SEASON_COL, "music": MUSIC_COL}
TEXT_COLS = {"food": FOOD_COL, "colour": COLOUR_COL, "song": SONG_COL, "feeling": FEELING_COL}
REQUIRED_COLS = [EASE_COL] + LIKERT_COLS + list(MULTI_COLS.values()) + \
                list(TEXT_COLS.values()) + [PRICE_COL]
NUMERIC_NAMES = ["ease", "strategy", "injury", "individual", "log_price"]

LIKERT_MAP = {"strongly disagree": 1, "disagree": 2, "neutral": 3, "agree": 4, "strongly agree": 5}

# ----------------------------------------------------------------------------
# Parsing helpers (price parser, normaliser, stop-words: identical to the exploration)
# ----------------------------------------------------------------------------
NUMBER = r"\d+(?:\.\d+)?"
RANGE_PATTERN = re.compile(r"\d\s*\$?\s*(?:-|\u2013|\u2014|to)\s*\$?\s*\d")
THOUSANDS_SPACE = re.compile(r"(\d) (\d{3})\b")
REFUSAL_WORDS = ("zero", "none", "nothing", "free", "not willing",
                 "not really willing", "wouldn't", "wouldnt", "would not",
                 "not pay", "never", "no ", "unwilling", "pay me")
STOPWORDS = {"a", "an", "and", "the", "of", "by", "to", "i", "it", "is", "in",
             "for", "with", "or", "my", "on", "at", "that", "this", "be",
             "are", "as", "so"}
PLACEHOLDERS = {"none", "idk", "no", "-", "n/a", "na"}      # "no answer" typed into the song box


def parse_price(raw):
    """Free-text ticket price -> (kind, value); kind in missing/single/range/refusal/unparsed."""
    if pd.isna(raw):
        return "missing", np.nan
    text = str(raw).replace("\xa0", " ").replace(",", "").lower().strip()
    text = THOUSANDS_SPACE.sub(r"\1\2", text)
    if "scale" in text:                                  # answered on a 1-10 scale, not in dollars
        return "unparsed", np.nan
    numbers = re.findall(NUMBER, text)
    if len(numbers) >= 2 and RANGE_PATTERN.search(text):
        return "range", (float(numbers[0]) + float(numbers[1])) / 2
    if numbers:
        return "single", float(numbers[0])
    if any(word in text + " " for word in REFUSAL_WORDS):
        return "refusal", 0.0
    return "unparsed", np.nan


def normalize_text(raw):
    """Lowercase, replace non-breaking spaces, strip punctuation."""
    text = str(raw).replace("\xa0", " ").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def tokenize(raw):
    """Tokens of one answer, stop-words removed."""
    return [tok for tok in normalize_text(raw).split() if tok not in STOPWORDS]


def is_placeholder(raw):
    """True for non-empty cells that really mean 'no answer' (idk, none, -, ...)."""
    return pd.notna(raw) and str(raw).strip().lower() in PLACEHOLDERS


def answer_tokens(raw, field):
    """Tokens used for the bag of words. Blank answers, and song placeholders,
    give an EMPTY list (an all-zero row) instead of being imputed."""
    if pd.isna(raw) or (field == "song" and is_placeholder(raw)):
        return []
    return tokenize(raw)


def split_multi(raw):
    """'Summer, Fall' -> ['summer', 'fall'] (blank -> [])."""
    if pd.isna(raw):
        return []
    return [part.strip().lower() for part in str(raw).split(",") if part.strip()]


def likert_value(raw):
    """'Strongly agree' -> 5 ... 'Strongly disagree' -> 1; blank/unknown -> NaN."""
    if pd.isna(raw):
        return np.nan
    return float(LIKERT_MAP.get(str(raw).strip().lower(), np.nan))


# ----------------------------------------------------------------------------
# Preprocessor: fit on training data only, then transform any dataframe
# ----------------------------------------------------------------------------
class Preprocessor:
    def __init__(self, min_count=3):
        self.min_count = min_count     # token must appear in >= this many training rows

    @staticmethod
    def _check_columns(df):
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise KeyError(f"CSV is missing expected columns: {missing}")

    @staticmethod
    def _numeric(df):
        """Returns (n x 5 raw numeric matrix with NaNs, price-unparseable flag)."""
        ease = pd.to_numeric(df[EASE_COL], errors="coerce").to_numpy(float)
        likert = [df[c].map(likert_value).to_numpy(float) for c in LIKERT_COLS]
        price = np.array([parse_price(raw)[1] for raw in df[PRICE_COL]], dtype=float)
        flag = np.isnan(price).astype(float)                 # blank or unparseable price
        num = np.column_stack([ease] + likert + [np.log1p(price)])
        return num, flag

    def fit(self, df):
        self._check_columns(df)
        num, _ = self._numeric(df)
        self.medians = np.nanmedian(num, axis=0)             # for imputation
        filled = np.where(np.isnan(num), self.medians, num)
        self.means = filled.mean(axis=0)                     # for standardisation
        self.stds = filled.std(axis=0)
        self.stds[self.stds == 0] = 1.0

        # One indicator per option seen in training
        self.multi = {k: sorted({o for raw in df[col] for o in split_multi(raw)})
                      for k, col in MULTI_COLS.items()}

        # Per-field vocabulary: tokens that occur in >= min_count training ROWS
        self.vocab = {}
        for k, col in TEXT_COLS.items():
            counts = Counter(tok for raw in df[col] for tok in set(answer_tokens(raw, k)))
            self.vocab[k] = sorted(t for t, n in counts.items() if n >= self.min_count)

        self.names = (NUMERIC_NAMES + ["price_unparsed"]
                      + [f"{k}={o}" for k in MULTI_COLS for o in self.multi[k]]
                      + [f"{k}:{t}" for k in TEXT_COLS for t in self.vocab[k]])
        return self

    def transform(self, df):
        self._check_columns(df)
        n = len(df)
        num, flag = self._numeric(df)
        num = np.where(np.isnan(num), self.medians, num)
        parts = [(num - self.means) / self.stds, flag[:, None]]

        for k, col in MULTI_COLS.items():
            index = {o: i for i, o in enumerate(self.multi[k])}
            block = np.zeros((n, len(index)))
            for r, raw in enumerate(df[col]):
                for o in split_multi(raw):
                    if o in index:
                        block[r, index[o]] = 1.0
            parts.append(block)

        for k, col in TEXT_COLS.items():
            index = {t: i for i, t in enumerate(self.vocab[k])}
            block = np.zeros((n, len(index)))
            for r, raw in enumerate(df[col]):
                for tok in answer_tokens(raw, k):
                    if tok in index:
                        block[r, index[tok]] = 1.0
            parts.append(block)

        return np.hstack(parts).astype(np.float64)

    # -- (de)serialisation (stored by train.py inside model.json)
    def get_state(self):
        return {"min_count": self.min_count, "medians": self.medians.tolist(),
                "means": self.means.tolist(), "stds": self.stds.tolist(),
                "multi": self.multi, "vocab": self.vocab, "names": self.names}

    @classmethod
    def from_state(cls, st):
        p = cls(st["min_count"])
        p.medians, p.means, p.stds = (np.array(st[k]) for k in ("medians", "means", "stds"))
        p.multi, p.vocab, p.names = st["multi"], st["vocab"], st["names"]
        return p


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def predict_proba(X, M):
    kind = str(M["kind"])
    if kind == "lr":
        return _softmax(X @ M["W"] + M["b"])
    if kind == "mlp":
        a, L = X, int(M["n_layers"])
        for i in range(L):
            z = a @ M[f"W{i}"] + M[f"b{i}"]
            a = np.maximum(z, 0) if i < L - 1 else z
        return _softmax(a)
    if kind == "rf":
        left, right, feat, thr, val = (M[k] for k in ("left", "right", "feat", "thr", "val"))
        X32 = X.astype(np.float32)           # sklearn compares in float32
        n = len(X)
        idx = np.tile(M["roots"][None, :], (n, 1))
        rows = np.arange(n)[:, None]
        for _ in range(int(M["max_depth"])):
            go_left = X32[rows, feat[idx]] <= thr[idx]
            idx = np.where(go_left, left[idx], right[idx])
        return val[idx].mean(axis=1)
    raise ValueError(kind)


_INT_KEYS = {"left", "right", "feat", "roots"}


def load_artifacts(path=None):
    """Reads model.json (default: next to this file, or a folder / file path)
    and returns (Preprocessor, model-parameter dict)."""
    path = HERE if path is None else path
    if os.path.isdir(path):
        path = os.path.join(path, "model.json")
    with open(path) as f:
        data = json.load(f)

    pre = Preprocessor.from_state(data["preproc"])

    M = {"kind": data["kind"], "classes": data["classes"]}
    for k, v in data["model"].items():
        if k in _INT_KEYS:
            M[k] = np.array(v, dtype=np.int32)
        elif k == "val":
            M[k] = np.array(v, dtype=np.float32)
        elif k in ("n_layers", "max_depth"):
            M[k] = int(v)
        else:
            M[k] = np.array(v, dtype=np.float64)
    return pre, M


def predict_all(filename, path=None):
    """Return a list of predicted labels (strings), one per row of the CSV."""
    pre, M = load_artifacts(path)
    df = pd.read_csv(filename)
    P = predict_proba(pre.transform(df), M)
    return [str(M["classes"][i]) for i in P.argmax(axis=1)]


if __name__ == "__main__":
    import sys
    print(predict_all(sys.argv[1])[:10])
