"""Code for the data exploration for our CSC311 Project"""

import re
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Initialization of csv related stuff
# ----------------------------------------------------------------------------
# This is our all our data stored in csv format
DATA_FILE = "csc311_project_training_202609.csv"

# Defining the columns for easier access later
# 1st Question, how easy it is to play
EASE_COL = ("How easy do you think it would be for someone with no prior experience "
            "to start playing this sport recreationally? Rate it on a scale from 0 to 10, where "
            "0 means extremely difficult and 10 means extremely easy.")

# 2nd, 3rd and 4th questions
# Grouped together because they all output a Likert value
LIKERT_COLS = ["Success in this sport depends heavily on strategy and planning.",
               "Players must be comfortable with a high risk of injury.",
               "Individual brilliance can outweigh teamwork in this sport.",
               ]
# Creating map for Likert answers to integer values
LIKERT_MAP = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly agree": 5,
    }
# Created a shorter version for representation
LIKERT_SHORT = ["Strategy\nmatters", "High injury\nrisk",
                "Individual >\nteamwork"]

# 5th Question
WHERE_COL = "Where might you play this sport?"

# 6th Question, and creating possible outputs list
SEASON_COL = "During which seasons would you be most likely to play this sport recreationally in Toronto?"
SEASONS = ["Winter", "Spring", "Summer", "Fall"]

# 7th Question
MUSIC_COL = "What types of music do you most associate with this sport?"

# 8th Question
FOOD_COL = ("What foods and drinks do you associate with watching this sport "
            "in the place or culture you know best?")

# 9th Question
COLOUR_COL = "What colour do you associate with this sport?"

# 10th Question
SONG_COL = "Which song do you associate with this sport?"

# 11th Question
FEELING_COL = "How do you feel when you watch this sport?"

# 12th Question
PRICE_COL = ("How much would you be willing to pay for a decent seat at a "
             "professional competition of this sport?")

# Columns that include text
TEXT_COLS = {"Food & drink": FOOD_COL, "Colour": COLOUR_COL,
             "Song": SONG_COL, "Feeling": FEELING_COL}

# Label
LABEL_COL = "label"

# List of Sports
SPORTS = ["Baseball", "Basketball", "Soccer"]


# ----------------------------------------------------------------------------
# Splitting our data
# ----------------------------------------------------------------------------
def split_by_student(df, n_test=44, seed=311):
    """Splits the unique student ids into train/test groups. All three rows
    from the same student end up in the same split, so the model is never
    tested on a student it has already seen during training."""
    rng = np.random.default_rng(seed)
    test_ids = rng.choice(df["id"].unique(), n_test, replace=False)
    is_test = df["id"].isin(test_ids)  # Series of bools, one per row

    # The ~ means not, so we return ids of training data and ids of test data
    return ~is_test, is_test


# ----------------------------------------------------------------------------
# Helper functions to parse through our data
# ----------------------------------------------------------------------------
# Defining regex patterns for understanding our price data
NUMBER = r"\d+(?:\.\d+)?"
RANGE_PATTERN = re.compile(r"\d\s*(-|to)\s*\$?\d")

# Defining words that mean a refusal
REFUSAL_WORDS = ("zero", "none", "nothing", "free", "not willing",
                 "not really willing", "wouldn't", "would not", "not pay",
                 "never", "no ")

# Stopwords are words that are useless since they are just for grammatical correctness
STOPWORDS = {"a", "an", "and", "the", "of", "by", "to", "i", "it", "is", "in",
             "for", "with", "or", "my", "on", "at", "that", "this", "be",
             "are", "as", "so"}

MIN_TOKEN_COUNT = 3  # a token must appear in at least this many training rows


def parse_price(raw):
    """
    Turns a free-text ticket price into (kind, value), where kind is one of
    "missing", "single", "range", "refusal" or "unparsed".

    We will follow the following conventions:
    - two numbers joined by a dash or "to": use their midpoint
    - one number: that number
    - a refusal such as "Zero" or "I would not pay": 0
    - anything else ("Hundreds of dollars"): NaN
    """
    # If missing, return NaN: Not a number
    if pd.isna(raw):
        return "missing", np.nan

    # Otherwise, strip text
    text = str(raw).replace("\xa0", " ").replace(",", "").lower().strip()

    # Find all occurences of a numeric value in the text
    numbers = re.findall(NUMBER, text)

    # If a range of numbers found, return midpoint
    if len(numbers) >= 2 and RANGE_PATTERN.search(text):
        return "range", (float(numbers[0]) + float(numbers[1])) / 2

    # If single number found, return it
    if numbers:
        return "single", float(numbers[0])

    # If a refusal word found, return 0
    if any(word in text + " " for word in REFUSAL_WORDS):
        return "refusal", 0.0

    # If some other type of text, return this tuple
    return "unparsed", np.nan


def agree_share(series):
    """Share of non-missing answers that are Agree / Strongly agree."""
    scores = series.map(LIKERT_MAP).dropna()
    return (scores >= 4).mean()


def normalize_text(raw):
    """Lowercase, replace non-breaking spaces, strip punctuation."""
    text = str(raw).replace("\xa0", " ").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def tokenize(raw):
    """Returns the list of tokens of one answer (stop words removed)."""
    return [tok for tok in normalize_text(raw).split()
            if tok not in STOPWORDS]


def build_vocabulary(series):
    """
    Returns a Counter mapping each token to the number of rows it appears in
    (a token counted once per row).
    """
    counts = Counter()
    for raw in series.dropna():
        counts.update(set(tokenize(raw)))
    return counts


# ----------------------------------------------------------------------------
# Calculations regarding our data
# ----------------------------------------------------------------------------
def report_overview(df, train, test):
    """Function that returns the basic overview: sizes of the full data, the train and test splits, etcetera."""
    counts = df["id"].value_counts()
    print("Basic overview:")
    print(f"rows: {len(df)}, students: {df['id'].nunique()}")
    print(f"rows per student: {sorted(int(n) for n in counts.unique())}")
    print(f"class sizes (entire dataset): {df[LABEL_COL].value_counts().to_dict()}")
    print(f"id range: {df['id'].min()} to {df['id'].max()}")
    print(f"test: {test['id'].nunique()} students, {len(test)} rows")
    print(f"train: {train['id'].nunique()} students, {len(train)} rows")
    print(f"class sizes (training data): {train[LABEL_COL].value_counts().to_dict()}")
    print(f"test ids: {sorted(int(i) for i in test['id'].unique())}")
    overlap = set(train["id"]) & set(test["id"])
    print(f"students in both splits: {len(overlap)}")


def report_missing(train):
    """Function that reports missing values per feature(question)."""
    print("\n\n\nMissing values per feature (training data)")
    features = {"Ease of starting": EASE_COL,
                "Strategy": LIKERT_COLS[0],
                "Injury risk": LIKERT_COLS[1],
                "Individual > team": LIKERT_COLS[2],
                "Where played": WHERE_COL,
                "Season": SEASON_COL,
                "Music": MUSIC_COL,
                "Food & drink": FOOD_COL,
                "Colour": COLOUR_COL,
                "Song": SONG_COL,
                "Feeling": FEELING_COL,
                "Price": PRICE_COL}
    for name, col in features.items():
        missing = int(train[col].isna().sum())
        print(f"{name:<20}{missing:>4}  ({missing / len(train):.1%})")


def report_distributions(train):
    """Distributions and patterns in our training data."""
    # How easy it is to start the sport
    print("\n\n\nDistributions and patterns in our training data:")
    print("(a) Ease of starting each sport on a scale of 10")
    print(train.groupby(LABEL_COL)[EASE_COL].agg(["mean", "median"]).round(2))

    # How many people agree with some statement about a sport
    print("\n(b) Share of Agreement (Agree/Strongly agree), by sport")
    for col, short in zip(LIKERT_COLS, LIKERT_SHORT):
        shares = train.groupby(LABEL_COL)[col].apply(agree_share)
        text = ", ".join(f"{s} {shares[s]:.0%}" for s in SPORTS)
        print(f"{short.replace(chr(10), ' '):<22}{text}")


def report_price(train):
    """Price information about each sport in our training data."""
    # Prints the missing, no. of distinct raw strings, and the kind of each price data
    # Kinds: single number, range midpoint, refusal, unparsed(Not a number)
    print("\n\n\nPrice data:")
    parsed = train[PRICE_COL].map(parse_price)
    kinds = Counter(kind for kind, _ in parsed)
    print(f"Types of price inputs:")
    print(f"single number: {kinds['single']}, range midpoint: "
          f"{kinds['range']}, refusal(0): {kinds['refusal']}, "
          f"unparsed(NaN): {kinds['unparsed']}")

    # Prints the median, 95th percentile, maximum, and values over $1000
    price = parsed.map(lambda pair: pair[1]).astype(float)
    print("\nPrice Distribution (dollars):")
    print(f"median: {price.median():.0f}, 95th percentile: "
          f"{price.quantile(0.95):.0f}, maximum: {price.max():.0f}, "
          f"values above $1000: {int((price > 1000).sum())}")

    print("\nMedian price by sport:")
    print(price.groupby(train[LABEL_COL]).median().to_string())
    skew = price.skew()
    skew_log = np.log1p(price).skew()
    print(f"skewness of raw data: {skew:.2f} -> after log(1+p) transformation: {skew_log:.2f}")


def report_text(train):
    """Text fields: frequent tokens, inconsistencies, vocabulary sizes."""
    # Prints the colour data
    # Drops blank colour answers.
    # Runs each answer through tokenize (lowercasing, stripping punctuation, splitting into words)
    # Then rejoins the tokens with spaces. "Dark Blue!" and "dark blue" become the same string.
    # Prints how many distinct answers there were before and after.
    # A big drop means lots of the variety was just formatting noise.
    print("\n\n\nData from Text fields")
    raw_colours = train[COLOUR_COL].dropna()
    normalized = raw_colours.map(lambda s: " ".join(tokenize(s)))
    print(f"colour: {raw_colours.nunique()} distinct raw answers -> "
          f"{normalized.nunique()} after normalization\n")

    # Loop over each text column
    # Counter of token =  number of rows containing it
    # We keep only Tokens that appear in at least 3 rows as features
    # Compute Fraction of the vocabulary that appears in exactly one row
    # A high value means most words are one-offs, including which would only cause overfitting
    # Compute how many total tokens we have in our binary bag of words
    total = 0
    for name, col in TEXT_COLS.items():
        vocab = build_vocabulary(train[col])
        kept = [tok for tok, n in vocab.items() if n >= MIN_TOKEN_COUNT]
        once = sum(1 for n in vocab.values() if n == 1) / len(vocab)
        total += len(kept)
        print(f"{name:<14} vocabulary {len(vocab):>4}, tokens in "
              f">= {MIN_TOKEN_COUNT} rows: {len(kept):>3}, "
              f"occur once: {once:.0%}")
    print(f"total bag-of-words features: {total}")

    # Prints most common tokens per sport (4)
    print("\nMost frequent tokens per sport (number of rows):")
    for name, col in TEXT_COLS.items():
        for sport in SPORTS:
            rows = train.loc[train[LABEL_COL] == sport, col]
            top = build_vocabulary(rows).most_common(4)
            text = ", ".join(f"{tok} ({n})" for tok, n in top)
            print(f"  {name:<13}{sport:<11}{text}")

    # Prints most common song per sport
    print("\nMost frequent whole song answers per sport:")
    for sport in SPORTS:
        rows = train.loc[train[LABEL_COL] == sport, SONG_COL].dropna()
        top = rows.map(normalize_text).value_counts().head(2)
        print(f"  {sport:<11}{top.to_dict()}")

    # Prints how common it is for a sport to be exciting or boring
    print("\nFeeling: share of rows mentioning a word, by sport")
    for word in ["excited", "bored"]:
        picked = train[FEELING_COL].map(
            lambda raw, w=word: w in tokenize(raw) if pd.notna(raw) else False)
        shares = picked.groupby(train[LABEL_COL]).mean()
        text = ", ".join(f"{s} {shares[s]:.0%}" for s in SPORTS)
        print(f"  {word:<9}{text}")


# Answers that mean "no answer" even though the cell is not empty
PLACEHOLDERS = {"none", "idk", "no", "-", "n/a", "na"}


def report_redundancy(train):
    """Near-constant multi-select options and correlations between the Likert/ease features."""
    # Share of rows choosing each option of a multi-select question
    # An option almost everyone (or no one) picks would carry little information
    print("\n\n\nShare of rows choosing each multi-select option:")
    for name, col in {"Where": WHERE_COL, "Season": SEASON_COL,
                      "Music": MUSIC_COL}.items():
        options = train[col].dropna().str.split(",").explode().str.strip()
        shares = options.value_counts() / len(train)
        text = ", ".join(f"{opt} {share:.0%}" for opt, share in shares.items())
        print(f"  {name:<8}{text}")

    # Spearman correlation suits ordinal answers like Likert values
    print("\nSpearman correlation between Likert and ease features:")
    ordinal = train[LIKERT_COLS].apply(lambda s: s.map(LIKERT_MAP))
    ordinal.columns = [short.replace("\n", " ") for short in LIKERT_SHORT]
    ordinal["Ease"] = train[EASE_COL]
    print(ordinal.corr(method="spearman").round(2).to_string())


def report_missing_by_sport(train):
    """Checks whether missing answers depend on the label."""
    # Placeholder answers such as "idk" are counted as missing too
    print("\n\n\nSong placeholder answers (none, idk, ...): ", end="")
    song = train[SONG_COL].astype(str).str.strip().str.lower()
    print(int((train[SONG_COL].notna() & song.isin(PLACEHOLDERS)).sum()))

    # If the missing rate differs a lot between sports, a missing answer is itself a signal
    print("Missing rate by sport:")
    for name, col in {"Song": SONG_COL, "Food & drink": FOOD_COL}.items():
        rates = train[col].isna().groupby(train[LABEL_COL]).mean()
        text = ", ".join(f"{s} {rates[s]:.1%}" for s in SPORTS)
        print(f"  {name:<14}{text}")


# ---------------------------------------------------------------------------
# Create the graphs
# ---------------------------------------------------------------------------
# Define colours for our figure
COLOURS = {"Baseball": "#2878d6", "Basketball": "#ee6c37", "Soccer": "#1fb07c"}

# How we will save our graph
FIGURE_FILE = "figure.png"     # Where we will save our graphs


# Create the boxplot
# All our figures will be a part of this
def style_boxplot(box, colours):
    """Colours the boxes of a matplotlib boxplot and makes medians black."""
    for patch, colour in zip(box["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_edgecolor(colour)
    for median in box["medians"]:
        median.set_color("black")


def plot_figure(train):
    """Draws the three panels of the figure and saves them to FIGURE_FILE."""
    fig = plt.figure(figsize=(11.0, 8.0))
    gs = fig.add_gridspec(2, 4)
    colours = [COLOURS[s] for s in SPORTS]
    x_pos = np.arange(len(SPORTS)) + 1

    # Our first graph (a) will compare the ease ratings of the sports
    # Sport on x-axis, Ease on y-axis
    # Will use 3 horizontal box and whisker diagrams, 1 for each sport
    ax_a = fig.add_subplot(gs[0, 0:2])
    data = [train.loc[train[LABEL_COL] == s, EASE_COL].dropna()
            for s in SPORTS]
    style_boxplot(ax_a.boxplot(data, patch_artist=True, widths=0.6), colours)
    ax_a.scatter(x_pos, [d.mean() for d in data], marker="D", color="black", zorder = 3)
    ax_a.set_xticks(x_pos, SPORTS)
    ax_a.set_ylabel("Ease rating (0-10)")
    ax_a.set_title("(a) Ease of starting (diamond = mean)", loc="left")

    # Our second graph (b) will compare each of the likert statements for each sport
    # We have 3 likert statements and 3 sports, so will plot a total of 9 items
    # Likert statements on x-axis, Agreement share on y-axis
    # Will use 3 mini bar graphs for this
    # Each bar graph will use 3 colours to represent the 3 sports
    ax_b = fig.add_subplot(gs[0, 2:4])
    width = 0.25
    for i, sport in enumerate(SPORTS):
        shares = [agree_share(train.loc[train[LABEL_COL] == sport, col])
                  for col in LIKERT_COLS]
        ax_b.bar(np.arange(3) + (i - 1) * width, shares, width, color=COLOURS[sport], label=sport)
        ax_b.set_xticks(range(3), LIKERT_SHORT)
        ax_b.set_ylim(0, 1)
        ax_b.set_ylabel("Share Agree / Strongly agree")
        ax_b.legend(ncol=3, loc="upper center", frameon=False)
        ax_b.set_title("(b) Likert statements", loc="left")

    # Our third graph and final graph (c) will compare the prices for the sports
    # Sport on x-axis, Price on y-axis (use log scale on y-axis)
    # Will use 3 horizontal box and whisker diagrams, 1 for each sport
    ax_c = fig.add_subplot(gs[1, 1:3])
    price = train[PRICE_COL].map(lambda raw: parse_price(raw)[1])
    price = np.log1p(price.astype(float))
    data = [price[train[LABEL_COL] == s].dropna() for s in SPORTS]
    style_boxplot(ax_c.boxplot(data, patch_artist=True, widths=0.6), colours)
    ticks = [0, 10, 100, 1000, 10000]
    ax_c.set_yticks(np.log1p(ticks), ["$0", "$10", "$100", "$1k", "$10k"])
    ax_c.set_xticks(x_pos, SPORTS)
    ax_c.set_ylabel("Willing to pay (log scale)")
    ax_c.set_title("(c) Parsed ticket price", loc="left")

    for ax in (ax_a, ax_b, ax_c):
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURE_FILE, dpi=200)
    plt.close(fig)
    print(f"\nSaved {FIGURE_FILE}")


# ----------------------------------------------------------------------------
# Running the script
# ----------------------------------------------------------------------------
def main():
    """The actual code"""
    # Load Data
    df = pd.read_csv(DATA_FILE)

    # Split the students into test and training sets
    train_ids, test_ids = split_by_student(df[["id"]])
    training_data = df[train_ids].reset_index(drop=True)
    testing_data = df[test_ids].reset_index(drop=True)

    # Reporting our findings of the data
    report_overview(df, training_data, testing_data)
    report_missing(training_data)
    report_distributions(training_data)
    report_price(training_data)
    report_text(training_data)
    report_redundancy(training_data)
    report_missing_by_sport(training_data)

    # Plot our data
    plot_figure(training_data)


if __name__ == "__main__":
    main()
