"""
Part 2 - Bayes' Theorem on IMDb movie reviews.
Basic Python only: csv and re from the standard library, no ML libraries.
We compute P(Positive | keyword) for each keyword.
"""

import csv
import re

csv.field_size_limit(10_000_000)   # some reviews are very long

HTML_TAG = re.compile(r"<[^>]+>")   # reviews contain tags like <br />
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")   # letters, keeps "don't" as one word

POSITIVE_LABELS = {"positive", "pos", "1"}
NEGATIVE_LABELS = {"negative", "neg", "0"}


# ---------------------------------------------------------------- loading
def tokenize(text):
    """Lowercase, remove HTML tags and punctuation, return the SET of words.
    A set means a word counts once per review, however often it repeats."""
    text = HTML_TAG.sub(" ", text.lower())
    return set(WORD.findall(text))


def _find_column(fieldnames, candidates):
    lookup = {name.strip().lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    raise KeyError(f"None of {candidates} found. Columns are: {fieldnames}")


def load_reviews(csv_path):
    """Read the IMDb CSV and return a list of (set_of_words, is_positive).
    Rows with an unknown label or empty text are skipped and counted."""
    reviews, skipped = [], 0
    with open(csv_path, encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        text_col = _find_column(reader.fieldnames, ["review", "text"])
        label_col = _find_column(reader.fieldnames, ["sentiment", "label"])
        for row in reader:
            text = (row.get(text_col) or "").strip()
            label = (row.get(label_col) or "").strip().lower()
            if not text or label not in POSITIVE_LABELS | NEGATIVE_LABELS:
                skipped += 1
                continue
            reviews.append((tokenize(text), label in POSITIVE_LABELS))
    if not reviews:
        raise ValueError("No usable reviews found in the file.")
    print(f"Loaded {len(reviews)} reviews ({skipped} rows skipped).")
    return reviews


# ---------------------------------------------------------------- Bayes
def bayes_posterior(prior, likelihood, marginal):
    """Bayes' Theorem: P(Pos | kw) = P(kw | Pos) * P(Pos) / P(kw).
    Returns None when the keyword never appears (P(kw) = 0, undefined)."""
    if marginal == 0:
        return None
    return likelihood * prior / marginal


def keyword_probabilities(reviews, keyword):
    """The four required probabilities for one keyword."""
    keyword = keyword.lower()
    total = len(reviews)
    positives = sum(1 for _, is_pos in reviews if is_pos)
    with_keyword = sum(1 for words, _ in reviews if keyword in words)
    positive_with_keyword = sum(1 for words, is_pos in reviews if is_pos and keyword in words)

    prior = positives / total
    likelihood = positive_with_keyword / positives if positives else 0.0
    marginal = with_keyword / total
    return {
        "keyword": keyword,
        "reviews_with_keyword": with_keyword,
        "prior": prior,
        "likelihood": likelihood,
        "marginal": marginal,
        "posterior": bayes_posterior(prior, likelihood, marginal),
    }


def direct_check(reviews, keyword):
    """Same answer counted directly: positive reviews with kw / all reviews with kw.
    Used only to prove our Bayes implementation is correct."""
    keyword = keyword.lower()
    with_kw = [is_pos for words, is_pos in reviews if keyword in words]
    return sum(with_kw) / len(with_kw) if with_kw else None


# ---------------------------------------------------------------- reporting
def markdown_table(result):
    """Small markdown table with the four probabilities for one keyword."""
    if result["posterior"] is None:
        return (f"**{result['keyword']}**: not found in any review, so "
                f"P(keyword) = 0 and the posterior is undefined.")
    return (
        f"**Keyword: {result['keyword']}** (in {result['reviews_with_keyword']} reviews)\n\n"
        "| Probability | Value |\n|---|---|\n"
        f"| Prior P(Positive) | {result['prior']:.4f} |\n"
        f"| Likelihood P({result['keyword']} \\| Positive) | {result['likelihood']:.4f} |\n"
        f"| Marginal P({result['keyword']}) | {result['marginal']:.4f} |\n"
        f"| Posterior P(Positive \\| {result['keyword']}) | {result['posterior']:.4f} |\n"
    )


def interpret(result):
    """One plain-English sentence comparing the posterior to the prior."""
    if result["posterior"] is None:
        return f"'{result['keyword']}' never appears, so it tells us nothing."
    post, prior = result["posterior"], result["prior"]
    if post > prior:
        direction = "more likely to be positive"
    elif post < prior:
        direction = "less likely to be positive (leans negative)"
    else:
        direction = "no more or less likely to be positive"
    return (f"A review containing '{result['keyword']}' is {post:.1%} likely to be positive, "
            f"versus {prior:.1%} for a random review, so it is {direction}.")
