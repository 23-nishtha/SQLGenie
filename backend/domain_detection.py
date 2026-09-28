"""Day 7: guesses an uploaded dataset's domain/theme from safe metadata.

Why this exists
----------------
The frontend picks a visual theme based on what KIND of data is active
(backend/dataset.py's `theme` field). For a built-in dataset a human chose
the theme by hand; for an upload, nobody did — so we guess, from metadata
that's safe to read without executing anything: the filename, the column
names, and a small sample of cell values already loaded during CSV
conversion (see backend/csv_upload.py).

No LLM is involved in this decision, on purpose — "never generate arbitrary
CSS/themes" (and more broadly, never let a model decide styling) is easiest
to guarantee by not calling a model for it at all. Instead this reuses the
same style of approach as backend/schema_retrieval.py: a small set of
hand-curated keyword lists, scored by simple overlap. Deterministic, cheap,
and when it guesses wrong for some CSV, it's a one-line fix to the keyword
list below, not a prompt to re-tune.

The result is always one of dataset.ThemeKey's fixed values — this module
picks a KEY, never a color, a font, or any styling content.
"""
import re

from backend.dataset import ThemeKey

# domain label -> (theme key, keywords). The theme key is always one of the
# six fixed values in backend/dataset.ThemeKey; the domain label is a freer
# human-readable string (matches backend/dataset.py's existing `domain`
# field, e.g. Olist's is "ecommerce").
_DOMAIN_PROFILES: dict[str, tuple[ThemeKey, frozenset[str]]] = {
    "sports": ("football", frozenset({
        "player", "players", "team", "teams", "match", "matches", "goal",
        "goals", "season", "club", "league", "coach", "stadium",
        "tournament", "fixture", "striker", "midfielder", "defender",
        "goalkeeper", "football", "soccer",
    })),
    "entertainment": ("entertainment", frozenset({
        "movie", "movies", "film", "films", "title", "director", "genre",
        "cast", "actor", "actress", "rating", "ratings", "boxoffice",
        "studio", "release", "runtime", "episode", "season", "imdb",
    })),
    "finance": ("finance", frozenset({
        "transaction", "transactions", "account", "accounts", "balance",
        "amount", "currency", "invoice", "invoices", "payment", "payments",
        "bank", "credit", "debit", "expense", "expenses", "budget",
        "revenue", "profit", "loan", "interest",
    })),
    "property": ("property", frozenset({
        "address", "bedroom", "bedrooms", "bathroom", "bathrooms", "price",
        "sqft", "squarefeet", "listing", "listings", "property",
        "properties", "rent", "lease", "realestate", "zipcode", "acreage",
    })),
    "ecommerce": ("commerce", frozenset({
        "order", "orders", "customer", "customers", "product", "products",
        "price", "cart", "seller", "sellers", "sku", "inventory",
        "checkout", "shipping", "purchase", "purchases",
    })),
}

# Below this score, the guess isn't confident enough to act on — better a
# known, neutral "default" theme than a wrong specific one.
_MIN_CONFIDENT_SCORE = 3

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def detect_domain_and_theme(
    filename: str,
    column_names: list[str],
    sample_values: list[str],
) -> tuple[str, ThemeKey]:
    """Guess (domain, theme) from safe metadata only.

    filename: the ORIGINAL uploaded filename, used only for its words —
        never as a path or identifier (see csv_upload.py for that rule).
    column_names: the CSV's header row.
    sample_values: a small number of cell values (already truncated by the
        caller — this function doesn't decide how many or which ones).

    Weighting: filename words count most (a file called "football.csv" is
    a strong signal), then column names, then sample values (the weakest
    signal — a "type" column full of "Comedy"/"Drama" only tips entertainment
    once combined with everything else).
    """
    filename_tokens = _tokens(filename.rsplit(".", 1)[0])
    column_tokens = [tok for col in column_names for tok in _tokens(col)]
    value_tokens = [tok for val in sample_values for tok in _tokens(str(val))]

    best_domain = "general"
    best_theme: ThemeKey = "default"
    best_score = 0

    for domain, (theme, keywords) in _DOMAIN_PROFILES.items():
        score = (
            4 * sum(1 for tok in filename_tokens if tok in keywords)
            + 2 * sum(1 for tok in column_tokens if tok in keywords)
            + 1 * sum(1 for tok in value_tokens if tok in keywords)
        )
        if score > best_score:
            best_domain, best_theme, best_score = domain, theme, score

    if best_score < _MIN_CONFIDENT_SCORE:
        return "general", "default"
    return best_domain, best_theme
