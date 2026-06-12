"""
steam_reviews.py
================
Fetch reviews for any Steam game using the Steam Store API.

USAGE
-----
  python steam_reviews.py <APP_ID> [OPTIONS]

EXAMPLES
  # Fetch 100 most-recent English reviews for Balatro (App ID 2379780)
  python steam_reviews.py 2379780

  # Fetch 500 reviews, all languages, sorted by helpfulness
  python steam_reviews.py 2379780 --num 500 --language all --filter all

  # Save to a custom CSV file
  python steam_reviews.py 2379780 --output my_reviews.csv

HOW TO FIND A GAME'S APP ID
  Open the game's Steam store page. The number in the URL is the App ID.
  Example: https://store.steampowered.com/app/2379780/Balatro/
                                                  ^^^^^^^^^^ App ID

REQUIREMENTS
  pip install requests
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Steam API helpers
# ---------------------------------------------------------------------------

REVIEW_URL = "https://store.steampowered.com/appreviews/{app_id}"
APP_DETAILS_URL = "https://store.steampowered.com/api/appdetails"

VALID_FILTERS   = ("recent", "updated", "all")
VALID_LANGUAGES = (
    "arabic", "bulgarian", "schinese", "tchinese", "czech", "danish",
    "dutch", "english", "finnish", "french", "german", "greek", "hungarian",
    "indonesian", "italian", "japanese", "koreana", "norwegian", "polish",
    "portuguese", "brazilian", "romanian", "russian", "spanish", "latam",
    "swedish", "thai", "turkish", "ukrainian", "vietnamese", "all",
)

_REVIEW_SESSION = requests.Session()


def get_app_name(app_id: int) -> str:
    """Return the game name from the Steam store, or a fallback string."""
    try:
        r = requests.get(APP_DETAILS_URL, params={"appids": app_id}, timeout=10)
        r.raise_for_status()
        data = r.json().get(str(app_id), {})
        if data.get("success"):
            return data["data"].get("name", f"App {app_id}")
    except Exception:
        pass
    return f"App {app_id}"


def fetch_reviews(
    app_id: int,
    num_reviews: int = 100,
    language: str = "english",
    review_filter: str = "recent",
    review_type: str = "all",
) -> list[dict]:
    """
    Page through the Steam review API until we have `num_reviews` reviews
    or there are no more pages.

    Parameters
    ----------
    app_id        : Steam App ID (integer)
    num_reviews   : Maximum reviews to retrieve (default 100)
    language      : Language code or "all" (default "english")
    review_filter : "recent" | "updated" | "all"  (default "recent")
    review_type   : "all" | "positive" | "negative"  (default "all")

    Returns
    -------
    List of review dicts with cleaned/flattened fields.
    """
    reviews: list[dict] = []
    seen_ids: set[str] = set()   # dedupe across pages — Steam's cursor can
                                 # re-serve reviews (always with filter="all")
    cursor = "*"          # Steam uses cursor-based pagination
    per_page = min(100, num_reviews)   # API max is 100 per request

    if review_filter == "all":
        print("[WARN] filter='all' is not cursor-stable and returns repeated "
              "reviews; use 'recent' or 'updated' to page through all reviews.",
              file=sys.stderr)

    print(f"\nFetching up to {num_reviews} reviews (language={language}, "
          f"filter={review_filter}, type={review_type}) …")

    while len(reviews) < num_reviews:
        params = {
            "json": 1,
            "language": language,
            "filter": review_filter,
            "review_type": review_type,
            "purchase_type": "all",
            "num_per_page": per_page,
            "cursor": cursor,
        }

        try:
            r = _REVIEW_SESSION.get(
                REVIEW_URL.format(app_id=app_id),
                params=params,
                timeout=15,
            )
            r.raise_for_status()
            payload = r.json()
        except requests.RequestException as exc:
            print(f"[ERROR] Request failed: {exc}", file=sys.stderr)
            break

        if payload.get("success") != 1:
            print("[ERROR] Steam API returned success=0. "
                  "Check that the App ID is correct.", file=sys.stderr)
            break

        batch = payload.get("reviews", [])
        if not batch:
            print("  → No more reviews available.")
            break

        new_in_batch = 0
        for raw in batch:
            rid = str(raw.get("recommendationid", ""))
            if not rid or rid in seen_ids:
                continue
            seen_ids.add(rid)
            reviews.append(_flatten_review(raw))
            new_in_batch += 1

        next_cursor = payload.get("cursor", "")
        fetched_so_far = len(reviews)
        total_available = payload.get("query_summary", {}).get("total_reviews")
        target = min(num_reviews, total_available) if total_available else num_reviews
        print(f"  Fetched {fetched_so_far} / {target} (+{new_in_batch} new)")

        if new_in_batch == 0:
            print("  → Cursor is cycling (no new reviews in page); stopping.")
            break
        if not next_cursor or next_cursor in ("*", cursor):
            break                       # Steam signals no next page
        cursor = next_cursor

        time.sleep(0.5)                 # polite delay between requests

    return reviews[:num_reviews]


def _flatten_review(raw: dict) -> dict:
    """Extract and flatten the fields we care about from a raw API review."""
    author = raw.get("author", {})
    ts_created = raw.get("timestamp_created", 0)
    ts_updated = raw.get("timestamp_updated", 0)

    return {
        "review_id":            raw.get("recommendationid", ""),
        "steam_id":             author.get("steamid", ""),
        "num_games_owned":      author.get("num_games_owned", 0),
        "num_reviews":          author.get("num_reviews", 0),
        "playtime_forever_hrs": round(author.get("playtime_forever", 0) / 60, 1),
        "playtime_at_review_hrs": round(author.get("playtime_at_review", 0) / 60, 1),
        "recommended":          raw.get("voted_up", False),
        "votes_up":             raw.get("votes_up", 0),
        "votes_funny":          raw.get("votes_funny", 0),
        "weighted_vote_score":  raw.get("weighted_vote_score", 0),
        "comment_count":        raw.get("comment_count", 0),
        "steam_purchase":       raw.get("steam_purchase", False),
        "received_for_free":    raw.get("received_for_free", False),
        "written_during_early_access": raw.get("written_during_early_access", False),
        "language":             raw.get("language", ""),
        "date_created":         _ts(ts_created),
        "date_updated":         _ts(ts_updated),
        "review_text":          raw.get("review", "").replace("\n", " ").strip(),
    }


def _ts(unix: int) -> str:
    """Convert a Unix timestamp to an ISO-8601 string (UTC)."""
    if not unix:
        return ""
    return datetime.fromtimestamp(unix, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_csv(reviews: list[dict], path: Path) -> None:
    if not reviews:
        print("No reviews to save.")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=reviews[0].keys())
        writer.writeheader()
        writer.writerows(reviews)
    print(f"\n✓ Saved {len(reviews)} reviews → {path}")


def save_json(reviews: list[dict], path: Path) -> None:
    if not reviews:
        print("No reviews to save.")
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Saved {len(reviews)} reviews → {path}")


def print_summary(reviews: list[dict], game_name: str) -> None:
    if not reviews:
        return
    positive = sum(1 for r in reviews if r["recommended"])
    negative = len(reviews) - positive
    pct = (positive / len(reviews)) * 100 if reviews else 0

    print("\n" + "=" * 50)
    print(f"  Game : {game_name}")
    print(f"  Total reviews fetched : {len(reviews)}")
    print(f"  Positive : {positive}  ({pct:.1f}%)")
    print(f"  Negative : {negative}  ({100 - pct:.1f}%)")
    print("=" * 50)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Steam game reviews via the Steam Store API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("app_id", type=int, help="Steam App ID of the game")
    parser.add_argument(
        "--num", type=int, default=100, metavar="N",
        help="Maximum number of reviews to fetch (default: 100)",
    )
    parser.add_argument(
        "--language", default="english", choices=VALID_LANGUAGES,
        help="Review language (default: english). Use 'all' for every language.",
    )
    parser.add_argument(
        "--filter", dest="review_filter", default="recent", choices=VALID_FILTERS,
        help="Sort order: recent | updated | all (default: recent)",
    )
    parser.add_argument(
        "--type", dest="review_type", default="all",
        choices=("all", "positive", "negative"),
        help="Which reviews to include (default: all)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file path. Extension sets format: .csv or .json "
             "(default: steam_reviews_<app_id>.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(f"steam_reviews_{args.app_id}.csv")

    fmt = out_path.suffix.lower()
    if fmt not in (".csv", ".json"):
        print(f"[WARN] Unknown extension '{fmt}', defaulting to CSV.")
        fmt = ".csv"
        out_path = out_path.with_suffix(".csv")

    # Fetch game name
    print(f"Looking up App ID {args.app_id} …")
    game_name = get_app_name(args.app_id)
    print(f"  → {game_name}")

    # Brief pause so Steam doesn't rate-limit back-to-back requests
    time.sleep(1)

    # Fetch reviews
    reviews = fetch_reviews(
        app_id=args.app_id,
        num_reviews=args.num,
        language=args.language,
        review_filter=args.review_filter,
        review_type=args.review_type,
    )

    # Print summary
    print_summary(reviews, game_name)

    # Save
    if fmt == ".json":
        save_json(reviews, out_path)
    else:
        save_csv(reviews, out_path)


if __name__ == "__main__":
    main()
