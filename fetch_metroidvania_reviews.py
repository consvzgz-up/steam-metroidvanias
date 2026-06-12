"""
fetch_metroidvania_reviews.py
=============================
Batch-fetch 500 Steam reviews for a curated list of metroidvania games.
Skips games whose output CSV already exists.
Imports helpers from steam_reviews.py (must be in the same directory).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from steam_reviews import fetch_reviews, get_app_name, save_csv, print_summary

# App IDs verified against store.steampowered.com/app/{appid}/ and the
# appreviews API (2026-06-09 audit). Fixed: blasphemous (was 833060),
# guacamelee_2 (was 534780), grime (was 1059650), deaths_gambit (was 1481085),
# record_of_lodoss_war (was 1277930). Added: axiom_verge_1, axiom_verge_2.
GAMES = [
    (367520,  "hollow_knight"),
    (774361,  "blasphemous"),
    (692850,  "bloodstained_ritual_of_the_night"),
    (275390,  "guacamelee_1"),            # Super Turbo Championship Edition
    (534550,  "guacamelee_2"),
    (283640,  "salt_and_sanctuary"),
    (1369630, "ender_lilies"),
    (1123050, "grime"),
    (356650,  "deaths_gambit_afterlife"),
    (2751000, "prince_of_persia_lost_crown"),
    (1809540, "nine_sols"),
    (2725260, "ender_magnolia"),
    (1030300, "hollow_knight_silksong"),
    (813230,  "animal_well"),
    (1672810, "mio_memories_in_orbit"),
    (1721060, "mandragora"),
    (2313700, "constance"),
    (1253920, "rogue_legacy_2"),
    (1203630, "record_of_lodoss_war"),
    (1701520, "afterimage"),
    (1517970, "aeterna_noctis"),
    (2014550, "voidwrought"),
    (332200,  "axiom_verge_1"),
    (946030,  "axiom_verge_2"),
]

# Fetch up to 1000 unique reviews per game; the analysis pipeline samples
# 500 per game (random_state=42), so the extra headroom improves diversity.
NUM_REVIEWS = 1000


def main():
    total = len(GAMES)
    completed = []
    skipped = []
    failed = []

    for i, (app_id, slug) in enumerate(GAMES, 1):
        out_path = Path(f"{slug}_reviews.csv")
        if out_path.exists():
            print(f"\n[{i}/{total}] Skipping {slug} — {out_path} already exists.")
            skipped.append(slug)
            continue

        print(f"\n{'='*60}")
        print(f"[{i}/{total}] Looking up App ID {app_id} …")
        game_name = get_app_name(app_id)
        print(f"  → {game_name}")

        time.sleep(1)

        # language="all" keeps the language column meaningful (choropleth);
        # filter="recent" is the cursor-stable mode that pages through
        # distinct reviews (filter="all" re-serves the same helpful reviews).
        reviews = fetch_reviews(
            app_id=app_id,
            num_reviews=NUM_REVIEWS,
            language="all",
            review_filter="recent",
            review_type="all",
        )

        if reviews:
            print_summary(reviews, game_name)
            save_csv(reviews, out_path)
            completed.append(game_name)
        else:
            print(f"  [WARN] No reviews returned for {game_name} (App {app_id}).")
            failed.append(slug)

        if i < total:
            time.sleep(2)

    print("\n" + "=" * 60)
    print("BATCH COMPLETE")
    print(f"  Fetched : {len(completed)} games")
    print(f"  Skipped : {len(skipped)} (CSV already existed)")
    print(f"  Failed  : {len(failed)}")
    if failed:
        print(f"  Failed games: {', '.join(failed)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
