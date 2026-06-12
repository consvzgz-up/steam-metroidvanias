"""One-off verification of suspected logic flaws in ProyectoFinal_Constantino.ipynb."""
import os
import re
import pandas as pd
import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))

import streamlit_app as app  # reuse registry/constants without running streamlit

# Build df without streamlit caching
frames = []
for filename, (appid, game_name) in app.GAME_REGISTRY.items():
    part = pd.read_csv(os.path.join(APP_DIR, filename))
    part['appid'] = appid
    part['game_name'] = game_name
    frames.append(part)
master = pd.concat(frames, ignore_index=True).dropna(subset=['review_text'])
master['playtime_band'] = pd.cut(master['playtime_at_review_hrs'],
                                 bins=app.PLAYTIME_BINS, labels=app.PLAYTIME_LABELS)

text_lower = master['review_text'].str.lower()
for tag, patterns in app.FRICTION_TAGS.items():
    master[tag] = text_lower.str.contains('|'.join(patterns), regex=True, na=False)

print("=== rows:", len(master))

# 1. \blost\b false positives from game title "The Lost Crown"
lost_hits = text_lower.str.contains(r'\blost\b', na=False)
print("\n[1] navigation_confusion '\\blost\\b' hits by game (top 8):")
print(master[lost_hits]['game_name'].value_counts().head(8).to_string())
pop = master[master['game_name'] == 'Prince of Persia: The Lost Crown']
pop_lost = pop[pop['review_text'].str.lower().str.contains(r'\blost\b', na=False)]
title_mention = pop_lost['review_text'].str.lower().str.contains(r'lost crown', na=False)
print(f"PoP reviews matching \\blost\\b: {len(pop_lost)}, of which mention 'lost crown' (title): {title_mention.sum()}")
print(f"navigation_confusion rate in PoP: {pop['navigation_confusion'].mean()*100:.1f}% vs corpus {master['navigation_confusion'].mean()*100:.1f}%")

# also "got lost" legit vs title across corpus
silksong = master[master['game_name'] == 'Hollow Knight: Silksong']

# 2. negative-review denominators per band (heatmap stability)
neg = master[master['recommended'] == False]
print("\n[2] negative reviews per band (heatmap denominators):")
print(neg.groupby('playtime_band', observed=True).size().to_string())

# 3. 'progression' & 'boss' bleed: share of POSITIVE reviews matching friction tags
print("\n[3] tag hit-rate in POSITIVE reviews (contextual bleed):")
pos = master[master['recommended'] == True]
for tag in app.FRICTION_TAGS:
    print(f"  {tag:<25} pos {pos[tag].mean()*100:5.1f}%   neg {neg[tag].mean()*100:5.1f}%")

# 4. duplicate review_ids across files (re-scrape overlap)
dups = master['review_id'].duplicated().sum()
print(f"\n[4] duplicated review_id rows: {dups}")

# 5. games.csv alignment: read subset and check Name/appid for our games
subset_path = os.path.join(APP_DIR, 'games_meta_subset.csv')
chunks = []
for chunk in pd.read_csv(os.path.join(APP_DIR, 'games.csv'), names=app.CORRECTED_COLUMNS,
                         skiprows=1, usecols=app.KEEP_COLS, chunksize=200_000):
    chunk['AppID'] = pd.to_numeric(chunk['AppID'], errors='coerce')
    chunks.append(chunk[chunk['AppID'].isin({a for a, _ in app.GAME_REGISTRY.values()})])
gm = pd.concat(chunks, ignore_index=True)
print(f"\n[5] games.csv matches: {len(gm)} / 24")
print(gm[['AppID', 'Name', 'Price', 'Metacritic score', 'Positive', 'Negative']].to_string(index=False))
gm.to_csv(subset_path, index=False)
print(f"saved subset -> {subset_path}")

# 6. world geojson iso_a3 check (France/Norway -99 bug)
try:
    import geopandas as gpd
    world = gpd.read_file(app.WORLD_GEOJSON_URL)
    print("\n[6] world geojson columns:", list(world.columns))
    iso_col = 'iso_a3' if 'iso_a3' in world.columns else None
    if iso_col:
        bad = world[world[iso_col].isin(['-99', -99])]
        name_col = next((c for c in ('name', 'NAME', 'admin', 'ADMIN') if c in world.columns), None)
        print("countries with iso_a3 == -99:", list(bad[name_col]) if name_col is not None and len(bad) else len(bad))
        used = set(app.LANG_TO_ISO.values())
        present = set(world[iso_col])
        print("LANG_TO_ISO codes missing from geojson:", used - present)
except Exception as e:
    print("[6] geojson check failed:", e)
