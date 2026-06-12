# Audit — ProyectoFinal_Constantino.ipynb

Audited 2026-06-09. Every finding below was verified empirically against the local
CSVs and, where relevant, against the live Steam API (see `audit_checks.py`).

## ✅ Fix status (applied 2026-06-10)

| Finding | Fix applied |
|---|---|
| C1 duplicates | Scraper fixed (dedupe by review_id, `filter=recent`); **all 24 games re-scraped**: 23,896 rows, 100% unique. Old CSVs in `old_scrape_backup/`. Notebook + app now `drop_duplicates(subset='review_id')` as safety net. Destructive sampling cells replaced by in-memory sampling. |
| C2 Guacamelee appid | `GAME_REGISTRY` corrected to 275390 (STCE) in notebook + app; `games_meta_subset.csv` regenerated. |
| C3 stop_words NameError | `stop_words` moved to module level in the notebook; already fixed in the app. |
| C4 English-only scrape | Re-scraped with `language="all"` (29 languages) — the choropleth is now representative. Friction/text analyses restricted to English reviews (regexes are English), with the rationale documented in the notebook. |
| H1 `\blost\b` vs "Lost Crown" | Pattern is now `\blost\b(?!\s*crown)`; verified: PoP's navigation tag fell from 10.0% to 0.8%. |
| H2 destructive sampling | Cells 5–6 no longer overwrite CSVs; sampling happens in memory (`random_state=42`). |
| H3 small denominators | Heatmap x-labels now show N per band; crosstab cell prints denominators. |

**Pending on your side:** upload the new `*_reviews.csv` files to
`Drive/ProyectoMetroidvanias` (replacing the old ones) and re-run the notebook
in Colab; then refresh the numbers quoted in your Google Sites text.

---

## CRITICAL

### C1. Massive review duplication from the scraper — affects nearly every number in the notebook

`steam_reviews.py::fetch_reviews` pages the Steam `appreviews` endpoint with
`filter="all"` and **never deduplicates**. With that filter Steam's cursor
re-serves the same "most helpful" reviews page after page, so for games with a
small review pool the loop keeps appending repeats until it reaches 500.

Unique reviews per 500-row file (worst offenders):

| File | Unique / rows |
|---|---|
| guacamelee_2 | **28 / 500** |
| deaths_gambit_afterlife | **36 / 500** |
| guacamelee_1 | **43 / 500** |
| aeterna_noctis | **44 / 500** |
| voidwrought | 58 / 500 |
| record_of_lodoss_war | 66 / 500 |
| rogue_legacy_2 | 72 / 500 |
| salt_and_sanctuary | 84 / 500 |
| (only hollow_knight, silksong, nine_sols, axiom_verge 1/2 are 100% unique) | |

Across the dataset the notebook analyzes as 11,962 rows, only **4,596 are unique
reviews (38%)**. One single review appears up to 22 times.

**The notebook's own output shows the symptom**: in cell 22, trigrams like
"shooting arrows because" (24×), "over over over" (24×), "got more more" (23×)
are one duplicated review counted ~24 times — not 24 players saying the same thing.

Impact: per-game statistics, friction-tag percentages, the heatmap, the n-grams,
and the wordcloud are all weighted by duplication count, not by player opinion.
The headline trend (negative rate falls monotonically with playtime band)
**survives deduplication** (39.2% → 2.6% deduped vs. 43.1% → 4.8% with dups),
but every magnitude changes.

Fix:
1. In the pipeline: `master_df = master_df.drop_duplicates(subset='review_id')`
   immediately after concatenation (the Streamlit app exposes this as a toggle).
2. In the scraper: track seen `recommendationid`s, stop when a page adds no new
   ones, and prefer `filter="recent"` (Steam's documented cursor-stable mode).
3. Ideally re-scrape the low-uniqueness games — 28 unique reviews can't support
   per-game conclusions.

### C2. Guacamelee! reviews are merged with the wrong product's metadata

`guacamelee_1_reviews.csv` actually contains reviews of **Guacamelee! Super
Turbo Championship Edition (appid 275390)** — verified by comparing the file's
`review_id`s against the live API: 40/42 overlap with 275390, **0/8** with
214770. The notebook's `GAME_REGISTRY` assigns appid **214770 (Gold Edition)**,
so the merge attaches Gold Edition's price, Metacritic score and vote counts to
STCE's reviews.

Fix (one line): `'guacamelee_1_reviews.csv': (275390, 'Guacamelee! STCE')`.
games.csv does contain appid 275390, so the merge will then match correctly.

(The other five scraper/notebook appid conflicts — blasphemous, guacamelee_2,
grime, deaths_gambit, record_of_lodoss — were checked the same way: those CSVs
match the **notebook's** appids, i.e. they were evidently re-fetched with
corrected IDs. Only guacamelee_1 is misattributed.)

### C3. Cell 23 crashes on a fresh "Restart & run all" (`NameError: stop_words`)

`stop_words` is defined **inside** the function `get_negative_phrases` (cell 22,
local scope) but used at top level in cell 23 (`stop_words.union(custom_stops)`).
The notebook only ran because a stale global from an earlier session existed.
Fix: define `stop_words` at module level (the Streamlit app does this).

### C4. The choropleth map is a scraping artifact, not a geographic finding

The batch scraper (`fetch_metroidvania_reviews.py`) fetched with
`language="english"` — **22 of 24 games contain only English reviews**. Only
Axiom Verge 1 & 2 (fetched separately) have 20 languages. So the map's
non-US shading reflects almost exclusively reviews of two games, and
"English → USA" funnels ~93% of the corpus into one country. As drawn, the map
answers "which languages did the Axiom Verge scrape include", not "where do
negative reviews come from". Recommendation: re-scrape with `language="all"`,
or rescope/caveat the map explicitly.

---

## HIGH

### H1. Friction-tag regex false positives

- `navigation_confusion` includes `\blost\b`, which matches the **title** of
  Prince of Persia: The Lost Crown. Measured: 31 of PoP's 50 `lost` matches are
  the phrase "lost crown"; PoP's navigation-confusion rate reads 10.0% vs. 4.5%
  corpus-wide. Fix: `\blost\b(?!\s*crown)` or `\b(get|got|getting|felt|feel|being)\s+lost\b`.
- `boss_wall` fires on any mention of "boss": 22.5% of **positive** reviews
  match it ("amazing boss fights"). Restricting the crosstab to negative
  reviews mitigates but does not remove this (praise-with-caveats reviews).
- `ability_gate_confusion` includes the bare word `progression`, which matches
  praise of progression systems.

### H2. Destructive, undocumented sampling (cells 5–6)

`df.sample(n=500).to_csv(file_path)` **overwrites the raw scraped CSVs in
place**. Original data is unrecoverable, the cell is not idempotent (errors with
`ValueError` if a file ever has <500 rows), and only 2 of 24 files are sampled
in the notebook — the other 22 were evidently sampled outside it, so the
notebook does not document its own input data. Fix: sample in memory at load
time (as the Streamlit app does) or write to a different filename.

### H3. Unstable percentages on tiny denominators

The `completionist (>60h)` band contains only **54 negative reviews** (and
that's before deduplication — ~17 unique). Heatmap cells like "3.7%" are 2
reviews; "70.4% boss_wall" is 38 duplicate-inflated rows. Conclusions that lean
on the right side of the heatmap should report N and be softened.

---

## MEDIUM / LOW

- **M1.** games.csv snapshot quirks: Silksong and Constance have
  `Positive=0, Negative=0`, and several titles have `Metacritic score = 0`
  (meaning "none", not zero). Any analysis using these columns must treat 0 as
  missing.
- **M2.** Equal allocation (500/game) means pooled statistics are per-game
  averages, not population estimates of "metroidvania reviews" — fine, but
  should be stated once.
- **M3.** "Playtime at review" is a proxy for friction point, not abandonment;
  reviewers are a self-selected subset of players (already partially
  acknowledged in the notebook).
- **M4.** `confus.*map` and `back.*save point` use unbounded `.*` which can
  bridge unrelated sentence parts in long reviews.
- **M5.** `steam_reviews_1875580.csv` is an unrelated leftover file (its
  review_ids overlap nothing in the project) — remove to avoid confusion.

## Verified OK (checked, not flaws)

- games.csv broken-header fix (`names=CORRECTED_COLUMNS, skiprows=1`) aligns
  correctly — 40 names for 40 fields; dtypes and spot-checked names/appids confirm.
- Left-join on `appid` does not multiply rows (no duplicate appids in metadata);
  Mio: Memories in Orbit (1672810) is genuinely absent from the snapshot, and its
  reviews do come from 1672810 (147/151 live-API overlap).
- The world GeoJSON contains all 21 ISO codes used by `LANG_TO_ISO`
  (the Natural-Earth `-99` bug only affects N. Cyprus/Kosovo/Somaliland, unused).
- `pd.cut` band edges lose no rows (no zero-playtime reviews).
- The crosstab percentage math (divide by negative-review count per band) is correct.
