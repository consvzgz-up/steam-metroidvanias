# =============================================================================
# Proyecto Final — ¿Por qué abandonamos los Metroidvanias?
# App de Streamlit que despliega el pipeline del notebook de Colab
# (ProyectoFinal_Constantino.ipynb) sin alterar su lógica ni resultados.
# =============================================================================

import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import streamlit as st

# -----------------------------------------------------------------------------
# Configuración general
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="¿Por qué abandonamos los Metroidvanias?",
    page_icon="🗺️",
    layout="wide",
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
GAMES_CSV_PATH = os.path.join(APP_DIR, "games.csv")
GAMES_META_SUBSET_PATH = os.path.join(APP_DIR, "games_meta_subset.csv")
ANIMATION_GIF_PATH = os.path.join(APP_DIR, "chart7_animation.gif")

# -----------------------------------------------------------------------------
# Registro de juegos — idéntico al notebook (cell 7)
# -----------------------------------------------------------------------------
GAME_REGISTRY = {
    'aeterna_noctis_reviews.csv':                   (1517970, 'Aeterna Noctis'),
    'afterimage_reviews.csv':                        (1701520, 'Afterimage'),
    'animal_well_reviews.csv':                       (813230,  'Animal Well'),
    'axiom_verge_1_reviews.csv':                     (332200,  'Axiom Verge'),
    'axiom_verge_2_reviews.csv':                     (946030,  'Axiom Verge 2'),
    'blasphemous_reviews.csv':                       (774361,  'Blasphemous'),
    'bloodstained_ritual_of_the_night_reviews.csv':  (692850,  'Bloodstained: Ritual of the Night'),
    'constance_reviews.csv':                         (2313700, 'Constance'),
    'deaths_gambit_afterlife_reviews.csv':           (356650,  "Death's Gambit: Afterlife"),
    'ender_lilies_reviews.csv':                      (1369630, 'Ender Lilies'),
    'ender_magnolia_reviews.csv':                    (2725260, 'Ender Magnolia'),
    'grime_reviews.csv':                             (1123050, 'Grime'),
    # Fix auditoría C2: las reseñas provienen de la Super Turbo Championship
    # Edition (275390), no de la Gold Edition (214770).
    'guacamelee_1_reviews.csv':                      (275390,  'Guacamelee!'),
    'guacamelee_2_reviews.csv':                      (534550,  'Guacamelee! 2'),
    'hollow_knight_reviews.csv':                     (367520,  'Hollow Knight'),
    'hollow_knight_silksong_reviews.csv':            (1030300, 'Hollow Knight: Silksong'),
    'mandragora_reviews.csv':                        (1721060, 'Mandragora'),
    'mio_memories_in_orbit_reviews.csv':             (1672810, 'Mio: Memories in Orbit'),
    'nine_sols_reviews.csv':                         (1809540, 'Nine Sols'),
    'prince_of_persia_lost_crown_reviews.csv':       (2751000, 'Prince of Persia: The Lost Crown'),
    'record_of_lodoss_war_reviews.csv':              (1203630, 'Record of Lodoss War'),
    'rogue_legacy_2_reviews.csv':                    (1253920, 'Rogue Legacy 2'),
    'salt_and_sanctuary_reviews.csv':                (283640,  'Salt and Sanctuary'),
    'voidwrought_reviews.csv':                       (2014550, 'Voidwrought'),
}

# Columnas corregidas del games.csv de fronkongames — idéntico al notebook
# (el encabezado original trae 'DiscountDLC count' pegado: 39 nombres para
# 40 campos, por eso se reasignan los nombres y se salta el encabezado).
CORRECTED_COLUMNS = [
    'AppID', 'Name', 'Release date', 'Estimated owners', 'Peak CCU',
    'Required age', 'Price', 'Discount', 'DLC count', 'About the game',
    'Supported languages', 'Full audio languages', 'Reviews', 'Header image',
    'Website', 'Support url', 'Support email', 'Windows', 'Mac', 'Linux',
    'Metacritic score', 'Metacritic url', 'User score', 'Positive', 'Negative',
    'Score rank', 'Achievements', 'Recommendations', 'Notes',
    'Average playtime forever', 'Average playtime two weeks',
    'Median playtime forever', 'Median playtime two weeks',
    'Developer', 'Publisher', 'Categories', 'Genres', 'Tags',
    'Screenshots', 'Movies'
]

KEEP_COLS = ['AppID', 'Name', 'Release date', 'Price', 'Genres',
             'Tags', 'Developer', 'Positive', 'Negative',
             'Metacritic score', 'Average playtime forever']

# Etiquetas de fricción — idéntico al notebook (cell 19)
FRICTION_TAGS = {
    'boss_wall': [
        'boss', 'difficulty spike', 'unfair', r'\bcheap\b',
        'one.?shot', 'punishing',
    ],
    'controls_platforming': [
        'control', 'clunky', 'floaty', 'hitbox',
        'slippery', 'unresponsive',
    ],
    'pacing_bloat': [
        r'\bboring\b', 'repetitiv', r'\bgrind',
        'too long', r'\bpadding\b', 'tedious',
        'way too much', 'over and over', 'waste your time',
    ],
    'navigation_confusion': [
        'where to go', 'confus.*map', 'no idea where',
        # Fix auditoría H1: el lookahead excluye el título "The Lost Crown"
        r'\blost\b(?!\s*crown)', 'no direction', 'figure out what',
    ],
    'backtracking_fatigue': [
        'backtrack', 'fast travel', 'running back',
        'trek back', 'back.*save point', 'save point',
    ],
    'ability_gate_confusion': [
        r'\bstuck\b', 'progression', 'need.*ability',
    ],
}

TAG_LABELS = {
    'boss_wall':              'Boss / difficulty wall',
    'controls_platforming':   'Controls & platforming',
    'pacing_bloat':           'Pacing & bloat',
    'navigation_confusion':   'Navigation confusion',
    'backtracking_fatigue':   'Backtracking fatigue',
    'ability_gate_confusion': 'Ability gate confusion',
}

TAG_LABELS_SHORT = {
    'boss_wall':              'Boss wall',
    'controls_platforming':   'Controls',
    'pacing_bloat':           'Pacing',
    'navigation_confusion':   'Navigation',
    'backtracking_fatigue':   'Backtracking',
    'ability_gate_confusion': 'Ability gate',
}

# Stop words usadas en n-gramas y nube de palabras (cells 22-23 del notebook)
STOP_WORDS = set(['the', 'and', 'this', 'that', 'with', 'for', 'you',
                  'have', 'was', 'but'])

# Mapeo idioma → país representativo (cell 31 del notebook)
LANG_TO_ISO = {
    'english':    'USA',   'russian':   'RUS',  'spanish':   'ESP',
    'schinese':   'CHN',   'brazilian': 'BRA',  'japanese':  'JPN',
    'french':     'FRA',   'german':    'DEU',  'latam':     'MEX',
    'italian':    'ITA',   'tchinese':  'TWN',  'polish':    'POL',
    'koreana':    'KOR',   'turkish':   'TUR',  'ukrainian': 'UKR',
    'swedish':    'SWE',   'czech':     'CZE',  'portuguese':'PRT',
    'finnish':    'FIN',   'dutch':     'NLD',  'hungarian': 'HUN',
}

WORLD_GEOJSON_URL = ("https://raw.githubusercontent.com/datasets/"
                     "geo-boundaries-world-110m/master/countries.geojson")

PLAYTIME_BINS = [0, 5, 15, 30, 60, np.inf]
PLAYTIME_LABELS = ['early drop (<5h)', 'short run (5–15h)',
                   'mid game (15–30h)', 'late game (30–60h)',
                   'completionist (>60h)']


# -----------------------------------------------------------------------------
# Carga y preparación de datos (Fases 1 y 2 del notebook)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Cargando reseñas de Steam…")
def load_reviews():
    frames = []
    for filename, (appid, game_name) in GAME_REGISTRY.items():
        path = os.path.join(APP_DIR, filename)
        part = pd.read_csv(path)
        # Muestreo de 500 reseñas por juego (cells 5-6 del notebook), aplicado
        # en memoria — sin sobrescribir el CSV original como hace el notebook.
        if len(part) > 500:
            part = part.sample(n=500, random_state=42)
        part['appid'] = appid
        part['game_name'] = game_name
        frames.append(part)
    master_df = pd.concat(frames, ignore_index=True)

    # Limpieza de nulos en review_text (cell 11)
    master_df = master_df.dropna(subset=['review_text'])

    # Columnas temporales (cell 12)
    master_df['date_created'] = pd.to_datetime(master_df['date_created'])
    master_df['date_updated'] = pd.to_datetime(master_df['date_updated'])
    master_df['review_year'] = master_df['date_created'].dt.year
    master_df['review_month'] = master_df['date_created'].dt.to_period('M')

    # Transformación logarítmica y bandas de horas jugadas (cell 13)
    master_df['playtime_log'] = np.log1p(master_df['playtime_at_review_hrs'])
    master_df['playtime_band'] = pd.cut(
        master_df['playtime_at_review_hrs'],
        bins=PLAYTIME_BINS,
        labels=PLAYTIME_LABELS,
    )
    return master_df


@st.cache_data(show_spinner="Cargando metadatos de juegos…")
def load_games_meta():
    """Carga games.csv con el mismo tratamiento del notebook (cell 14).

    Como games.csv pesa ~390 MB, la primera vez se lee por bloques
    filtrando solo los appids del proyecto y se guarda un extracto
    (games_meta_subset.csv). Filtrar antes del merge no cambia el
    resultado del left join. En despliegues basta con subir el extracto.
    """
    registry_appids = {appid for appid, _ in GAME_REGISTRY.values()}

    if os.path.exists(GAMES_META_SUBSET_PATH):
        games_meta = pd.read_csv(GAMES_META_SUBSET_PATH)
    elif os.path.exists(GAMES_CSV_PATH):
        chunks = []
        reader = pd.read_csv(
            GAMES_CSV_PATH,
            names=CORRECTED_COLUMNS,
            skiprows=1,            # saltamos el encabezado original roto
            usecols=KEEP_COLS,
            chunksize=100_000,
        )
        for chunk in reader:
            chunk['AppID'] = pd.to_numeric(chunk['AppID'], errors='coerce')
            chunks.append(chunk[chunk['AppID'].isin(registry_appids)])
        games_meta = pd.concat(chunks, ignore_index=True)
        games_meta.to_csv(GAMES_META_SUBSET_PATH, index=False)
    else:
        st.error(
            "No se encontró ni `games_meta_subset.csv` ni `games.csv` junto a "
            "la app. Coloca alguno de los dos archivos para continuar."
        )
        st.stop()

    games_meta = (
        games_meta[KEEP_COLS]
        .rename(columns={'AppID': 'appid'})
        .copy()
    )
    games_meta['appid'] = pd.to_numeric(games_meta['appid'], errors='coerce')
    games_meta = games_meta.dropna(subset=['appid'])
    games_meta['appid'] = games_meta['appid'].astype(int)
    return games_meta


@st.cache_data(show_spinner="Uniendo datasets y etiquetando reseñas…")
def build_master():
    master_df = load_reviews()

    # Fix auditoría C1 (igual que el notebook): el scraper original repetía
    # reseñas al paginar con filter='all'. Con los datos re-descargados esto
    # elimina ~0 filas, pero se mantiene como red de seguridad.
    master_df = master_df.drop_duplicates(subset='review_id')

    games_meta = load_games_meta()

    # Left join sobre appid (cell 16): se conserva toda reseña aunque el
    # juego no exista en los metadatos (limitación de recencia del snapshot).
    df = master_df.merge(games_meta, on='appid', how='left')

    # Etiquetado de fricción por regex (cell 20)
    text_lower = df['review_text'].str.lower()
    for tag, patterns in FRICTION_TAGS.items():
        combined_pattern = '|'.join(patterns)
        df[tag] = text_lower.str.contains(combined_pattern, regex=True, na=False)

    return df


@st.cache_data
def friction_crosstabs(df):
    """Crosstab fricción × banda de horas en reseñas negativas.

    Solo reseñas en inglés: las regex de fricción son en inglés, y los datos
    incluyen todos los idiomas (necesarios para el mapa coroplético).
    """
    neg_df = df[(df['recommended'] == False) & (df['language'] == 'english')]  # noqa: E712
    crosstab = pd.DataFrame({
        tag: neg_df.groupby('playtime_band', observed=True)[tag].sum()
        for tag in FRICTION_TAGS
    }).T
    band_totals = neg_df.groupby('playtime_band', observed=True).size()
    crosstab_pct = (crosstab.div(band_totals, axis=1) * 100).round(1)
    return crosstab, crosstab_pct, band_totals


@st.cache_data
def negative_phrases(df, n=3, top_k=25):
    """Top n-gramas en reseñas negativas (cell 22). Devuelve DataFrame."""
    from nltk.util import ngrams

    neg_df = df[(df['recommended'] == False) & (df['language'] == 'english')]  # noqa: E712
    text_blob = " ".join(neg_df['review_text'].astype(str).str.lower())
    words = re.findall(r'\b[a-z]{3,}\b', text_blob)
    words = [w for w in words if w not in STOP_WORDS]
    phrase_counts = Counter(ngrams(words, n))
    rows = [{'Frase': ' '.join(p), 'Apariciones': c}
            for p, c in phrase_counts.most_common(top_k)]
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Visualizaciones (Fase 3 del notebook) — misma lógica, una función por gráfico
# -----------------------------------------------------------------------------
def chart_playtime_histogram(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle('Playtime distribution at review — recommended vs. not recommended',
                 fontsize=13, fontweight='bold', y=1.01)
    for ax, (rec, label, color) in zip(axes, [
        (False, 'Not recommended', '#e05c5c'),
        (True,  'Recommended',     '#5b8dd9'),
    ]):
        data = df[df['recommended'] == rec]['playtime_at_review_hrs']
        data_capped = data[data <= 100]
        ax.hist(data_capped, bins=40, color=color, edgecolor='white', linewidth=0.4)
        ax.axvline(data.median(), color='black', linestyle='--', linewidth=1.2,
                   label=f'Median: {data.median():.1f}h')
        ax.set_title(label, fontsize=11)
        ax.set_xlabel('Playtime at review (hours)')
        ax.set_ylabel('Number of reviews')
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return fig


def chart_negative_rate_by_band(df):
    band_stats = df.groupby('playtime_band', observed=True).agg(
        total=('recommended', 'count'),
        negative=('recommended', lambda x: (~x).sum())
    ).assign(neg_rate=lambda d: (d.negative / d.total * 100).round(1))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(band_stats.index, band_stats['neg_rate'],
                  color=['#e05c5c', '#e8845c', '#e8b45c', '#8db87a', '#5b8dd9'],
                  edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, band_stats['neg_rate']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title('Negative review rate by playtime band', fontsize=13, fontweight='bold')
    ax.set_xlabel('Playtime at review')
    ax.set_ylabel('Negative review rate (%)')
    ax.set_ylim(0, max(55, band_stats['neg_rate'].max() + 5))
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return fig


CUMULATIVE_GAMES = [
    'Hollow Knight: Silksong',
    'Ender Magnolia',
    'Record of Lodoss War',
    'Guacamelee!',
    'Hollow Knight',
    'Blasphemous',
]


def chart_abandonment_curves(df):
    fig, ax = plt.subplots(figsize=(11, 6))
    colors_line = ['#e05c5c', '#e8845c', '#5b8dd9', '#5bbdd9', '#2d6a4f', '#9b5de5']
    for game, color in zip(CUMULATIVE_GAMES, colors_line):
        game_df = df[df['game_name'] == game].copy()
        game_df = game_df.sort_values('playtime_at_review_hrs')
        game_df['cumulative_pct'] = (np.arange(1, len(game_df) + 1) / len(game_df)) * 100
        ax.plot(game_df['playtime_at_review_hrs'].clip(upper=80),
                game_df['cumulative_pct'],
                label=game, color=color, linewidth=1.8, alpha=0.85)
    for x, lbl in [(5, '<5h band'), (15, '15h'), (30, '30h')]:
        ax.axvline(x, color='grey', linestyle=':', linewidth=1, alpha=0.6)
        ax.text(x + 0.3, 2, lbl, fontsize=8, color='grey')
    ax.set_title('Cumulative review accumulation by playtime\n'
                 '(steeper early rise = more early-quitter reviews)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Playtime at review (hours, capped at 80h)')
    ax.set_ylabel('Cumulative % of reviews')
    ax.legend(fontsize=9, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return fig


def chart_boxplot_sentiment(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=df[df['playtime_at_review_hrs'] <= 120],
        x='recommended',
        y='playtime_at_review_hrs',
        hue='recommended',
        palette={True: '#5b8dd9', False: '#e05c5c'},
        width=0.45,
        linewidth=1.2,
        flierprops=dict(marker='o', markersize=2, alpha=0.3),
        legend=False,
        ax=ax,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Not recommended', 'Recommended'])
    ax.set_title('Playtime at review by sentiment\n(capped at 120h for readability)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Playtime at review (hours)')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return fig


def chart_friction_heatmap(crosstab_pct, band_totals):
    data = crosstab_pct.copy()
    data.index = [TAG_LABELS[t] for t in data.index]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        data,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        linewidths=0.4,
        linecolor='white',
        cbar_kws={'label': '% of negative reviews in band'},
        ax=ax,
    )
    # Denominadores visibles: N de reseñas negativas por banda
    ax.set_xticklabels([f"{band}\n(N={band_totals[band]})"
                        for band in data.columns], fontsize=8)
    ax.set_title('Friction complaint themes by playtime band\n'
                 '(% of English negative reviews per band mentioning each theme)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Playtime at review')
    ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    return fig


KDE_GAMES = [
    'Record of Lodoss War',
    'Guacamelee!',
    'Animal Well',
    'Hollow Knight',
    'Nine Sols',
    'Hollow Knight: Silksong',
]


def chart_kde_playtime(df):
    palette = ['#e05c5c', '#e8845c', '#e8c45c', '#5b8dd9', '#5bbdd9', '#9b5de5']
    fig, ax = plt.subplots(figsize=(11, 5))
    for game, color in zip(KDE_GAMES, palette):
        game_data = df[df['game_name'] == game]['playtime_at_review_hrs']
        game_data = game_data[game_data <= 100]
        sns.kdeplot(game_data, ax=ax, label=game, color=color,
                    linewidth=2, fill=True, alpha=0.08)
    ax.set_title('Playtime distribution shape by game\n(KDE — capped at 100h)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Playtime at review (hours)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return fig


@st.cache_data(show_spinner="Generando animación…")
def build_animation_gif(crosstab):
    """Barra animada de quejas por banda (cell 30). Guarda y devuelve un GIF."""
    anim_data = crosstab.T.copy()  # filas = bandas, columnas = tags
    anim_data.index = ['<5h', '5–15h', '15–30h', '30–60h', '>60h']
    tag_cols = list(FRICTION_TAGS.keys())
    bar_colors = ['#e05c5c', '#e8845c', '#e8c45c', '#5b8dd9', '#5bbdd9', '#9b5de5']
    bands = anim_data.index.tolist()
    tags_short = [TAG_LABELS_SHORT[t] for t in tag_cols]

    fig, ax = plt.subplots(figsize=(10, 6))

    def update(frame):
        ax.clear()
        band = bands[frame]
        vals = anim_data.iloc[frame][tag_cols].values
        bars = ax.barh(tags_short, vals, color=bar_colors, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    str(int(val)), va='center', fontsize=10)
        ax.set_xlim(0, anim_data[tag_cols].values.max() * 1.15)
        ax.set_title(f'Friction complaints in negative reviews\nPlaytime band: {band}',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of negative reviews mentioning theme')
        ax.spines[['top', 'right']].set_visible(False)

    ani = animation.FuncAnimation(fig, update, frames=len(bands),
                                  interval=1200, repeat=True)
    ani.save(ANIMATION_GIF_PATH, writer='pillow', fps=0.9, dpi=120)
    plt.close(fig)
    return ANIMATION_GIF_PATH


@st.cache_data(show_spinner="Construyendo mapa coroplético…")
def build_choropleth(df):
    """Mapa coroplético de reseñas negativas por idioma (cell 31)."""
    import geopandas as gpd

    neg_df = df[df['recommended'] == False]  # noqa: E712
    neg_lang = neg_df.copy()
    neg_lang['iso'] = neg_lang['language'].map(LANG_TO_ISO)
    neg_lang = neg_lang.dropna(subset=['iso'])
    lang_counts = neg_lang.groupby('iso').size().reset_index(name='neg_reviews')

    world = gpd.read_file(WORLD_GEOJSON_URL)
    world = world.rename(columns={'iso_a3': 'iso'})
    merged = world.merge(lang_counts, on='iso', how='left')

    fig, ax = plt.subplots(figsize=(15, 8))
    world.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.3)
    merged[merged['neg_reviews'].notna()].plot(
        ax=ax, column='neg_reviews', cmap='YlOrRd',
        legend=True,
        legend_kwds={'label': 'Negative reviews (by review language)',
                     'shrink': 0.5},
        edgecolor='#cccccc', linewidth=0.3,
    )
    ax.set_title('Geographic distribution of negative reviews\n'
                 '(mapped by review language to representative country)',
                 fontsize=12, fontweight='bold')
    ax.axis('off')
    fig.text(0.5, 0.02,
             'Note: language mapped to a single representative country — '
             'English→USA, Spanish→Spain, etc. This is an approximation.',
             ha='center', fontsize=8, color='grey')
    plt.tight_layout()
    return fig


def chart_scatter_merge_validation(df):
    """Validación cruzada muestra (API) vs. historial global (games.csv).

    Ambos ejes y el color usan columnas que solo existen gracias al merge:
    X = % votos negativos globales, Y = % negativas en la muestra reciente,
    color = precio. La diagonal marca el acuerdo perfecto; puntos por encima
    indican sentimiento reciente peor que el histórico.
    """
    game_scatter = (
        df.groupby('game_name', observed=True)
          .agg(sample_neg_rate=('recommended', lambda x: (~x).mean() * 100),
               global_pos=('Positive', 'first'),
               global_neg=('Negative', 'first'),
               price=('Price', 'first'))
          .reset_index()
    )
    # Sin metadatos (Mio) o con contadores en cero en el snapshot
    # (Silksong, Constance — el dump es anterior a sus reseñas)
    game_scatter = game_scatter.dropna(subset=['global_pos', 'global_neg'])
    game_scatter = game_scatter[
        (game_scatter['global_pos'] + game_scatter['global_neg']) > 0]
    game_scatter['global_neg_pct'] = (
        game_scatter['global_neg']
        / (game_scatter['global_pos'] + game_scatter['global_neg']) * 100
    )

    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(game_scatter['global_neg_pct'],
                    game_scatter['sample_neg_rate'],
                    c=game_scatter['price'], cmap='viridis', s=140,
                    edgecolor='white', linewidth=0.8, zorder=3)
    fig.colorbar(sc, ax=ax, label='Precio actual (USD, games.csv)', shrink=0.8)

    lim = max(game_scatter['global_neg_pct'].max(),
              game_scatter['sample_neg_rate'].max()) * 1.1
    ax.plot([0, lim], [0, lim], color='grey', linestyle='--',
            linewidth=1, zorder=1)
    ax.text(lim * 0.72, lim * 0.78, 'reseñas recientes peores\nque la historia global',
            fontsize=8, color='grey', ha='center')
    ax.text(lim * 0.78, lim * 0.55, 'reseñas recientes mejores\nque la historia global',
            fontsize=8, color='grey', ha='center')

    for _, row in game_scatter.iterrows():
        ax.annotate(row['game_name'],
                    (row['global_neg_pct'], row['sample_neg_rate']),
                    textcoords='offset points', xytext=(6, 4), fontsize=7.5)

    ax.set_title('Validación cruzada: % negativo global (games.csv) vs. '
                 '% negativo en la muestra (API)\n'
                 'Color = precio — tres columnas que solo existen gracias al merge por appid',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('% de votos negativos globales del juego (games.csv, todo su historial)')
    ax.set_ylabel('% de reseñas negativas en la muestra (API, reseñas recientes)')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return fig


@st.cache_data(show_spinner="Generando nube de palabras…")
def build_wordcloud(df):
    from wordcloud import WordCloud, STOPWORDS

    neg_df = df[(df['recommended'] == False) & (df['language'] == 'english')]  # noqa: E712
    neg_text = " ".join(neg_df['review_text'].astype(str).str.lower())
    custom_stops = {'game', 'play', 'played', 'really', 'much', 'get',
                    'even', 'one', 'make', 'made'}
    # STOPWORDS es la lista integrada de la librería (~190 stop words en
    # inglés); sin la unión, el set propio la reemplaza y se cuelan
    # palabras como "it", "is", "a".
    final_stop_words = STOPWORDS.union(STOP_WORDS).union(custom_stops)
    wc = WordCloud(
        width=800, height=400,
        background_color='black', colormap='Reds',
        stopwords=final_stop_words, max_words=100,
    ).generate(neg_text)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Conceptos Dominantes en Reseñas Negativas', fontsize=16)
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Navegación y datos base
# -----------------------------------------------------------------------------
st.sidebar.title("🗺️ Metroidvanias en Steam")
page = st.sidebar.radio(
    "Navegación",
    ["🏠 Inicio", "🧪 Metodología", "📊 Dashboard", "📝 Conclusiones"],
)
st.sidebar.markdown("---")
df = build_master()
neg_df = df[df['recommended'] == False]  # noqa: E712
crosstab, crosstab_pct, band_totals = friction_crosstabs(df)
st.sidebar.caption(
    "Proyecto Final — Herramientas Enfocadas a la Ciencia de Datos\n\n"
    "Constantino · Universidad Panamericana"
)

# =============================================================================
# 🏠 INICIO
# =============================================================================
if page == "🏠 Inicio":
    st.title("¿Por qué abandonamos los Metroidvanias?")
    st.subheader("Análisis de fricción y abandono en reseñas de Steam")

    st.markdown(
        """
        Los **metroidvanias** son juegos de exploración no lineal donde el
        progreso depende de habilidades que se desbloquean gradualmente. Ese
        diseño genera fricciones características: *muros de jefes*, confusión
        de navegación, backtracking excesivo. Este proyecto cruza **dos fuentes
        de datos** para estudiar en qué momento de la partida esas fricciones
        se vuelven decisivas y el jugador deja una reseña negativa:

        1. **Reseñas de Steam** (24 juegos × 500 reseñas, obtenidas vía la
           [API pública de Steam](https://partner.steamgames.com/doc/store/getreviews)).
        2. **Metadatos de juegos** del dataset
           [Steam Games Dataset (fronkongames, Kaggle)](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset)
           — precio, género, fecha de lanzamiento, puntuaciones.
        """
    )

    st.markdown("### 📌 Estadísticas clave")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Reseñas analizadas", f"{len(df):,}")
    c2.metric("Juegos", f"{df['game_name'].nunique()}")
    neg_rate = (~df['recommended']).mean() * 100
    c3.metric("Reseñas negativas", f"{neg_rate:.1f}%")
    c4.metric("Mediana horas jugadas", f"{df['playtime_at_review_hrs'].median():.1f} h")
    tagged_pct = df[list(FRICTION_TAGS)].any(axis=1).mean() * 100
    c5.metric("Reseñas con tema de fricción", f"{tagged_pct:.0f}%")

    st.markdown("### 🎮 Juegos incluidos")
    games_table = (
        df.groupby('game_name', observed=True)
        .agg(
            reseñas=('review_id', 'count'),
            negativas=('recommended', lambda x: int((~x).sum())),
            mediana_horas=('playtime_at_review_hrs', 'median'),
        )
        .assign(pct_negativas=lambda d: (d.negativas / d.reseñas * 100).round(1))
        .sort_values('pct_negativas', ascending=False)
    )
    st.dataframe(games_table, width='stretch')

# =============================================================================
# 🧪 METODOLOGÍA
# =============================================================================
elif page == "🧪 Metodología":
    st.title("🧪 Metodología — Procesamiento de datos")

    st.markdown(
        f"""
        ### 1. Adquisición y muestreo
        Para cada uno de los 24 juegos se descargaron reseñas con la API de
        Steam y se tomó una **muestra aleatoria de 500 reseñas por juego**
        (`random_state=42`), evitando que los juegos más populares dominen
        el análisis.

        ### 2. Limpieza (Pandas / NumPy)
        - Se eliminaron filas duplicadas por `review_id` (el identificador
          único de cada reseña en Steam) y las reseñas con `review_text` nulo
          (quedaron **{len(df):,}** filas).
        - Fechas convertidas con `pd.to_datetime`; se derivaron
          `review_year` y `review_month`.
        - Las horas jugadas están muy sesgadas a la derecha, por lo que se
          aplicó **`np.log1p`** (operación vectorizada de NumPy) para crear
          `playtime_log`.
        - Se discretizó el tiempo de juego en **bandas** con `pd.cut`:
          `<5h`, `5–15h`, `15–30h`, `30–60h`, `>60h`.

        ### 3. Unión de tablas (Merge)
        Se hizo un **left join** entre las reseñas y los metadatos de Kaggle
        usando **`appid` como llave primaria** — es el identificador único y
        estable que Steam asigna a cada juego, presente en ambas fuentes.
        El left join conserva todas las reseñas aunque el juego no aparezca
        en el snapshot de metadatos (p. ej. *Mio: Memories in Orbit*, título
        de 2025 ausente del snapshot).

        ### 4. Etiquetado de fricción (regex)
        Cada reseña se etiquetó con 6 temas de fricción mediante expresiones
        regulares sobre el texto en minúsculas (boss wall, controles,
        pacing, navegación, backtracking, ability gates).
        """
    )

    st.markdown("### Vista previa del DataFrame unido")
    preview_cols = ['game_name', 'recommended', 'playtime_at_review_hrs',
                    'playtime_band', 'language', 'Price', 'Genres',
                    'Metacritic score']
    st.dataframe(df[preview_cols].head(50), width='stretch')

    st.markdown("### Cobertura del merge")
    matched = df['Genres'].notna().sum()
    st.write(
        f"- Filas con metadatos tras el merge: **{matched:,} / {len(df):,}** "
        f"({matched / len(df) * 100:.1f}%)"
    )
    unmatched = df[df['Genres'].isna()]['game_name'].value_counts()
    if len(unmatched):
        st.write("- Juegos sin match en los metadatos:")
        st.dataframe(unmatched)

    st.markdown("### Cobertura de etiquetas de fricción")
    tag_cov = pd.DataFrame({
        'Tema': [TAG_LABELS[t] for t in FRICTION_TAGS],
        'Reseñas': [int(df[t].sum()) for t in FRICTION_TAGS],
        '% del total': [round(df[t].mean() * 100, 1) for t in FRICTION_TAGS],
    })
    st.dataframe(tag_cov, width='stretch', hide_index=True)

# =============================================================================
# 📊 DASHBOARD
# =============================================================================
elif page == "📊 Dashboard":
    st.title("📊 Dashboard — Análisis visual")

    tab_names = [
        "1· Histograma", "2· Tasa negativa", "3· Curvas de abandono",
        "4· Boxplot", "5· Heatmap fricción", "6· KDE por juego",
        "7· Animación", "8· Mapa mundial", "9· Validación (merge)",
        "10· Texto (extra)",
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.markdown("**Distribución de horas jugadas al reseñar** — la "
                    "divergencia entre poblaciones revela la ventana donde la "
                    "fricción se vuelve decisiva.")
        st.pyplot(chart_playtime_histogram(df))

    with tabs[1]:
        st.markdown("**Tasa de reseñas negativas por banda de horas** — el "
                    "patrón de abandono en su forma más directa.")
        st.pyplot(chart_negative_rate_by_band(df))

    with tabs[2]:
        st.markdown("**Acumulación de reseñas según horas jugadas** — una "
                    "subida temprana pronunciada implica más jugadores que "
                    "reseñan (y abandonan) pronto.")
        st.pyplot(chart_abandonment_curves(df))

    with tabs[3]:
        st.markdown("**Horas jugadas según sentimiento** — mediana, dispersión "
                    "y outliers en un solo gráfico (Seaborn boxplot).")
        st.pyplot(chart_boxplot_sentiment(df))

    with tabs[4]:
        st.markdown("**Temas de fricción × banda de horas** (Seaborn heatmap) "
                    "— % de reseñas negativas *en inglés* de cada banda que "
                    "mencionan cada tema (las regex de fricción son en inglés).")
        st.pyplot(chart_friction_heatmap(crosstab_pct, band_totals))

    with tabs[5]:
        st.markdown("**Forma de la distribución de horas por juego** "
                    "(Seaborn KDE) — juegos con perfiles de abandono "
                    "contrastantes.")
        st.pyplot(chart_kde_playtime(df))

    with tabs[6]:
        st.markdown("**Animación**: evolución de las quejas de fricción "
                    "conforme avanza la partida (Matplotlib FuncAnimation, "
                    "exportada a GIF).")
        gif_path = build_animation_gif(crosstab)
        st.image(gif_path)

    with tabs[7]:
        st.markdown("**Mapa coroplético (GeoPandas)** — reseñas negativas por "
                    "idioma, asignadas a un país representativo.")
        try:
            st.pyplot(build_choropleth(df))
        except Exception as exc:
            st.warning(
                "No se pudo construir el mapa (se necesita conexión a "
                f"internet para descargar el GeoJSON mundial): {exc}"
            )

    with tabs[8]:
        st.markdown("**Validación cruzada de la muestra (el valor del merge)** "
                    "— % de votos negativos en TODO el historial del juego "
                    "(games.csv) vs. % negativo en nuestra muestra reciente "
                    "(API). La diagonal = acuerdo perfecto; arriba de la línea "
                    "= sentimiento reciente peor que el histórico. "
                    "Color = precio. Tres columnas del merge en un solo gráfico.")
        st.pyplot(chart_scatter_merge_validation(df))

    with tabs[9]:
        st.markdown("**Análisis de texto en reseñas negativas en inglés** — "
                    "frases más repetidas (trigramas, NLTK) y nube de palabras.")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(negative_phrases(df, n=3, top_k=25),
                         width='stretch', hide_index=True)
        with col2:
            st.pyplot(build_wordcloud(df))

# =============================================================================
# 📝 CONCLUSIONES
# =============================================================================
elif page == "📝 Conclusiones":
    st.title("📝 Conclusiones")

    band_stats = df.groupby('playtime_band', observed=True).agg(
        total=('recommended', 'count'),
        negative=('recommended', lambda x: (~x).sum())
    ).assign(neg_rate=lambda d: (d.negative / d.total * 100).round(1))

    st.markdown(
        f"""
        ### Hallazgos principales

        1. **El abandono temprano concentra el sentimiento negativo.** La tasa
           de reseñas negativas es máxima en la banda `<5h`
           ({band_stats['neg_rate'].iloc[0]:.1f}%) y desciende conforme
           aumenta el tiempo jugado
           ({band_stats['neg_rate'].iloc[-1]:.1f}% en `>60h`): quien supera la
           fricción inicial tiende a recomendar el juego.

        2. **La queja cambia con las horas.** En reseñas negativas tempranas
           dominan controles y navegación; en bandas tardías crecen los muros
           de jefes ({crosstab_pct.loc['boss_wall'].iloc[-1]:.0f}% de las
           negativas `>60h`) y el *pacing/bloat*
           ({crosstab_pct.loc['pacing_bloat'].iloc[-1]:.0f}%): el jugador
           comprometido no abandona por confusión, sino por desgaste.

        3. **El valor del merge.** Unir las reseñas con los metadatos de
           Kaggle (llave `appid`) permitió contextualizar el sentimiento con
           precio, género y puntuaciones, y contrastar percepción de la
           crítica (Metacritic) contra fricción reportada por jugadores.

        ### Limitaciones

        - El etiquetado por palabras clave es una aproximación: una reseña que
          dice *"the bosses are hard but fair"* cuenta como mención de
          `boss_wall`; por eso el análisis se restringe a reseñas negativas.
        - Las bandas altas tienen pocos casos negativos (denominadores
          pequeños), así que sus porcentajes son menos estables.
        - La muestra de 500 reseñas por juego iguala pesos entre títulos, pero
          no es proporcional a la población real de reseñas.
        - El playtime al momento de la reseña no equivale a abandono — es un
          *proxy* del punto de fricción.
        """
    )

st.sidebar.markdown("---")
st.sidebar.caption(f"Total de reseñas cargadas: {len(df):,}")
