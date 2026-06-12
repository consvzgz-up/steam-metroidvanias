# =============================================================================
# Proyecto Final — ¿Por qué abandonamos los Metroidvanias?
# App de Streamlit que despliega el pipeline del notebook de Colab
# (ProyectoFinal_Constantino.ipynb) sin alterar su lógica ni resultados.
# =============================================================================

import os
import re
from collections import Counter
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Configuración general
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="¿Por qué abandonamos los Metroidvanias?",
    page_icon="🗺️",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Estilo global — tema oscuro minimalista (paleta inspirada en dashboards
# tipo Frame.io: fondo grafito, tarjetas con borde sutil, acentos saturados)
# -----------------------------------------------------------------------------
ACCENT = {
    'blue':   '#4FC1FF',
    'orange': '#FFA13D',
    'pink':   '#F2589C',
    'purple': '#9D5CF5',
    'red':    '#FF5C5C',
    'green':  '#39D98A',
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], [data-testid="stAppViewContainer"] * {
    font-family: 'Inter', 'Source Sans Pro', sans-serif;
}
h1, h2, h3 { letter-spacing: -0.02em; font-weight: 700; }

/* Tarjetas de métricas */
[data-testid="stMetric"] {
    background: #1C1F26;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 16px 18px 12px 18px;
}
[data-testid="stMetricLabel"] p {
    color: #9AA0AB;
    font-size: 0.82rem;
    font-weight: 500;
}
[data-testid="stMetricValue"] {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
}
[data-testid="stMetricDelta"] { font-size: 0.85rem; font-weight: 600; }

/* Pestañas minimalistas */
[data-baseweb="tab-list"] { gap: 4px; }
[data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 6px 14px;
}

/* Contenedor principal más compacto */
[data-testid="stAppViewBlockContainer"],
.block-container { padding-top: 2.2rem; }

/* Separadores sutiles */
hr { border-color: rgba(255, 255, 255, 0.08); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

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
    # Con el filtro de juegos una banda puede quedar vacía: se reindexa a las
    # 5 bandas para que heatmap y animación siempre tengan ejes completos.
    crosstab = (crosstab.reindex(columns=PLAYTIME_LABELS)
                .fillna(0).astype(int))
    band_totals = band_totals.reindex(PLAYTIME_LABELS).fillna(0).astype(int)
    crosstab_pct = (crosstab
                    .div(band_totals.replace(0, np.nan), axis=1) * 100).round(1)
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
# Helpers de Plotly — estilo compartido por todos los gráficos interactivos
# -----------------------------------------------------------------------------
def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'


def style_fig(fig, height=440, title=None):
    """Aplica el tema oscuro minimalista común a una figura de Plotly."""
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, Source Sans Pro, sans-serif',
                  size=12, color='#C9CDD6'),
        margin=dict(l=10, r=10, t=56 if title else 28, b=10),
        height=height,
        hoverlabel=dict(bgcolor='#23262E', bordercolor='rgba(255,255,255,0.15)',
                        font_size=12),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0, xanchor='left',
                                     font=dict(size=15, color='#E8EAED')))
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.06)', zeroline=False)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.06)', zeroline=False)
    return fig


def mini_cumulative_chart(series, color, kind, hover_label, date_fmt='%b %Y'):
    """Mini gráfico acumulado para las tarjetas de métricas (estilo
    'Cumulative Stats'): barras, línea o área según el selector."""
    x = series.index
    y = series.values
    hover = ('%{x|' + date_fmt + '}: %{y:,.0f} ' + hover_label
             + '<extra></extra>')
    if kind == 'Barras':
        trace = go.Bar(x=x, y=y, marker=dict(color=color, cornerradius=2),
                       hovertemplate=hover)
    elif kind == 'Área':
        trace = go.Scatter(x=x, y=y, mode='lines',
                           line=dict(color=color, width=2),
                           fill='tozeroy', fillcolor=hex_to_rgba(color, 0.25),
                           hovertemplate=hover)
    else:  # Línea
        trace = go.Scatter(x=x, y=y, mode='lines',
                           line=dict(color=color, width=2.4),
                           hovertemplate=hover)
    fig = go.Figure(trace)
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=180,
        margin=dict(l=0, r=0, t=8, b=0),
        showlegend=False,
        hoverlabel=dict(bgcolor='#23262E', font_size=11),
        bargap=0.25,
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=9, color='#9AA0AB'),
                     zeroline=False)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=9, color='#9AA0AB'),
                     zeroline=False, nticks=4)
    return fig


def gaussian_kde_curve(values, grid):
    """KDE gaussiana con regla de Scott (equivalente a sns.kdeplot) en NumPy
    puro — evita depender de scipy solo para este gráfico."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    bandwidth = values.std(ddof=1) * n ** (-1 / 5)
    if bandwidth == 0 or n < 2:
        return np.zeros_like(grid)
    diffs = (grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * diffs ** 2).sum(axis=1) / (n * bandwidth * np.sqrt(2 * np.pi))


# -----------------------------------------------------------------------------
# Visualizaciones (Fase 3 del notebook) — misma lógica, ahora interactivas
# con Plotly (tooltips, zoom y paneo). Las versiones Matplotlib/Seaborn del
# rúbrico viven en el notebook de Colab enlazado en Metodología.
# -----------------------------------------------------------------------------
def chart_playtime_histogram(df):
    fig = go.Figure()
    for rec, label, color in [
        (False, 'No recomendado', ACCENT['red']),
        (True,  'Recomendado',    ACCENT['blue']),
    ]:
        data = df[df['recommended'] == rec]['playtime_at_review_hrs']
        fig.add_trace(go.Histogram(
            x=data[data <= 100], nbinsx=40, name=label,
            marker_color=color, opacity=0.72,
            hovertemplate='%{x} h → %{y} reseñas<extra>' + label + '</extra>',
        ))
        fig.add_vline(
            x=data.median(), line_dash='dash', line_color=color, line_width=1.4,
            annotation_text=f'mediana: {data.median():.1f}h',
            annotation_font=dict(size=10, color=color),
            annotation_position='top right' if rec else 'top left',
        )
    fig.update_layout(
        barmode='overlay',
        xaxis_title='Horas jugadas al reseñar (≤100h)',
        yaxis_title='Número de reseñas',
        legend=dict(orientation='h', y=1.08),
    )
    return style_fig(fig, title='Distribución de horas jugadas — recomendado vs. no recomendado')


def chart_negative_rate_by_band(df):
    band_stats = df.groupby('playtime_band', observed=True).agg(
        total=('recommended', 'count'),
        negative=('recommended', lambda x: (~x).sum())
    ).assign(neg_rate=lambda d: (d.negative / d.total * 100).round(1))

    fig = go.Figure(go.Bar(
        x=band_stats.index.astype(str),
        y=band_stats['neg_rate'],
        marker=dict(
            color=[ACCENT['red'], ACCENT['orange'], '#FFC34D',
                   ACCENT['green'], ACCENT['blue']],
            cornerradius=6,
        ),
        text=[f'{v}%' for v in band_stats['neg_rate']],
        textposition='outside',
        textfont=dict(size=13, color='#E8EAED'),
        customdata=band_stats[['negative', 'total']].values,
        hovertemplate=('<b>%{x}</b><br>%{y}% negativas<br>'
                       '%{customdata[0]} de %{customdata[1]} reseñas'
                       '<extra></extra>'),
    ))
    fig.update_layout(
        xaxis_title='Horas jugadas al reseñar',
        yaxis_title='Tasa de reseñas negativas (%)',
        yaxis_range=[0, max(55, band_stats['neg_rate'].max() + 8)],
    )
    return style_fig(fig, title='Tasa de reseñas negativas por banda de horas')


CUMULATIVE_GAMES = [
    'Hollow Knight: Silksong',
    'Ender Magnolia',
    'Record of Lodoss War',
    'Guacamelee!',
    'Hollow Knight',
    'Blasphemous',
]


def chart_abandonment_curves(df):
    fig = go.Figure()
    colors_line = [ACCENT['red'], ACCENT['orange'], ACCENT['blue'],
                   '#5BBDD9', ACCENT['green'], ACCENT['purple']]
    for game, color in zip(CUMULATIVE_GAMES, colors_line):
        game_df = df[df['game_name'] == game].copy()
        if game_df.empty:
            continue
        game_df = game_df.sort_values('playtime_at_review_hrs')
        game_df['cumulative_pct'] = (np.arange(1, len(game_df) + 1) / len(game_df)) * 100
        fig.add_trace(go.Scatter(
            x=game_df['playtime_at_review_hrs'].clip(upper=80),
            y=game_df['cumulative_pct'],
            name=game, mode='lines',
            line=dict(color=color, width=2),
            hovertemplate='%{x:.1f} h → %{y:.0f}% de reseñas<extra>' + game + '</extra>',
        ))
    for x, lbl in [(5, '<5h'), (15, '15h'), (30, '30h')]:
        fig.add_vline(x=x, line_dash='dot', line_color='rgba(255,255,255,0.25)',
                      annotation_text=lbl,
                      annotation_font=dict(size=10, color='#9AA0AB'),
                      annotation_position='bottom right')
    fig.update_layout(
        xaxis_title='Horas jugadas al reseñar (tope 80h)',
        yaxis_title='% acumulado de reseñas',
        legend=dict(orientation='h', y=1.1),
    )
    return style_fig(
        fig, height=480,
        title='Acumulación de reseñas según horas jugadas '
              '(subida temprana = más reseñas de abandono)')


def chart_boxplot_sentiment(df):
    data = df[df['playtime_at_review_hrs'] <= 120].copy()
    data['sentimiento'] = np.where(data['recommended'],
                                   'Recomendado', 'No recomendado')
    fig = px.box(
        data, x='sentimiento', y='playtime_at_review_hrs',
        color='sentimiento',
        color_discrete_map={'Recomendado': ACCENT['blue'],
                            'No recomendado': ACCENT['red']},
        category_orders={'sentimiento': ['No recomendado', 'Recomendado']},
        points='outliers',
    )
    fig.update_traces(marker=dict(size=3, opacity=0.35), width=0.45,
                      showlegend=False)
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Horas jugadas al reseñar',
    )
    return style_fig(fig, height=480,
                     title='Horas jugadas según sentimiento (tope 120h)')


def chart_friction_heatmap(crosstab_pct, band_totals):
    # Denominadores visibles: N de reseñas negativas por banda
    x_labels = [f'{band}<br><span style="font-size:10px">N={band_totals[band]}</span>'
                for band in crosstab_pct.columns]
    y_labels = [TAG_LABELS[t] for t in crosstab_pct.index]
    fig = px.imshow(
        crosstab_pct.values,
        x=x_labels, y=y_labels,
        text_auto='.1f',
        color_continuous_scale='YlOrRd',
        aspect='auto',
    )
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>%{x}<br>%{z:.1f}% de las negativas'
                      '<extra></extra>')
    fig.update_layout(
        xaxis_title='Horas jugadas al reseñar',
        coloraxis_colorbar=dict(title='% negativas<br>de la banda'),
    )
    fig.update_xaxes(side='bottom', gridcolor='rgba(0,0,0,0)')
    fig.update_yaxes(gridcolor='rgba(0,0,0,0)')
    return style_fig(
        fig, height=460,
        title='Temas de fricción × banda de horas '
              '(% de reseñas negativas en inglés que mencionan cada tema)')


KDE_GAMES = [
    'Record of Lodoss War',
    'Guacamelee!',
    'Animal Well',
    'Hollow Knight',
    'Nine Sols',
    'Hollow Knight: Silksong',
]


def chart_kde_playtime(df):
    palette = [ACCENT['red'], ACCENT['orange'], '#E8C45C',
               ACCENT['blue'], '#5BBDD9', ACCENT['purple']]
    grid = np.linspace(0, 100, 400)
    fig = go.Figure()
    for game, color in zip(KDE_GAMES, palette):
        game_data = df[df['game_name'] == game]['playtime_at_review_hrs']
        game_data = game_data[game_data <= 100]
        if len(game_data) < 2:
            continue
        density = gaussian_kde_curve(game_data, grid)
        fig.add_trace(go.Scatter(
            x=grid, y=density, name=game, mode='lines',
            line=dict(color=color, width=2),
            fill='tozeroy', fillcolor=hex_to_rgba(color, 0.08),
            hovertemplate='%{x:.0f} h — densidad %{y:.4f}<extra>' + game + '</extra>',
        ))
    fig.update_layout(
        xaxis_title='Horas jugadas al reseñar',
        yaxis_title='Densidad',
        legend=dict(orientation='h', y=1.1),
        hovermode='x unified',
    )
    return style_fig(fig, height=460,
                     title='Forma de la distribución de horas por juego '
                           '(KDE — tope 100h)')


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


def chart_friction_animated(crosstab):
    """Versión interactiva de la animación: barras de quejas con botón ▶
    que recorre las bandas de horas (frames de Plotly)."""
    anim_data = crosstab.T.copy()
    anim_data.index = ['<5h', '5–15h', '15–30h', '30–60h', '>60h']
    tag_cols = list(FRICTION_TAGS.keys())
    long_df = (
        anim_data[tag_cols]
        .reset_index(names='banda')
        .melt(id_vars='banda', var_name='tag', value_name='quejas')
    )
    long_df['Tema'] = long_df['tag'].map(TAG_LABELS_SHORT)

    bar_colors = [ACCENT['red'], ACCENT['orange'], '#E8C45C',
                  ACCENT['blue'], '#5BBDD9', ACCENT['purple']]
    fig = px.bar(
        long_df, x='quejas', y='Tema', color='Tema',
        animation_frame='banda', orientation='h',
        color_discrete_sequence=bar_colors,
        range_x=[0, long_df['quejas'].max() * 1.15],
        text='quejas',
    )
    fig.update_traces(
        textposition='outside',
        hovertemplate='<b>%{y}</b>: %{x} reseñas negativas<extra></extra>')
    fig.update_layout(
        showlegend=False,
        xaxis_title='Reseñas negativas que mencionan el tema',
        yaxis_title='',
    )
    # Ritmo de la animación (ms por frame) y transición suave entre bandas
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1100
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500
    fig.layout.sliders[0].currentvalue['prefix'] = 'Banda de horas: '
    return style_fig(fig, height=480,
                     title='Evolución de las quejas de fricción conforme avanza la partida')


def build_choropleth(df):
    """Mapa coroplético interactivo de reseñas negativas por idioma (cell 31
    del notebook usa GeoPandas; aquí Plotly trae las geometrías integradas,
    sin descargar GeoJSON, con zoom y tooltips)."""
    neg_df = df[df['recommended'] == False]  # noqa: E712
    neg_lang = neg_df.copy()
    neg_lang['iso'] = neg_lang['language'].map(LANG_TO_ISO)
    neg_lang = neg_lang.dropna(subset=['iso'])
    lang_counts = (neg_lang.groupby(['iso', 'language']).size()
                   .reset_index(name='neg_reviews'))

    fig = px.choropleth(
        lang_counts,
        locations='iso',
        color='neg_reviews',
        hover_name='language',
        color_continuous_scale='YlOrRd',
        labels={'neg_reviews': 'Reseñas negativas'},
    )
    fig.update_traces(
        marker_line_color='rgba(255,255,255,0.2)', marker_line_width=0.4,
        hovertemplate='<b>%{hovertext}</b> → %{location}<br>'
                      '%{z} reseñas negativas<extra></extra>')
    fig.update_geos(
        bgcolor='rgba(0,0,0,0)',
        showframe=False,
        showcoastlines=False,
        landcolor='#23262E',
        oceancolor='rgba(0,0,0,0)',
        showocean=True,
        projection_type='natural earth',
    )
    fig.update_layout(coloraxis_colorbar=dict(title='Negativas', len=0.7))
    return style_fig(
        fig, height=520,
        title='Distribución geográfica de reseñas negativas '
              '(idioma → país representativo, aproximación)')


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

    lim = max(game_scatter['global_neg_pct'].max(),
              game_scatter['sample_neg_rate'].max()) * 1.1

    fig = px.scatter(
        game_scatter,
        x='global_neg_pct', y='sample_neg_rate',
        color='price', color_continuous_scale='viridis',
        text='game_name',
        labels={'price': 'Precio (USD)'},
        custom_data=['game_name', 'price'],
    )
    fig.update_traces(
        marker=dict(size=14, line=dict(color='rgba(255,255,255,0.7)', width=1)),
        textposition='top center',
        textfont=dict(size=9, color='#9AA0AB'),
        hovertemplate=('<b>%{customdata[0]}</b><br>'
                       'Global (games.csv): %{x:.1f}% negativas<br>'
                       'Muestra (API): %{y:.1f}% negativas<br>'
                       'Precio: $%{customdata[1]:.2f}<extra></extra>'),
    )
    fig.add_shape(type='line', x0=0, y0=0, x1=lim, y1=lim,
                  line=dict(color='rgba(255,255,255,0.3)', dash='dash', width=1))
    fig.add_annotation(x=lim * 0.70, y=lim * 0.86, showarrow=False,
                       text='reseñas recientes<br>peores que la historia',
                       font=dict(size=10, color='#9AA0AB'))
    fig.add_annotation(x=lim * 0.86, y=lim * 0.55, showarrow=False,
                       text='reseñas recientes<br>mejores que la historia',
                       font=dict(size=10, color='#9AA0AB'))
    fig.update_layout(
        xaxis_title='% votos negativos globales (games.csv, todo el historial)',
        yaxis_title='% reseñas negativas en la muestra (API, recientes)',
        coloraxis_colorbar=dict(title='Precio<br>(USD)'),
    )
    return style_fig(fig, height=560,
                     title='Validación cruzada: % negativo global vs. muestra '
                           '(color = precio, tres columnas del merge)')


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
        background_color='#14161B', colormap='Reds',
        stopwords=final_stop_words, max_words=100,
    ).generate(neg_text)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_alpha(0)  # se integra con el fondo oscuro de la app
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Conceptos Dominantes en Reseñas Negativas',
                 fontsize=16, color='#E8EAED')
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

# Filtro global de juegos: alimenta todo el Dashboard en vivo.
# Vacío = los 24 juegos (evita llenar la barra lateral de chips).
ALL_GAMES = sorted(df['game_name'].unique())
selected_games = st.sidebar.multiselect(
    "🎮 Filtrar juegos",
    options=ALL_GAMES,
    default=[],
    placeholder="Todos los juegos",
    help="Filtra todas las gráficas del Dashboard. Vacío = todos.",
)
df_dash = df[df['game_name'].isin(selected_games)] if selected_games else df

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
    st.dataframe(
        games_table, width='stretch',
        column_config={
            'pct_negativas': st.column_config.ProgressColumn(
                '% negativas', format='%.1f%%',
                min_value=0, max_value=float(games_table['pct_negativas'].max()),
            ),
            'mediana_horas': st.column_config.NumberColumn(
                'mediana horas', format='%.1f h'),
        })

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
    if selected_games:
        st.caption("Filtrado a " + ", ".join(selected_games))

    # -------------------------------------------------------------------------
    # Estadísticas acumuladas — encabezado interactivo estilo "Cumulative Stats"
    # -------------------------------------------------------------------------
    dates = df_dash['date_created'].dt.date
    min_d, max_d = dates.min(), dates.max()

    c1, c2, c3, c4 = st.columns(4)
    start_date = c1.date_input("Fecha inicial", value=min_d,
                               min_value=min_d, max_value=max_d)
    end_date = c2.date_input("Fecha final", value=max_d,
                             min_value=min_d, max_value=max_d)
    freq_label = c3.selectbox("Granularidad", ["Mensual", "Semanal", "Diaria"])
    chart_kind = c4.selectbox("Tipo de gráfico", ["Barras", "Área", "Línea"])

    if start_date > end_date:
        st.warning("La fecha inicial es posterior a la final.")
    else:
        in_window = (dates >= start_date) & (dates <= end_date)
        window = df_dash[in_window]

        # Periodo anterior de igual duración → deltas de las métricas
        span = end_date - start_date
        prev_start = start_date - span - timedelta(days=1)
        prev_end = start_date - timedelta(days=1)
        prev = df_dash[(dates >= prev_start) & (dates <= prev_end)]

        freq = {"Mensual": "MS", "Semanal": "W", "Diaria": "D"}[freq_label]
        date_fmt = '%b %Y' if freq_label == "Mensual" else '%d %b %Y'
        wi = window.set_index('date_created').sort_index()
        cum_reviews = wi['review_id'].resample(freq).count().cumsum()
        cum_negatives = (~wi['recommended']).resample(freq).sum().cumsum()
        cum_hours = wi['playtime_at_review_hrs'].resample(freq).sum().cumsum()
        cum_friction = (wi[list(FRICTION_TAGS)].any(axis=1)
                        .resample(freq).sum().cumsum())

        cards = [
            ("Reseñas", len(window), len(prev),
             'normal', ACCENT['blue'], cum_reviews, 'reseñas'),
            ("Reseñas negativas",
             int((~window['recommended']).sum()),
             int((~prev['recommended']).sum()),
             'inverse', ACCENT['orange'], cum_negatives, 'negativas'),
            ("Horas jugadas",
             window['playtime_at_review_hrs'].sum(),
             prev['playtime_at_review_hrs'].sum(),
             'normal', ACCENT['pink'], cum_hours, 'horas'),
            ("Menciones de fricción",
             int(window[list(FRICTION_TAGS)].any(axis=1).sum()),
             int(prev[list(FRICTION_TAGS)].any(axis=1).sum()),
             'inverse', ACCENT['purple'], cum_friction, 'menciones'),
        ]
        for col, (label, val, prev_val, dcolor, color, series, unit) in zip(
                st.columns(4), cards):
            with col:
                # Sin periodo previo comparable (rango completo) no hay delta
                st.metric(
                    label, f"{val:,.0f}",
                    delta=f"{val - prev_val:+,.0f}" if len(prev) else None,
                    delta_color=dcolor,
                    help=(f"Cambio vs. el periodo anterior de igual duración "
                          f"({prev_start:%d %b %Y} – {prev_end:%d %b %Y})"),
                )
                st.plotly_chart(
                    mini_cumulative_chart(series, color, chart_kind,
                                          unit, date_fmt),
                    width='stretch', key=f"mini_{unit}")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Gráficas interactivas — todas reaccionan al filtro de juegos del sidebar
    # -------------------------------------------------------------------------
    crosstab_f, crosstab_pct_f, band_totals_f = friction_crosstabs(df_dash)

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
        st.plotly_chart(chart_playtime_histogram(df_dash), width='stretch')

    with tabs[1]:
        st.markdown("**Tasa de reseñas negativas por banda de horas** — el "
                    "patrón de abandono en su forma más directa.")
        st.plotly_chart(chart_negative_rate_by_band(df_dash), width='stretch')

    with tabs[2]:
        st.markdown("**Acumulación de reseñas según horas jugadas** — una "
                    "subida temprana pronunciada implica más jugadores que "
                    "reseñan (y abandonan) pronto.")
        st.plotly_chart(chart_abandonment_curves(df_dash), width='stretch')

    with tabs[3]:
        st.markdown("**Horas jugadas según sentimiento** — mediana, dispersión "
                    "y outliers en un solo gráfico interactivo.")
        st.plotly_chart(chart_boxplot_sentiment(df_dash), width='stretch')

    with tabs[4]:
        st.markdown("**Temas de fricción × banda de horas** — % de reseñas "
                    "negativas *en inglés* de cada banda que mencionan cada "
                    "tema (las regex de fricción son en inglés).")
        st.plotly_chart(chart_friction_heatmap(crosstab_pct_f, band_totals_f),
                        width='stretch')

    with tabs[5]:
        st.markdown("**Forma de la distribución de horas por juego** (KDE) — "
                    "juegos con perfiles de abandono contrastantes.")
        st.plotly_chart(chart_kde_playtime(df_dash), width='stretch')

    with tabs[6]:
        st.markdown("**Animación interactiva**: pulsa ▶ para recorrer cómo "
                    "evolucionan las quejas de fricción conforme avanza la "
                    "partida. La versión Matplotlib (FuncAnimation → GIF) del "
                    "rúbrico está en el notebook y en el desplegable.")
        st.plotly_chart(chart_friction_animated(crosstab_f), width='stretch')
        with st.expander("Ver GIF generado con Matplotlib (FuncAnimation)"):
            st.image(build_animation_gif(crosstab))

    with tabs[7]:
        st.markdown("**Mapa coroplético interactivo** — reseñas negativas por "
                    "idioma, asignadas a un país representativo (la versión "
                    "GeoPandas del rúbrico vive en el notebook).")
        st.plotly_chart(build_choropleth(df_dash), width='stretch')

    with tabs[8]:
        st.markdown("**Validación cruzada de la muestra (el valor del merge)** "
                    "— % de votos negativos en TODO el historial del juego "
                    "(games.csv) vs. % negativo en nuestra muestra reciente "
                    "(API). La diagonal = acuerdo perfecto; arriba de la línea "
                    "= sentimiento reciente peor que el histórico. "
                    "Color = precio. Tres columnas del merge en un solo gráfico.")
        st.plotly_chart(chart_scatter_merge_validation(df_dash), width='stretch')

    with tabs[9]:
        st.markdown("**Análisis de texto en reseñas negativas en inglés** — "
                    "frases más repetidas (trigramas, NLTK) y nube de palabras.")
        eng_neg = df_dash[(df_dash['recommended'] == False)  # noqa: E712
                          & (df_dash['language'] == 'english')]
        if eng_neg.empty:
            st.info("No hay reseñas negativas en inglés con el filtro actual.")
        else:
            col1, col2 = st.columns([1, 2])
            with col1:
                phrases = negative_phrases(df_dash, n=3, top_k=25)
                st.dataframe(
                    phrases, width='stretch', hide_index=True,
                    column_config={
                        'Apariciones': st.column_config.ProgressColumn(
                            'Apariciones', format='%d', min_value=0,
                            max_value=int(phrases['Apariciones'].max()),
                        ),
                    })
            with col2:
                st.pyplot(build_wordcloud(df_dash))

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
