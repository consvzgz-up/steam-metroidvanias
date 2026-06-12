# ¿Por qué abandonamos los Metroidvanias? — Streamlit App

Despliegue en Streamlit del análisis del notebook `ProyectoFinal_Constantino.ipynb`
(reseñas de Steam de 24 metroidvanias + metadatos de Kaggle).

## Ejecutar localmente

```powershell
# 1. Crear entorno e instalar dependencias (una sola vez)
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 2. Lanzar la app
.venv\Scripts\streamlit run streamlit_app.py
```

## Archivos de datos

- `*_reviews.csv` — reseñas por juego (obligatorios).
- `games_meta_subset.csv` — extracto de metadatos ya generado (23 juegos).
- `games.csv` (389 MB) — **opcional**: solo se usa para regenerar el extracto
  si `games_meta_subset.csv` no existe. **No lo subas al deploy.**

## Despliegue en Streamlit Community Cloud

Sube a GitHub: `streamlit_app.py`, `requirements.txt`, los 24 `*_reviews.csv`
y `games_meta_subset.csv` (NO subir `games.csv` ni `.venv`). Luego en
share.streamlit.io apunta a `streamlit_app.py`.

## Paridad con el notebook

La app replica el pipeline del notebook (versión corregida tras la auditoría
2026-06-09): muestreo en memoria de hasta 500 reseñas/juego (`random_state=42`),
deduplicación por `review_id`, merge con `appid` como llave y las mismas 8+
visualizaciones. Los datos se re-descargaron con el scraper corregido
(`fetch_metroidvania_reviews.py`: reseñas únicas, todos los idiomas,
`filter=recent`). Los CSV del scrape original (con duplicados) están en
`old_scrape_backup/`. Detalle de hallazgos y fixes en `AUDIT_NOTEBOOK.md`.
