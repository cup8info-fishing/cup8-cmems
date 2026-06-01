# CMEMS onde — deploy GRATIS (GitHub Actions + Pages), niente PC acceso

Obiettivo: il render onde gira ogni 6h su **GitHub Actions** e pubblica i file statici su
**GitHub Pages** (CDN gratis). L'app legge da lì → **il tuo PC non serve più acceso**, **0 €**, **nessuna carta**.

Già pronto in questa cartella: `requirements.txt`, `run_pipeline.py`, `.github/workflows/cmems-render.yml`,
`med-land-merged.geojson`, `.gitignore` + fix di portabilità (PORT, path geojson).

## Passi che devi fare TU (una volta)

### 0. 🔴 Ruota la password Copernicus (sicurezza)
La vecchia password era salvata in chiaro sul PC. Vai su https://data.marine.copernicus.eu → account → **cambia password**. Userai la nuova nei secret qui sotto.

### 1. Crea il repo GitHub (root = questa cartella `cmems-service`)
Nel tuo terminale, dentro `cup8all\cmems-service`:
```
git init
git add .
git commit -m "CMEMS render pipeline (GitHub Actions → Pages)"
gh repo create cup8info-fishing/cup8-cmems --private --source=. --push
```
*(Se non hai `gh`: crea il repo a mano su github.com e fai `git remote add origin … && git push -u origin main`.)*
> Il `.gitignore` esclude già `data/`, `cache/`, `public/`, `*.nc` → niente file pesanti committati.

### 2. Aggiungi i Secret (repo → Settings → Secrets and variables → Actions)
- `COPERNICUSMARINE_SERVICE_USERNAME` = il tuo user Copernicus
- `COPERNICUSMARINE_SERVICE_PASSWORD` = la **nuova** password (punto 0)

### 3. Abilita Pages (repo → Settings → Pages)
- **Source: GitHub Actions**

### 4. Lancia il workflow a mano per testare (repo → Actions → "CMEMS render → Pages" → Run workflow)
Al termine, nell'environment `github-pages` trovi l'**URL** (tipo
`https://cup8info-fishing.github.io/cup8-cmems/`). I dati saranno a `…/waves/meta.json`.

## Poi: collega l'app (DOPO che Pages è live) e ricompila l'APK
I file statici hanno questo layout (diverso dagli endpoint vivi → serve un piccolo ritocco all'app):
| Prima (API viva) | Dopo (statico su Pages) |
|---|---|
| `/api/cmems/waves-png/meta` | `<PAGES>/waves/meta.json` |
| `/api/cmems/waves-png/{h}` (full) | `<PAGES>/waves/full/h{NN}.png` |
| `/api/cmems/waves-png/{h}?eroded=1` | `<PAGES>/waves/eroded/h{NN}.png` |
| `/api/cmems/forecast?hours=72&step=0.2` | `<PAGES>/waves/forecast_72h_0p2.json` |

Modifiche app (in `Cup8 - Copia/src/features/meteo/`):
1. `.env.production` → `VITE_CMEMS_URL=<PAGES base>` (es. `https://cup8info-fishing.github.io/cup8-cmems`).
2. In `WavePngOverlay.tsx` / `MeteoSection.tsx` / dove si fetch-a il forecast: cambiare i 3 path come da tabella (eroded da query-param → sottocartella `eroded/`, hour zero-padded `hNN.png`, forecast nome fisso). Il **versioning/cache-bust** resta in query (`?v=<render_ts da meta>`), non nel nome file.
3. `npm run build && npx cap sync android && (cd android && gradlew assembleDebug)` → nuovo APK.

> Posso farti io queste 3 modifiche all'app quando l'URL Pages è attivo (così non rompo il wave map nel frattempo).

## Note / limiti del piano gratis
- I **workflow schedulati si fermano dopo 60 gg** di inattività del repo → un commit li riattiva.
- Il cron GitHub può **ritardare** di 10-30 min (irrilevante a 6h).
- GitHub Pages: limiti **soft** (~100 GB banda/mese, ~1 GB sito) → ottimo per beta/amici. Quando l'app **parte davvero** → migra a **Cloud Run + Firebase Hosting/CDN** (vedi `ARCHITETTURA-PRODUZIONE-cup8.md`).
- Test locale del render (facoltativo): `pip install -r requirements.txt` poi `python run_pipeline.py` con le env Copernicus settate.
