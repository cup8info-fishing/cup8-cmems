# Nuovi layer meteo cap8 — spec di integrazione app (pipeline → mappa)

> La pipeline CMEMS ora produce 3 nuovi prodotti "snapshot giornaliero" e li pubblica sul
> CDN. Questo documento dice all'app COME consumarli (URL, schema meta, rendering) e le
> **didascalie "come si legge"** da mostrare. La parte pipeline è fatta e testata il 12/06.

## CDN — base e percorsi

Base (come gli altri layer): `https://cup8info-fishing.github.io/cup8-cmems`
(in `src/features/meteo/cmems.ts` → `CMEMS_BASE`).

| Layer | meta | PNG (full / eroded) |
|---|---|---|
| **SST satellitare** (Mare° HD) | `/sst-uhr/meta.json` | `/sst-uhr/full/current.png`, `/sst-uhr/eroded/current.png` |
| **Clorofilla** | `/chl/meta.json` | `/chl/full/current.png`, `/chl/eroded/current.png` |
| **Termoclino** (temp in profondità) | `/temp3d/meta.json` | `/temp3d/full/d00.png … dNN.png` (+ `_eroded`) |

`full` = per basemap "Mappa" (terra opaca), `eroded` = per "Satellite/Nautica" (terra ritagliata) — **identico a onde/SST**.

## Schema meta.json

Snapshot singolo (sst-uhr, chl):
```json
{
  "bbox": { "lat_min": 30.0, "lat_max": 46.0, "lng_min": -6.0, "lng_max": 36.0 },
  "width": 4000, "height": 1920,
  "dataset": "…", "time": "2026-06-11T00:00:00",
  "legend": { "kind": "thermal", "unit": "°C", "min": 18, "max": 26 },
  "levels_count": 1, "render_ts": 1781… 
}
```
Termoclino (`/temp3d/meta.json`): come sopra + `"depths": [5.5, 10.5, 16.3, 19.4, 29.9, 42.1, 51.4]` e `"levels_count": 7`. Il file `d{NN}.png` corrisponde a `depths[NN]` (stesso ordine).

`legend.kind`:
- `"thermal"` (sst-uhr, temp3d): `min`/`max` in °C → riusa la rampa termica già esistente per SST.
- `"chl_log"` (chl): `min`/`max` in mg/m³ + `ticks: number[]` (bande in scala log) → palette blu→verde→giallo. **La scala è logaritmica**, non lineare.

## Come renderizzarli nell'app

**SST-UHR e Clorofilla** = `SstPngOverlay.tsx` semplificato: 1 solo `L.imageOverlay` (niente serie ore, niente slider tempo). bbox dal meta, cache-bust `?v=render_ts`, `eroded` dal basemap, `useSeaClip` per il ritaglio costa. Sono mappe del giorno: lo slider temporale NON le muove (vedi nota più sotto).

**Termoclino** = riusa ESATTAMENTE il pattern "monta N overlay con opacity 0, toggle istantaneo": qui i N overlay sono le **profondità** (`d00..dNN`), e lo slider — per questo layer — diventa uno **slider di profondità** (5,5 → 51 m) invece che di ore. La legenda mostra la profondità attiva (`depths[idx]` m).

Nota slider: SST-UHR/chl sono snapshot → quando attivi, disabilita/ignora lo slider ore per quel layer (mostra "dato di oggi"). Per il termoclino, lo slider pilota la profondità. Gli altri layer (onde/correnti/vento) restano sul forecast orario: nessun conflitto, basta che il layer attivo decida cosa fa lo slider.

## Didascalie "come si legge" (testi pronti — richiesta esplicita di Giuseppe)

- **SST satellitare (Mare° HD)**:
  «Temperatura reale della superficie del mare, misurata da satellite a ~0,9 km. Dal blu
  (freddo) al rosso (caldo). Cerca i **fronti**: dove il colore cambia in fretta, acqua
  calda e fredda si incontrano e lì si concentra il pesce (tonni, lampughe, ricciole).
  Si aggiorna una volta al giorno.»

- **Clorofilla (plancton)**:
  «Quanta vita c'è nell'acqua. **Blu** = acqua povera e limpida; **verde/giallo** = acqua
  ricca di plancton. Il pesce foraggio sta nel verde, i predatori cacciano sui **bordi**
  tra blu e verde. Aggiornata ogni giorno.»

- **Termoclino (temperatura in profondità)**:
  «Temperatura del mare alla profondità scelta: sposta lo slider per scendere (5 → 50 m).
  Dove, scendendo, la temperatura crolla di colpo c'è il **termoclino**: spesso i pesci si
  fermano lì. Confronta le profondità per trovare il salto termico.»

## Note
- Tutti e 3 sono **best-effort** nella pipeline: se un giorno un download fallisce, il
  meta resta quello del giorno prima (il `render_ts` lo segnala). L'app può mostrare la
  data da `meta.time`.
- Sorgenti: SST-UHR `SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_c_V2`; chl `cmems_mod_med_bgc-pft_anfc_4.2km_P1D-m`; termoclino `cmems_mod_med_phy-tem_anfc_4.2km_P1D-m`. Tutte CMEMS, gratuite.
- File pipeline nuovi: `download_daily.py`, `render_daily_png.py`, e gli step in `run_pipeline.py` (+ timeout workflow 110 min). Da committare nel repo `cup8-cmems` (cartella `cmems-service/`) quando Giuseppe dà l'ok.
