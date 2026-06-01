"""
render_waves_png.py — Renderizza PNG (stile LaMMA) di altezza onde da NetCDF.

Per ogni ora del forecast genera un PNG (~1500×600px, ~80KB) con:
  - Contour fill marching squares (matplotlib.contourf)
  - Scala colori LaMMA discreta
  - Trasparenza dove NaN (terra)
  - Bbox bounds compatibile con Leaflet imageOverlay

Output: cache/waves_h{0..23}.png + cache/waves_meta.json

Uso: python render_waves_png.py [--hours 24] [--width 1500]
"""
import sys
import os
import json
import argparse
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")  # backend non interattivo
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image, ImageDraw

DATA_NC = os.path.join(os.path.dirname(__file__), "data", "forecast_med_waves.nc")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
# Maschera terra: GeoJSON ISTAT 1:50000 rasterizzato per ritagliare l'overlay
# DENTRO la PNG (incisa al pixel, zero clip-path runtime).
LAND_GEOJSON = "C:/Users/giuse/cup8all/Cup8 - Copia/public/med-land-merged.geojson"


def build_land_mask(width, height, x_min_merc, x_max_merc, y_min_merc, y_max_merc):
    """Rasterizza i polygons terra ISTAT in una maschera PIL (L mode).
    Terra = 255, mare = 0. Coords in Web Mercator (radianti) come il contourf."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    try:
        with open(LAND_GEOJSON, "r", encoding="utf-8") as f:
            geo = json.load(f)
    except Exception as e:
        print(f"[render-png] WARN land mask: {e}", file=sys.stderr)
        return None

    def lat_to_y(lat):
        return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))

    def project(lng, lat):
        xm = np.radians(lng)
        ym = lat_to_y(lat)
        px = (xm - x_min_merc) / (x_max_merc - x_min_merc) * width
        py = (y_max_merc - ym) / (y_max_merc - y_min_merc) * height
        return (px, py)

    def draw_ring(ring):
        pts = [project(c[0], c[1]) for c in ring]
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)

    for feat in geo.get("features", []):
        g = feat.get("geometry")
        if not g:
            continue
        if g["type"] == "Polygon":
            for ring in g["coordinates"]:
                draw_ring(ring)
        elif g["type"] == "MultiPolygon":
            for poly in g["coordinates"]:
                for ring in poly:
                    draw_ring(ring)
    return mask

# Scala colori LaMMA (height in m → RGB)
LAMMA_LEVELS = [0.0, 0.1, 0.3, 0.5, 0.8, 1.25, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 12.0]
LAMMA_COLORS_RGB = [
    (50,  255, 90),   # 0-0.1 quasi piatto — VERDE ACCESO (vivido/saturo, non "pisello"):
                      # distinto dal ciano 0.1-0.3 e dal bianco
    (0,   255, 255),  # 0.1-0.3 cyan
    (0,   140, 255),  # 0.3-0.5 azzurro netto (più "blu" del ciano → distinguibile;
                      #         era (0,191,255), troppo simile al ciano sopra)
    (65,  105, 225),  # 0.5-0.8 blu chiaro
    (0,   0,   255),  # 0.8-1.25 blu medio
    (0,   0,   205),  # 1.25-1.6 blu scuro
    (0,   0,   139),  # 1.6-2.0 blu navy
    (75,  0,   130),  # 2.0-2.5 indaco
    (128, 0,   128),  # 2.5-3.0 viola
    (199, 21,  133),  # 3.0-4.0 viola-magenta
    (255, 0,   0),    # 4.0-5.0 rosso
    (255, 69,  0),    # 5.0-6.0 rosso-arancio
    (255, 165, 0),    # 6.0-7.0 arancio
    (255, 215, 0),    # 7.0-8.0 giallo
    (211, 211, 211),  # 8.0-9.0 grigio
    (255, 255, 255),  # 9.0+ bianco (mareggiata estrema)
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=int, default=72, help="Numero ore da renderizzare (1-72)")
    p.add_argument("--width", type=int, default=3000, help="Larghezza PNG in pixel")
    args = p.parse_args()

    if not os.path.exists(DATA_NC):
        print(f"ERROR: NetCDF non trovato: {DATA_NC}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(CACHE_DIR, exist_ok=True)

    ds = xr.open_dataset(DATA_NC)
    total_hours = ds.sizes["time"]
    hours_count = min(args.hours, total_hours)

    lats = ds.latitude.values   # ascending
    lngs = ds.longitude.values  # ascending
    lng_min, lng_max = float(lngs.min()), float(lngs.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())

    # CRITICO: rendiamo la PNG in proiezione **Web Mercator** (EPSG:3857), la stessa
    # usata da Leaflet. Senza questo, Leaflet vede una PNG in lat/lng lineare
    # (EPSG:4326) e la rimappa stirando le coste → i confini delle chiazze non
    # combaciano con la mappa di base (terra OSM/Voyager).
    def lat_to_mercator_y(lat_deg):
        # Web Mercator: y = ln(tan(π/4 + φ/2)), φ in radianti.
        # Unità: radianti (coerenti con lng_rad sotto).
        return np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2))

    # Upsample lats/lngs 4x per matchare data upsampled (vedi sotto in loop)
    lats_us = np.linspace(lats[0], lats[-1], len(lats) * 4)
    lngs_us = np.linspace(lngs[0], lngs[-1], len(lngs) * 4)
    lats_merc = lat_to_mercator_y(lats_us)
    y_min_merc = lat_to_mercator_y(lat_min)
    y_max_merc = lat_to_mercator_y(lat_max)
    lngs_rad = np.radians(lngs_us)
    x_min_merc = np.radians(lng_min)
    x_max_merc = np.radians(lng_max)

    # Aspect ratio Web Mercator (entrambi in radianti, unità omogenee)
    aspect = (x_max_merc - x_min_merc) / (y_max_merc - y_min_merc)
    width = args.width
    height = int(round(width / aspect))

    # Colormap discreta + BoundaryNorm = ogni bucket = colore solido (no interpolazione)
    cmap = ListedColormap([(r / 255, g / 255, b / 255) for r, g, b in LAMMA_COLORS_RGB])
    cmap.set_bad(alpha=0)  # NaN = trasparente
    norm = BoundaryNorm(LAMMA_LEVELS, cmap.N)

    hour_strs = []
    print(f"[render-png] {hours_count} ore × {width}×{height}px, output → {CACHE_DIR}")

    # Cmap "eroded" (per Satellite/Nautica): uso tutti i livelli LaMMA ma sostituisco
    # il primo colore (quasi bianco) con un turchese saturo distinguibile su satellite.
    # Così le zone con onde piccole (0-0.3m) NON sono trasparenti — l'utente vede
    # sempre il mare coperto da onde con colore. Il taglio costa è identico alla
    # versione full (maschera terra ISTAT incisa nel PNG, non più clip-path).
    LAMMA_COLORS_ERODED = [(135, 220, 245)] + LAMMA_COLORS_RGB[1:]  # primo livello: turchese saturo
    cmap_eroded = ListedColormap([(r / 255, g / 255, b / 255) for r, g, b in LAMMA_COLORS_ERODED])
    cmap_eroded.set_bad(alpha=0)
    norm_eroded = BoundaryNorm(LAMMA_LEVELS, cmap_eroded.N)

    # Maschera terra rasterizzata UNA volta (ISTAT 1:50000) → incisa nel PNG
    land_mask = build_land_mask(width, height, x_min_merc, x_max_merc, y_min_merc, y_max_merc)
    land_mask_arr = np.array(land_mask) if land_mask is not None else None

    def render_one(out_path, data, eroded=False, incise=False):
        # Figure SENZA padding/assi/bordi
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_xlim(x_min_merc, x_max_merc)
        ax.set_ylim(y_min_merc, y_max_merc)
        ax.contourf(
            lngs_rad, lats_merc, data,
            levels=LAMMA_LEVELS,
            cmap=(cmap_eroded if eroded else cmap),
            norm=(norm_eroded if eroded else norm),
            extend="neither",
            antialiased=True,
            corner_mask=False,
        )
        fig.savefig(out_path, dpi=100, transparent=True, pad_inches=0, bbox_inches=None)
        plt.close(fig)

        # Incisione maschera terra SOLO per la versione "eroded" (Satellite/Nautica):
        # lì la basemap (Esri/OpenSeaMap) mostra la terra fotografica, quindi l'overlay
        # dev'essere trasparente sulla terra.
        # Per "mappa" invece la PNG è PIENA (copre TUTTO il mare, anche la striscia
        # costiera dove la basemap Positron aveva il mare azzurro): la terra viene
        # disegnata SOPRA dal frontend (LandCoverLayer vettoriale ISTAT). Così sotto
        # non resta nessun mare di basemap = niente "alone" di pixel attorno alla terra.
        if incise and land_mask_arr is not None:
            img = Image.open(out_path).convert("RGBA")
            arr = np.array(img)
            arr[..., 3][land_mask_arr > 127] = 0  # alpha=0 sulla terra
            Image.fromarray(arr).save(out_path)
        return os.path.getsize(out_path)

    for i in range(hours_count):
        data_raw = ds.VHM0.isel(time=i).values  # (lat, lng)
        time_iso = str(np.datetime_as_string(ds.time.values[i], unit="s"))

        # Pipeline ANTI-ALONE (2026-05-28).
        # L'alone ciano costiero nasceva da: (1) celle terra riempite con valori
        # bassi da tratti di costa lontani (nearest 2D), (2) blur che li spande nel
        # mare costiero abbassandolo sotto la soglia 0.3 → banda ciano fittizia.
        # FIX: blur NaN-aware (SOLO il mare contribuisce, la terra NON bleeda), POI
        # estensione PIATTA oltre costa col valore mare già smussato. Così la cella
        # costiera resta al suo valore reale fino al taglio maschera ISTAT.
        try:
            from scipy import ndimage

            sea = ~np.isnan(data_raw)
            if not sea.any():
                data_filled = np.zeros_like(data_raw)
            else:
                # Upsample 4x (lineare = no overshoot bicubico) di dati e maschera mare
                data_up = ndimage.zoom(np.where(sea, data_raw, 0.0), 4, order=1)
                sea_up = ndimage.zoom(sea.astype(float), 4, order=1) > 0.5
                # Gaussian NORMALIZZATO: blur(dato·mare)/blur(mare) = media pesata
                # solo su celle di mare → la terra non contamina il mare costiero.
                num = ndimage.gaussian_filter(np.where(sea_up, data_up, 0.0), sigma=1.5)
                den = ndimage.gaussian_filter(sea_up.astype(float), sigma=1.5)
                blurred = np.where(den > 1e-6, num / np.maximum(den, 1e-9), np.nan)
                blurred = np.where(sea_up, blurred, np.nan)
                # Estensione oltre costa col MASSIMO locale (grey_dilation), NON la media.
                # La striscia "no-data" tra il bordo dato CMEMS (4.2km, si ferma ~3km al
                # largo) e la costa ISTAT veniva riempita col valore costiero basso →
                # rim ciano UNIFORME (artefatto risoluzione). Col massimo prende il
                # valore del MARE APERTO vicino → rim blu coerente. I golfi davvero
                # calmi restano ciano (dato reale, non toccato perché è già nel mare).
                tmp = np.where(np.isnan(blurred), -1e9, blurred)
                dil = ndimage.grey_dilation(tmp, size=15)
                filled = np.where(np.isnan(blurred), dil, blurred)
                filled = np.where(filled < -1e8, np.nan, filled)
                resid = np.isnan(filled)
                if resid.any():
                    idx = ndimage.distance_transform_edt(
                        resid, return_distances=False, return_indices=True
                    )
                    filled = filled[tuple(idx)]
                data_filled = filled
            data_eroded = data_filled.copy()
        except ImportError:
            data_filled = data_raw
            data_eroded = data_raw

        # Genero 2 PNG:
        #  - "full"   → Mappa: PIENA, non incisa (copre tutto il mare). La terra la
        #    disegna il frontend sopra (LandCoverLayer ISTAT) = niente mare basemap.
        #  - "eroded" → Satellite/Nautica: INCISA (trasparente sulla terra), così la
        #    foto satellitare/nautica sotto mostra la terra. Colormap turchese saturo.
        out_full = os.path.join(CACHE_DIR, f"waves_h{i:02d}.png")
        out_eroded = os.path.join(CACHE_DIR, f"waves_h{i:02d}_eroded.png")
        render_one(out_full, data_filled, eroded=False, incise=False)
        render_one(out_eroded, data_eroded, eroded=True, incise=True)
        hour_strs.append(time_iso)

    # Metadata per il frontend
    import time
    meta = {
        "hours_count": hours_count,
        "hours": hour_strs,
        "bbox": {"lat_min": lat_min, "lat_max": lat_max, "lng_min": lng_min, "lng_max": lng_max},
        "width": width,
        "height": height,
        "dataset": "cmems_mod_med_wav_anfc_4.2km_PT1H-i",
        "render_ts": int(time.time()),  # epoch sec - cache-bust per browser PNG
    }
    meta_path = os.path.join(CACHE_DIR, "waves_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Stats (full + eroded)
    sizes_full = [os.path.getsize(os.path.join(CACHE_DIR, f"waves_h{i:02d}.png")) for i in range(hours_count)]
    sizes_eroded = [os.path.getsize(os.path.join(CACHE_DIR, f"waves_h{i:02d}_eroded.png")) for i in range(hours_count)]
    total_kb = (sum(sizes_full) + sum(sizes_eroded)) / 1024
    avg_kb = total_kb / (hours_count * 2) if hours_count else 0
    print(f"[render-png] ✓ {hours_count*2} PNG (full + eroded) = {total_kb:.0f} KB totali ({avg_kb:.0f} KB media)")
    print(f"[render-png] meta saved → {meta_path}")


if __name__ == "__main__":
    main()
