"""
render_bathy_png.py — Renderizza UNA PNG papercut della batimetria di TUTTO il
Mediterraneo dal batimetrico statico CMEMS (deptho, 4.2km).

Look "papercut": bande di profondità piatte (blu chiaro sottocosta → blu navy al
largo) con un'ombra direzionale morbida sul bordo delle bande più profonde, così i
fondali sembrano "buchi" scavati nella carta (luce da alto-sinistra). Stessa tecnica
del render onde (shift+blur del campo a bande).

Proiezione: EQUIRETTANGOLARE (lat/lng lineare, EPSG:4326). Il prototipo windy-proto
usa la stessa proiezione lineare → la PNG si mappa 1:1 con proj(). (Per l'app Leaflet
servirà in futuro una variante Web Mercator, come per le onde.)

Output: cache/bathy.png  +  cache/bathy_meta.json
Uso:    python render_bathy_png.py [--width 4000]
"""
import os
import json
import time
import argparse
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(__file__)
DATA_NC = os.path.join(HERE, "data", "med_bathy.nc")
CACHE_DIR = os.path.join(HERE, "cache")

# Soglie profondità (m) e palette blu — IDENTICHE alla legenda del prototipo
# (BATI_BANDS / BATI_COL in sea.html) così la legenda combacia col render.
DEPTH_LEVELS = [0, 20, 50, 100, 200, 500, 1000, 2000, 3000, 6000]   # 10 bordi → 9 bande
DEPTH_COLORS_RGB = [
    (207, 234, 247),  # 0-20      bassofondo (chiarissimo)
    (169, 214, 236),  # 20-50
    (127, 189, 224),  # 50-100
    (85, 159, 208),   # 100-200
    (53, 127, 190),   # 200-500
    (35, 95, 158),    # 500-1000
    (22, 68, 125),    # 1000-2000
    (14, 47, 94),     # 2000-3000
    (8, 31, 68),      # 3000+     abisso (navy scurissimo)
]

# Papercut: ombra sul lato profondo dei gradini (buchi). Luce da alto-sinistra.
PAPERCUT_OFFSET_PX = 6     # spostamento ombra in px PNG
PAPERCUT_BLUR = 5.0        # morbidezza (sigma gauss)
PAPERCUT_DARK = 0.42       # intensità ombra 0-1
SUPERSAMPLE = 2            # render Nx poi LANCZOS → bordi banda antialiasati


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=4000, help="Larghezza PNG (tutto il Med)")
    args = ap.parse_args()

    if not os.path.exists(DATA_NC):
        raise SystemExit(f"ERRORE: manca {DATA_NC} (scaricalo con copernicusmarine subset ... deptho)")
    os.makedirs(CACHE_DIR, exist_ok=True)

    ds = xr.open_dataset(DATA_NC)
    lats = ds.latitude.values   # ascending
    lngs = ds.longitude.values  # ascending
    depth = ds.deptho.values.astype(np.float32)   # (lat, lng), positivo = profondità (m), NaN = terra
    lat_min, lat_max = float(lats.min()), float(lats.max())
    lng_min, lng_max = float(lngs.min()), float(lngs.max())

    sea = np.isfinite(depth)
    vmin, vmax = float(np.nanmin(depth)), float(np.nanmax(depth))

    # Riempio la terra (NaN) col valore di mare PIÙ VICINO → l'upsample cubico non
    # sotto-oscilla alla costa (niente bande spurie), poi rimaschereremo la terra.
    idx_nn = ndimage.distance_transform_edt(~sea, return_distances=False, return_indices=True)
    depth_nn = depth[tuple(idx_nn)]

    # Upsample 4x cubico → contorni lisci e curvi (non faccettati ai punti 4.2km)
    UP = 4
    depth_up = ndimage.zoom(depth_nn, UP, order=3)
    depth_up = np.clip(depth_up, vmin, vmax)
    sea_up = ndimage.zoom(sea.astype(float), UP, order=1) > 0.5
    lats_us = np.linspace(lats[0], lats[-1], len(lats) * UP)
    lngs_us = np.linspace(lngs[0], lngs[-1], len(lngs) * UP)

    # Risoluzione PNG (equirettangolare: x∝lng, y∝lat lineari)
    width = args.width
    height = int(round(width * (lat_max - lat_min) / (lng_max - lng_min)))

    cmap = ListedColormap([(r / 255, g / 255, b / 255) for r, g, b in DEPTH_COLORS_RGB])
    cmap.set_bad(alpha=0)
    norm = BoundaryNorm(DEPTH_LEVELS, cmap.N)

    print(f"[bathy] render {width}x{height}px | profondita {vmin:.0f}-{vmax:.0f} m | {UP}x upsample")

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100 * SUPERSAMPLE, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(lng_min, lng_max); ax.set_ylim(lat_min, lat_max)   # equirettangolare
    ax.contourf(lngs_us, lats_us, depth_up, levels=DEPTH_LEVELS, cmap=cmap, norm=norm,
                extend="neither", antialiased=True, corner_mask=False)
    tmp = os.path.join(CACHE_DIR, "_bathy_tmp.png")
    fig.savefig(tmp, dpi=100 * SUPERSAMPLE, transparent=True, pad_inches=0)
    plt.close(fig)

    img = Image.open(tmp).convert("RGBA")
    if img.size != (width, height):
        img = img.resize((width, height), Image.LANCZOS)
    arr = np.array(img)

    # ── PAPERCUT: banda discreta per pixel-immagine (resample del campo profondità con
    #    la STESSA mappatura equirettangolare) → l'ombra cade esatta sui bordi banda.
    col_lng = lng_min + ((np.arange(width) + 0.5) / width) * (lng_max - lng_min)
    row_lat = lat_max - ((np.arange(height) + 0.5) / height) * (lat_max - lat_min)
    col_idx = np.clip((col_lng - lngs_us[0]) / (lngs_us[-1] - lngs_us[0]) * (len(lngs_us) - 1), 0, len(lngs_us) - 1)
    row_idx = np.clip((row_lat - lats_us[0]) / (lats_us[-1] - lats_us[0]) * (len(lats_us) - 1), 0, len(lats_us) - 1)
    rows = np.repeat(row_idx[:, None], width, axis=1)
    cols = np.repeat(col_idx[None, :], height, axis=0)
    samp = ndimage.map_coordinates(depth_up, [rows, cols], order=1, mode="nearest")
    band = np.clip(np.searchsorted(DEPTH_LEVELS, samp, side="right") - 1, 0, len(DEPTH_LEVELS) - 2).astype(np.int16)

    off = PAPERCUT_OFFSET_PX
    # vicino alto-sinistra (carta "sopra"); se è più SUPERFICIALE (banda minore) di
    # questo pixel più profondo → questo è dentro un "buco" → ombra.
    upleft = np.roll(np.roll(band, off, axis=0), off, axis=1)
    shadow = (upleft < band).astype(np.float32)
    shadow = np.clip(ndimage.gaussian_filter(shadow, sigma=PAPERCUT_BLUR), 0, 1) * PAPERCUT_DARK
    rgb = arr[..., :3].astype(np.float32) * (1.0 - shadow[..., None])
    arr[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)

    # Terra trasparente (maschera mare del modello, resample equirettangolare).
    sea_img = ndimage.map_coordinates(sea_up.astype(np.float32), [rows, cols], order=1, mode="nearest") > 0.5
    arr[..., 3][~sea_img] = 0

    out = os.path.join(CACHE_DIR, "bathy.png")
    Image.fromarray(arr).save(out)
    try:
        os.remove(tmp)
    except OSError:
        pass

    meta = {
        "bbox": {"lat_min": lat_min, "lat_max": lat_max, "lng_min": lng_min, "lng_max": lng_max},
        "width": width, "height": height,
        "depth_levels": DEPTH_LEVELS,
        "colors": DEPTH_COLORS_RGB,
        "dataset": "cmems_mod_med_phy_anfc_4.2km_static (deptho)",
        "projection": "equirectangular_epsg4326",
        "render_ts": int(time.time()),
    }
    with open(os.path.join(CACHE_DIR, "bathy_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    kb = os.path.getsize(out) / 1024
    print(f"[bathy] OK -> {out}  ({kb:.0f} KB)  +  bathy_meta.json")


if __name__ == "__main__":
    main()
