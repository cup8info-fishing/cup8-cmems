"""
render_sst_png.py — Renderizza PNG di TEMPERATURA SUPERFICIALE del mare (SST) + FRONTI TERMICI.

Clone di render_waves_png.py: stessa pipeline "soft" (proiezione Web Mercator, upsample
cubico 4×, blur NaN-aware, supersample+LANCZOS, palette PNG) ma:
  - dato = thetao (°C) invece di VHM0 (m)
  - scala colori TERMICA (turbo: blu freddo → rosso caldo), range fisso 15-29 °C
  - FRONTI TERMICI incisi nella PNG: dove |∇SST| è alta (confine caldo/freddo) si disegna
    un bordo luminoso. Lì si radunano i pelagici (tonni, lampughe).

Per ogni ora genera cache/sst_h{NN}.png (Mappa, piena) + cache/sst_h{NN}_eroded.png
(Satellite, incisa sulla terra) + cache/sst_meta.json.

Uso: python render_sst_png.py [--hours 72] [--width 3000]
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

DATA_NC = os.path.join(os.path.dirname(__file__), "data", "forecast_med_sst.nc")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
# Maschera terra: stesso GeoJSON ISTAT 1:50000 delle onde (rasterizzato per incidere
# l'overlay sulla terra nella variante "eroded"/Satellite). Path portabile (CI/cloud).
LAND_GEOJSON = os.environ.get("CUP8_LAND_GEOJSON") or next(
    (p for p in [
        os.path.join(os.path.dirname(__file__), "med-land-merged.geojson"),
        "C:/Users/giuse/cup8all/Cup8 - Copia/public/med-land-merged.geojson",
    ] if os.path.exists(p)),
    os.path.join(os.path.dirname(__file__), "med-land-merged.geojson"),
)

# ── SCALA TERMICA ──────────────────────────────────────────────────────────────
# Range FISSO (tutte le ore con la stessa scala → il colore di una temperatura NON
# cambia frame-to-frame e il riscaldamento del forecast è leggibile). 15-29 °C copre
# bene l'estate mediterranea (giugno ~18-28 °C) con un filo di margine. Il dato fuori
# range viene "clippato" agli estremi PRIMA del render (niente buchi trasparenti),
# ma i FRONTI sono calcolati sul dato reale non clippato. Range tarabile per stagione.
TEMP_MIN = 16.0
TEMP_MAX = 28.0
TEMP_STEP = 1.0   # bande da 1°C = zone di temperatura NETTE e leggibili (non un blur)
TEMP_LEVELS = list(np.round(np.arange(TEMP_MIN, TEMP_MAX + 1e-6, TEMP_STEP), 2))

def _build_thermal_ramp(n):
    """n colori RGB 0-255 da una palette SST RAFFINATA (NON neon): blu profondo (freddo) →
    blu → teal → verde-mare → sabbia → arancio → terracotta (caldo). Tinte desaturate =
    professionale e LEGGIBILE (il turbo rainbow era psichedelico)."""
    anchors = [
        (0.00, (38, 92, 150)),    # blu profondo (freddo)
        (0.22, (64, 140, 186)),   # blu
        (0.43, (96, 186, 188)),   # teal
        (0.60, (158, 206, 170)),  # verde-mare (medio)
        (0.75, (232, 184, 104)),  # sabbia
        (0.87, (224, 132, 78)),   # arancio
        (1.00, (190, 78, 60)),    # terracotta (caldo)
    ]
    out = []
    for k in range(n):
        t = k / max(1, n - 1)
        for i in range(1, len(anchors)):
            if t <= anchors[i][0] or i == len(anchors) - 1:
                t0, c0 = anchors[i - 1]; t1, c1 = anchors[i]
                f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                f = max(0.0, min(1.0, f))
                out.append(tuple(int(round(c0[j] + (c1[j] - c0[j]) * f)) for j in range(3)))
                break
    return out

SST_COLORS_RGB = _build_thermal_ramp(len(TEMP_LEVELS) - 1)  # N livelli → N-1 colori

# ── FRONTI TERMICI ─────────────────────────────────────────────────────────────
# |∇SST| in °C/km sul campo liscio; soglia ADATTIVA (top ~8% dei gradienti) con un
# pavimento FISICO (frame piatto → nessun fronte fittizio). I fronti si disegnano come
# cresta luminosa + alone scuro morbido per staccarli dalla heatmap, su ENTRAMBE le PNG.
FRONT_ENABLE = False         # fronti DISATTIVATI: il groviglio di linee bianche rendeva la
                             # mappa illeggibile. La temperatura si legge dai colori; i fronti
                             # (confini caldo/freddo) si vedono dove il colore cambia in fretta.
FRONT_PERCENTILE = 90        # solo il top ~10% dei gradienti = fronte
FRONT_MIN_GRAD = 0.05        # °C/km: soglia fisica minima (sotto = mare "uniforme", niente fronte)
FRONT_SOFT = 0.5             # ampiezza smoothstep sopra soglia (basso = cresta più decisa)
FRONT_THICKEN = 4            # px: ispessisce la nervatura del fronte (max_filter) così si vede
FRONT_FIELD_BLUR = 0.5       # morbidezza del campo-fronte (px griglia dati)
FRONT_EROSION = 2            # px erosione maschera mare (niente fronti fittizi lungo la costa)
FRONT_COLOR = (255, 255, 255)  # colore cresta del fronte
FRONT_CASING_BLUR = 3.2      # alone scuro: morbidezza (px immagine) — largo = stacca bene
FRONT_CASING_DARK = 0.55     # alone scuro: intensità 0-1 (forte = la cresta bianca "spicca")

# SST: bordi NETTI (no supersample/LANCZOS): la SST vuole zone DEFINITE, non sfumate come
# le onde. SUPERSAMPLE=1 + contourf antialiased=False → confini banda crisp.
SUPERSAMPLE = 1


def build_land_mask(width, height, x_min_merc, x_max_merc, y_min_merc, y_max_merc):
    """Rasterizza i polygons terra ISTAT in una maschera PIL (L mode).
    Terra = 255, mare = 0. Coords in Web Mercator (radianti) come il contourf."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    try:
        with open(LAND_GEOJSON, "r", encoding="utf-8") as f:
            geo = json.load(f)
    except Exception as e:
        print(f"[render-sst] WARN land mask: {e}", file=sys.stderr)
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


def save_quantized(img, out_path, colors=255):
    """Salva un'immagine RGBA come PNG a PALETTE (mode P) con 1 indice trasparente.
    ~2.5× più leggero di un RGBA → si renderizza a risoluzione alta restando leggeri."""
    rgba = np.asarray(img.convert("RGBA"))
    alpha = rgba[..., 3]
    pal = Image.fromarray(rgba[..., :3], "RGB").quantize(colors=colors)
    idx = np.array(pal, dtype=np.uint8)
    idx[alpha < 128] = colors                              # land/NaN → indice trasparente
    out = Image.fromarray(idx, "P")
    out.putpalette(pal.getpalette()[: colors * 3] + [0, 0, 0])
    out.info["transparency"] = colors
    out.save(out_path, optimize=True)
    return os.path.getsize(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=int, default=72, help="Numero ore da renderizzare (1-72)")
    p.add_argument("--width", type=int, default=3000, help="Larghezza PNG in pixel (tutto il Med)")
    args = p.parse_args()

    if not os.path.exists(DATA_NC):
        print(f"ERROR: NetCDF non trovato: {DATA_NC}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(CACHE_DIR, exist_ok=True)

    ds = xr.open_dataset(DATA_NC)
    total_hours = ds.sizes["time"]
    hours_count = min(args.hours, total_hours)

    # ── SCALA ADATTIVA (soluzione garbata): il range colore = percentili robusti del dato
    #    (tutte le ore) → la palette blu→rosso copre SEMPRE il range reale del giorno. Niente
    #    "tutto teal" né taratura fissa sbagliata per stagione. Sempre leggibile e bilanciata.
    _t = ds.thetao
    if "depth" in _t.dims:
        _t = _t.isel(depth=0)
    _arr = _t.isel(time=slice(0, hours_count)).values.astype("float32")
    if np.isfinite(_arr).any() and np.nanmax(_arr) > 100:   # difensivo Kelvin
        _arr = _arr - 273.15
    _fin = _arr[np.isfinite(_arr)]
    _lo = float(np.percentile(_fin, 2)); _hi = float(np.percentile(_fin, 98))
    vmin = float(np.floor(_lo)); vmax = float(np.ceil(_hi))
    if vmax - vmin < 6:                       # span minimo: non comprimere troppo i colori
        _mid = (vmin + vmax) / 2.0; vmin = round(_mid - 3); vmax = round(_mid + 3)
    vmin = max(8.0, vmin); vmax = min(32.0, vmax)
    step = 1.0 if (vmax - vmin) <= 13 else 2.0
    levels = list(np.round(np.arange(vmin, vmax + 1e-6, step), 2))
    if len(levels) < 3:
        levels = [vmin, (vmin + vmax) / 2.0, vmax]
    colors_rgb = _build_thermal_ramp(len(levels) - 1)
    print(f"[render-sst] scala ADATTIVA {vmin:.0f}-{vmax:.0f}°C (p2={_lo:.1f} p98={_hi:.1f}) — {len(levels)-1} bande", flush=True)

    lats = ds.latitude.values   # ascending
    lngs = ds.longitude.values  # ascending
    # PNG su TUTTO il Mediterraneo (bbox = dati interi) → al minimo zoom non si vede mai
    # il bordo rettangolare del PNG. Nitidezza gestita lato frontend (maxZoom).
    lng_min, lng_max = float(lngs.min()), float(lngs.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())

    # PNG in proiezione Web Mercator (EPSG:3857), come Leaflet (vedi render_waves_png).
    def lat_to_mercator_y(lat_deg):
        return np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2))

    # Upsample lats/lngs 4× per matchare il dato upsamplato (come onde)
    lats_us = np.linspace(lats[0], lats[-1], len(lats) * 4)
    lngs_us = np.linspace(lngs[0], lngs[-1], len(lngs) * 4)
    lats_merc = lat_to_mercator_y(lats_us)
    y_min_merc = lat_to_mercator_y(lat_min)
    y_max_merc = lat_to_mercator_y(lat_max)
    lngs_rad = np.radians(lngs_us)
    x_min_merc = np.radians(lng_min)
    x_max_merc = np.radians(lng_max)

    aspect = (x_max_merc - x_min_merc) / (y_max_merc - y_min_merc)
    width = args.width
    height = int(round(width / aspect))

    # Colormap discreta termica + BoundaryNorm = bande di colore solide (no interpolazione),
    # poi smussate dall'upsample/blur/supersample → look "soft" identico alle onde.
    cmap = ListedColormap([(r / 255, g / 255, b / 255) for r, g, b in colors_rgb])
    cmap.set_bad(alpha=0)  # NaN = trasparente
    norm = BoundaryNorm(levels, cmap.N)

    hour_strs = []
    print(f"[render-sst] {hours_count} ore × {width}×{height}px, scala {vmin:.0f}-{vmax:.0f}°C, output → {CACHE_DIR}")

    # Maschera terra rasterizzata UNA volta (ISTAT 1:50000) → incisa nella variante eroded
    land_mask = build_land_mask(width, height, x_min_merc, x_max_merc, y_min_merc, y_max_merc)
    land_mask_arr = np.array(land_mask) if land_mask is not None else None

    # Mappa pixel-immagine → indici griglia dati (stessa proiezione del contourf): serve a
    # ricampionare il campo-FRONTE in spazio immagine, allineato pixel-per-pixel alla heatmap.
    def _merc_y_to_lat(my):
        return np.degrees(2.0 * np.arctan(np.exp(my)) - np.pi / 2.0)
    _col_lng = np.degrees(x_min_merc + ((np.arange(width) + 0.5) / width) * (x_max_merc - x_min_merc))
    _col_idx = (_col_lng - lngs_us[0]) / (lngs_us[-1] - lngs_us[0]) * (len(lngs_us) - 1)
    _row_merc = y_max_merc - ((np.arange(height) + 0.5) / height) * (y_max_merc - y_min_merc)
    _row_lat = _merc_y_to_lat(_row_merc)
    _row_idx = (_row_lat - lats_us[0]) / (lats_us[-1] - lats_us[0]) * (len(lats_us) - 1)
    pc_rows = np.repeat(np.clip(_row_idx, 0, len(lats_us) - 1)[:, None], width, axis=1)
    pc_cols = np.repeat(np.clip(_col_idx, 0, len(lngs_us) - 1)[None, :], height, axis=0)

    # Spaziatura griglia upsamplata (per |∇SST| in °C/km): la longitudine "si stringe" con
    # la latitudine (cos φ), quindi dx_km è per-riga.
    dlat_deg = abs((lats_us[-1] - lats_us[0]) / (len(lats_us) - 1))
    dlon_deg = abs((lngs_us[-1] - lngs_us[0]) / (len(lngs_us) - 1))
    dy_km = 111.195 * dlat_deg
    dx_km = np.maximum(111.195 * dlon_deg * np.cos(np.radians(lats_us)), 1e-3)  # per riga

    def render_one(out_path, data, front_a, front_casing, incise=False):
        # Figure SENZA padding/assi — render a SUPERSAMPLE× poi LANCZOS (AA bordi banda).
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100 * SUPERSAMPLE, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_xlim(x_min_merc, x_max_merc)
        ax.set_ylim(y_min_merc, y_max_merc)
        ax.contourf(
            lngs_rad, lats_merc, data,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend="neither",   # dato già clippato nel range → niente buchi trasparenti
            antialiased=False,  # bordi banda NETTI (zone di temperatura definite)
            corner_mask=False,
        )
        fig.savefig(out_path, dpi=100 * SUPERSAMPLE, transparent=True, pad_inches=0, bbox_inches=None)
        plt.close(fig)

        img = Image.open(out_path).convert("RGBA")
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)

        # front_a / front_casing sono già in spazio immagine (precalcolati 1 volta per ora,
        # riusati da full+eroded → niente map_coordinates/gaussian ripetuti = render molto più veloce).
        do_front = FRONT_ENABLE and front_a is not None
        do_incise = incise and land_mask_arr is not None
        if do_front or do_incise:
            arr = np.array(img)
            if do_front:
                a = front_a
                rgb = arr[..., :3].astype(np.float32)
                rgb *= (1.0 - FRONT_CASING_DARK * front_casing[..., None])  # alone scuro = stacca il fronte
                fc = np.array(FRONT_COLOR, dtype=np.float32)
                rgb = rgb * (1.0 - a[..., None]) + fc * a[..., None]        # cresta luminosa nitida
                arr[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
            if do_incise:
                arr[..., 3][land_mask_arr > 127] = 0   # alpha=0 sulla terra (Satellite)
            img = Image.fromarray(arr)
        return save_quantized(img, out_path)

    for i in range(hours_count):
        da = ds.thetao.isel(time=i)
        if "depth" in da.dims:           # difensivo: se puntassimo al dataset 3D
            da = da.isel(depth=0)
        data_raw = da.values             # (lat, lng), °C, NaN su terra
        # Difensivo Kelvin: il prodotto NEMO è in °C, ma se max>100 è Kelvin → -273.15
        if np.isfinite(data_raw).any() and np.nanmax(data_raw) > 100:
            data_raw = data_raw - 273.15
        time_iso = str(np.datetime_as_string(ds.time.values[i], unit="s"))

        front = None
        try:
            from scipy import ndimage

            sea = ~np.isnan(data_raw)
            if not sea.any():
                data_filled = np.full((len(lats) * 4, len(lngs) * 4), vmin)
            else:
                # Stessa pipeline ANTI-ALONE delle onde: riempi terra col mare più vicino,
                # upsample cubico 4×, blur NaN-aware, estensione oltre costa, riempi residui.
                _vmin = float(np.nanmin(data_raw)); _vmax = float(np.nanmax(data_raw))
                _idx_nn = ndimage.distance_transform_edt(~sea, return_distances=False, return_indices=True)
                _data_nn = data_raw[tuple(_idx_nn)]
                data_up = ndimage.zoom(_data_nn, 4, order=3)
                data_up = np.clip(data_up, _vmin, _vmax)
                sea_up = ndimage.zoom(sea.astype(float), 4, order=1) > 0.5
                num = ndimage.gaussian_filter(np.where(sea_up, data_up, 0.0), sigma=0.9)
                den = ndimage.gaussian_filter(sea_up.astype(float), sigma=0.9)
                blurred = np.where(den > 1e-6, num / np.maximum(den, 1e-9), np.nan)
                blurred = np.where(sea_up, blurred, np.nan)
                # Oltre costa: media locale (per SST va bene la media, non il massimo come onde)
                tmp = np.where(np.isnan(blurred), 0.0, blurred)
                wgt = ndimage.gaussian_filter((~np.isnan(blurred)).astype(float), sigma=6)
                ext = ndimage.gaussian_filter(tmp, sigma=6) / np.maximum(wgt, 1e-6)
                filled = np.where(np.isnan(blurred), ext, blurred)
                resid = np.isnan(filled)
                if resid.any():
                    idx = ndimage.distance_transform_edt(
                        resid, return_distances=False, return_indices=True
                    )
                    filled = filled[tuple(idx)]
                data_filled = filled

                # ── FRONTI TERMICI: |∇SST| in °C/km sul campo liscio, soglia adattiva ──
                if FRONT_ENABLE:
                    gy, gx = np.gradient(data_filled)                 # °C/cella
                    grad = np.hypot(gy / dy_km, gx / dx_km[:, None])  # °C/km
                    sea_front = ndimage.binary_erosion(sea_up, iterations=FRONT_EROSION)
                    grad = np.where(sea_front, grad, np.nan)
                    finite = np.isfinite(grad)
                    if finite.any():
                        thr = max(FRONT_MIN_GRAD, float(np.nanpercentile(grad[finite], FRONT_PERCENTILE)))
                        f = np.clip((grad - thr) / (thr * FRONT_SOFT + 1e-9), 0.0, 1.0)
                        f = np.where(np.isfinite(f), f, 0.0)
                        if FRONT_THICKEN > 1:           # ispessisci la nervatura → fronte visibile
                            f = ndimage.maximum_filter(f, size=FRONT_THICKEN)
                        front = ndimage.gaussian_filter(f, FRONT_FIELD_BLUR)
        except ImportError:
            data_filled = data_raw

        # Clip al range di scala SOLO per il render (i fronti sono già calcolati sul dato reale)
        data_disp = np.clip(data_filled, levels[0] + 1e-6, levels[-1] - 1e-6)

        # Fronti in spazio immagine: calcolati UNA volta per ora (map_coordinates + gaussian
        # sull'immagine sono costosi → non ripeterli per full E eroded). Riusati da entrambi.
        front_a = front_casing = None
        if FRONT_ENABLE and front is not None:
            from scipy import ndimage as _ndi
            fimg = _ndi.map_coordinates(front, [pc_rows, pc_cols], order=1, mode="nearest")
            front_a = np.clip(fimg, 0.0, 1.0)
            front_casing = np.clip(_ndi.gaussian_filter(front_a, FRONT_CASING_BLUR), 0.0, 1.0)

        out_full = os.path.join(CACHE_DIR, f"sst_h{i:02d}.png")
        out_eroded = os.path.join(CACHE_DIR, f"sst_h{i:02d}_eroded.png")
        render_one(out_full, data_disp, front_a, front_casing, incise=False)
        render_one(out_eroded, data_disp, front_a, front_casing, incise=True)
        hour_strs.append(time_iso)

    # Metadata per il frontend (mirror waves_meta + range termico per la legenda)
    import time
    meta = {
        "hours_count": hours_count,
        "hours": hour_strs,
        "bbox": {"lat_min": lat_min, "lat_max": lat_max, "lng_min": lng_min, "lng_max": lng_max},
        "width": width,
        "height": height,
        "dataset": "cmems_mod_med_phy-tem_anfc_4.2km-2D_PT1H-m",
        "temp_min": vmin,
        "temp_max": vmax,
        "fronts": FRONT_ENABLE,
        "render_ts": int(time.time()),
    }
    meta_path = os.path.join(CACHE_DIR, "sst_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    sizes_full = [os.path.getsize(os.path.join(CACHE_DIR, f"sst_h{i:02d}.png")) for i in range(hours_count)]
    sizes_eroded = [os.path.getsize(os.path.join(CACHE_DIR, f"sst_h{i:02d}_eroded.png")) for i in range(hours_count)]
    total_kb = (sum(sizes_full) + sum(sizes_eroded)) / 1024
    avg_kb = total_kb / (hours_count * 2) if hours_count else 0
    print(f"[render-sst] ✓ {hours_count*2} PNG (full + eroded) = {total_kb:.0f} KB totali ({avg_kb:.0f} KB media)")
    print(f"[render-sst] meta saved → {meta_path}")


if __name__ == "__main__":
    main()
