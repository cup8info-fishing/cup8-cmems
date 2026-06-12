"""
render_daily_png.py — Renderizza i prodotti "snapshot giornaliero" di cap8 in PNG overlay
(stesso schema dei layer onde/SST: full + eroded + meta.json), ma per UNA mappa del giorno.

Prodotti:
  - sst-uhr : SST satellitare ~0.9 km, scala termica adattiva (1 PNG: current).
  - chl     : Clorofilla mg/m³, scala LOGARITMICA blu→verde (1 PNG: current).
  - temp3d  : Temperatura a varie PROFONDITÀ (termoclino): 1 PNG per livello (slider profondità
              lato app), scala termica adattiva COMUNE a tutti i livelli (per confronto).

Riusa la proiezione Web Mercator (EPSG:3857) e la maschera terra ISTAT di render_sst_png.

Uso: python render_daily_png.py --product {sst-uhr,chl,temp3d} [--width N]

Output: cache/daily_<product>[_d{NN}].png (+ _eroded) + cache/daily_<product>_meta.json
"""
import sys
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
from PIL import Image, ImageDraw

ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(ROOT, "cache")
LAND_GEOJSON = os.environ.get("CUP8_LAND_GEOJSON") or next(
    (p for p in [
        os.path.join(ROOT, "med-land-merged.geojson"),
        "C:/Users/giuse/cup8all/Cup8 - Copia/public/med-land-merged.geojson",
    ] if os.path.exists(p)),
    os.path.join(ROOT, "med-land-merged.geojson"),
)

PRODUCTS = {
    "sst-uhr": {"nc": "daily_sst_uhr.nc", "var": "analysed_sst", "scale": "thermal",
                "upsample": 1, "sigma": 0.6, "width": 4000, "dataset": "SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_c_V2"},
    "chl":     {"nc": "daily_chl.nc", "var": "chl", "scale": "chl_log",
                "upsample": 4, "sigma": 1.0, "width": 3000, "dataset": "cmems_mod_med_bgc-pft_anfc_4.2km_P1D-m"},
    "temp3d":  {"nc": "daily_temp3d.nc", "var": "thetao", "scale": "thermal",
                "upsample": 4, "sigma": 0.9, "width": 3000, "dataset": "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m"},
}


def thermal_ramp(n):
    """Palette SST: blu profondo (freddo) → teal → verde-mare → sabbia → terracotta (caldo)."""
    anchors = [(0.00, (38, 92, 150)), (0.22, (64, 140, 186)), (0.43, (96, 186, 188)),
               (0.60, (158, 206, 170)), (0.75, (232, 184, 104)), (0.87, (224, 132, 78)),
               (1.00, (190, 78, 60))]
    return _interp(anchors, n)


def chl_ramp(n):
    """Palette clorofilla: blu profondo (acqua povera/oligotrofica) → ciano → verde → giallo-verde
    (acqua ricca di plancton). Stile NASA chlorophyll, leggibile."""
    anchors = [(0.00, (28, 50, 110)), (0.25, (32, 110, 178)), (0.45, (40, 170, 170)),
               (0.62, (70, 190, 120)), (0.80, (150, 210, 80)), (1.00, (225, 225, 70))]
    return _interp(anchors, n)


def _interp(anchors, n):
    out = []
    for k in range(n):
        t = k / max(1, n - 1)
        for i in range(1, len(anchors)):
            if t <= anchors[i][0] or i == len(anchors) - 1:
                t0, c0 = anchors[i - 1]; t1, c1 = anchors[i]
                f = max(0.0, min(1.0, (t - t0) / (t1 - t0) if t1 > t0 else 0.0))
                out.append(tuple(int(round(c0[j] + (c1[j] - c0[j]) * f)) for j in range(3)))
                break
    return out


def lat_to_y(lat_deg):
    return np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2))


def build_land_mask(width, height, x_min, x_max, y_min, y_max):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    try:
        with open(LAND_GEOJSON, "r", encoding="utf-8") as f:
            geo = json.load(f)
    except Exception as e:
        print(f"[render-daily] WARN land mask: {e}", file=sys.stderr)
        return None

    def project(lng, lat):
        px = (np.radians(lng) - x_min) / (x_max - x_min) * width
        py = (y_max - lat_to_y(lat)) / (y_max - y_min) * height
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
    rgba = np.asarray(img.convert("RGBA"))
    alpha = rgba[..., 3]
    pal = Image.fromarray(rgba[..., :3], "RGB").quantize(colors=colors)
    idx = np.array(pal, dtype=np.uint8)
    idx[alpha < 128] = colors
    out = Image.fromarray(idx, "P")
    out.putpalette(pal.getpalette()[: colors * 3] + [0, 0, 0])
    out.info["transparency"] = colors
    out.save(out_path, optimize=True)
    return os.path.getsize(out_path)


def get_coord(ds, names):
    for n in names:
        if n in ds.coords or n in ds.variables:
            return ds[n].values
    raise KeyError(f"coordinata non trovata fra {names}")


def fill_and_smooth(data_raw, upsample, sigma):
    """Riempi terra col mare più vicino, upsample cubico, blur NaN-aware (anti-alone costiero).
    Identico alla pipeline onde/SST. Ritorna il campo pronto per il contourf."""
    try:
        from scipy import ndimage
    except ImportError:
        return data_raw
    sea = ~np.isnan(data_raw)
    if not sea.any():
        return data_raw
    vmn, vmx = float(np.nanmin(data_raw)), float(np.nanmax(data_raw))
    idx_nn = ndimage.distance_transform_edt(~sea, return_distances=False, return_indices=True)
    data_nn = data_raw[tuple(idx_nn)]
    if upsample > 1:
        data_up = np.clip(ndimage.zoom(data_nn, upsample, order=3), vmn, vmx)
        sea_up = ndimage.zoom(sea.astype(float), upsample, order=1) > 0.5
    else:
        data_up, sea_up = data_nn, sea
    num = ndimage.gaussian_filter(np.where(sea_up, data_up, 0.0), sigma=sigma)
    den = ndimage.gaussian_filter(sea_up.astype(float), sigma=sigma)
    blurred = np.where(den > 1e-6, num / np.maximum(den, 1e-9), np.nan)
    blurred = np.where(sea_up, blurred, np.nan)
    tmp = np.where(np.isnan(blurred), 0.0, blurred)
    wgt = ndimage.gaussian_filter((~np.isnan(blurred)).astype(float), sigma=6)
    ext = ndimage.gaussian_filter(tmp, sigma=6) / np.maximum(wgt, 1e-6)
    filled = np.where(np.isnan(blurred), ext, blurred)
    resid = np.isnan(filled)
    if resid.any():
        idx = ndimage.distance_transform_edt(resid, return_distances=False, return_indices=True)
        filled = filled[tuple(idx)]
    return filled


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--product", required=True, choices=list(PRODUCTS.keys()))
    p.add_argument("--width", type=int, default=None)
    args = p.parse_args()
    cfg = PRODUCTS[args.product]
    width = args.width or cfg["width"]
    var = cfg["var"]

    nc_path = os.path.join(DATA_DIR, cfg["nc"])
    if not os.path.exists(nc_path):
        print(f"ERROR: NetCDF non trovato: {nc_path}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(CACHE_DIR, exist_ok=True)

    ds = xr.open_dataset(nc_path)
    da = ds[var]
    # Prendi l'ultimo istante disponibile (il più recente)
    if "time" in da.dims:
        da = da.isel(time=-1)
        time_iso = str(np.datetime_as_string(ds.time.values[-1], unit="s"))
    else:
        time_iso = ""

    # Solo temp3d è multi-profondità. Gli altri prodotti (chl) possono avere una dim depth
    # con un solo livello di superficie: la schiaccio così sono snapshot singoli.
    is_depth_product = (args.product == "temp3d")
    if "depth" in da.dims and not is_depth_product:
        da = da.isel(depth=0)

    lats = get_coord(ds, ["latitude", "lat"])
    lngs = get_coord(ds, ["longitude", "lon"])
    asc_lat = lats[0] < lats[-1]
    lat_min, lat_max = float(lats.min()), float(lats.max())
    lng_min, lng_max = float(lngs.min()), float(lngs.max())

    # Livelli di profondità (solo temp3d): un PNG per livello.
    # Il NC contiene TUTTI i livelli nativi NEMO fra 5 e 60 m (~15): renderizzarli tutti
    # = 30 PNG e minuti di CI sprecati. Selezioniamo solo le profondità utili al pescatore
    # (chips lato app), prendendo il livello nativo più vicino a ciascun target.
    TARGET_DEPTHS = [5, 10, 15, 20, 30, 40, 50]
    depth_vals = []          # [(indice_nel_nc, profondità_m), ...]
    if is_depth_product and "depth" in da.dims:
        native = [float(d) for d in ds.depth.values]
        seen = set()
        for tgt in TARGET_DEPTHS:
            di = min(range(len(native)), key=lambda i: abs(native[i] - tgt))
            if di not in seen:
                seen.add(di)
                depth_vals.append((di, native[di]))

    # ── Determina scala/palette dai dati ──────────────────────────────────────────
    def to_celsius(a):
        a = a.astype("float32")
        if np.isfinite(a).any() and np.nanmax(a) > 100:  # Kelvin → °C
            a = a - 273.15
        return a

    # Raccogli i valori per la scala adattiva (per temp3d: sui livelli SELEZIONATI insieme,
    # così la palette è confrontabile fra profondità diverse)
    if depth_vals:
        sample = to_celsius(da.isel(depth=[di for di, _ in depth_vals]).values)  # (depth, lat, lng)
    else:
        sample = da.values.astype("float32")
        if cfg["scale"] == "thermal":
            sample = to_celsius(sample)

    fin = sample[np.isfinite(sample)]
    if fin.size == 0:
        print("ERROR: nessun dato valido nel NetCDF", file=sys.stderr)
        sys.exit(1)

    legend = {}
    if cfg["scale"] == "thermal":
        lo, hi = float(np.percentile(fin, 2)), float(np.percentile(fin, 98))
        vmin, vmax = float(np.floor(lo)), float(np.ceil(hi))
        if vmax - vmin < 6:
            mid = (vmin + vmax) / 2.0; vmin, vmax = round(mid - 3), round(mid + 3)
        vmin, vmax = max(0.0, vmin), min(32.0, vmax)
        step = 1.0 if (vmax - vmin) <= 13 else 2.0
        levels = list(np.round(np.arange(vmin, vmax + 1e-6, step), 2))
        colors_rgb = thermal_ramp(len(levels) - 1)
        legend = {"kind": "thermal", "unit": "°C", "min": vmin, "max": vmax}
        print(f"[{args.product}] scala termica {vmin:.0f}-{vmax:.0f}°C ({len(levels)-1} bande)")
        transform = to_celsius
    else:  # chl_log
        pos = fin[fin > 0]
        lo = max(0.02, float(np.percentile(pos, 5)))
        hi = max(lo * 4, float(np.percentile(pos, 98)))
        l0, l1 = np.log10(lo), np.log10(hi)
        levels = list(np.round(10 ** np.linspace(l0, l1, 9), 4))   # 8 bande in log
        colors_rgb = chl_ramp(len(levels) - 1)
        legend = {"kind": "chl_log", "unit": "mg/m³", "min": round(lo, 3), "max": round(hi, 2),
                  "ticks": [round(x, 3) for x in levels]}
        print(f"[{args.product}] scala clorofilla LOG {lo:.2f}-{hi:.2f} mg/m³")
        transform = lambda a: a.astype("float32")

    # ── Proiezione Web Mercator ───────────────────────────────────────────────────
    up = cfg["upsample"]
    lats_us = np.linspace(lats[0], lats[-1], len(lats) * up)
    lngs_us = np.linspace(lngs[0], lngs[-1], len(lngs) * up)
    lats_merc = lat_to_y(lats_us)
    y_min, y_max = lat_to_y(lat_min), lat_to_y(lat_max)
    lngs_rad = np.radians(lngs_us)
    x_min, x_max = np.radians(lng_min), np.radians(lng_max)
    aspect = (x_max - x_min) / (y_max - y_min)
    height = int(round(width / aspect))

    cmap = ListedColormap([(r / 255, g / 255, b / 255) for r, g, b in colors_rgb])
    cmap.set_bad(alpha=0)
    norm = BoundaryNorm(levels, cmap.N)

    land_mask = build_land_mask(width, height, x_min, x_max, y_min, y_max)
    land_arr = np.array(land_mask) if land_mask is not None else None

    def render_one(out_path, field2d, incise=False):
        data = field2d
        if not asc_lat:
            data = data[::-1, :]   # contourf vuole lat ascendente come lats_us
        filled = fill_and_smooth(data, up, cfg["sigma"])
        disp = np.clip(filled, levels[0] + 1e-9, levels[-1] - 1e-9)
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.contourf(lngs_rad, lats_merc, disp, levels=levels, cmap=cmap, norm=norm,
                    extend="neither", antialiased=(cfg["scale"] == "chl_log"), corner_mask=False)
        fig.savefig(out_path, dpi=100, transparent=True, pad_inches=0)
        plt.close(fig)
        img = Image.open(out_path).convert("RGBA")
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        if incise and land_arr is not None:
            arr = np.array(img)
            arr[..., 3][land_arr > 127] = 0
            img = Image.fromarray(arr)
        return save_quantized(img, out_path)

    base_meta = {
        "bbox": {"lat_min": lat_min, "lat_max": lat_max, "lng_min": lng_min, "lng_max": lng_max},
        "width": width, "height": height, "dataset": cfg["dataset"],
        "time": time_iso, "legend": legend, "render_ts": int(time.time()),
    }

    if depth_vals:
        # temp3d: un PNG per profondità selezionata (slider profondità lato app).
        # I file sono numerati 0..N-1 nell'ORDINE di meta.depths (non per indice nel NC).
        depths_out = []
        for k, (di, dval) in enumerate(depth_vals):
            field = transform(da.isel(depth=di).values)
            out_full = os.path.join(CACHE_DIR, f"daily_{args.product}_d{k:02d}.png")
            out_eroded = os.path.join(CACHE_DIR, f"daily_{args.product}_d{k:02d}_eroded.png")
            render_one(out_full, field, incise=False)
            render_one(out_eroded, field, incise=True)
            depths_out.append(round(dval, 1))
        base_meta["depths"] = depths_out
        base_meta["levels_count"] = len(depths_out)
        print(f"[{args.product}] ✓ {len(depths_out)} livelli profondità: {depths_out} m")
    else:
        field = transform(da.values)
        render_one(os.path.join(CACHE_DIR, f"daily_{args.product}.png"), field, incise=False)
        render_one(os.path.join(CACHE_DIR, f"daily_{args.product}_eroded.png"), field, incise=True)
        base_meta["levels_count"] = 1
        print(f"[{args.product}] ✓ snapshot ({width}×{height}px)")

    meta_path = os.path.join(CACHE_DIR, f"daily_{args.product}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(base_meta, f, indent=2)
    print(f"[{args.product}] meta → {meta_path}")


if __name__ == "__main__":
    main()
