"""
render_bathy_merc.py — Renderizza la batimetria papercut del Mediterraneo in
proiezione WEB MERCATOR (EPSG:3857), per usarla come overlay Leaflet nell'app cup8
(L.imageOverlay, stesso schema delle PNG onde).

Legge il dato fine EMODnet 115m (data/med_emodnet_115m.dat) e produce:
  <out>/bathy.png        (PNG papercut Mercator, terra trasparente)
  <out>/bathy_meta.json  (bbox lat/lng per le bounds dell'imageOverlay)

Uso: python render_bathy_merc.py [--width 6000] [--out <dir>]
Default out = Cup8 - Copia/public/bathy (servito da Vite in dev).
"""
import os
import json
import time
import argparse
import numpy as np
from scipy import ndimage
from PIL import Image

HERE = os.path.dirname(__file__)
STORE_DAT = os.path.join(HERE, "data", "med_emodnet_115m.dat")
STORE_META = os.path.join(HERE, "data", "med_emodnet_115m.json")
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "Cup8 - Copia", "public", "bathy"))

# Bande/colori IDENTICI al prototipo (21 bande, rampa blu)
DEPTH_LEVELS = np.array([0, 5, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 600, 800, 1200, 1700, 2300, 3000, 4000, 5500], dtype=np.float32)
_cp = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_cr = np.array([[224, 242, 250], [130, 190, 228], [50, 120, 185], [18, 60, 115], [5, 18, 42]], dtype=np.float32)
_t = np.linspace(0.0, 1.0, len(DEPTH_LEVELS) - 1)
DEPTH_COLORS = np.stack([np.interp(_t, _cp, _cr[:, i]) for i in range(3)], axis=1).astype(np.float32)


def merc_y(lat_deg):
    return np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2))


def merc_y_inv(y):
    return np.degrees(2.0 * np.arctan(np.exp(y)) - np.pi / 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=6000)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()

    m = json.load(open(STORE_META))
    store = np.memmap(STORE_DAT, dtype=np.int16, mode="r", shape=(m["H"], m["W"]))
    lo0, lo1, la0, la1 = m["lng_min"], m["lng_max"], m["lat_min"], m["lat_max"]

    W = a.width
    yN, yS = merc_y(la1), merc_y(la0)
    xrad = np.radians(lo1 - lo0)
    H = int(round(W * (yN - yS) / xrad))
    print(f"[bathy-merc] {W}x{H}px Web Mercator | bbox lng {lo0}..{lo1} lat {la0}..{la1}")

    # lat/lng di ogni pixel (Mercator lineare in y)
    col_lng = lo0 + (np.arange(W) + 0.5) / W * (lo1 - lo0)
    row_y = yN - (np.arange(H) + 0.5) / H * (yN - yS)
    row_lat = merc_y_inv(row_y)
    # indici nello store (EMODnet 115m, equirettangolare)
    cc = (col_lng - lo0) / m["res_deg"]
    rr = (la1 - row_lat) / m["res_deg"]
    CC = np.repeat(cc[None, :], H, axis=0)
    RR = np.repeat(rr[:, None], W, axis=1)
    depth = ndimage.map_coordinates(store, [np.clip(RR, 0, m["H"] - 1), np.clip(CC, 0, m["W"] - 1)], order=1, mode="nearest").astype(np.float32)
    sea = depth > 0.5

    band = np.clip(np.searchsorted(DEPTH_LEVELS, depth, side="right") - 1, 0, len(DEPTH_LEVELS) - 2).astype(np.int16)
    rgb = DEPTH_COLORS[band]
    off = max(2, W // 1200)   # ombra proporzionale alla risoluzione
    upleft = np.roll(np.roll(band, off, 0), off, 1)
    shadow = np.clip(ndimage.gaussian_filter((upleft < band).astype(np.float32), off * 0.8), 0, 1) * 0.34
    rgb = rgb * (1.0 - shadow[..., None])
    alpha = np.where(sea, 235, 0).astype(np.uint8)
    out_arr = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), alpha])

    os.makedirs(a.out, exist_ok=True)
    Image.fromarray(out_arr, "RGBA").save(os.path.join(a.out, "bathy.png"))
    meta = {"bbox": {"lat_min": la0, "lat_max": la1, "lng_min": lo0, "lng_max": lo1},
            "width": W, "height": H, "projection": "web_mercator_epsg3857",
            "dataset": "EMODnet 115m + 21 bande papercut", "render_ts": int(time.time())}
    json.dump(meta, open(os.path.join(a.out, "bathy_meta.json"), "w"), indent=2)
    kb = os.path.getsize(os.path.join(a.out, "bathy.png")) / 1024
    print(f"[bathy-merc] OK -> {a.out}/bathy.png ({kb:.0f} KB) + bathy_meta.json")


if __name__ == "__main__":
    main()
