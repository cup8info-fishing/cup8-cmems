"""
render_bathy_xyz.py — Tile batimetria papercut in WEB MERCATOR standard XYZ
(z/x/y, 256px), per usarli come L.tileLayer nell'app cup8 → batimetria NITIDA a
ogni zoom (ogni livello ha il suo dettaglio nativo, limite = dato EMODnet 115m).

Legge data/med_emodnet_115m.dat e scrive <out>/{z}/{x}/{y}.png (terra trasparente,
tile solo-terra saltate). Render con padding + crop → niente cuciture sull'ombra.

Uso: python render_bathy_xyz.py [--zmin 5] [--zmax 10] [--out <dir>] [--workers N]
Default out = Cup8 - Copia/public/bathy-tiles (servito da Vite in dev).
"""
import os
import math
import json
import argparse
import numpy as np
from scipy import ndimage
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(__file__)
STORE_DAT = os.path.join(HERE, "data", "med_emodnet_115m.dat")
STORE_META = os.path.join(HERE, "data", "med_emodnet_115m.json")
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "Cup8 - Copia", "public", "bathy-tiles"))

TILE = 256
PAD = 6  # px di margine per calcolare l'ombra senza cuciture ai bordi

DEPTH_LEVELS = np.array([0, 5, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 600, 800, 1200, 1700, 2300, 3000, 4000, 5500], dtype=np.float32)
_cp = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_cr = np.array([[224, 242, 250], [130, 190, 228], [50, 120, 185], [18, 60, 115], [5, 18, 42]], dtype=np.float32)
_t = np.linspace(0.0, 1.0, len(DEPTH_LEVELS) - 1)
DEPTH_COLORS = np.stack([np.interp(_t, _cp, _cr[:, i]) for i in range(3)], axis=1).astype(np.float32)

# stato per-worker (memmap aperta una volta per processo)
_S = {}


def _init(meta):
    _S["m"] = meta
    _S["store"] = np.memmap(STORE_DAT, dtype=np.int16, mode="r", shape=(meta["H"], meta["W"]))


def _render_one(args):
    z, x, y, out = args
    m = _S["m"]; store = _S["store"]
    n = 2 ** z
    world_px = TILE * n
    px = np.arange(-PAD, TILE + PAD)
    X = x * TILE + px
    lng = X / world_px * 360.0 - 180.0
    Y = y * TILE + px
    ymerc = math.pi - 2.0 * math.pi * Y / world_px
    lat = np.degrees(np.arctan(np.sinh(ymerc)))
    # indici nello store (equirettangolare 115m)
    ccrow = (lng - m["lng_min"]) / m["res_deg"]          # per colonna
    rrcol = (m["lat_max"] - lat) / m["res_deg"]          # per riga
    sz = TILE + 2 * PAD
    CC = np.repeat(ccrow[None, :], sz, axis=0)
    RR = np.repeat(rrcol[:, None], sz, axis=1)
    depth = ndimage.map_coordinates(
        store, [np.clip(RR, 0, m["H"] - 1), np.clip(CC, 0, m["W"] - 1)],
        order=1, mode="nearest").astype(np.float32)
    sea = depth > 0.5
    # salta subito se il CROP centrale non ha mare (la terra resta trasparente / tile assente)
    if not sea[PAD:PAD + TILE, PAD:PAD + TILE].any():
        return 0
    band = np.clip(np.searchsorted(DEPTH_LEVELS, depth, side="right") - 1, 0, len(DEPTH_LEVELS) - 2).astype(np.int16)
    rgb = DEPTH_COLORS[band].astype(np.float32)
    # Ombra papercut NETTA (offset giù-destra, blur minimo) → dà profondità SENZA sfocare.
    off = 2
    upleft = np.roll(np.roll(band, off, 0), off, 1)
    shadow = np.clip(ndimage.gaussian_filter((upleft < band).astype(np.float32), 0.5), 0, 1) * 0.30
    rgb = rgb * (1.0 - shadow[..., None])
    # Linea di contorno NITIDA (1px) a ogni cambio di fascia → confini definiti "carta nautica".
    edge = (band != np.roll(band, 1, 0)) | (band != np.roll(band, 1, 1))
    rgb[edge] *= 0.62
    alpha = np.where(sea, 235, 0).astype(np.uint8)
    out_arr = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), alpha])
    crop = out_arr[PAD:PAD + TILE, PAD:PAD + TILE]
    d = os.path.join(out, str(z), str(x))
    os.makedirs(d, exist_ok=True)
    Image.fromarray(crop, "RGBA").save(os.path.join(d, str(y) + ".png"))
    return 1


def _lat_to_ytile(lat, n):
    r = math.radians(lat)
    return int((1.0 - math.log(math.tan(math.pi / 4 + r / 2)) / math.pi) / 2.0 * n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zmin", type=int, default=5)
    ap.add_argument("--zmax", type=int, default=10)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 1))
    a = ap.parse_args()
    m = json.load(open(STORE_META))
    lo0, lo1, la0, la1 = m["lng_min"], m["lng_max"], m["lat_min"], m["lat_max"]

    jobs = []
    for z in range(a.zmin, a.zmax + 1):
        n = 2 ** z
        x0 = int((lo0 + 180) / 360 * n); x1 = int((lo1 + 180) / 360 * n)
        ytop = _lat_to_ytile(la1, n); ybot = _lat_to_ytile(la0, n)
        for x in range(x0, x1 + 1):
            for y in range(ytop, ybot + 1):
                jobs.append((z, x, y, a.out))
    print(f"[bathy-xyz] {len(jobs)} tile candidate z{a.zmin}-{a.zmax} | {a.workers} worker -> {a.out}")
    os.makedirs(a.out, exist_ok=True)
    written = 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(m,)) as ex:
        for i, r in enumerate(ex.map(_render_one, jobs, chunksize=16)):
            written += r
            if (i + 1) % 2000 == 0:
                print(f"  ...{i+1}/{len(jobs)} processate, {written} scritte")
    # meta per il componente
    json.dump({"zmin": a.zmin, "zmax": a.zmax, "bbox": {"lat_min": la0, "lat_max": la1, "lng_min": lo0, "lng_max": lo1},
               "tile": TILE, "projection": "EPSG:3857 XYZ"}, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(f"[bathy-xyz] FATTO: {written} tile mare scritte (su {len(jobs)} candidate)")


if __name__ == "__main__":
    main()
