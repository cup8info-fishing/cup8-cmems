"""
build_emodnet_store.py — Scarica la batimetria EMODnet di TUTTO il Mediterraneo
UNA volta in un singolo array locale (memmap int16), a ~150 m/px. Serve al
bathy_tile_server per renderizzare i tile dal dato LOCALE (niente fetch per-tile
da EMODnet → caricamento istantaneo) e a piena nitidezza.

Profondità in metri (int16): 0 = terra/no-data, >=1 = profondità mare.
Scarica EMODnet WCS in chunk 2°x2° (sotto il limite di download) in parallelo.

Output: data/med_emodnet_150m.dat (~740 MB) + data/med_emodnet_150m.json
Uso:    python build_emodnet_store.py
"""
import os
import io
import sys
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tifffile

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
LNG_MIN, LNG_MAX = -6.0, 36.0
LAT_MIN, LAT_MAX = 30.0, 46.0
CHUNK_DEG = 2.0
CHUNK_PX = 1485                       # px per chunk 2° → ~150 m/px
NX = int(round((LNG_MAX - LNG_MIN) / CHUNK_DEG))   # 21
NY = int(round((LAT_MAX - LAT_MIN) / CHUNK_DEG))   # 8
W = NX * CHUNK_PX                     # 31185
H = NY * CHUNK_PX                     # 11880
RES = CHUNK_DEG / CHUNK_PX            # gradi/px
OUT_DAT = os.path.join(DATA, "med_emodnet_150m.dat")
OUT_META = os.path.join(DATA, "med_emodnet_150m.json")
WORKERS = 6


def fetch_chunk(ci, cj):
    lo0 = LNG_MIN + ci * CHUNK_DEG
    la1 = LAT_MAX - cj * CHUNK_DEG     # alto
    lo1 = lo0 + CHUNK_DEG
    la0 = la1 - CHUNK_DEG
    url = ("https://ows.emodnet-bathymetry.eu/wcs?SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
           "&COVERAGEID=emodnet__mean&FORMAT=image/tiff"
           f"&SUBSET=Long({lo0:.5f},{lo1:.5f})&SUBSET=Lat({la0:.5f},{la1:.5f})"
           f"&SCALESIZE=i({CHUNK_PX}),j({CHUNK_PX})")
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            elev = tifffile.imread(io.BytesIO(data)).astype(np.float32)   # (px,px), <0 = mare
            if elev.shape != (CHUNK_PX, CHUNK_PX):
                from PIL import Image
                elev = np.array(Image.fromarray(elev).resize((CHUNK_PX, CHUNK_PX)))
            depth = np.where(elev < 0, np.clip(-elev, 1, 32760), 0).astype(np.int16)  # 0=terra
            return ci, cj, depth
        except Exception as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    sys.stderr.write(f"[store] chunk {ci},{cj} FALLITO: {last}\n")
    return ci, cj, None


def main():
    os.makedirs(DATA, exist_ok=True)
    print(f"[store] memmap {W}x{H} int16 (~{W*H*2/1e6:.0f} MB), {NX*NY} chunk @ {RES*111320:.0f} m/px")
    arr = np.memmap(OUT_DAT, dtype=np.int16, mode="w+", shape=(H, W))
    tasks = [(ci, cj) for ci in range(NX) for cj in range(NY)]
    done = 0; fails = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(fetch_chunk, ci, cj) for ci, cj in tasks]
        for f in as_completed(futs):
            ci, cj, depth = f.result()
            done += 1
            if depth is None:
                fails += 1
            else:
                r0 = cj * CHUNK_PX; c0 = ci * CHUNK_PX
                arr[r0:r0 + CHUNK_PX, c0:c0 + CHUNK_PX] = depth
            if done % 12 == 0 or done == len(tasks):
                print(f"[store] {done}/{len(tasks)} chunk ({fails} falliti) — {time.time()-t0:.0f}s", flush=True)
    arr.flush(); del arr
    meta = {"path": os.path.basename(OUT_DAT), "W": W, "H": H, "res_deg": RES,
            "lng_min": LNG_MIN, "lng_max": LNG_MAX, "lat_min": LAT_MIN, "lat_max": LAT_MAX,
            "dtype": "int16", "nodata": 0, "source": "EMODnet mean (WCS)"}
    with open(OUT_META, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)
    print(f"[store] OK -> {OUT_DAT} ({fails} chunk falliti) + {OUT_META}")


if __name__ == "__main__":
    main()
