"""
fondali_coste.py — Il MARE di OpenStreetMap nella zona della carta di Spot, una volta sola.

Dai «water polygons» di osmdata.openstreetmap.de (il mare ricavato dalle coste di OSM, in Web
Mercator, divisi in quadrati da 100 km; ODbL) tiene solo i pezzi che toccano la zona e li salva in
un file compatto per il generatore delle tessere (fondali_tessere.py). Il mare di OSM è lo stesso
che la carta disegna sotto le fasce: così la riva delle fasce combacia con quella della carta, e
saline, lagune e laghi (che per OSM non sono mare) restano fuori.

Uso: python fondali_coste.py <zip-water-polygons-split-3857> <file-uscita.npz>
"""
import math
import sys
import zipfile

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # la console di Windows non è in UTF-8
import shapefile

R = 6378137.0
OVEST, SUD, EST, NORD = 5.3, 34.8, 19.7, 47.4      # la zona (come fondali_scarica.py)


def merc(lon, lat):
    return R * math.radians(lon), R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def main():
    zp, uscita = sys.argv[1], sys.argv[2]
    x0, y0 = merc(OVEST, SUD)
    x1, y1 = merc(EST, NORD)
    import os
    cartella = os.path.dirname(os.path.abspath(zp))
    with zipfile.ZipFile(zp) as z:
        nomi = z.namelist()
        base = [n for n in nomi if n.endswith(".shp")][0][:-4]
        print("dentro lo zip:", base, flush=True)
        for est in (".shp", ".shx"):                  # letti da disco: lo zip non si scorre a salti
            if not os.path.exists(os.path.join(cartella, base + est)):
                z.extract(base + est, cartella)
    with open(os.path.join(cartella, base + ".shp"), "rb") as shp, open(os.path.join(cartella, base + ".shx"), "rb") as shx:
        if True:
            r = shapefile.Reader(shp=shp, shx=shx)
            anelli, inizi = [], [0]
            n = 0
            for s in r.iterShapes(bbox=[x0, y0, x1, y1]):
                pts = np.asarray(s.points, dtype=np.float64)
                parti = list(s.parts) + [len(pts)]
                for a, b in zip(parti[:-1], parti[1:]):
                    anello = pts[a:b]
                    if len(anello) < 4:
                        continue
                    anelli.append(anello)
                    inizi.append(inizi[-1] + len(anello))
                n += 1
                if n % 50 == 0:
                    print(f"{n} pezzi, {len(anelli)} anelli, {inizi[-1]} punti", flush=True)
    tutti = np.concatenate(anelli) if anelli else np.zeros((0, 2))
    bb = np.array([[a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()] for a in anelli])
    np.savez_compressed(uscita, punti=tutti, inizi=np.asarray(inizi, dtype=np.int64), bbox=bb)
    print(f"FINE: {n} pezzi, {len(anelli)} anelli, {len(tutti)} punti → {uscita}", flush=True)


if __name__ == "__main__":
    main()
