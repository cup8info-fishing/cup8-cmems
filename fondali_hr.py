"""
fondali_hr.py — Le griglie fini di EMODnet (HR-DTM) pronte per il generatore delle tessere.

Il DTM generale di EMODnet ha una cella di 1/16 di primo (~115 m). In alcune zone EMODnet pubblica anche
griglie più fini (da 1/32 a 1/256 di primo: da ~58 a ~7 m), fatte dai singoli istituti con i loro rilievi
(scaricate in cache/hr con la licenza scritta nei metadati di ognuna: tutte CC BY 4.0). Qui ognuna diventa:

  profondità   in metri, positiva (alcune griglie danno la quota negativa, altre la profondità: il segno
               si riconosce dalla mediana), col nord in alto come il DTM generale; la terra e i buchi fuori;
  lisciata     appena (gaussiana di una cella, normalizzata: i buchi non entrano nella media);
  peso         0…1, che sale dal bordo dei dati verso l'interno in RAGGIO celle: dove i rilievi finiscono
               la griglia fine sfuma nel DTM generale invece di fare un gradino;
  piramide     la stessa griglia 2, 4, 8… volte più larga, fino alla cella del DTM generale: una tessera
               vista da lontano prende il livello con la cella più vicina al suo pixel (niente sfarfallio).

Uscita: <uscita>/<nome>/L<k>.npy (profondità × peso, float32), L<k>_p.npy (peso, float16) e
        <uscita>/indice.json (zona, cella e livelli di ogni griglia, chi l'ha fatta, licenza).
Uso: python fondali_hr.py <cartella con i .nc> <licenze.json> <uscita> [--processi N]
"""
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import xarray as xr
from scipy import ndimage

sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # la console di Windows non è in UTF-8

PASSO_DTM = 1 / 960        # gradi: la cella del DTM generale (1/16 di primo)
RAGGIO = 4                 # celle: in quanto la griglia fine passa da 0 a tutto il suo peso (almeno)
RAGGIO_M = 120             # metri: e comunque non meno di così (sulle griglie da 7 m quattro celle erano
                           # 28 m: dove finiva una striscia di rilievo si vedeva lo scalino col DTM)
MINIMA = 0.05              # metri: sotto, il punto è riva o terra (lo decide la costa di OpenStreetMap)


def una(args):
    percorso, uscita = args
    nome = os.path.basename(percorso)[:-3]
    t0 = time.time()
    ds = xr.open_dataset(percorso, engine="h5netcdf")
    lat = ds["lat"].values.astype(np.float64)
    lon = ds["lon"].values.astype(np.float64)
    el = ds["elevation"].values.astype(np.float32)
    ds.close()
    if lat[1] > lat[0]:                                           # il nord in alto
        lat, el = lat[::-1], el[::-1]
    if lon[1] < lon[0]:                                           # l'ovest a sinistra
        lon, el = lon[::-1], el[:, ::-1]
    passo_lat, passo_lon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
    ok = np.isfinite(el)
    if not ok.any():
        return {"nome": nome, "vuota": True}
    prof = el if float(np.median(el[ok])) > 0 else -el           # profondità positiva
    ok &= prof > MINIMA
    # solo il riquadro coi dati (più un margine): le griglie grandi sono quasi vuote ai lati
    righe, colonne = np.where(ok.any(axis=1))[0], np.where(ok.any(axis=0))[0]
    r0, r1 = max(0, righe[0] - 2), min(ok.shape[0], righe[-1] + 3)
    c0, c1 = max(0, colonne[0] - 2), min(ok.shape[1], colonne[-1] + 3)
    ok, prof, lat, lon = ok[r0:r1, c0:c1], prof[r0:r1, c0:c1], lat[r0:r1], lon[c0:c1]
    m = ok.astype(np.float32)
    d = np.where(ok, prof, 0.0).astype(np.float32)
    # lisciata normalizzata: la media pesata solo sui punti coi dati
    num = ndimage.gaussian_filter(d * m, 1.0, mode="constant")
    den = ndimage.gaussian_filter(m, 1.0, mode="constant")
    d = np.where(ok, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)
    # il peso: distanza dal bordo dei dati (il fuori griglia conta come buco)
    dist = ndimage.distance_transform_edt(np.pad(ok, 1, constant_values=False))[1:-1, 1:-1]
    raggio = max(RAGGIO, math.ceil(RAGGIO_M / (passo_lat * 111320.0)))
    t = np.clip((dist - 1.0) / raggio, 0.0, 1.0)
    w = (t * t * (3 - 2 * t)).astype(np.float32)                 # smoothstep
    del dist, t, num, den, m
    cart = os.path.join(uscita, nome)
    os.makedirs(cart, exist_ok=True)
    livelli = []
    ovest, nord = float(lon[0] - passo_lon / 2), float(lat[0] + passo_lat / 2)
    dw = d * w
    k = 0
    while True:
        np.save(os.path.join(cart, f"L{k}.npy"), dw.astype(np.float32))
        np.save(os.path.join(cart, f"L{k}_p.npy"), w.astype(np.float16))
        livelli.append({"k": k, "passo_lat": passo_lat * 2 ** k, "passo_lon": passo_lon * 2 ** k,
                        "ovest": ovest, "nord": nord, "righe": int(dw.shape[0]), "colonne": int(dw.shape[1])})
        if passo_lat * 2 ** k >= PASSO_DTM * 0.99:
            break
        # il livello dopo: blocchi 2×2 (media della profondità pesata e del peso)
        H, W = dw.shape
        H2, W2 = (H + 1) // 2, (W + 1) // 2
        dw = np.pad(dw, ((0, 2 * H2 - H), (0, 2 * W2 - W))).reshape(H2, 2, W2, 2).mean(axis=(1, 3))
        w = np.pad(w, ((0, 2 * H2 - H), (0, 2 * W2 - W))).reshape(H2, 2, W2, 2).mean(axis=(1, 3))
        k += 1
    valide = prof[ok]
    return {"nome": nome, "cella_primi": round(passo_lat * 60, 6), "livelli": livelli,
            "bbox": [float(lon[0]), float(lat[-1]), float(lon[-1]), float(lat[0])],
            "dati_pct": round(100 * float(ok.mean()), 1),
            "profondita": [round(float(valide.min()), 1), round(float(valide.max()), 1)],
            "secondi": round(time.time() - t0, 1)}


def principale():
    cart_nc, licenze, uscita = sys.argv[1], sys.argv[2], sys.argv[3]
    processi = int(sys.argv[sys.argv.index("--processi") + 1]) if "--processi" in sys.argv else 4
    os.makedirs(uscita, exist_ok=True)
    lic = {r["id"]: r for r in json.load(open(licenze, encoding="utf-8"))}
    files = sorted(os.path.join(cart_nc, f) for f in os.listdir(cart_nc) if f.endswith(".nc"))
    # le più grandi per prime: il giro finisce prima
    files.sort(key=lambda f: -os.path.getsize(f))
    indice = []
    with Pool(processi) as pool:
        for r in pool.imap_unordered(una, [(f, uscita) for f in files]):
            if r.get("vuota"):
                print(f"{r['nome']}: vuota, saltata", flush=True)
                continue
            l = lic.get(r["nome"], {})
            r["produttori"] = l.get("produttori", [])
            r["licenza"] = "; ".join(l.get("licenza_cc", [])) or "?"
            indice.append(r)
            print(f"{r['nome']:52s} {r['cella_primi'] * 60:5.1f}\" {len(r['livelli'])} livelli  dati {r['dati_pct']:5.1f}%  "
                  f"{r['profondita'][0]}…{r['profondita'][1]} m  ({r['secondi']} s)", flush=True)
    # dalla più larga alla più fine: nel generatore la più fine passa per ultima e vince
    indice.sort(key=lambda r: -r["cella_primi"])
    json.dump(indice, open(os.path.join(uscita, "indice.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("griglie pronte:", len(indice), flush=True)


if __name__ == "__main__":
    principale()
