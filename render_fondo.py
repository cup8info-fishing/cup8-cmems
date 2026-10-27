"""render_fondo.py — Il fondo caldo e il fondo freddo, per la carta di Spot (cap8).

Dove il fondo sta almeno 1 °C sotto la superficie, il fondo è entrato nell'acqua fredda che sta
sotto il termoclino. Dalla temperatura del modello del Mediterraneo (download_daily.py --product
fondo: thetao fino a ~200 m, un valore al giorno) e dal fondale della carta (fondo_griglia.npz,
EMODnet a 1/480°), per ogni giorno:
  - la LINEA del fronte: per ogni colonna d'acqua del modello (maglie di 4 km) si cerca la
    profondità dove l'acqua è già 1 °C più fredda della superficie, e la linea è dove il fondale
    VERO arriva a quella profondità. Così segue le batimetriche della carta anche sulle coste
    ripide (calcolata a maglie di 900 m, in Sardegna e in Liguria si appoggiava sulla costa).
    Solo sui fondali fino a 150 m: più giù non si pesca, e d'inverno vagava in mare aperto;
  - le SCRITTE ai due lati della linea, «21° sul fondo» verso costa e «16°» al largo: la
    temperatura tipica del fondo da quel lato nel raggio di ~10 km (la stessa a ogni zoom). Sono
    in tre fitte, una per gruppo di zoom (più vicino, più fitte e più accostate alla linea), e
    ognuna sa da che parte della linea sta (`ancora`): la scritta si allunga lontano dalla linea.

Uscita in cache/fondo/, pubblicata così com'è in public/fondo/ (compatta: la scarica il telefono
quando accendi lo strato):
  <giorno>.json  {"giorno", "fitte": [[zoom da, zoom a]…],
                  "linee": [[x0, y0, dx1, dy1, …]…]  (gradi × 10000, poi le differenze),
                  "scritte": [[x, y, gradi, lato 0 caldo/1 freddo, fitta, ancora]…]}
  meta.json      {"giorni": […], "render_ts"}
Uso: python render_fondo.py [--nc data/daily_fondo.nc]
"""
import argparse
import json
import math
import os
import sys
import time

import contourpy
import numpy as np
import xarray as xr
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
SOGLIA = 1.0            # °C fra superficie e fondo
FONDO_MAX = 150.0       # la linea solo fin qui (m)
TIPICO_KM = 5.0         # sigma della temperatura tipica di un lato (~10 km di raggio)
LINEA_MIN_KM = 4.0      # pezzi di linea più corti: rumore
TOLLERANZA = 0.25       # semplificazione della linea, in celle fini (~60 m)
LISCIO = 1.3            # sigma del fondale fine prima della linea, in celle (~300 m)
GROSSA = 4              # la griglia delle temperature tipiche: 4 × 1/480° = 1/120°
MAI = 400.0             # «qui l'acqua fredda non c'è» (più giù di qualunque fondale della linea)
# le fitte delle scritte: (zoom da, zoom a, una coppia ogni km, a quanti km dalla linea)
# ~280 px fra una coppia e l'altra e ~20 px dalla linea, a metà di ogni gruppo di zoom
FITTE = [(7.75, 9.0, 50.0, 3.6), (9.0, 10.25, 21.0, 1.5), (10.25, 24.0, 9.0, 0.63)]
ANCORE = {(0, 0): 0, (1, 0): 1, (-1, 0): 2, (0, -1): 3, (0, 1): 4, (1, -1): 5, (-1, -1): 6, (1, 1): 7, (-1, 1): 8}


def riempi(a):
    """I buchi (terra o sotto il fondo del modello) presi dalla cella d'acqua più vicina."""
    buco = ~np.isfinite(a)
    if not buco.any():
        return a
    _, (ii, jj) = ndimage.distance_transform_edt(buco, return_indices=True)
    return a[ii, jj]


def chaikin(p, giri=1):
    chiusa = len(p) > 3 and np.allclose(p[0], p[-1])
    for _ in range(giri):
        q = 0.75 * p[:-1] + 0.25 * p[1:]
        r = 0.25 * p[:-1] + 0.75 * p[1:]
        s = np.empty((2 * len(q), 2))
        s[0::2], s[1::2] = q, r
        p = np.vstack([s, s[:1]]) if chiusa else np.vstack([p[:1], s, p[-1:]])
    return p


def rdp(p, tol):
    """Douglas-Peucker: toglie i punti che stanno a meno di tol dalla linea."""
    n = len(p)
    if n < 3:
        return p
    tieni = np.zeros(n, bool)
    tieni[0] = tieni[-1] = True
    pila = [(0, n - 1)]
    while pila:
        a, b = pila.pop()
        if b - a < 2:
            continue
        seg = p[b] - p[a]
        L = math.hypot(seg[0], seg[1])
        q = p[a + 1:b] - p[a]
        d = np.hypot(q[:, 0], q[:, 1]) if L == 0 else np.abs(q[:, 0] * seg[1] - q[:, 1] * seg[0]) / L
        k = int(np.argmax(d))
        if d[k] > tol:
            m = a + 1 + k
            tieni[m] = True
            pila += [(a, m), (m, b)]
    return p[tieni]


def pesi(dest, orig):
    """Cella di sinistra e peso, per interpolare da orig (crescente, passo fisso) ai punti dest."""
    x = np.clip((dest - orig[0]) / (orig[1] - orig[0]), 0, len(orig) - 1.000001)
    i0 = np.floor(x).astype(np.int64)
    return i0, (x - i0).astype(np.float32)


def sulla_griglia(a, rig, col):
    """Bilineare separabile dalla griglia del modello a un'altra griglia regolare."""
    (i0, wi), (j0, wj) = rig, col
    r = a[:, j0] * (1 - wj) + a[:, j0 + 1] * wj
    return (r[i0, :] * (1 - wi)[:, None] + r[i0 + 1, :] * wi[:, None]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default=os.path.join(HERE, "data", "daily_fondo.nc"))
    args = ap.parse_args()
    t0 = time.time()

    # ── il fondale: fine (1/480°) per la linea, grosso (1/120°) per le temperature tipiche ──
    g = np.load(os.path.join(HERE, "fondo_griglia.npz"))
    fine = g["fine"]
    OV, NO, PF = float(g["ovest"]), float(g["nord"]), float(g["passo"])
    HF, WF = fine.shape
    a = fine.reshape(HF // GROSSA, GROSSA, WF // GROSSA, GROSSA).astype(np.float32)
    quanti = (a > 0).sum(axis=(1, 3))
    fondo = np.where(quanti * 2 >= GROSSA * GROSSA, a.sum(axis=(1, 3)) / np.maximum(quanti, 1), 0).astype(np.float32)
    del a
    PG = PF * GROSSA
    H, W = fondo.shape
    mare = fondo > 0
    lat_media = math.radians((NO - HF * PF / 2))
    km_y, km_x = PG * 110.57, PG * 111.32 * math.cos(lat_media)
    kmf_y, kmf_x = PF * 110.57, PF * 111.32 * math.cos(lat_media)
    maschera = (fine == 0) | (fine > FONDO_MAX)                           # la linea solo sul mare fino a 150 m
    # il fondale fine lisciato appena (~300 m, solo il mare): salvato in metri interi, sui fondali in
    # leggera pendenza (la Puglia) faceva gradini, e la linea veniva a scalini
    sul_mare = (fine > 0).astype(np.float32)
    liscio = ndimage.gaussian_filter(fine.astype(np.float32), LISCIO)
    liscio /= np.maximum(ndimage.gaussian_filter(sul_mare, LISCIO), 1e-6)
    del sul_mare

    ds = xr.open_dataset(args.nc)
    prof = ds["depth"].values.astype(np.float32)
    mlat = ds["latitude"].values.astype(np.float64)
    mlon = ds["longitude"].values.astype(np.float64)
    K = len(prof)
    grossa = (pesi(NO - (np.arange(H) + 0.5) * PG, mlat), pesi(OV + (np.arange(W) + 0.5) * PG, mlon))
    fitta = (pesi(NO - (np.arange(HF) + 0.5) * PF, mlat), pesi(OV + (np.arange(WF) + 0.5) * PF, mlon))
    # il fondo (griglia grossa) fra quali livelli del modello cade
    z = np.clip(fondo, prof[0], prof[-1])
    kz = np.clip(np.searchsorted(prof, z) - 1, 0, K - 2)
    wz = ((z - prof[kz]) / (prof[kz + 1] - prof[kz])).astype(np.float32)

    uscita = os.path.join(CACHE, "fondo")
    os.makedirs(uscita, exist_ok=True)
    for vecchio in os.listdir(uscita):                                    # solo i giorni di questo giro
        os.remove(os.path.join(uscita, vecchio))
    giorni = []
    for it, tempo in enumerate(ds["time"].values):
        giorno = str(np.datetime_as_string(tempo, unit="D"))
        T = ds["thetao"].isel(time=it).values.astype(np.float32)          # (depth, lat, lon)
        for k in range(K):
            if not np.isfinite(T[k]).any():                               # livello tutto vuoto: quello sopra
                T[k] = T[k - 1]
            T[k] = riempi(T[k])

        # ── per colonna del modello: la profondità dove l'acqua è già 1 °C più fredda della superficie ──
        bersaglio = T[0] - SOGLIA
        fredda = T <= bersaglio[None]
        c_e = fredda.any(axis=0)
        k1 = np.maximum(np.argmax(fredda, axis=0), 1)                     # il primo livello già freddo
        k0 = k1 - 1
        ii, jj = np.indices(bersaglio.shape)
        Ta, Tb = T[k0, ii, jj], T[k1, ii, jj]
        frac = np.clip((Ta - bersaglio) / np.maximum(Ta - Tb, 1e-6), 0, 1)
        Zc = np.where(c_e, prof[k0] + frac * (prof[k1] - prof[k0]), MAI).astype(np.float32)

        # ── la linea: dove il fondale vero arriva a quella profondità ──
        d = liscio - sulla_griglia(Zc, *fitta)
        cg = contourpy.contour_generator(z=np.ma.masked_array(d, maschera), name="serial", line_type="Separate")
        del d
        linee = []
        for p in cg.lines(0.0):
            p = np.asarray(p, np.float64)
            lung = np.hypot(np.diff(p[:, 0]) * kmf_x, np.diff(p[:, 1]) * kmf_y).sum()
            if lung < LINEA_MIN_KM:
                continue
            linee.append(rdp(chaikin(p, 2), TOLLERANZA))
        del cg

        # ── la temperatura del fondo (griglia grossa) e quella tipica di ciascun lato, nel raggio di ~10 km ──
        Tsup = sulla_griglia(T[0], *grossa)
        Tf = np.zeros((H, W), np.float32)
        for k in range(K):
            liv = sulla_griglia(T[k], *grossa)
            Tf += np.where(kz == k, liv * (1 - wz), 0) + np.where(kz + 1 == k, liv * wz, 0)
        dT = Tsup - Tf
        sig = (TIPICO_KM / km_y, TIPICO_KM / km_x)
        tipico = {}
        for lato, m in ((0, mare & (dT < SOGLIA)), (1, mare & (dT >= SOGLIA) & (fondo <= 200))):
            n = ndimage.gaussian_filter(np.where(m, Tf, 0), sig)
            q = ndimage.gaussian_filter(m.astype(np.float32), sig)
            tipico[lato] = np.where(q > 0.02, n / np.maximum(q, 1e-6), np.nan).astype(np.float32)

        def oltre(lo, la):
            """Quanti metri il fondo sta sotto l'acqua fredda in questo punto (>= 0: fondo freddo); nan = terra."""
            i, j = (NO - la) / PF - 0.5, (lo - OV) / PF - 0.5
            if not (0 <= i <= HF - 1 and 0 <= j <= WF - 1):
                return float("nan")
            f = float(fine[int(round(i)), int(round(j))])
            if f == 0:
                return float("nan")
            zi, zj = (la - mlat[0]) / (mlat[1] - mlat[0]), (lo - mlon[0]) / (mlon[1] - mlon[0])
            return f - float(ndimage.map_coordinates(Zc, [[zi], [zj]], order=1, mode="nearest")[0])

        def tipica(lato, lo, la):
            i, j = (NO - la) / PG - 0.5, (lo - OV) / PG - 0.5
            if not (0 <= i <= H - 1 and 0 <= j <= W - 1):
                return float("nan")
            return float(ndimage.map_coordinates(tipico[lato], [[i], [j]], order=1, mode="nearest")[0])

        # ── le scritte: una coppia ogni tanto lungo la linea, una per lato ──
        scritte = []
        for f, (_, _, ogni_km, dist_km) in enumerate(FITTE):
            for p in linee:
                lo = OV + (p[:, 0] + 0.5) * PF
                la = NO - (p[:, 1] + 0.5) * PF
                xk = (lo - OV) * 111.32 * np.cos(np.radians(la))
                yk = la * 110.57
                s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xk), np.diff(yk)))])
                n = int(s[-1] / ogni_km + 0.5)
                for q in range(n):
                    sq = (q + 0.5) * s[-1] / n
                    px, py = np.interp(sq, s, xk), np.interp(sq, s, yk)
                    a0, a1 = max(0.0, sq - 1.0), min(s[-1], sq + 1.0)        # la direzione della linea su ~2 km
                    tx = np.interp(a1, s, xk) - np.interp(a0, s, xk)
                    ty = np.interp(a1, s, yk) - np.interp(a0, s, yk)
                    L = math.hypot(tx, ty)
                    if L == 0:
                        continue
                    ue, un = -ty / L, tx / L
                    cosp = math.cos(math.radians(py / 110.57))
                    punto = lambda e_, n_: (OV + (px + e_) / (111.32 * cosp), (py + n_) / 110.57)
                    for verso in (1, -1):
                        for dd in (dist_km, dist_km / 2):
                            e_, n_ = ue * verso * dd, un * verso * dd
                            lo_s, la_s = punto(e_, n_)
                            v = oltre(lo_s, la_s)
                            if not math.isfinite(v):
                                continue
                            lato = 1 if v >= 0 else 0
                            w = oltre(*punto(-e_, -n_))                    # dall'altra parte, l'altro lato
                            if math.isfinite(w) and (w >= 0) == (v >= 0):
                                continue
                            t = tipica(lato, lo_s, la_s)
                            if not math.isfinite(t):
                                continue
                            ax = (1 if ue * verso > 0 else -1) if abs(ue) > 0.38 else 0
                            ay = (1 if un * verso > 0 else -1) if abs(un) > 0.38 else 0
                            scritte.append([round(lo_s * 1e4), round(la_s * 1e4), int(round(t)), lato, f, ANCORE[(ax, ay)]])
                            break

        out_linee = []
        for p in linee:
            x = np.round((OV + (p[:, 0] + 0.5) * PF) * 1e4).astype(np.int64)
            y = np.round((NO - (p[:, 1] + 0.5) * PF) * 1e4).astype(np.int64)
            dx, dy = np.diff(x), np.diff(y)
            tieni = (dx != 0) | (dy != 0)
            seq = [int(x[0]), int(y[0])]
            for a_, b_ in zip(dx[tieni], dy[tieni]):
                seq += [int(a_), int(b_)]
            out_linee.append(seq)
        doc = {"giorno": giorno, "fitte": [[a_, b_] for a_, b_, _, _ in FITTE], "linee": out_linee, "scritte": scritte}
        dest = os.path.join(uscita, f"{giorno}.json")
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, separators=(",", ":"))
        giorni.append(giorno)
        caldi = [x[2] for x in scritte if x[3] == 0 and x[4] == 0]
        freddi = [x[2] for x in scritte if x[3] == 1 and x[4] == 0]
        print(f"[fondo] {giorno}: {len(out_linee)} linee, {sum(len(x) for x in out_linee) // 2} punti, "
              f"{len(scritte)} scritte, fondo caldo ~{np.median(caldi) if caldi else '-'}°, "
              f"freddo ~{np.median(freddi) if freddi else '-'}° · {os.path.getsize(dest) / 1024:.0f} KB "
              f"({time.time() - t0:.0f} s)", flush=True)

    with open(os.path.join(uscita, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"giorni": giorni, "render_ts": int(time.time())}, fh)
    print(f"[fondo] fatto in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
