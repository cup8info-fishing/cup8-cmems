"""
fondali_scarica.py — Scarica UNA volta le profondità EMODnet (DTM, servizio WCS `emodnet__mean`) della
zona della carta di Spot alla loro risoluzione VERA (1/16 di primo = 1/960 di grado, ~115 m), senza
che il servizio le rimpicciolisca (SCALESIZE prendeva una cella sì e una no: bordi a dentini).

Uscita (nella cartella data/, fuori da git):
  fondali_nativo.f32   griglia float32, righe da nord a sud; metri: negativi in mare, positivi in terra,
                       NaN dove il servizio non ha dato
  fondali_nativo.json  dove sta la griglia (angolo nord-ovest, passo, righe, colonne)

Uso: python fondali_scarica.py <cartella-data>
"""
import json
import math
import os
import struct
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # la console di Windows non è in UTF-8

# la zona di zona.ts (5.5, 35, 19.5, 47.2) più un margine per le lisciate sul bordo
OVEST, SUD, EST, NORD = 5.3, 34.8, 19.7, 47.4
PASSO = 1.0 / 960.0                       # la cella di EMODnet
BLOCCO = 1.0                              # gradi per richiesta
WCS = ("https://ows.emodnet-bathymetry.eu/wcs?SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
       "&COVERAGEID=emodnet__mean&FORMAT=image/tiff&GEOTIFF:COMPRESSION=Deflate")
INSIEME = 6


def leggi_tiff(buf):
    """GeoTIFF float32 (strisce o riquadri, deflate) → (griglia HxW, x0, y0, dx, dy) con x0, y0 = bordo nord-ovest."""
    le = buf[:2] == b"II"
    e = "<" if le else ">"
    u16 = lambda o: struct.unpack_from(e + "H", buf, o)[0]
    u32 = lambda o: struct.unpack_from(e + "I", buf, o)[0]
    SZ = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 11: 4, 12: 8, 16: 8}
    ifd = u32(4)
    tag = {}
    for k in range(u16(ifd)):
        p = ifd + 2 + k * 12
        tid, ty, cnt = u16(p), u16(p + 2), u32(p + 4)
        sz = SZ.get(ty, 1)
        off = p + 8 if sz * cnt <= 4 else u32(p + 8)
        v = []
        for i in range(min(cnt, 1 << 20)):
            o = off + i * sz
            if ty == 3:
                v.append(u16(o))
            elif ty == 4:
                v.append(u32(o))
            elif ty == 12:
                v.append(struct.unpack_from(e + "d", buf, o)[0])
            elif ty == 11:
                v.append(struct.unpack_from(e + "f", buf, o)[0])
            elif ty == 16:
                v.append(struct.unpack_from(e + "Q", buf, o)[0])
            else:
                v.append(buf[o])
        tag[tid] = v
    W, H = tag[256][0], tag[257][0]
    comp = tag.get(259, [1])[0]
    tw, th = tag.get(322, [None])[0], tag.get(323, [None])[0]
    offs = tag.get(324) or tag[273]
    cnts = tag.get(325) or tag[279]
    rps = tag.get(278, [H])[0]
    v = np.full((H, W), np.nan, dtype=np.float32)
    for p in range(len(offs)):
        parte = buf[offs[p]:offs[p] + cnts[p]]
        if comp in (8, 32946):
            parte = zlib.decompress(parte)
        a = np.frombuffer(parte, dtype=(e + "f4"))
        if tw and th:
            per = math.ceil(W / tw)
            ox, oy = (p % per) * tw, (p // per) * th
            blocco = a[: tw * th].reshape(th, tw)
            hh, ww = min(th, H - oy), min(tw, W - ox)
            v[oy:oy + hh, ox:ox + ww] = blocco[:hh, :ww]
        else:
            y0 = p * rps
            q = min(rps, H - y0)
            v[y0:y0 + q, :] = a[: q * W].reshape(q, W)
    v[~(np.abs(v) < 20000)] = np.nan
    mt = tag.get(34264)
    if mt:
        dx, x0, dy, y0 = mt[0], mt[3], -mt[5], mt[7]
    else:
        sc, tp = tag[33550], tag[33922]
        dx, dy = sc[0], sc[1]
        x0, y0 = tp[3] - tp[0] * dx, tp[4] + tp[1] * dy
    ch = tag.get(34735)
    if ch:
        for k in range(4, len(ch) - 3, 4):
            if ch[k] == 1025 and ch[k + 3] == 2:      # valori «al punto»: l'angolo è il centro della cella
                x0 -= dx / 2
                y0 += dy / 2
    return v, x0, y0, dx, dy


def scarica(url, tentativi=4):
    ultimo = None
    for t in range(tentativi):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                b = r.read()
            if b[:2] not in (b"II", b"MM"):
                raise RuntimeError("risposta non tiff: " + b[:200].decode("latin-1"))
            return b
        except Exception as ex:          # il servizio a volte non risponde: si riprova con calma
            ultimo = ex
            time.sleep(5 * (t + 1))
    raise ultimo


def main():
    data = sys.argv[1]
    os.makedirs(data, exist_ok=True)
    W = int(round((EST - OVEST) / PASSO))
    H = int(round((NORD - SUD) / PASSO))
    percorso = os.path.join(data, "fondali_nativo.f32")
    griglia = np.lib.format.open_memmap(percorso + ".npy", mode="w+", dtype=np.float32, shape=(H, W))
    griglia[:] = np.nan
    blocchi = [(lo, la) for la in np.arange(SUD, NORD, BLOCCO) for lo in np.arange(OVEST, EST, BLOCCO)]
    print(f"griglia {W}x{H} celle, {len(blocchi)} blocchi da {BLOCCO} gradi", flush=True)
    fatti = 0

    def uno(lo, la):
        lo1, la1 = min(lo + BLOCCO, EST), min(la + BLOCCO, NORD)
        url = WCS + f"&SUBSET=Long({lo:.6f},{lo1:.6f})&SUBSET=Lat({la:.6f},{la1:.6f})"
        g, x0, y0, dx, dy = leggi_tiff(scarica(url))
        return g, x0, y0, dx, dy

    with ThreadPoolExecutor(INSIEME) as ex:
        futuri = {ex.submit(uno, lo, la): (lo, la) for lo, la in blocchi}
        for f in as_completed(futuri):
            lo, la = futuri[f]
            try:
                g, x0, y0, dx, dy = f.result()
            except Exception as e:
                print(f"BLOCCO {lo:.1f},{la:.1f} FALLITO: {e}", flush=True)
                continue
            if abs(dx - PASSO) > 1e-9 or abs(dy - PASSO) > 1e-9:
                print(f"BLOCCO {lo:.1f},{la:.1f}: passo inatteso {dx} {dy}", flush=True)
            c0 = int(round((x0 - OVEST) / PASSO))
            r0 = int(round((NORD - y0) / PASSO))
            h, w = g.shape
            # ritaglio dentro la griglia
            rr0, cc0 = max(0, r0), max(0, c0)
            rr1, cc1 = min(H, r0 + h), min(W, c0 + w)
            if rr1 > rr0 and cc1 > cc0:
                pezzo = g[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0]
                dove = griglia[rr0:rr1, cc0:cc1]
                np.copyto(dove, pezzo, where=~np.isnan(pezzo))
            fatti += 1
            if fatti % 10 == 0 or fatti == len(blocchi):
                print(f"{fatti}/{len(blocchi)} blocchi", flush=True)
    griglia.flush()
    mare = int(np.count_nonzero(griglia < 0))
    vuote = int(np.count_nonzero(np.isnan(griglia)))
    meta = {"ovest": OVEST, "nord": NORD, "passo": PASSO, "colonne": W, "righe": H,
            "file": os.path.basename(percorso) + ".npy", "celle_mare": mare, "celle_vuote": vuote,
            "fonte": "EMODnet Bathymetry DTM (WCS emodnet__mean), risoluzione nativa", "quando": time.strftime("%Y-%m-%d %H:%M")}
    with open(os.path.join(data, "fondali_nativo.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print("FINE", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
