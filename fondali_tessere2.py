"""
fondali_tessere2.py — Le tessere GIÀ PRONTE del mare della carta di Spot (app cap8), seconda versione
(fondali-2, 23/10/2026). La prima (fondali_tessere.py → fondali-1, 16/10/2026) resta com'era.

Cosa cambia rispetto alla prima:
  più precisa  dove EMODnet ha griglie più fini del DTM generale (fondali_hr.py: 41 griglie da ~58 a ~7 m,
               tutte CC BY 4.0), la profondità viene da quelle, sfumata ai bordi dei rilievi; e lì le
               tessere arrivano fino allo zoom che la griglia regge: z11 per 1/32 di primo, z12 per 1/64,
               z13 per 1/128, z14 per 1/256 (altrove si ferma a z12 come prima e la carta ingrandisce
               la tessera madre);
  più leggera  (1) dentro una fascia, lontano dall'ombra dell'isobata, la carta colora pieno: lì la
               profondità esatta non serve, e il pixel prende un valore fisso per fascia se anche i suoi
               8 vicini stanno nella stessa zona piena. Ogni interpolazione bilineare che lo tocca resta
               nella zona piena, quindi il disegno è IDENTICO (controllato su un campione di tessere a
               ogni giro). Il WebP senza perdite comprime le zone uniformi quasi a zero: circa il 60% in
               meno agli zoom vicini (prova del 23/10/2026 su 400 tessere di fondali-1);
               (2) WebP col metodo 6 (più lento da scrivere, più piccolo);
               (3) disponibili.json: quali tessere esistono, zoom per zoom (un bit a tessera). L'app non
               chiede più quelle che non ci sono: ogni 404 di GitHub Pages era una pagina da 9 KB.
  ⚠️ LIVELLI e OMBRA devono essere quelli della carta (scene3d/fondale.ts LIVELLI, cartaStile.ts
     OMBRA_NAUTICA): se cambiano, queste tessere vanno rifatte, perché la «zona piena» cambia con loro.

Il resto come la prima: profondità EMODnet lisciata (gaussiana di una cella) e campionata con la B-spline
cubica; la riva è il mare di OpenStreetMap disegnato 4 volte più fine; lungo la riva il mare non va oltre
il metro; terra = +1 (trasparente nella carta); le linee coi metri (isobate) z6-z12, MVT scritto a mano.

Uscita:
  <uscita>/profondita/{z}/{x}/{y}.webp  terrarium (metri = R·256 + G + B/256 − 32768), WebP senza perdite
  <uscita>/isobate/{z}/{x}/{y}.pbf      MVT, livello «isobate», campo m: z6-z12
  <uscita>/disponibili.json             per zoom: riquadro di tessere e bit (base64) di quelle che esistono
  <uscita>/meta.json                    conteggi, pesi, fonti (con chi ha fatto le griglie fini)

Uso: python fondali_tessere2.py <data> <coste.npz> <hr pronti> <uscita> [--processi N] [--prova]
     --prova: solo attorno a Ischia-Procida, Trani e alla laguna di Venezia
"""
import base64
import io
import json
import math
import os
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # la console di Windows non è in UTF-8
from PIL import Image, ImageDraw
import contourpy

T = 256
R = 6378137.0
MONDO = 2 * math.pi * R
LIVELLI = [0, 5, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 600, 800, 1200, 1700, 2300, 3000, 4000, 5500]
OMBRA = 0.12                                    # cartaStile.ts OMBRA_NAUTICA
LINEE = LIVELLI[1:-1]
ZONA = (5.5, 35.0, 19.5, 47.2)                  # zona.ts: ovest, sud, est, nord
Z_PIENE = (3, 10)                               # da z3 a z10 tutto il mare
Z_COSTA = (11, 12)                              # z11 e z12 sulla piattaforma
Z_MAX = 14                                      # dove le griglie fini lo reggono
PIATTAFORMA = 200.0                             # metri: sotto, da vicino servono le tessere fini
SOVRA = 4                                       # la maschera della riva disegnata 4 volte più fine
PESO_MINIMO = 0.02                              # sotto, una tessera «solo per la griglia fine» non serve
CONTROLLO = 0.05                                # quota delle tessere su cui si rifà il controllo del disegno

# ── condivisi nei processi ──────────────────────────────────────────────────────────────────
G = {}


def apri(data, coste, hr):
    meta = json.load(open(os.path.join(data, "fondali_nativo.json"), encoding="utf-8"))
    G["meta"] = meta
    G["F"] = np.load(os.path.join(data, "fondali_liscio.npy"), mmap_mode="r")
    c = np.load(coste)
    punti, inizi, bbox = c["punti"], c["inizi"], c["bbox"]
    anelli = [punti[inizi[i]:inizi[i + 1]] for i in range(len(inizi) - 1)]
    # verso: nei shapefile l'anello esterno gira in senso orario (area negativa), i buchi al contrario
    buco = np.array([(np.dot(a[:-1, 0], a[1:, 1]) - np.dot(a[1:, 0], a[:-1, 1])) > 0 for a in anelli])
    G["anelli"], G["bbox"], G["buco"] = anelli, bbox, buco
    # indice: secchi da 50 km per trovare subito gli anelli che toccano una tessera
    S = 50000.0
    secchi = {}
    for i, (x0, y0, x1, y1) in enumerate(bbox):
        for gx in range(int(math.floor(x0 / S)), int(math.floor(x1 / S)) + 1):
            for gy in range(int(math.floor(y0 / S)), int(math.floor(y1 / S)) + 1):
                secchi.setdefault((gx, gy), []).append(i)
    G["secchi"], G["S"] = secchi, S
    # le griglie fini, dalla più larga alla più fine (la più fine passa per ultima e vince)
    G["hr"] = []
    if hr:
        for g in json.load(open(os.path.join(hr, "indice.json"), encoding="utf-8")):
            for L in g["livelli"]:
                L["DW"] = np.load(os.path.join(hr, g["nome"], f"L{L['k']}.npy"), mmap_mode="r")
                L["W"] = np.load(os.path.join(hr, g["nome"], f"L{L['k']}_p.npy"), mmap_mode="r")
            G["hr"].append(g)


def limiti_tessera(z, x, y):
    lato = MONDO / (1 << z)
    x0 = -MONDO / 2 + x * lato
    y1 = MONDO / 2 - y * lato
    return x0, y1 - lato, x0 + lato, y1, lato


def mare_osm(z, x, y, n):
    """Quanto di ogni pixel (n×n) è mare per OSM, 0…1 (disegnato SOVRA volte più fine)."""
    x0, y0, x1, y1, lato = limiti_tessera(z, x, y)
    S = G["S"]
    cand = set()
    for gx in range(int(math.floor(x0 / S)), int(math.floor(x1 / S)) + 1):
        for gy in range(int(math.floor(y0 / S)), int(math.floor(y1 / S)) + 1):
            cand.update(G["secchi"].get((gx, gy), ()))
    N = n * SOVRA
    img = Image.new("L", (N, N), 0)
    if cand:
        d = ImageDraw.Draw(img)
        bb = G["bbox"]
        for i in sorted(cand):                       # in ordine: l'esterno prima dei suoi buchi
            if bb[i, 2] < x0 or bb[i, 0] > x1 or bb[i, 3] < y0 or bb[i, 1] > y1:
                continue
            a = G["anelli"][i]
            px = (a[:, 0] - x0) / lato * N
            py = (y1 - a[:, 1]) / lato * N
            d.polygon(list(zip(px.tolist(), py.tolist())), fill=0 if G["buco"][i] else 255)
    m = np.asarray(img, dtype=np.float32).reshape(n, SOVRA, n, SOVRA).mean(axis=(1, 3)) / 255.0
    return m


def pesi(t):
    t2 = t * t
    t3 = t2 * t
    return ((1 - t) ** 3 / 6, (3 * t3 - 6 * t2 + 4) / 6, (-3 * t3 + 3 * t2 + 3 * t + 1) / 6, t3 / 6)


def griglie_fini(lon, lat):
    """Le griglie fini sui pixel (lon: colonne, lat: righe, centri dei pixel): per ognuna che tocca la
    tessera, (peso, profondità) n×n. Il livello della piramide è quello con la cella più vicina al pixel
    (mai più fine del pixel: niente sfarfallio da lontano)."""
    out = []
    pix = abs(float(lat[0] - lat[-1])) / max(len(lat) - 1, 1)
    lo0, lo1, la0, la1 = float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())
    for g in G["hr"]:
        o, s, e, n = g["bbox"]
        if e < lo0 or o > lo1 or n < la0 or s > la1:
            continue
        livs = g["livelli"]
        cand = [L for L in livs if L["passo_lat"] <= pix * 1.01]
        L = cand[-1] if cand else livs[0]
        u = (lon - L["ovest"]) / L["passo_lon"] - 0.5
        v = (L["nord"] - lat) / L["passo_lat"] - 0.5
        H, W = L["righe"], L["colonne"]
        cu = (u >= 0) & (u <= W - 1)
        cv = (v >= 0) & (v <= H - 1)
        if not cu.any() or not cv.any() or H < 2 or W < 2:
            continue
        iu = np.clip(np.floor(u).astype(np.int64), 0, W - 2)
        iv = np.clip(np.floor(v).astype(np.int64), 0, H - 2)
        fu = np.clip(u - iu, 0.0, 1.0)
        fv = np.clip(v - iv, 0.0, 1.0)
        r0, r1 = int(iv[cv].min()), int(iv[cv].max()) + 2
        c0, c1 = int(iu[cu].min()), int(iu[cu].max()) + 2
        dw = np.asarray(L["DW"][r0:r1, c0:c1], dtype=np.float32)      # una lettura sola dal disco
        ww = np.asarray(L["W"][r0:r1, c0:c1], dtype=np.float32)
        ivr = np.clip(iv - r0, 0, r1 - r0 - 2)
        iuc = np.clip(iu - c0, 0, c1 - c0 - 2)

        def bil(A):
            a = A[np.ix_(ivr, iuc)] * (1 - fu)[None, :] + A[np.ix_(ivr, iuc + 1)] * fu[None, :]
            b = A[np.ix_(ivr + 1, iuc)] * (1 - fu)[None, :] + A[np.ix_(ivr + 1, iuc + 1)] * fu[None, :]
            return a * (1 - fv)[:, None] + b * fv[:, None]

        sw = np.where(cv[:, None] & cu[None, :], bil(ww), 0.0)
        if float(sw.max()) <= 0:
            continue
        d = np.where(sw > 1e-4, bil(dw) / np.maximum(sw, 1e-6), 0.0)
        out.append((np.clip(sw, 0.0, 1.0), d))
    return out


def profondita(z, x, y, bordo, lato=T):
    """La profondità (m, ≥ 0; terra 0) ai centri dei pixel della tessera (lato×lato), con `bordo` pixel
    in più per lato (presi dalle vicine: le linee continuano da una tessera all'altra), dove sta fuori
    dalla zona dei dati, e il peso più alto delle griglie fini nella tessera."""
    meta, F = G["meta"], G["F"]
    n = lato + 2 * bordo
    mondo = lato * (1 << z)
    k = np.arange(n, dtype=np.float64) - bordo + 0.5
    lon = (x * lato + k) / mondo * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y * lato + k) / mondo))))
    u = (lon - meta["ovest"]) / meta["passo"] - 0.5
    v = (meta["nord"] - lat) / meta["passo"] - 0.5
    fuori = (lon[None, :] < ZONA[0]) | (lon[None, :] > ZONA[2]) | (lat[:, None] < ZONA[1]) | (lat[:, None] > ZONA[3])
    iu, iv = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    wu, wv = pesi(u - iu), pesi(v - iv)
    H, W = F.shape
    r0, r1 = max(0, int(iv.min()) - 1), min(H, int(iv.max()) + 3)
    c0, c1 = max(0, int(iu.min()) - 1), min(W, int(iu.max()) + 3)
    pezzo = np.asarray(F[r0:r1, c0:c1], dtype=np.float32)       # una lettura sola dal disco
    out = np.zeros((n, n), dtype=np.float64)
    for j in range(4):
        rr = np.clip(iv - 1 + j, r0, r1 - 1) - r0
        for i in range(4):
            cc = np.clip(iu - 1 + i, c0, c1 - 1) - c0
            out += np.outer(wv[j], wu[i]) * pezzo[np.ix_(rr, cc)]
    # le griglie fini sopra il DTM generale, sfumate col loro peso
    hrmax = 0.0
    for sw, d in griglie_fini(lon, lat):
        out = out * (1 - sw) + d * sw
        hrmax = max(hrmax, float(sw.max()))
    out[fuori] = 0.0
    return out.astype(np.float32), fuori, hrmax


# ── più leggera: la zona piena di ogni fascia ─────────────────────────────────────────────────
M = 0.01                                        # margine (m) dai bordi delle zone


def _zone_piene():
    """[(profondità minima, massima, valore fisso)]: dove la carta colora pieno (fuori dall'ombra
    dell'isobata e dal passaggio di 2 cm fra le fasce). Valori fissi interi (esatti in terrarium)."""
    z = [(1.2 + M, LIVELLI[1] - M)]             # fascia 0: piena da 1,2 m (sopra sfuma nella riva) a 5
    for i in range(1, len(LIVELLI) - 1):
        lo, hi = LIVELLI[i], LIVELLI[i + 1]
        z.append((lo + OMBRA * (hi - lo) + M, hi - M))
    z[-1] = (z[-1][0], 1e9)                     # l'ultima fascia non ha fondo
    return [(lo, hi, -float(round((lo + min(hi, lo + 500)) / 2))) for lo, hi in z]


ZONE = _zone_piene()


def indice_zona(v):
    """Per ogni quota: l'indice della zona piena, -1 trasparente (terra e riva oltre il metro), -2 altro."""
    d = -v
    idx = np.full(v.shape, -2, dtype=np.int16)
    idx[v >= -1 + M] = -1
    for k, (lo, hi, _) in enumerate(ZONE):
        idx[(d >= lo) & (d <= hi)] = k
    return idx


def alleggerisci(v):
    """Il valore fisso della fascia ai pixel che stanno, con tutti gli 8 vicini, nella stessa zona piena.
    Il bordo della tessera resta esatto (i suoi vicini stanno nella tessera accanto)."""
    idx = indice_zona(v)
    p = np.pad(idx, 1, mode="edge")
    H, W = v.shape
    stessa = idx != -2
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            stessa &= p[dr:dr + H, dc:dc + W] == idx
    stessa[0, :] = stessa[-1, :] = stessa[:, 0] = stessa[:, -1] = False
    fisso = np.where(idx == -1, 1.0, 0.0)
    for k, (_, _, val) in enumerate(ZONE):
        fisso = np.where(idx == k, val, fisso)
    return np.where(stessa, fisso, v).astype(np.float32)


def _regole_colore():
    """Le regole di colore della carta (cartaStile.ts: carta-nautica con l'ombra e carta-nautica-piena),
    con colori di fascia a caso ma tutti diversi: due quote danno lo stesso colore solo se la carta le
    disegna uguali. (Il primo controllo guardava solo le zone col margine: segnalava come «cambiate»
    tessere dove i colori erano identici — 11 su 537 nel giro del 23/10/2026, ricontrollate a mano.)"""
    nb = len(LIVELLI) - 1
    tinta = np.random.default_rng(1).uniform(0.1, 1.0, size=(nb, 3))
    rgba = lambda c, a=1.0: np.array([c[0], c[1], c[2], a])
    regole = []
    for ombra in (True, False):
        s = [(-LIVELLI[nb], rgba(tinta[nb - 1]))]
        for i in range(nb - 1, 0, -1):
            sopra, spessore = LIVELLI[i], LIVELLI[i + 1] - LIVELLI[i]
            if ombra:
                s += [(-(sopra + OMBRA * spessore), rgba(tinta[i])), (-(sopra + 0.02), rgba(tinta[i] * 0.66))]
            else:
                s.append((-(sopra + 0.02), rgba(tinta[i])))
            s.append((-sopra, rgba(tinta[i - 1])))
        s += [(-1.2, rgba(tinta[0])), (-1.0, np.zeros(4)), (0.0, np.zeros(4))]
        regole.append((np.array([q for q, _ in s]), np.array([c for _, c in s])))
    return regole


REGOLE = _regole_colore()


def disegno(v, fattore=3):
    """I colori della carta sulla tessera, campionati con l'interpolazione bilineare su una griglia
    `fattore` volte più fitta (come li disegna la scheda video)."""
    H, W = v.shape
    t = np.arange(fattore * (H - 1) + 1) / fattore
    i0 = np.minimum(np.floor(t).astype(int), H - 2)
    a = v[i0, :] * (1 - (t - i0))[:, None] + v[i0 + 1, :] * (t - i0)[:, None]
    s = np.arange(fattore * (W - 1) + 1) / fattore
    j0 = np.minimum(np.floor(s).astype(int), W - 2)
    e = a[:, j0] * (1 - (s - j0))[None, :] + a[:, j0 + 1] * (s - j0)[None, :]
    out = []
    for q, c in REGOLE:
        k = np.clip(np.searchsorted(q, e, side="right") - 1, 0, len(q) - 2)
        f = np.clip((e - q[k]) / (q[k + 1] - q[k]), 0.0, 1.0)
        col = c[k] * (1 - f)[..., None] + c[k + 1] * f[..., None]
        col[e <= q[0]] = c[0]
        col[e >= q[-1]] = c[-1]
        out.append(col)
    return np.stack(out)


def terrarium(v):
    """metri → immagine terrarium. Il mare con una precisione che segue il fondo: 1/256 di metro sotto
    i 50 m, poi 1/64, 1/16, 1/4 e mezzo metro oltre i 1500 m (come in fondali-1: niente gradini)."""
    v = v.astype(np.float64)
    d = np.maximum(-v, 0.0)
    passo = np.where(d < 50, 1 / 256, np.where(d < 150, 1 / 64, np.where(d < 500, 1 / 16, np.where(d < 1500, 1 / 4, 1 / 2))))
    q = np.where(v < 0, -np.round(d / passo) * passo, v)
    w = q + 32768.0
    r = np.floor(w / 256)
    g = np.floor(w - r * 256)
    b = np.round((w - r * 256 - g) * 256)
    over = b >= 256
    g = np.where(over, g + 1, g)
    b = np.where(over, 0, b)
    rgb = np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)
    return rgb


def da_terrarium(rgb):
    rgb = rgb.astype(np.float64)
    return rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768


def webp_leggera(v, z, x, y):
    """Il WebP della tessera alleggerita, e se va controllata: True se il disegno è rimasto uguale."""
    rgb = terrarium(alleggerisci(v))
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, format="WEBP", lossless=True, quality=100, method=6)
    controllo = None
    if (hash((z, x, y)) % 1000) < CONTROLLO * 1000:
        piena = da_terrarium(terrarium(v))
        controllo = bool(np.allclose(disegno(piena), disegno(da_terrarium(rgb)), rtol=0, atol=1e-9))
    return buf.getvalue(), controllo


def semplifica(p, tol):
    """Douglas–Peucker (p: N×2)."""
    if len(p) < 3:
        return p
    tieni = np.zeros(len(p), dtype=bool)
    tieni[0] = tieni[-1] = True
    pila = [(0, len(p) - 1)]
    while pila:
        a, b = pila.pop()
        if b <= a + 1:
            continue
        d = p[b] - p[a]
        L = math.hypot(d[0], d[1]) or 1e-9
        seg = p[a + 1:b]
        dist = np.abs((seg[:, 0] - p[a, 0]) * d[1] - (seg[:, 1] - p[a, 1]) * d[0]) / L
        k = int(np.argmax(dist))
        if dist[k] > tol:
            m = a + 1 + k
            tieni[m] = True
            pila.append((a, m))
            pila.append((m, b))
    return p[tieni]


def varint(n, out):
    while n > 0x7F:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    out.append(n)


def campo(numero, tipo, out):
    varint((numero << 3) | tipo, out)


def mvt_linee(linee):
    """[(m, punti N×2 in pixel della tessera)] → MVT (livello «isobate», campo m, estensione 4096)."""
    S = 4096 / T
    zz = lambda v: v * 2 if v >= 0 else -v * 2 - 1
    valori = sorted({m for m, _ in linee})
    strato = bytearray()
    campo(15, 0, strato); varint(2, strato)
    nome = b"isobate"
    campo(1, 2, strato); varint(len(nome), strato); strato.extend(nome)
    for m, p in linee:
        q = np.round(p * S).astype(np.int64)
        keep = np.ones(len(q), dtype=bool)
        keep[1:] = np.any(q[1:] != q[:-1], axis=1)
        q = q[keep]
        if len(q) < 2:
            continue
        geo = [9, zz(int(q[0, 0])), zz(int(q[0, 1])), 2 | ((len(q) - 1) << 3)]
        dq = np.diff(q, axis=0)
        for dx, dy in dq.tolist():
            geo.append(zz(dx)); geo.append(zz(dy))
        el = bytearray()
        tags = bytearray(); varint(0, tags); varint(valori.index(m), tags)
        campo(2, 2, el); varint(len(tags), el); el.extend(tags)
        campo(3, 0, el); varint(2, el)
        g = bytearray()
        for v in geo:
            varint(v, g)
        campo(4, 2, el); varint(len(g), el); el.extend(g)
        campo(2, 2, strato); varint(len(el), strato); strato.extend(el)
    chiave = b"m"
    campo(3, 2, strato); varint(len(chiave), strato); strato.extend(chiave)
    for v in valori:
        val = bytearray(); campo(5, 0, val); varint(int(v), val)
        campo(4, 2, strato); varint(len(val), strato); strato.extend(val)
    campo(5, 0, strato); varint(4096, strato)
    tessera = bytearray()
    campo(3, 2, tessera); varint(len(strato), tessera); tessera.extend(strato)
    return bytes(tessera)


def linee_di(D, bordo, scala=1):
    """Le curve dei confini di fascia su una griglia di profondità (con `bordo` pixel per lato), in pixel
    della tessera (la griglia può essere `scala` volte più fitta della tessera)."""
    cg = contourpy.contour_generator(z=D, name="serial", line_type="Separate")
    lo, hi = float(D.min()), float(D.max())
    out = []
    for L in LINEE:
        if L <= lo or L >= hi:
            continue
        for p in cg.lines(L):
            if len(p) < 4:
                continue
            p = (p - bordo + 0.5) / scala                   # pixel della tessera (centri a i+0.5)
            if np.hypot(*np.diff(p, axis=0).T).sum() < 14:  # pezzetti: lì una scritta non ci sta
                continue
            out.append((L, semplifica(p, 0.25)))
    return out


def combina(D, mare):
    """Profondità (≥0) + maschera OSM → metri da scrivere (mare negativo, terra +1)."""
    acqua = mare >= 0.5
    terra = ~acqua
    v = np.where(acqua, -np.maximum(D, 0.5), 1.0).astype(np.float32)
    # lungo la riva al più un metro: la fascia piena comincia un pixel dentro l'acqua
    vicino = np.zeros_like(terra)
    tp = np.pad(terra, 1, mode="edge")
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            vicino |= tp[dr:dr + terra.shape[0], dc:dc + terra.shape[1]]
    v = np.where(acqua & vicino, np.maximum(v, -1.0), v)
    return v, acqua


def scrivi(percorso, dati):
    os.makedirs(os.path.dirname(percorso), exist_ok=True)
    with open(percorso, "wb") as f:
        f.write(dati)


def fai_fine(args):
    """Una tessera da z10 in su, dal dato. `obbligata`: serve comunque (tutto il mare a z10, la
    piattaforma a z11-z12); altrimenti è lì solo per una griglia fine, e se quella non c'è si salta.
    `fine`: z12 con una griglia da 1/128 di primo o più fine (le linee si calcolano due volte più fitte).
    Restituisce (z, x, y, fatta, minima profondità, controllo, profondità e maschera se z10)."""
    z, x, y, uscita, obbligata, fine = args
    B = 1
    D, fuori, hrmax = profondita(z, x, y, B)
    if not obbligata and hrmax < PESO_MINIMO:
        return (z, x, y, False, None, None, None, None)
    mare_c = mare_osm(z, x, y, T)
    mare_c[fuori[B:-B, B:-B]] = 0.0
    Dc = D[B:-B, B:-B]
    v, acqua = combina(Dc, mare_c)
    if not acqua.any():
        return (z, x, y, False, None, None, None, None)
    minimo = float(Dc[acqua].min())
    dati, controllo = webp_leggera(v, z, x, y)
    scrivi(os.path.join(uscita, "profondita", str(z), str(x), f"{y}.webp"), dati)
    if z <= Z_COSTA[1]:
        # le linee: la profondità con la terra a zero (OSM), col bordo preso dalle vicine
        if fine and z == Z_COSTA[1]:
            S2 = 2
            D2, _, _ = profondita(z, x, y, B, T * S2)
            m2 = mare_osm(z, x, y, T * S2) >= 0.5
            Dl = D2.copy()
            Dl[B:-B, B:-B] = np.where(m2, D2[B:-B, B:-B], 0.0)
            linee = linee_di(Dl, B, S2)
        else:
            Dl = D.copy()
            Dl[B:-B, B:-B] = np.where(acqua, Dc, 0.0)
            linee = linee_di(Dl, B)
        scrivi(os.path.join(uscita, "isobate", str(z), str(x), f"{y}.pbf"), mvt_linee(linee))
    ritorno = (Dc.astype(np.float32), mare_c.astype(np.float32)) if z == Z_PIENE[1] else (None, None)
    return (z, x, y, True, minimo, controllo, ritorno[0], ritorno[1])


def tessera_di(z, lon, lat):
    n = 1 << z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def tessere_zona(z, zona=ZONA):
    o, s, e, nn = zona
    x0, y0 = tessera_di(z, o, nn)
    x1, y1 = tessera_di(z, e, s)
    return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def zoom_massimo(cella_primi):
    """Fino a che zoom una griglia fine regge: 1/32 di primo → z11, 1/64 → z12, 1/128 → z13, 1/256 → z14."""
    return min(Z_MAX, 10 + max(0, round(math.log2(1 / 16 / cella_primi))))


def disponibili(uscita):
    """Per zoom: il riquadro delle tessere esistenti e un bit per tessera (base64 di zlib)."""
    out = {}
    for cart in ("profondita", "isobate"):
        out[cart] = {}
        radice = os.path.join(uscita, cart)
        if not os.path.isdir(radice):
            continue
        for zs in sorted(os.listdir(radice), key=int):
            tess = []
            for xs in os.listdir(os.path.join(radice, zs)):
                for f in os.listdir(os.path.join(radice, zs, xs)):
                    tess.append((int(xs), int(f.split(".")[0])))
            if not tess:
                continue
            xs_, ys_ = [t[0] for t in tess], [t[1] for t in tess]
            x0, y0 = min(xs_), min(ys_)
            w, h = max(xs_) - x0 + 1, max(ys_) - y0 + 1
            bit = bytearray((w * h + 7) // 8)
            for x, y in tess:
                i = (y - y0) * w + (x - x0)
                bit[i >> 3] |= 1 << (i & 7)
            out[cart][zs] = {"x0": x0, "y0": y0, "w": w, "h": h, "n": len(tess),
                             "bit": base64.b64encode(zlib.compress(bytes(bit), 9)).decode("ascii")}
    return out


def principale():
    data, coste, hr, uscita = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    processi = int(sys.argv[sys.argv.index("--processi") + 1]) if "--processi" in sys.argv else 8
    prova = "--prova" in sys.argv
    t0 = time.time()
    os.makedirs(uscita, exist_ok=True)
    if not os.path.exists(os.path.join(data, "fondali_liscio.npy")):
        sys.exit("manca fondali_liscio.npy: lo fa fondali_tessere.py (la prima versione) al primo giro")
    indice_hr = json.load(open(os.path.join(hr, "indice.json"), encoding="utf-8"))
    apri(data, coste, None)                      # qui servono solo le tessere della zona
    controlli = [0, 0]                           # (fatti, disegno cambiato)

    if prova:
        def vicine(z, lon, lat, r):
            cx, cy = tessera_di(z, lon, lat)
            return [(cx + dx, cy + dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)]
        posti = [(13.95, 40.73), (16.43, 41.30), (12.33, 45.37)]          # Ischia-Procida, Trani, Venezia
        dieci = sorted({t for lon, lat in posti for t in vicine(10, lon, lat, 2)})
    else:
        dieci = tessere_zona(10)

    def conta(r):
        if r[5] is not None:
            controlli[0] += 1
            if not r[5]:
                controlli[1] += 1
                print(f"⚠️ disegno cambiato in z{r[0]}/{r[1]}/{r[2]}", flush=True)

    with Pool(processi, initializer=apri, initargs=(data, coste, hr)) as pool:
        # 1) z10: tutte le tessere della zona (dal dato)
        risultati = {10: {}}
        lavori = [(10, x, y, uscita, True, False) for x, y in dieci]
        for i, r in enumerate(pool.imap_unordered(fai_fine, lavori, chunksize=4)):
            conta(r)
            if r[3]:
                risultati[10][(r[1], r[2])] = r
            if (i + 1) % 200 == 0:
                print(f"z10: {i + 1}/{len(lavori)} ({len(risultati[10])} col mare, {time.time() - t0:.0f} s)", flush=True)
        print(f"z10 fatte: {len(risultati[10])} col mare ({time.time() - t0:.0f} s)", flush=True)

        # 2) z11 … z14: la piattaforma (z11-z12, fondo sotto i 200 m nella tessera madre) e le griglie fini
        for z in range(Z_COSTA[0], Z_MAX + 1):
            obbligate = set()
            if z <= Z_COSTA[1]:
                for (x, y), r in risultati[z - 1].items():
                    if r[4] is not None and r[4] <= PIATTAFORMA:
                        obbligate.update((2 * x + dx, 2 * y + dy) for dx in (0, 1) for dy in (0, 1))
            per_fini, fini12 = set(), set()
            for g in indice_hr:
                if zoom_massimo(g["cella_primi"]) < z:
                    continue
                tt = tessere_zona(z, tuple(g["bbox"]))
                per_fini.update(tt)
                if z == Z_COSTA[1] and g["cella_primi"] <= 1 / 128 + 1e-9:
                    fini12.update(tt)
            # solo dentro le tessere madri fatte (la carta ingrandisce la madre dove la figlia manca)
            madri = risultati[z - 1]
            per_fini = {(x, y) for x, y in per_fini if (x // 2, y // 2) in madri}
            if prova:
                obbligate = {t for t in obbligate if (t[0] // 2, t[1] // 2) in madri}
            tutte = obbligate | per_fini
            lavori = [(z, x, y, uscita, (x, y) in obbligate, (x, y) in fini12) for x, y in sorted(tutte)]
            fatte = {}
            for i, r in enumerate(pool.imap_unordered(fai_fine, lavori, chunksize=8)):
                conta(r)
                if r[3]:
                    fatte[(r[1], r[2])] = r
                if (i + 1) % 1000 == 0:
                    print(f"z{z}: {i + 1}/{len(lavori)} ({time.time() - t0:.0f} s)", flush=True)
            risultati[z] = fatte
            print(f"z{z} fatte: {len(fatte)} su {len(lavori)} (piattaforma {len(obbligate)}, griglie fini "
                  f"{len(per_fini - obbligate)}) ({time.time() - t0:.0f} s)", flush=True)

    # 3) z9 … z3: dalle z10, a due a due (media della profondità e della maschera)
    livello = {k: (r[6], r[7]) for k, r in risultati[10].items()}
    for z in range(Z_PIENE[1] - 1, Z_PIENE[0] - 1, -1):
        madri = {}
        for (x, y) in livello:
            madri.setdefault((x // 2, y // 2), []).append((x, y))
        nuovo = {}
        for (X, Y), figlie in madri.items():
            D2 = np.zeros((2 * T, 2 * T), dtype=np.float32)
            M2 = np.zeros((2 * T, 2 * T), dtype=np.float32)
            for (x, y) in figlie:
                d, m = livello[(x, y)]
                ox, oy = (x - 2 * X) * T, (y - 2 * Y) * T
                D2[oy:oy + T, ox:ox + T] = d
                M2[oy:oy + T, ox:ox + T] = m
            # la media della profondità solo sull'acqua (la terra non tira su il fondo)
            W2 = (M2 >= 0.5).astype(np.float32)
            somma = (D2 * W2).reshape(T, 2, T, 2).sum(axis=(1, 3))
            peso = W2.reshape(T, 2, T, 2).sum(axis=(1, 3))
            D = np.where(peso > 0, somma / np.maximum(peso, 1e-6), 0.0).astype(np.float32)
            Mm = M2.reshape(T, 2, T, 2).mean(axis=(1, 3))
            v, acqua = combina(D, Mm)
            if not acqua.any():
                continue
            nuovo[(X, Y)] = (D, Mm)
            dati, controllo = webp_leggera(v, z, X, Y)
            conta((z, X, Y, True, None, controllo))
            scrivi(os.path.join(uscita, "profondita", str(z), str(X), f"{Y}.webp"), dati)
            if z >= 6:
                Dl = np.pad(np.where(acqua, D, 0.0), 1, mode="edge")
                scrivi(os.path.join(uscita, "isobate", str(z), str(X), f"{Y}.pbf"), mvt_linee(linee_di(Dl, 1)))
        print(f"z{z} fatte: {len(nuovo)} ({time.time() - t0:.0f} s)", flush=True)
        livello = nuovo

    # 4) quali tessere esistono, e il riassunto
    disp = disponibili(uscita)
    disp["zoom"] = {"profondita": [Z_PIENE[0], Z_MAX], "isobate": [6, Z_COSTA[1]]}
    with open(os.path.join(uscita, "disponibili.json"), "w", encoding="utf-8") as f:
        json.dump(disp, f, separators=(",", ":"))
    conta_t, peso_t = {}, {}
    for cart in ("profondita", "isobate"):
        for radice, _, files in os.walk(os.path.join(uscita, cart)):
            for f in files:
                zz = int(os.path.relpath(radice, os.path.join(uscita, cart)).split(os.sep)[0])
                conta_t[f"{cart}/{zz}"] = conta_t.get(f"{cart}/{zz}", 0) + 1
                peso_t[cart] = peso_t.get(cart, 0) + os.path.getsize(os.path.join(radice, f))
    produttori = sorted({p for g in indice_hr for p in g.get("produttori", [])})
    meta = {"quando": time.strftime("%Y-%m-%d %H:%M"), "zona": ZONA, "fondali_zoom": [Z_PIENE[0], Z_MAX],
            "isobate_zoom": [6, Z_COSTA[1]], "tessere": conta_t, "mb": {k: round(v / 1e6, 1) for k, v in peso_t.items()},
            "controllo_disegno": {"tessere": controlli[0], "cambiate": controlli[1]},
            "griglie_fini": [{"nome": g["nome"], "cella_primi": g["cella_primi"], "produttori": g.get("produttori", []),
                              "licenza": g.get("licenza", "")} for g in indice_hr],
            "fonti": ["EMODnet Bathymetry DTM 2024 (CC BY 4.0)",
                      "EMODnet Bathymetry, griglie ad alta risoluzione (CC BY 4.0): " + "; ".join(produttori),
                      "OpenStreetMap water polygons, osmdata.openstreetmap.de (ODbL)"],
            "minuti": round((time.time() - t0) / 60, 1)}
    with open(os.path.join(uscita, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print("FINE", json.dumps({k: meta[k] for k in ("tessere", "mb", "controllo_disegno", "minuti")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    principale()
