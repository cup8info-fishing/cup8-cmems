"""
fondali_tessere.py — Le tessere GIÀ PRONTE del mare della carta di Spot (app cap8).

Prima il telefono si calcolava il mare tessera per tessera (worker, servizio WCS di EMODnet chiamato
dal vivo, maschera della riva da OpenStreetMap): lento, e quando il servizio non rispondeva un pezzo
usciva dalle fasce e non combaciava. Qui si fa tutto UNA volta, sul PC, con calma e con tutti i dati:

  profondità   EMODnet DTM alla risoluzione vera (fondali_scarica.py), lisciata appena (gaussiana di una
               cella) e campionata con la B-spline cubica: bordi delle fasce lisci, niente dentini;
  riva         il mare di OpenStreetMap (fondali_coste.py): dove OSM non dice mare è terra, quindi saline,
               lagune e laghi restano col colore d'acqua della carta; lungo la riva il mare non va oltre
               il metro (la fascia piena comincia un pixel dentro l'acqua e la riva è quella di OSM);
  metri        le linee dei confini di fascia (5 … 4000 m) dalle stesse profondità, già pronte.

Uscita:
  <uscita>/profondita/{z}/{x}/{y}.webp  «terrarium» (WebP senza perdite: il 37% meno del PNG) (metri = R·256 + G + B/256 − 32768): mare negativo,
                                     terra +1 (trasparente nella carta). z3–z10 tutto il mare della zona,
                                     z11–z12 solo dove c'è fondo sotto i 200 m (piattaforma, sotto costa)
  <uscita>/isobate/{z}/{x}/{y}.pbf   MVT, livello «isobate», campo m (metri): z6–z12, tutto il mare
  <uscita>/meta.json

Uso: python fondali_tessere.py <data> <coste.npz> <uscita> [--processi N] [--prova]
     --prova: solo le tessere attorno a Trani e a Zapponeta (per guardarle prima del giro intero)
"""
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # la console di Windows non è in UTF-8
from PIL import Image, ImageDraw
import contourpy

T = 256
R = 6378137.0
MONDO = 2 * math.pi * R
LIVELLI = [0, 5, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 600, 800, 1200, 1700, 2300, 3000, 4000, 5500]
LINEE = LIVELLI[1:-1]
ZONA = (5.5, 35.0, 19.5, 47.2)                  # zona.ts: ovest, sud, est, nord
Z_PIENE = (3, 10)                               # da z3 a z10 tutto il mare
Z_COSTA = (11, 12)                              # z11 e z12 solo sulla piattaforma
PIATTAFORMA = 200.0                             # metri: sotto, da vicino servono le tessere fini
SOVRA = 4                                       # la maschera della riva disegnata 4 volte più fine

# ── condivisi nei processi ──────────────────────────────────────────────────────────────────
G = {}


def apri(data, coste):
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


def profondita(z, x, y, bordo):
    """La profondità lisciata (m, ≥ 0; terra 0) ai centri dei pixel della tessera, con `bordo` pixel
    in più per lato (presi dalle vicine: le linee continuano da una tessera all'altra), e dove sta
    fuori dalla zona dei dati."""
    meta, F = G["meta"], G["F"]
    n = T + 2 * bordo
    mondo = T * (1 << z)
    k = np.arange(n, dtype=np.float64) - bordo + 0.5
    lon = (x * T + k) / mondo * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y * T + k) / mondo))))
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
    out[fuori] = 0.0
    return out.astype(np.float32), fuori


def terrarium(v):
    """metri → immagine terrarium. Il mare con una precisione che segue il fondo: 1/256 di metro sotto
    i 50 m (dove le fasce sono larghe 5 m), poi 1/64, 1/16, 1/4 e mezzo metro oltre i 1500 m — sempre
    molto più fine dell'ombra della fascia, quindi niente gradini (provato il 16/10/2026: fasce e ombre
    identiche alla precisione piena), e le tessere pesano il 44% in meno. Un arrotondamento uguale per
    tutti (al decimetro) faceva gradini dove il fondo scende piano: bordi a dentini e ombre a scala."""
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


def linee_di(D, bordo):
    """Le curve dei confini di fascia su una griglia di profondità (con `bordo` pixel per lato)."""
    cg = contourpy.contour_generator(z=D, name="serial", line_type="Separate")
    lo, hi = float(D.min()), float(D.max())
    out = []
    for L in LINEE:
        if L <= lo or L >= hi:
            continue
        for p in cg.lines(L):
            if len(p) < 4:
                continue
            p = p - bordo                                   # pixel della tessera (centri a i+0.5 → +0.5)
            p = p + 0.5
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
    """Una tessera da z10 in su: dal dato. Restituisce (z, x, y, c'è mare, minima profondità,
    profondità e maschera 256×256 se z10 — servono per le tessere più larghe)."""
    z, x, y, uscita, con_png = args
    B = 1
    D, fuori = profondita(z, x, y, B)
    mare_c = mare_osm(z, x, y, T)
    mare_c[fuori[B:-B, B:-B]] = 0.0
    Dc = D[B:-B, B:-B]
    v, acqua = combina(Dc, mare_c)
    if not acqua.any():
        return (z, x, y, False, None, None, None)
    minimo = float(Dc[acqua].min())
    if con_png:
        buf = __import__("io").BytesIO()
        Image.fromarray(terrarium(v), "RGB").save(buf, format="WEBP", lossless=True, quality=100, method=4)
        scrivi(os.path.join(uscita, "profondita", str(z), str(x), f"{y}.webp"), buf.getvalue())
    # le linee: la profondità con la terra a zero (OSM), col bordo preso dalle vicine
    Dl = D.copy()
    terra_b = np.ones_like(D, dtype=bool)
    terra_b[B:-B, B:-B] = ~acqua
    # sul bordo (fuori dalla tessera) vale il dato: la riva OSM lì non si conosce, si usa la profondità
    Dl[B:-B, B:-B] = np.where(acqua, Dc, 0.0)
    linee = linee_di(Dl, B)
    scrivi(os.path.join(uscita, "isobate", str(z), str(x), f"{y}.pbf"), mvt_linee(linee))
    ritorno = (Dc.astype(np.float32), mare_c.astype(np.float32)) if z == Z_PIENE[1] else (None, None)
    return (z, x, y, True, minimo, ritorno[0], ritorno[1])


def tessere_zona(z):
    n = 1 << z
    o, s, e, nn = ZONA
    x0 = int((o + 180) / 360 * n)
    x1 = int((e + 180) / 360 * n)
    ty = lambda lat: int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    return [(x, y) for x in range(x0, x1 + 1) for y in range(ty(nn), ty(s) + 1)]


def principale():
    data, coste, uscita = sys.argv[1], sys.argv[2], sys.argv[3]
    processi = int(sys.argv[sys.argv.index("--processi") + 1]) if "--processi" in sys.argv else 8
    prova = "--prova" in sys.argv
    t0 = time.time()
    os.makedirs(uscita, exist_ok=True)

    # 1) la profondità lisciata, una volta per tutte (gaussiana σ = una cella; la terra conta zero)
    liscio = os.path.join(data, "fondali_liscio.npy")
    if not os.path.exists(liscio):
        meta = json.load(open(os.path.join(data, "fondali_nativo.json"), encoding="utf-8"))
        Gr = np.load(os.path.join(data, meta["file"]), mmap_mode="r")
        H, W = Gr.shape
        Fo = np.lib.format.open_memmap(liscio + ".tmp.npy", mode="w+", dtype=np.float32, shape=(H, W))
        k = np.exp(-0.5 * (np.arange(-3, 4) / 1.0) ** 2)
        k /= k.sum()
        passo = 1024
        for r in range(0, H, passo):
            a, b = max(0, r - 3), min(H, r + passo + 3)
            g = np.asarray(Gr[a:b], dtype=np.float32)
            dep = np.where(g < 0, -g, 0.0).astype(np.float32)          # NaN e terra → 0
            p = np.pad(dep, ((0, 0), (3, 3)), mode="edge")
            h = sum(k[i] * p[:, i:i + W] for i in range(7))
            p = np.pad(h, ((3, 3), (0, 0)), mode="edge")
            v = sum(k[i] * p[i:i + h.shape[0], :] for i in range(7))
            Fo[r:min(H, r + passo)] = v[r - a:r - a + min(passo, H - r)]
            print(f"lisciata {min(H, r + passo)}/{H} righe", flush=True)
        Fo.flush()
        del Fo
        os.replace(liscio + ".tmp.npy", liscio)
    print(f"profondità lisciata pronta ({time.time() - t0:.0f} s)", flush=True)

    apri(data, coste)
    if prova:
        def vicine(z, lon, lat, r):
            n = 1 << z
            cx = int((lon + 180) / 360 * n)
            cy = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
            return [(cx + dx, cy + dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)]
        dieci = sorted(set(vicine(10, 16.43, 41.30, 2) + vicine(10, 15.96, 41.45, 2)))
    else:
        dieci = tessere_zona(10)

    # 2) z10: tutte le tessere della zona (dal dato)
    risultati10 = {}
    lavori = [(10, x, y, uscita, True) for x, y in dieci]
    with Pool(processi, initializer=apri, initargs=(data, coste)) as pool:
        for i, r in enumerate(pool.imap_unordered(fai_fine, lavori, chunksize=4)):
            if r[3]:
                risultati10[(r[1], r[2])] = r
            if (i + 1) % 100 == 0:
                print(f"z10: {i + 1}/{len(lavori)} ({len(risultati10)} col mare, {time.time() - t0:.0f} s)", flush=True)
    print(f"z10 fatte: {len(risultati10)} col mare ({time.time() - t0:.0f} s)", flush=True)

    # 3) z11 e z12 solo sulla piattaforma (fondo sotto i 200 m nella tessera madre)
    for z in Z_COSTA:
        madri = risultati10 if z == 11 else risultati11
        lavori = []
        for (x, y), r in madri.items():
            if r[4] is None or r[4] > PIATTAFORMA:
                continue
            for dx in (0, 1):
                for dy in (0, 1):
                    lavori.append((z, 2 * x + dx, 2 * y + dy, uscita, True))
        fatte = {}
        with Pool(processi, initializer=apri, initargs=(data, coste)) as pool:
            for i, r in enumerate(pool.imap_unordered(fai_fine, lavori, chunksize=8)):
                if r[3]:
                    fatte[(r[1], r[2])] = r
                if (i + 1) % 500 == 0:
                    print(f"z{z}: {i + 1}/{len(lavori)} ({time.time() - t0:.0f} s)", flush=True)
        print(f"z{z} fatte: {len(fatte)} col mare su {len(lavori)} ({time.time() - t0:.0f} s)", flush=True)
        if z == 11:
            risultati11 = fatte

    # 4) z9 … z5: dalle z10, a due a due (media della profondità e della maschera)
    livello = {k: (r[5], r[6]) for k, r in risultati10.items()}
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
            M = M2.reshape(T, 2, T, 2).mean(axis=(1, 3))
            v, acqua = combina(D, M)
            if not acqua.any():
                continue
            nuovo[(X, Y)] = (D, M)
            buf = __import__("io").BytesIO()
            Image.fromarray(terrarium(v), "RGB").save(buf, format="WEBP", lossless=True, quality=100, method=4)
            scrivi(os.path.join(uscita, "profondita", str(z), str(X), f"{Y}.webp"), buf.getvalue())
            if z >= 6:
                Dl = np.pad(np.where(acqua, D, 0.0), 1, mode="edge")
                scrivi(os.path.join(uscita, "isobate", str(z), str(X), f"{Y}.pbf"), mvt_linee(linee_di(Dl, 1)))
        print(f"z{z} fatte: {len(nuovo)} ({time.time() - t0:.0f} s)", flush=True)
        livello = nuovo

    # 5) il riassunto
    conta, peso = {}, {}
    for cart in ("profondita", "isobate"):
        for radice, _, files in os.walk(os.path.join(uscita, cart)):
            for f in files:
                zz = int(os.path.relpath(radice, os.path.join(uscita, cart)).split(os.sep)[0])
                conta[f"{cart}/{zz}"] = conta.get(f"{cart}/{zz}", 0) + 1
                peso[cart] = peso.get(cart, 0) + os.path.getsize(os.path.join(radice, f))
    meta = {"quando": time.strftime("%Y-%m-%d %H:%M"), "zona": ZONA, "fondali_zoom": [Z_PIENE[0], Z_COSTA[1]],
            "isobate_zoom": [6, Z_COSTA[1]], "tessere": conta, "mb": {k: round(v / 1e6, 1) for k, v in peso.items()},
            "fonti": ["EMODnet Bathymetry DTM (CC BY 4.0)", "OpenStreetMap water polygons, osmdata.openstreetmap.de (ODbL)"],
            "minuti": round((time.time() - t0) / 60, 1)}
    with open(os.path.join(uscita, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print("FINE", json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    principale()
