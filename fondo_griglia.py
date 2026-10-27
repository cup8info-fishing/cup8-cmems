"""fondo_griglia.py — Il fondale leggero per il «fondo caldo / fondo freddo» (render_fondo.py).

La Action di GitHub non ha data/ (il fondale EMODnet a 115 m pesa 670 MB): render_fondo.py usa
questa copia a 1/480° (~230 m), abbastanza fine perché la linea segua le batimetriche della carta
anche sulle coste ripide (a 1/120° in Sardegna e in Liguria si appoggiava sulla costa). Si rifà
solo se cambia il fondale (data/fondali_liscio.npy, fatto da fondali_scarica.py), e si committa.

Ogni cella = 2×2 celle EMODnet: è mare se lo è almeno metà, e la profondità è la media del mare,
in metri interi fino a 254 (255 = più giù: per la linea contano i fondali fino a 150 m).
Uscita: fondo_griglia.npz (fine uint8, 0 = terra; ovest, nord, passo).
Uso: python fondo_griglia.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OVEST, SUD, EST, NORD = 5.5, 35.0, 19.5, 47.2        # la zona della carta (app: zona.ts)
BLOCCO = 2                                            # 2 × 1/960° = 1/480°


def main():
    meta = json.load(open(os.path.join(DATA, "fondali_nativo.json"), encoding="utf-8"))
    F = np.load(os.path.join(DATA, "fondali_liscio.npy"), mmap_mode="r")
    passo = meta["passo"]
    c0 = round((OVEST - meta["ovest"]) / passo)
    c1 = round((EST - meta["ovest"]) / passo)
    r0 = round((meta["nord"] - NORD) / passo)
    r1 = round((meta["nord"] - SUD) / passo)
    assert (c1 - c0) % BLOCCO == 0 and (r1 - r0) % BLOCCO == 0, "la zona non cade sui blocchi"
    righe, colonne = (r1 - r0) // BLOCCO, (c1 - c0) // BLOCCO
    out = np.zeros((righe, colonne), np.uint8)
    for i in range(0, righe, 256):                   # a fette: il file intero non sta in memoria
        n = min(256, righe - i)
        a = np.asarray(F[r0 + i * BLOCCO:r0 + (i + n) * BLOCCO, c0:c1], dtype=np.float32)
        a = a.reshape(n, BLOCCO, colonne, BLOCCO)
        mare = a > 0.5
        quanti = mare.sum(axis=(1, 3))
        media = np.where(mare, a, 0).sum(axis=(1, 3)) / np.maximum(quanti, 1)
        out[i:i + n] = np.where(quanti * 2 >= BLOCCO * BLOCCO, np.clip(np.round(media), 1, 255), 0).astype(np.uint8)
    dest = os.path.join(HERE, "fondo_griglia.npz")
    np.savez_compressed(dest, fine=out, ovest=OVEST, nord=NORD, passo=passo * BLOCCO)
    print(f"fondo_griglia.npz: {colonne}x{righe} celle, mare {int((out > 0).sum())}, "
          f"{os.path.getsize(dest) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
