"""
publish_bathy.py — Prepara i tile batimetrici per la pubblicazione su GitHub Pages.

Copia i tile renderizzati (tiles/, cache locale gitignorata) nella cartella COMMITTATA
bathy_tiles/, SALTANDO i tile completamente vuoti (tutta terra/trasparenti): cosi
finiscono in git e l'Action li pubblica in public/bathy/ (CDN) → visti da chiunque,
senza server ne dati locali.

Uso: python publish_bathy.py
"""
import os
import shutil
import numpy as np
from PIL import Image

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "tiles")
DST = os.path.join(HERE, "bathy_tiles")


def has_sea(path):
    try:
        a = np.array(Image.open(path).convert("RGBA"))
        return bool((a[..., 3] > 10).any())
    except Exception:
        return False


def main():
    if not os.path.isdir(SRC):
        raise SystemExit("manca tiles/ (genera prima con prerender_tiles.py)")
    if os.path.isdir(DST):
        shutil.rmtree(DST)
    n = skip = 0
    total = 0
    for root, _, files in os.walk(SRC):
        for f in files:
            if not f.endswith(".png"):
                continue
            sp = os.path.join(root, f)
            if not has_sea(sp):
                skip += 1
                continue
            dp = os.path.join(DST, os.path.relpath(sp, SRC))
            os.makedirs(os.path.dirname(dp), exist_ok=True)
            shutil.copyfile(sp, dp)
            n += 1
            total += os.path.getsize(sp)
    print(f"[publish] {n} tile con mare -> bathy_tiles/ ({total/1e6:.1f} MB), {skip} vuoti (terra) saltati")


if __name__ == "__main__":
    main()
