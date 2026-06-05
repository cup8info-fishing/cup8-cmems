"""
prerender_tiles.py — Pre-renderizza (warm cache) i tile batimetrici del Mediterraneo
fino a un certo zoom, così le viste panoramiche-regionali sono ISTANTANEE.
Richiede il bathy_tile_server attivo (porta 8097): chiede ogni tile → il server lo
renderizza dal dato locale e lo cacha su disco.

Uso: python prerender_tiles.py [--maxz 8]
Lo zoom alto (oltre maxz) resta on-demand (comunque veloce dal dato locale + prefetch).
"""
import argparse
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

PORT = 8097
LNG_MIN, LNG_MAX = -6.0, 36.0
LAT_MIN, LAT_MAX = 30.0, 46.0


def tiles_for(z):
    n = 180.0 / (2 ** z)
    x0 = math.floor((LNG_MIN + 180) / n); x1 = math.floor((LNG_MAX + 180) / n)
    y0 = math.floor((90 - LAT_MAX) / n); y1 = math.floor((90 - LAT_MIN) / n)
    return [(z, x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def get(t):
    z, x, y = t
    try:
        urllib.request.urlopen(f"http://localhost:{PORT}/btile/{z}/{x}/{y}.png", timeout=60).read()
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minz", type=int, default=3)
    ap.add_argument("--maxz", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    all_t = []
    for z in range(a.minz, a.maxz + 1):
        all_t += tiles_for(z)
    print(f"[prerender] {len(all_t)} tile (z{a.minz}-{a.maxz})", flush=True)
    t0 = time.time(); done = 0; ok = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for r in as_completed([ex.submit(get, t) for t in all_t]):
            done += 1; ok += 1 if r.result() else 0
            if done % 100 == 0 or done == len(all_t):
                print(f"[prerender] {done}/{len(all_t)} ({ok} ok) — {time.time()-t0:.0f}s", flush=True)
    print(f"[prerender] OK {ok}/{len(all_t)} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
