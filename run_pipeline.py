#!/usr/bin/env python3
"""
Pipeline CMEMS per CI/cloud (GitHub Actions) — gira ogni 6h.

Scarica il NetCDF onde (Copernicus Marine), renderizza le PNG (full + eroded) e il
forecast JSON, poi prepara `public/waves/` come SITO STATICO per GitHub Pages.

Credenziali Copernicus: via env (GitHub Secrets), lette da copernicusmarine:
  COPERNICUSMARINE_SERVICE_USERNAME / COPERNICUSMARINE_SERVICE_PASSWORD

Layout statico prodotto (= ciò che leggerà l'app):
  public/waves/meta.json
  public/waves/full/h00.png .. h71.png
  public/waves/eroded/h00.png .. h71.png
  public/waves/forecast_72h_0p2.json
"""
import os
import sys
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
CACHE = os.path.join(HERE, "cache")
PUBLIC = os.path.join(HERE, "public", "waves")
HOURS = 72
STEP = "0.2"
FORECAST_NAME = "forecast_72h_0p2.json"


def run(args):
    print("→ python", *args, flush=True)
    r = subprocess.run([PY] + args, cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"FAILED ({r.returncode}): {' '.join(args)}")


def main():
    os.makedirs(CACHE, exist_ok=True)

    # 1) Download NetCDF (3 giorni = 72h di forecast)
    run(["download_cmems.py", "--days", "3"])
    # 2) Render PNG (full + eroded) + cache/waves_meta.json
    # 6000px = quella attualmente deployata sul CDN (era 3000 qui → avrebbe regredito la
    # risoluzione alla prossima rigenerazione). Con la maschera-costa anti-aliasata =
    # bordi lisci. Se la CI va in timeout a 6000+SUPERSAMPLE=3, abbassare SUPERSAMPLE a 2.
    run(["render_waves_png.py", "--hours", str(HOURS), "--width", "6000"])
    # 3) Forecast JSON compatto (frecce direzione + popup), step 0.2°
    run(["extract_forecast.py", str(HOURS), os.path.join(CACHE, FORECAST_NAME), "--step", STEP])

    # 4) Componi il sito statico public/waves/
    full = os.path.join(PUBLIC, "full")
    eroded = os.path.join(PUBLIC, "eroded")
    for d in (full, eroded):
        os.makedirs(d, exist_ok=True)
    shutil.copyfile(os.path.join(CACHE, "waves_meta.json"), os.path.join(PUBLIC, "meta.json"))
    shutil.copyfile(os.path.join(CACHE, FORECAST_NAME), os.path.join(PUBLIC, FORECAST_NAME))

    n = 0
    for i in range(HOURS):
        for suffix, dest in (("", full), ("_eroded", eroded)):
            src = os.path.join(CACHE, f"waves_h{i:02d}{suffix}.png")
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(dest, f"h{i:02d}.png"))
                n += 1

    # Pagina indice/health (root del sito)
    with open(os.path.join(HERE, "public", "index.html"), "w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset=utf-8><title>cup8 CMEMS</title>"
                "<p>cup8 CMEMS static data — vedi <a href='waves/meta.json'>waves/meta.json</a></p>")

    print(f"OK: public/waves pronta — {n} PNG + meta.json + {FORECAST_NAME}", flush=True)


if __name__ == "__main__":
    main()
