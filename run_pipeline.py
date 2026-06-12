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
CURRENTS_NAME = "currents_72h_0p2.json"


def run(args):
    print("→ python", *args, flush=True)
    r = subprocess.run([PY] + args, cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"FAILED ({r.returncode}): {' '.join(args)}")


def run_soft(args):
    """Come run() ma NON-FATALE: se fallisce logga e prosegue. Per gli step ADDITIVI
    (es. SST): un intoppo transitorio su SST non deve buttare giù onde/correnti/bathy."""
    print("→ python", *args, "(best-effort)", flush=True)
    r = subprocess.run([PY] + args, cwd=HERE)
    if r.returncode != 0:
        print(f"⚠ STEP NON-FATALE fallito ({r.returncode}): {' '.join(args)} — proseguo senza.", flush=True)
        return False
    return True


def publish_daily(product, web_name):
    """Pubblica uno snapshot giornaliero (sst-uhr/chl/temp3d) in public/<web_name>/.
    Layout: meta.json + current.png/current_eroded.png (snapshot singolo) oppure
    d{NN}.png/d{NN}_eroded.png (multi-profondità). Best-effort: se manca il meta, salta."""
    meta_src = os.path.join(CACHE, f"daily_{product}_meta.json")
    if not os.path.exists(meta_src):
        print(f"⚠ {product}: meta mancante — salto la pubblicazione.", flush=True)
        return False
    import json
    with open(meta_src, encoding="utf-8") as f:
        meta = json.load(f)
    pub = os.path.join(HERE, "public", web_name)
    full = os.path.join(pub, "full")
    eroded = os.path.join(pub, "eroded")
    for d in (full, eroded):
        os.makedirs(d, exist_ok=True)
    shutil.copyfile(meta_src, os.path.join(pub, "meta.json"))
    n = 0
    if meta.get("depths"):                       # multi-profondità (termoclino)
        for k in range(len(meta["depths"])):
            for suffix, dest in (("", full), ("_eroded", eroded)):
                src = os.path.join(CACHE, f"daily_{product}_d{k:02d}{suffix}.png")
                if os.path.exists(src):
                    shutil.copyfile(src, os.path.join(dest, f"d{k:02d}.png"))
                    n += 1
    else:                                        # snapshot singolo (SST UHR, clorofilla)
        for suffix, dest in (("", full), ("_eroded", eroded)):
            src = os.path.join(CACHE, f"daily_{product}{suffix}.png")
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(dest, "current.png"))
                n += 1
    print(f"OK: public/{web_name} pronta — {n} PNG + meta.json", flush=True)
    return True


def main():
    os.makedirs(CACHE, exist_ok=True)

    # 1) Download NetCDF onde (3 giorni = 72h di forecast)
    run(["download_cmems.py", "--days", "3"])
    # 1b) Download NetCDF CORRENTI (uo/vo, stesso modello fisico 4km)
    run(["download_currents.py", "--days", "3"])
    # 1c) Download NetCDF SST (thetao 2D, stesso modello fisico 4km) — ADDITIVO/best-effort
    sst_ok = run_soft(["download_sst.py", "--days", "3"])
    # 2) Render PNG (full + eroded) + cache/waves_meta.json
    #    Width 6000 (era 3000): cattura tutto il dettaglio del dato 4km upsamplato 4× →
    #    a zoom alto si ingrandisce ~7× invece di ~13× = molto più nitido, stessa forma.
    #    PNG a palette (save_quantized) → resta leggero (~640 KB/PNG, come prima a 3000px RGBA).
    run(["render_waves_png.py", "--hours", str(HOURS), "--width", "6000"])
    # 2b) Render SST PNG (heatmap temperatura + FRONTI termici) — ADDITIVO/best-effort.
    #     Width 3000 (NON 6000 come le onde): la temperatura è liscia e i fronti sono
    #     feature da km → 3000 basta ed è ~4× più veloce sul composito fronti (altrimenti
    #     il render SST sfora il timeout del job). Palette PNG leggera.
    if sst_ok:
        sst_ok = run_soft(["render_sst_png.py", "--hours", str(HOURS), "--width", "3000"])
    # 3) Forecast JSON compatto (frecce direzione + popup), step 0.2°
    run(["extract_forecast.py", str(HOURS), os.path.join(CACHE, FORECAST_NAME), "--step", STEP])
    # 3b) Correnti JSON (u/v per punto) per il flusso animato delle correnti
    run(["extract_currents.py", str(HOURS), os.path.join(CACHE, CURRENTS_NAME), "--step", STEP])

    # 4) Componi il sito statico public/waves/
    full = os.path.join(PUBLIC, "full")
    eroded = os.path.join(PUBLIC, "eroded")
    for d in (full, eroded):
        os.makedirs(d, exist_ok=True)
    shutil.copyfile(os.path.join(CACHE, "waves_meta.json"), os.path.join(PUBLIC, "meta.json"))
    shutil.copyfile(os.path.join(CACHE, FORECAST_NAME), os.path.join(PUBLIC, FORECAST_NAME))
    cur_src = os.path.join(CACHE, CURRENTS_NAME)
    if os.path.exists(cur_src):
        shutil.copyfile(cur_src, os.path.join(PUBLIC, CURRENTS_NAME))
        print(f"OK: public/waves/{CURRENTS_NAME} pronto (correnti)", flush=True)

    n = 0
    for i in range(HOURS):
        for suffix, dest in (("", full), ("_eroded", eroded)):
            src = os.path.join(CACHE, f"waves_h{i:02d}{suffix}.png")
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(dest, f"h{i:02d}.png"))
                n += 1

    # 4b) Componi il sito statico public/sst/ (heatmap temperatura + fronti termici).
    #     Stesso layout delle onde: meta.json + full/h{NN}.png + eroded/h{NN}.png.
    #     Best-effort: se SST non c'è (download/render falliti), pubblico solo onde/correnti.
    sst_meta_src = os.path.join(CACHE, "sst_meta.json")
    if sst_ok and os.path.exists(sst_meta_src):
        sst_pub = os.path.join(HERE, "public", "sst")
        sst_full = os.path.join(sst_pub, "full")
        sst_eroded = os.path.join(sst_pub, "eroded")
        for d in (sst_full, sst_eroded):
            os.makedirs(d, exist_ok=True)
        shutil.copyfile(sst_meta_src, os.path.join(sst_pub, "meta.json"))
        ns = 0
        for i in range(HOURS):
            for suffix, dest in (("", sst_full), ("_eroded", sst_eroded)):
                src = os.path.join(CACHE, f"sst_h{i:02d}{suffix}.png")
                if os.path.exists(src):
                    shutil.copyfile(src, os.path.join(dest, f"h{i:02d}.png"))
                    ns += 1
        print(f"OK: public/sst pronta — {ns} PNG + meta.json (SST + fronti termici)", flush=True)
    else:
        print("⚠ SST saltato (download/render non riusciti) — pubblico solo onde/correnti.", flush=True)

    # 4c) SNAPSHOT GIORNALIERI (ADDITIVI/best-effort): cambiano lentamente → 1 mappa/giorno.
    #     - sst-uhr: SST satellitare ~0.9 km (fronti termici nitidi), osservata da satellite.
    #     - chl: clorofilla (plancton) — acqua verde = ricca di vita, i break = zone di caccia.
    #     - temp3d: temperatura del mare a varie profondità (termoclino), slider profondità.
    #     Ognuno è indipendente: se uno fallisce, gli altri (e tutta la pipeline) proseguono.
    for product, web_name in (("sst-uhr", "sst-uhr"), ("chl", "chl"), ("temp3d", "temp3d")):
        if run_soft(["download_daily.py", "--product", product]):
            if run_soft(["render_daily_png.py", "--product", product]):
                publish_daily(product, web_name)

    # 5) Batimetria: tile statici XYZ Web Mercator (committati in bathy_xyz/, lo SCHEMA
    #    che usa L.tileLayer dell'app) → public/bathy/. Servite dal CDN così NON pesano
    #    nel bundle app. (Le vecchie bathy_tiles erano WGS84-quad, schema sbagliato per l'app.)
    bathy_src = os.path.join(HERE, "bathy_xyz")
    if os.path.isdir(bathy_src):
        bathy_dst = os.path.join(HERE, "public", "bathy")
        if os.path.isdir(bathy_dst):
            shutil.rmtree(bathy_dst)
        shutil.copytree(bathy_src, bathy_dst)
        nb = sum(len(fs) for _, _, fs in os.walk(bathy_dst))
        print(f"OK: public/bathy pronta — {nb} tile batimetrici", flush=True)

    # Pagina indice/health (root del sito)
    with open(os.path.join(HERE, "public", "index.html"), "w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset=utf-8><title>cup8 CMEMS</title>"
                "<p>cup8 CMEMS static data — vedi <a href='waves/meta.json'>waves/meta.json</a></p>")

    print(f"OK: public/waves pronta — {n} PNG + meta.json + {FORECAST_NAME}", flush=True)


if __name__ == "__main__":
    main()
