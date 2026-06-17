"""
download_daily.py — Scarica i prodotti CMEMS "snapshot giornaliero" per cap8.

A differenza di onde/correnti/SST (forecast orario 72h), questi prodotti cambiano
lentamente e si guardano come UNA mappa del giorno (non un film ora-per-ora):

  - sst-uhr  : SST satellitare ULTRA HIGH RES (~0.9 km, osservata da satellite, gap-free).
               Dataset SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_c_V2, var analysed_sst (Kelvin).
               È il "Cloudfree SST" per leggere i FRONTI termici offshore.
  - chl      : Clorofilla superficiale (plancton). Dataset bio P1D-m, var chl (mg/m³).
               Acqua verde = ricca di vita; i break di colore = zone di caccia.
  - temp3d   : Temperatura del mare in PROFONDITÀ (termoclino). Dataset phy-tem P1D-m,
               var thetao su livelli di profondità (0 → ~50 m).

Scarica una piccola finestra temporale e il render prende l'ULTIMO timestep disponibile
(il più recente), così non dipendiamo dal timestamp esatto del prodotto.

Uso: python download_daily.py --product {sst-uhr,chl,temp3d}

Output: data/daily_<product>.nc
"""
import sys
import os
import argparse
import subprocess
import time
from datetime import datetime, timezone, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BBOX = {"lng_min": -6, "lng_max": 36, "lat_min": 30, "lat_max": 46}

PRODUCTS = {
    "sst-uhr": {
        "dataset_id": "SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_c_V2",
        "variables": ["analysed_sst"],
        "out": "daily_sst_uhr.nc",
        "max_depth": None,   # prodotto satellitare di superficie: nessuna dimensione depth
        # SST L4 osservata: latenza ~1 giorno. Scarico gli ultimi 4 giorni, prendo l'ultimo.
        "back_days": 4, "fwd_days": 0,
    },
    "chl": {
        "dataset_id": "cmems_mod_med_bgc-pft_anfc_4.2km_P1D-m",
        "variables": ["chl"],
        "out": "daily_chl.nc",
        # Il primo livello del modello è ~1.02 m: NON passo --minimum-depth (darebbe
        # "out of bounds"), solo il massimo → CMEMS prende il primo livello = superficie.
        "max_depth": 2.0,
        "back_days": 1, "fwd_days": 1,
    },
    "temp3d": {
        "dataset_id": "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
        "variables": ["thetao"],
        "out": "daily_temp3d.nc",
        "max_depth": 55.0,    # dal primo livello (~1 m) a ~50 m: copre il termoclino estivo
        "back_days": 0, "fwd_days": 1,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", required=True, choices=list(PRODUCTS.keys()))
    args = parser.parse_args()
    cfg = PRODUCTS[args.product]

    os.makedirs(DATA_DIR, exist_ok=True)
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = now - timedelta(days=cfg["back_days"])
    end = now + timedelta(days=cfg["fwd_days"])

    cmd = [
        "copernicusmarine", "subset",
        "--dataset-id", cfg["dataset_id"],
        "--minimum-longitude", str(BBOX["lng_min"]),
        "--maximum-longitude", str(BBOX["lng_max"]),
        "--minimum-latitude", str(BBOX["lat_min"]),
        "--maximum-latitude", str(BBOX["lat_max"]),
        "--start-datetime", start.strftime("%Y-%m-%dT%H:%M:%S"),
        "--end-datetime", end.strftime("%Y-%m-%dT%H:%M:%S"),
        "--output-directory", DATA_DIR,
        "--output-filename", cfg["out"],
        "--overwrite",
        "--disable-progress-bar",
    ]
    if cfg.get("max_depth") is not None:
        cmd.extend(["--maximum-depth", str(cfg["max_depth"])])
    for v in cfg["variables"]:
        cmd.extend(["--variable", v])

    tag = args.product
    print(f"[{tag}] download → {cfg['out']}")
    print(f"[{tag}] range: {start.isoformat()} → {end.isoformat()}")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    ATTEMPTS = 3
    PER_ATTEMPT_TIMEOUT = 600
    RETRY_DELAY = 90
    TRANSIENT = ("could not connect", "authentication system", "connection", "timed out",
                 "timeout", "temporar", "max retries", "ssl", "502", "503", "504",
                 "remotedisconnect", "reset by peer", "gateway")

    result = None
    for i in range(1, ATTEMPTS + 1):
        print(f"[{tag}] tentativo {i}/{ATTEMPTS}…", flush=True)
        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace",
                                    timeout=PER_ATTEMPT_TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"[{tag}] ⚠ tentativo {i}: timeout {PER_ATTEMPT_TIMEOUT}s.", file=sys.stderr)
            if i < ATTEMPTS:
                time.sleep(RETRY_DELAY)
            continue
        if result.stdout:
            print(result.stdout)
        if result.returncode == 0:
            break
        err = (result.stderr or "").strip()
        print(f"[{tag}] ⚠ tentativo {i} fallito (rc={result.returncode}):\n{err}", file=sys.stderr)
        if not any(k in err.lower() for k in TRANSIENT):
            sys.exit(result.returncode)
        if i < ATTEMPTS:
            print(f"[{tag}] errore transitorio → ritento tra {RETRY_DELAY}s…", flush=True)
            time.sleep(RETRY_DELAY)

    if result is None or result.returncode != 0:
        print(f"[{tag}] ERROR: download fallito dopo {ATTEMPTS} tentativi.", file=sys.stderr)
        sys.exit(1)

    out_path = os.path.join(DATA_DIR, cfg["out"])
    if not os.path.exists(out_path):
        print(f"[{tag}] ERROR: file {out_path} non creato", file=sys.stderr)
        sys.exit(1)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[{tag}] ✓ {cfg['out']} = {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
