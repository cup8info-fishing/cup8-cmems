"""
download_sst.py — Scarica forecast TEMPERATURA SUPERFICIALE del mare (SST) Mediterraneo
da Copernicus Marine.

Clone di download_currents.py per il dataset FISICO temperatura (thetao) 2D di superficie.
La SST serve all'overlay "Temperatura mare + Fronti termici": i confini caldo/freddo
(fronti) dove si radunano i pelagici (tonni, lampughe).

Uso: python download_sst.py [--days N]

Output: data/forecast_med_sst.nc  (NetCDF, N giorni di forecast)
"""
import sys
import os
import argparse
import subprocess
import time
from datetime import datetime, timezone, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
# Sea Water Potential Temperature 2D, Hourly Mean (MEDSEA_ANALYSISFORECAST_PHY_006_013).
# Dataset 2D di SUPERFICIE: dims (time, lat, lon), NESSUNA depth → niente --minimum-depth.
# Stessa griglia 4.2km di onde (cmems_mod_med_wav_*) e correnti (cmems_mod_med_phy-cur_*),
# quindi la SST è allineata pixel-per-pixel con quelle.
DATASET_ID = "cmems_mod_med_phy-tem_anfc_4.2km-2D_PT1H-m"
# thetao = sea_water_potential_temperature. Sul dataset 2D è il livello di SUPERFICIE = SST.
# Unità: °C (NON Kelvin per questo prodotto NEMO). Il render è comunque difensivo (>100 → -273.15).
VARIABLES = ["thetao"]
# Stesso BBox Mediterraneo del download onde/correnti (allineato alla mappa cap8).
BBOX = {"lng_min": -6, "lng_max": 36, "lat_min": 30, "lat_max": 46}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=3, help="Giorni di forecast da scaricare (1-10)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    end = now + timedelta(days=args.days)
    out_file = "forecast_med_sst.nc"

    cmd = [
        "copernicusmarine", "subset",
        "--dataset-id", DATASET_ID,
        "--minimum-longitude", str(BBOX["lng_min"]),
        "--maximum-longitude", str(BBOX["lng_max"]),
        "--minimum-latitude", str(BBOX["lat_min"]),
        "--maximum-latitude", str(BBOX["lat_max"]),
        "--start-datetime", now.strftime("%Y-%m-%dT%H:%M:%S"),
        "--end-datetime", end.strftime("%Y-%m-%dT%H:%M:%S"),
        "--output-directory", DATA_DIR,
        "--output-filename", out_file,
        "--overwrite",
        "--disable-progress-bar",
    ]
    for v in VARIABLES:
        cmd.extend(["--variable", v])

    print(f"[sst] download {args.days}d forecast → {out_file}")
    print(f"[sst] range: {now.isoformat()} → {end.isoformat()}")

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
        print(f"[sst] tentativo {i}/{ATTEMPTS}…", flush=True)
        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace",
                                    timeout=PER_ATTEMPT_TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"[sst] ⚠ tentativo {i}: superato il timeout di {PER_ATTEMPT_TIMEOUT}s, interrotto.", file=sys.stderr)
            if i < ATTEMPTS:
                time.sleep(RETRY_DELAY)
            continue
        if result.stdout:
            print(result.stdout)
        if result.returncode == 0:
            break
        err = (result.stderr or "").strip()
        print(f"[sst] ⚠ tentativo {i} fallito (rc={result.returncode}):\n{err}", file=sys.stderr)
        if not any(k in err.lower() for k in TRANSIENT):
            sys.exit(result.returncode)
        if i < ATTEMPTS:
            print(f"[sst] errore transitorio → ritento tra {RETRY_DELAY}s…", flush=True)
            time.sleep(RETRY_DELAY)

    if result is None or result.returncode != 0:
        print(f"[sst] ERROR: download fallito dopo {ATTEMPTS} tentativi (Copernicus irraggiungibile).", file=sys.stderr)
        sys.exit(1)

    out_path = os.path.join(DATA_DIR, out_file)
    if not os.path.exists(out_path):
        print(f"[sst] ERROR: file {out_path} non creato", file=sys.stderr)
        sys.exit(1)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[sst] ✓ {out_file} = {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
