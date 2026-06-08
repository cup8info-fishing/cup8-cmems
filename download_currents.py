"""
download_currents.py — Scarica forecast CORRENTI superficiali Mediterraneo da Copernicus Marine.

Clone di download_cmems.py per il dataset FISICO correnti (uo/vo) 2D di superficie.

Uso: python download_currents.py [--days N]

Output: data/forecast_med_currents.nc  (NetCDF, N giorni di forecast)
"""
import sys
import os
import argparse
import subprocess
import time
from datetime import datetime, timezone, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
# Sea Surface Horizontal Velocity 2D, Hourly Mean (MEDSEA_ANALYSISFORECAST_PHY_006_013).
# Dataset 2D di SUPERFICIE: dims (time, lat, lon), NESSUNA depth → niente --minimum-depth.
DATASET_ID = "cmems_mod_med_phy-cur_anfc_4.2km-2D_PT1H-m"
# uo = eastward_sea_water_velocity (m/s), vo = northward_sea_water_velocity (m/s)
VARIABLES = ["uo", "vo"]
# Stesso BBox Mediterraneo del download onde (allineato alla mappa cap8).
BBOX = {"lng_min": -6, "lng_max": 36, "lat_min": 30, "lat_max": 46}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=3, help="Giorni di forecast da scaricare (1-10)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    end = now + timedelta(days=args.days)
    out_file = "forecast_med_currents.nc"

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

    print(f"[currents] download {args.days}d forecast → {out_file}")
    print(f"[currents] range: {now.isoformat()} → {end.isoformat()}")

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
        print(f"[currents] tentativo {i}/{ATTEMPTS}…", flush=True)
        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace",
                                    timeout=PER_ATTEMPT_TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"[currents] ⚠ tentativo {i}: superato il timeout di {PER_ATTEMPT_TIMEOUT}s, interrotto.", file=sys.stderr)
            if i < ATTEMPTS:
                time.sleep(RETRY_DELAY)
            continue
        if result.stdout:
            print(result.stdout)
        if result.returncode == 0:
            break
        err = (result.stderr or "").strip()
        print(f"[currents] ⚠ tentativo {i} fallito (rc={result.returncode}):\n{err}", file=sys.stderr)
        if not any(k in err.lower() for k in TRANSIENT):
            sys.exit(result.returncode)
        if i < ATTEMPTS:
            print(f"[currents] errore transitorio → ritento tra {RETRY_DELAY}s…", flush=True)
            time.sleep(RETRY_DELAY)

    if result is None or result.returncode != 0:
        print(f"[currents] ERROR: download fallito dopo {ATTEMPTS} tentativi (Copernicus irraggiungibile).", file=sys.stderr)
        sys.exit(1)

    out_path = os.path.join(DATA_DIR, out_file)
    if not os.path.exists(out_path):
        print(f"[currents] ERROR: file {out_path} non creato", file=sys.stderr)
        sys.exit(1)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[currents] ✓ {out_file} = {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
