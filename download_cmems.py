"""
download_cmems.py — Scarica forecast onde Mediterraneo da Copernicus Marine.

Uso: python download_cmems.py [--days N]

Output: data/forecast_med_waves.nc  (NetCDF ~100 MB, N giorni di previsione)

Eseguito automaticamente dal cron del server Node ogni 6h.
"""
import sys
import os
import argparse
import subprocess
from datetime import datetime, timezone, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATASET_ID = "cmems_mod_med_wav_anfc_4.2km_PT1H-i"
# Variabili scaricate dal modello MED Waves CMEMS:
#   VHM0  = altezza significativa onde totali (m)
#   VMDR  = direzione media onde (°)
#   VTM10 = periodo medio onde (s)
#   VHM0_SW1 = altezza swell (m)
#   VHM0_WW = altezza wind-wave (m)
VARIABLES = ["VHM0", "VMDR", "VTM10", "VHM0_SW1", "VHM0_WW"]
# BBox Mediterraneo (= lo stesso usato dall'app cap8 per il render)
BBOX = {"lng_min": -6, "lng_max": 36, "lat_min": 30, "lat_max": 46}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=3, help="Giorni di forecast da scaricare (1-10)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    # Allineo l'inizio all'ora UTC corrente arrotondata in giù
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    end = now + timedelta(days=args.days)
    out_file = "forecast_med_waves.nc"

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
        "--overwrite",                # forza sovrascrittura, no suffisso _(N)
        "--disable-progress-bar",
    ]
    for v in VARIABLES:
        cmd.extend(["--variable", v])

    print(f"[cmems] download {args.days}d forecast → {out_file}")
    print(f"[cmems] range: {now.isoformat()} → {end.isoformat()}")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(result.stdout)
    if result.returncode != 0:
        print(f"[cmems] ERROR stderr:\n{result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)

    out_path = os.path.join(DATA_DIR, out_file)
    if not os.path.exists(out_path):
        print(f"[cmems] ERROR: file {out_path} non creato", file=sys.stderr)
        sys.exit(1)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[cmems] ✓ {out_file} = {size_mb:.1f} MB")

    # Touch file per indicare ora di completion
    with open(os.path.join(DATA_DIR, ".last_download"), "w") as f:
        f.write(datetime.now(timezone.utc).isoformat())


if __name__ == "__main__":
    main()
