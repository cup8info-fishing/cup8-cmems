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
import time
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

    # Retry sui blip di connessione/autenticazione Copernicus: il server di auth a volte
    # è irraggiungibile dai runner CI e il client resta appeso ~20 min prima di arrendersi.
    # Limitiamo ogni tentativo nel tempo e ritentiamo (solo su errori transitori).
    ATTEMPTS = 3
    PER_ATTEMPT_TIMEOUT = 600   # 10 min per tentativo (un download normale dura pochi min)
    RETRY_DELAY = 90            # pausa tra i tentativi
    TRANSIENT = ("could not connect", "authentication system", "connection", "timed out",
                 "timeout", "temporar", "max retries", "ssl", "502", "503", "504",
                 "remotedisconnect", "reset by peer", "gateway")

    result = None
    for i in range(1, ATTEMPTS + 1):
        print(f"[cmems] tentativo {i}/{ATTEMPTS}…", flush=True)
        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace",
                                    timeout=PER_ATTEMPT_TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"[cmems] ⚠ tentativo {i}: superato il timeout di {PER_ATTEMPT_TIMEOUT}s, interrotto.", file=sys.stderr)
            if i < ATTEMPTS:
                time.sleep(RETRY_DELAY)
            continue
        if result.stdout:
            print(result.stdout)
        if result.returncode == 0:
            break
        err = (result.stderr or "").strip()
        print(f"[cmems] ⚠ tentativo {i} fallito (rc={result.returncode}):\n{err}", file=sys.stderr)
        if not any(k in err.lower() for k in TRANSIENT):
            # Errore NON transitorio (es. credenziali errate, dataset inesistente) → inutile ritentare.
            sys.exit(result.returncode)
        if i < ATTEMPTS:
            print(f"[cmems] errore transitorio → ritento tra {RETRY_DELAY}s…", flush=True)
            time.sleep(RETRY_DELAY)

    if result is None or result.returncode != 0:
        print(f"[cmems] ERROR: download fallito dopo {ATTEMPTS} tentativi (Copernicus irraggiungibile).", file=sys.stderr)
        sys.exit(1)

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
