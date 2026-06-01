"""
extract_hour.py — Estrae una singola ora dal NetCDF forecast e la converte in JSON
                  compatto (gzipabile) per il frontend cap8.

Uso: python extract_hour.py <hour_iso> <output_json> [--step 0.05]

  hour_iso     ISO datetime UTC, es. "2026-05-25T14:00:00"
  output_json  Path del file JSON da generare (verrà overwritten)
  --step       Risoluzione downsample in gradi (default 0.05° = ~5 km)
               L'app cap8 a 0.05° vede ~60k punti = ~3 MB JSON.
               LaMMA nativa è 0.042° (4.2 km) ma per mobile è eccessivo.

Output JSON:
{
  "hour":   "2026-05-25T14:00:00Z",
  "bbox":   {"lat_min": 30, "lat_max": 46, "lng_min": -6, "lng_max": 36},
  "step":   0.05,
  "gridW":  840,
  "gridH":  320,
  "points": [{"lat": 41.0, "lng": 12.5, "wh": 0.42, "wd": 215, "wp": 4.2, "sw": 0.30, "ww": 0.28}, ...]
}
"""
import sys
import os
import json
import argparse
import numpy as np
import xarray as xr

DATA_NC = os.path.join(os.path.dirname(__file__), "data", "forecast_med_waves.nc")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("hour", help="ISO datetime UTC, es. 2026-05-25T14:00:00")
    p.add_argument("output", help="Path output JSON")
    p.add_argument("--step", type=float, default=0.05, help="Downsample step (gradi)")
    args = p.parse_args()

    if not os.path.exists(DATA_NC):
        print(f"ERROR: NetCDF non trovato: {DATA_NC}", file=sys.stderr)
        sys.exit(1)

    ds = xr.open_dataset(DATA_NC)

    # Trovo l'ora più vicina nel dataset (forecast hourly)
    target = np.datetime64(args.hour)
    try:
        sel = ds.sel(time=target, method="nearest")
    except Exception as e:
        print(f"ERROR sel time: {e}", file=sys.stderr)
        sys.exit(1)

    actual_time = str(np.datetime_as_string(sel.time.values, unit="s"))
    # Downsample con coarsen (media bilineare delle celle dentro lo step)
    # Native è 0.042°, vogliamo 0.05° → ratio ~1.2 (sostanzialmente 1:1, no perdita)
    # Per step più grandi (es. 0.1°) il coarsen aggrega davvero.
    native_step = 0.041666  # ~4.2 km
    factor = max(1, int(round(args.step / native_step)))
    if factor > 1:
        sel = sel.coarsen(latitude=factor, longitude=factor, boundary="trim").mean(skipna=True)

    lats = sel.latitude.values
    lngs = sel.longitude.values
    vhm0 = sel.VHM0.values        # (lat, lng)
    vmdr = sel.VMDR.values
    vtm10 = sel.VTM10.values
    vhm0_sw1 = sel.VHM0_SW1.values if "VHM0_SW1" in sel else None
    vhm0_ww = sel.VHM0_WW.values if "VHM0_WW" in sel else None

    points = []
    for iy in range(len(lats)):
        for ix in range(len(lngs)):
            wh = vhm0[iy, ix]
            if np.isnan(wh):
                continue
            p_obj = {
                "lat": round(float(lats[iy]), 4),
                "lng": round(float(lngs[ix]), 4),
                "wh": round(float(wh), 3),
            }
            if not np.isnan(vmdr[iy, ix]):
                p_obj["wd"] = round(float(vmdr[iy, ix]), 1)
            if not np.isnan(vtm10[iy, ix]):
                p_obj["wp"] = round(float(vtm10[iy, ix]), 2)
            if vhm0_sw1 is not None and not np.isnan(vhm0_sw1[iy, ix]):
                p_obj["sw"] = round(float(vhm0_sw1[iy, ix]), 3)
            if vhm0_ww is not None and not np.isnan(vhm0_ww[iy, ix]):
                p_obj["ww"] = round(float(vhm0_ww[iy, ix]), 3)
            points.append(p_obj)

    output = {
        "hour": actual_time,
        "bbox": {
            "lat_min": float(lats.min()),
            "lat_max": float(lats.max()),
            "lng_min": float(lngs.min()),
            "lng_max": float(lngs.max()),
        },
        "step": args.step,
        "gridW": len(lngs),
        "gridH": len(lats),
        "points_count": len(points),
        "points": points,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f)

    size_kb = os.path.getsize(args.output) / 1024
    print(f"[extract] {actual_time} → {args.output} ({len(points)} punti, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
