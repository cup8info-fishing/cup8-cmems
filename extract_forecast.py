"""
extract_forecast.py — Estrae N ore di forecast dal NetCDF in UN SOLO JSON,
                      in formato compatto (per ogni punto, array di valori per ogni ora).

Uso: python extract_forecast.py <hours_count> <output_json> [--step 0.10]

Output:
{
  "start_hour": "2026-05-25T10:00:00",
  "hours_count": 24,
  "hours": ["2026-05-25T10:00:00", ..., "2026-05-26T09:00:00"],
  "step": 0.10,
  "bbox": {...},
  "gridW": 504, "gridH": 190,
  "points": [
    {"lat": 30.29, "lng": 18.93, "wh": [0.65, 0.71, ...], "wd": [...], "wp": [...]},
    ...
  ]
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
    p.add_argument("hours", type=int, help="Numero di ore di forecast da estrarre (1-72)")
    p.add_argument("output", help="Path output JSON")
    p.add_argument("--step", type=float, default=0.10)
    args = p.parse_args()

    if not os.path.exists(DATA_NC):
        print(f"ERROR: NetCDF non trovato: {DATA_NC}", file=sys.stderr)
        sys.exit(1)

    ds = xr.open_dataset(DATA_NC)
    total_hours = ds.sizes["time"]
    hours_count = min(args.hours, total_hours)
    sel = ds.isel(time=slice(0, hours_count))

    native_step = 0.041666
    factor = max(1, int(round(args.step / native_step)))
    if factor > 1:
        sel = sel.coarsen(latitude=factor, longitude=factor, boundary="trim").mean(skipna=True)

    lats = sel.latitude.values
    lngs = sel.longitude.values
    times = sel.time.values
    vhm0 = sel.VHM0.values        # (time, lat, lng)
    vmdr = sel.VMDR.values
    vtm10 = sel.VTM10.values
    vhm0_sw1 = sel.VHM0_SW1.values if "VHM0_SW1" in sel else None
    vhm0_ww = sel.VHM0_WW.values if "VHM0_WW" in sel else None

    # Per ogni punto (lat, lng), creo un object con array temporale.
    # Skip i punti dove TUTTE le ore sono NaN (terra).
    points = []
    for iy in range(len(lats)):
        for ix in range(len(lngs)):
            wh_series = vhm0[:, iy, ix]
            if np.all(np.isnan(wh_series)):
                continue
            # Sostituisco NaN nelle serie con 0 (mare calmo) → frontend gestisce
            wh_clean = np.where(np.isnan(wh_series), 0, wh_series)
            p_obj = {
                "lat": round(float(lats[iy]), 4),
                "lng": round(float(lngs[ix]), 4),
                "wh": [round(float(v), 3) for v in wh_clean],
            }
            wd_series = np.where(np.isnan(vmdr[:, iy, ix]), 0, vmdr[:, iy, ix])
            wp_series = np.where(np.isnan(vtm10[:, iy, ix]), 0, vtm10[:, iy, ix])
            p_obj["wd"] = [round(float(v), 1) for v in wd_series]
            p_obj["wp"] = [round(float(v), 2) for v in wp_series]
            if vhm0_sw1 is not None:
                sw_series = np.where(np.isnan(vhm0_sw1[:, iy, ix]), 0, vhm0_sw1[:, iy, ix])
                p_obj["sw"] = [round(float(v), 3) for v in sw_series]
            if vhm0_ww is not None:
                ww_series = np.where(np.isnan(vhm0_ww[:, iy, ix]), 0, vhm0_ww[:, iy, ix])
                p_obj["ww"] = [round(float(v), 3) for v in ww_series]
            points.append(p_obj)

    hour_strs = [str(np.datetime_as_string(t, unit="s")) for t in times]
    output = {
        "start_hour": hour_strs[0],
        "hours_count": hours_count,
        "hours": hour_strs,
        "step": args.step,
        "bbox": {
            "lat_min": float(lats.min()),
            "lat_max": float(lats.max()),
            "lng_min": float(lngs.min()),
            "lng_max": float(lngs.max()),
        },
        "gridW": len(lngs),
        "gridH": len(lats),
        "points_count": len(points),
        "points": points,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f)

    size_kb = os.path.getsize(args.output) / 1024
    print(f"[forecast] {hours_count}h, {len(points)} punti → {args.output} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
