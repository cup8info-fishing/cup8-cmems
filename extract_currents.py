"""
extract_currents.py — Estrae N ore di CORRENTI (uo/vo) dal NetCDF in UN SOLO JSON
                      compatto (per ogni punto di mare, array orari di u e v).

Clone di extract_forecast.py per il dataset correnti 2D (nessuna depth).

Uso: python extract_currents.py <hours_count> <output_json> [--step 0.20]

Output (stessa struttura del forecast onde, con campi u/v invece di wh/wd):
{
  "start_hour": "...", "hours_count": 72, "hours": [...], "step": 0.20,
  "bbox": {...}, "gridW": ..., "gridH": ..., "points_count": ...,
  "points": [ {"lat":.., "lng":.., "u":[...], "v":[...]}, ... ]
}
u = eastward velocity (m/s), v = northward velocity (m/s). NaN (terra) → 0.
"""
import sys
import os
import json
import argparse
import numpy as np
import xarray as xr

DATA_NC = os.path.join(os.path.dirname(__file__), "data", "forecast_med_currents.nc")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("hours", type=int, help="Numero di ore di forecast da estrarre (1-72)")
    p.add_argument("output", help="Path output JSON")
    p.add_argument("--step", type=float, default=0.20)
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
    uo = sel.uo.values        # (time, lat, lng)  eastward m/s
    vo = sel.vo.values        # (time, lat, lng)  northward m/s

    from scipy import ndimage
    # Il modello correnti CMEMS MASCHERA le celle COSTIERE (acqua bassa non risolta) →
    # buchi senza corrente a ridosso della costa, dove però le onde ci sono. Estendo la
    # corrente fino a riva riempiendo quei buchi col valore offshore più vicino (STIMA,
    # non misura: il modello lì non ha dati). Il clip-costa ISTAT del frontend rifinisce.
    valid = ~np.all(np.isnan(uo), axis=0)                  # (lat,lng) celle con dato corrente
    emit = ndimage.binary_dilation(valid, iterations=2)    # + estensione costiera (~2 celle)
    uo_f = np.array(uo, dtype=float)
    vo_f = np.array(vo, dtype=float)
    for t in range(uo_f.shape[0]):                         # nearest-fill per ora
        mu = np.isnan(uo_f[t])
        if mu.all():
            uo_f[t] = 0.0; vo_f[t] = 0.0; continue
        if mu.any():
            idx = ndimage.distance_transform_edt(mu, return_distances=False, return_indices=True)
            uo_f[t] = uo_f[t][tuple(idx)]
        mv = np.isnan(vo_f[t])
        if mv.any():
            idxv = ndimage.distance_transform_edt(mv, return_distances=False, return_indices=True)
            vo_f[t] = vo_f[t][tuple(idxv)]

    # Emetto le celle "emit" (mare + estensione costiera), con i valori riempiti.
    nt = uo_f.shape[0]
    points = []
    for iy in range(len(lats)):
        for ix in range(len(lngs)):
            if not emit[iy, ix]:
                continue
            points.append({
                "lat": round(float(lats[iy]), 4),
                "lng": round(float(lngs[ix]), 4),
                "u": [round(float(uo_f[t, iy, ix]), 3) for t in range(nt)],
                "v": [round(float(vo_f[t, iy, ix]), 3) for t in range(nt)],
            })

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

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f)

    size_kb = os.path.getsize(args.output) / 1024
    print(f"[currents] {hours_count}h, {len(points)} punti → {args.output} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
