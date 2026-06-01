"""
buffer_land.py — Genera versione bufferizzata di med-land-merged.geojson.

Applica buffer geometrico di ~100m ai polygons terra. Output:
  cap8/public/med-land-buffered-100m.geojson

Frontend WavePngOverlay usa questo file in modalità eroded (Satellite/Nautica),
così il clip-path "estende" la terra di 100m oltre la costa reale → overlay onde
INIZIA a 100m dal bordo costa.

Buffer in lat/lng degrees: 100m ≈ 0.0009° (= 100m / 111000 m/° media).
Applico 0.001° per leggera tolleranza (~111m a sicurezza).
"""
import sys
import os
import json

INPUT = "C:/Users/giuse/cup8all/Cup8 - Copia/public/med-land-merged.geojson"
PUBLIC_DIR = "C:/Users/giuse/cup8all/Cup8 - Copia/public"
# Genero 4 versioni bufferizzate (utenti possono scegliere distanza min onda-costa)
BUFFER_METERS = [500]
DEGREES_PER_METER = 1.0 / 111000  # approx Mediterraneo

try:
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
except ImportError:
    print("ERROR: shapely non installato. Run: pip install shapely", file=sys.stderr)
    sys.exit(1)


def buffer_features(geo, buffer_deg):
    buffered = []
    for i, feat in enumerate(geo["features"]):
        try:
            g = shape(feat["geometry"])
            b = g.buffer(buffer_deg, resolution=4)
            if b.is_empty:
                continue
            buffered.append({
                "type": "Feature",
                "properties": feat.get("properties", {}),
                "geometry": mapping(b),
            })
        except Exception as e:
            print(f"  feat {i} skip: {e}", file=sys.stderr)
    return buffered


def main():
    with open(INPUT, "r", encoding="utf-8") as f:
        geo = json.load(f)
    print(f"[buffer-land] {len(geo['features'])} features → genero {len(BUFFER_METERS)} versioni...")

    for meters in BUFFER_METERS:
        buffer_deg = meters * DEGREES_PER_METER
        out_path = os.path.join(PUBLIC_DIR, f"med-land-buffered-{meters}m.geojson")
        bf = buffer_features(geo, buffer_deg)
        out = {"type": "FeatureCollection", "features": bf}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f)
        size_kb = os.path.getsize(out_path) / 1024
        print(f"  ✓ {meters}m ({buffer_deg:.4f}°) → {len(bf)} polygons, {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
