"""
bathy_tile_server.py — Tile server PAPERCUT batimetrico (EPSG:4326), porta 8097.

Tile /btile/{z}/{x}/{y}.png nitidi a OGNI zoom (piramide tipo Google Maps), 512px.
Profondità:
  - z>=6: store LOCALE EMODnet 150m (data/med_emodnet_150m.dat) → render dal dato
    locale = ISTANTANEO. Se lo store non c'è ancora, fallback EMODnet WCS per-tile.
  - z<6 : batimetrico statico CMEMS (med_bathy.nc) per la panoramica.
Ogni tile è cachato su disco (poi servito al volo).

Schema WGS84 quad: zoom z → tile = 180/2^z gradi; lng=-180+x*t, lat=90-y*t.
Avvio: python bathy_tile_server.py
"""
import os
import io
import sys
import json
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage
import xarray as xr

HERE = os.path.dirname(__file__)
TILE_DIR = os.path.join(HERE, "tiles")
CMEMS_NC = os.path.join(HERE, "data", "med_bathy.nc")
STORE_DAT = os.path.join(HERE, "data", "med_emodnet_115m.dat")
STORE_META = os.path.join(HERE, "data", "med_emodnet_115m.json")
PORT = int(os.environ.get("BATHY_TILE_PORT", "8097"))

TILE = 512          # px del tile (supersample → nitido anche su display ad alta densità)
PAD = 12            # margine px per continuità ombra tra tile
SHADOW_OFF = 3
SHADOW_BLUR = 2.2
SHADOW_DARK = 0.34

# Bande FINI (21), fittissime sul basso fondale (0-100m, zona pesca) → max definizione.
# Colori generati da una rampa blu (chiaro→navy) per avere tanti step morbidi e distinti.
DEPTH_LEVELS = np.array([0, 5, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 600, 800, 1200, 1700, 2300, 3000, 4000, 5500], dtype=np.float32)
_ctrl_pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_ctrl_rgb = np.array([[224, 242, 250], [130, 190, 228], [50, 120, 185], [18, 60, 115], [5, 18, 42]], dtype=np.float32)
_t = np.linspace(0.0, 1.0, len(DEPTH_LEVELS) - 1)
DEPTH_COLORS = np.stack([np.interp(_t, _ctrl_pos, _ctrl_rgb[:, i]) for i in range(3)], axis=1).astype(np.float32)

# ── Store locale EMODnet (memmap, lazy) ──
_store = None
def store():
    global _store
    if _store is None and os.path.exists(STORE_DAT) and os.path.exists(STORE_META):
        m = json.load(open(STORE_META))
        arr = np.memmap(STORE_DAT, dtype=np.int16, mode="r", shape=(m["H"], m["W"]))
        _store = (arr, m)
        sys.stderr.write("[tile] store EMODnet locale caricato\n")
    return _store

# ── CMEMS deptho (panoramica) ──
_cmems = None
def cmems():
    global _cmems
    if _cmems is None:
        ds = xr.open_dataset(CMEMS_NC)
        dep = ds.deptho.values.astype(np.float32)
        la = ds.latitude.values.astype(np.float64); lo = ds.longitude.values.astype(np.float64)
        sea = np.isfinite(dep)
        idx = ndimage.distance_transform_edt(~sea, return_distances=False, return_indices=True)
        _cmems = dict(dep=dep[tuple(idx)], sea=sea.astype(np.float32), la=la, lo=lo)
    return _cmems


def tile_bbox(z, x, y):
    n = 180.0 / (2 ** z)
    lo0 = -180.0 + x * n
    la1 = 90.0 - y * n
    return lo0, la1 - n, lo0 + n, la1


def _grid(lo0, la0, lo1, la1, n):
    col_lng = np.linspace(lo0, lo1, n)
    row_lat = np.linspace(la1, la0, n)   # alto = lat_max
    return col_lng, row_lat


def depth_from_store(lo0, la0, lo1, la1, n):
    s = store()
    if s is None:
        return None
    arr, m = s
    col_lng, row_lat = _grid(lo0, la0, lo1, la1, n)
    cc = (col_lng - m["lng_min"]) / m["res_deg"]
    rr = (m["lat_max"] - row_lat) / m["res_deg"]
    CC = np.repeat(cc[None, :], n, 0); RR = np.repeat(rr[:, None], n, 1)
    inside = (CC >= 0) & (CC <= m["W"] - 1) & (RR >= 0) & (RR <= m["H"] - 1)
    depth = ndimage.map_coordinates(arr, [np.clip(RR, 0, m["H"] - 1), np.clip(CC, 0, m["W"] - 1)],
                                    order=1, mode="nearest").astype(np.float32)
    sea = (depth > 0.5) & inside
    return depth, sea.astype(np.float32)


def depth_from_emodnet(lo0, la0, lo1, la1, n):
    url = ("https://ows.emodnet-bathymetry.eu/wcs?SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
           "&COVERAGEID=emodnet__mean&FORMAT=image/tiff"
           f"&SUBSET=Long({lo0:.6f},{lo1:.6f})&SUBSET=Lat({la0:.6f},{la1:.6f})&SCALESIZE=i({n}),j({n})")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            elev = tifffile.imread(io.BytesIO(r.read())).astype(np.float32)
        if elev.shape != (n, n):
            elev = np.array(Image.fromarray(elev).resize((n, n)))
        sea = elev < 0
        return np.where(sea, -elev, 0.0).astype(np.float32), sea.astype(np.float32)
    except Exception as e:
        sys.stderr.write(f"[tile] EMODnet fail {e}\n")
        return None


def depth_from_cmems(lo0, la0, lo1, la1, n):
    C = cmems(); lo = C["lo"]; la = C["la"]
    col_lng, row_lat = _grid(lo0, la0, lo1, la1, n)
    cc = (col_lng - lo[0]) / (lo[-1] - lo[0]) * (len(lo) - 1)
    rr = (row_lat - la[0]) / (la[-1] - la[0]) * (len(la) - 1)
    CC = np.repeat(cc[None, :], n, 0); RR = np.repeat(rr[:, None], n, 1)
    inside = (CC >= 0) & (CC <= len(lo) - 1) & (RR >= 0) & (RR <= len(la) - 1)
    depth = ndimage.map_coordinates(C["dep"], [RR, CC], order=1, mode="nearest")
    sea = ndimage.map_coordinates(C["sea"], [RR, CC], order=1, mode="nearest")
    return depth.astype(np.float32), ((sea > 0.5) & inside).astype(np.float32)


def render_tile(z, x, y):
    lo0, la0, lo1, la1 = tile_bbox(z, x, y)
    dlo = (lo1 - lo0) * PAD / TILE; dla = (la1 - la0) * PAD / TILE
    bo0, ba0, bo1, ba1 = lo0 - dlo, la0 - dla, lo1 + dlo, la1 + dla
    N = TILE + 2 * PAD

    res = None
    if z >= 6:
        res = depth_from_store(bo0, ba0, bo1, ba1, N) or depth_from_emodnet(bo0, ba0, bo1, ba1, N)
    if res is None:
        res = depth_from_cmems(bo0, ba0, bo1, ba1, N)
    depth, sea = res

    band = np.clip(np.searchsorted(DEPTH_LEVELS, depth, side="right") - 1, 0, len(DEPTH_LEVELS) - 2).astype(np.int16)
    rgb = DEPTH_COLORS[band]
    upleft = np.roll(np.roll(band, SHADOW_OFF, 0), SHADOW_OFF, 1)
    shadow = (upleft < band).astype(np.float32)
    shadow = np.clip(ndimage.gaussian_filter(shadow, SHADOW_BLUR), 0, 1) * SHADOW_DARK
    rgb = rgb * (1.0 - shadow[..., None])
    a = np.where(sea > 0.5, 255, 0).astype(np.uint8)
    out = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), a])[PAD:PAD + TILE, PAD:PAD + TILE]
    return Image.fromarray(out, "RGBA")


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"   # keep-alive: il browser riusa la connessione → molto piu' veloce

    def log_message(self, *a):
        pass

    def _send(self, data, ctype="image/png"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "public, max-age=604800")
        self.end_headers(); self.wfile.write(data)

    def _empty(self, code):
        self.send_response(code)
        self.send_header("Content-Length", "0")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/health"):
            self._send(b"bathy-tile-server ok", "text/plain"); return
        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[0] == "btile" and parts[3].endswith(".png"):
            try:
                z, x, y = int(parts[1]), int(parts[2]), int(parts[3][:-4])
            except ValueError:
                self._empty(400); return
            cp = os.path.join(TILE_DIR, str(z), str(x), f"{y}.png")
            if os.path.exists(cp):
                with open(cp, "rb") as f:
                    self._send(f.read()); return
            try:
                img = render_tile(z, x, y)
            except Exception as e:
                sys.stderr.write(f"[tile] render fail z{z}/{x}/{y}: {e}\n")
                img = Image.new("RGBA", (TILE, TILE), (0, 0, 0, 0))
            buf = io.BytesIO(); img.save(buf, "PNG"); data = buf.getvalue()
            try:
                os.makedirs(os.path.dirname(cp), exist_ok=True)
                with open(cp, "wb") as f:
                    f.write(data)
            except OSError:
                pass
            self._send(data); return
        self._empty(404)


if __name__ == "__main__":
    os.makedirs(TILE_DIR, exist_ok=True)
    cmems()
    st = "presente" if (os.path.exists(STORE_DAT) and os.path.exists(STORE_META)) else "ASSENTE (fallback EMODnet finché non pronto)"
    print(f"[tile] server su http://localhost:{PORT}/btile/{{z}}/{{x}}/{{y}}.png | store EMODnet: {st}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
