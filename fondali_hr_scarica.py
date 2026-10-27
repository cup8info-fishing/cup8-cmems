"""
fondali_hr_scarica.py — Scarica le griglie fini di EMODnet (HR-DTM) che toccano la zona della carta di Spot,
e per ognuna legge nei metadati la licenza e chi l'ha fatta (servono per i crediti: CC BY 4.0).

  1) il servizio WFS di EMODnet Bathymetry (livello emodnet:hr_bathymetry_area) dà le griglie nel riquadro,
     col link per scaricarle;
  2) di ogni zip si tengono solo il NetCDF (il .emo dentro è lo stesso dato in testo, 20 volte più grande)
     e il metadata.xml (ISO 19139) dello zip interno;
  3) licenze.json: per griglia la licenza (Creative Commons …), chi l'ha prodotta e i vincoli d'uso.
     Una griglia senza «Creative Commons Attribution» si scarta (stampata a parte): in vendita non entra.

Uscita: <cartella>/<nome>.emo.zip, <cartella>/nc/<nome>.nc, <cartella>/licenze.json
Uso: python fondali_hr_scarica.py <cartella> [ovest sud est nord]
Poi: python fondali_hr.py <cartella>/nc <cartella>/licenze.json <cartella>/pronti
"""
import io
import json
import os
import re
import sys
import urllib.parse
import urllib.request
import zipfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # la console di Windows non è in UTF-8
ZONA = (5.5, 35.0, 19.5, 47.2)                                 # zona.ts: ovest, sud, est, nord
WFS = "https://ows.emodnet-bathymetry.eu/wfs"


def griglie(zona):
    o, s, e, n = zona
    q = {"SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature", "TYPENAMES": "emodnet:hr_bathymetry_area",
         "OUTPUTFORMAT": "application/json", "SRSNAME": "EPSG:4326", "BBOX": f"{s},{o},{n},{e},urn:ogc:def:crs:EPSG::4326"}
    with urllib.request.urlopen(f"{WFS}?{urllib.parse.urlencode(q)}", timeout=120) as r:
        j = json.load(r)
    uniche = {}
    for f in j["features"]:
        p = f["properties"]
        uniche.setdefault(p["identifier"], p)                  # una griglia può avere più poligoni
    return list(uniche.values())


def licenza(xml):
    parti = sorted({(o.strip(), ru) for o, ru in re.findall(
        r"organisationName>\s*<(?:gco:CharacterString|gmx:Anchor)[^>]*>([^<]+)<.*?CI_RoleCode[^>]*codeListValue=\"([^\"]+)\"",
        xml, flags=re.S)})
    return {
        "produttori": sorted({o for o, ru in parti if ru in ("originator", "owner", "resourceProvider", "principalInvestigator", "author")}),
        "tutti": [f"{o} ({ru})" for o, ru in parti],
        "licenza_cc": sorted(set(re.findall(r"Creative Commons[^<\"]{0,80}", xml))),
        "vincoli": sorted(set(t.strip() for t in re.findall(
            r"<(?:gco:CharacterString|gmx:Anchor)[^>]*>([^<]*(?:navigation|licen[cs]e|copyright|restrict)[^<]*)<", xml, flags=re.I)))[:6],
    }


def principale():
    cart = sys.argv[1]
    zona = tuple(map(float, sys.argv[2:6])) if len(sys.argv) >= 6 else ZONA
    os.makedirs(os.path.join(cart, "nc"), exist_ok=True)
    elenco = griglie(zona)
    print(f"griglie fini nella zona: {len(elenco)}", flush=True)
    righe, scartate = [], []
    for p in sorted(elenco, key=lambda p: p["identifier"]):
        url = p["download_url"]
        nome = os.path.basename(urllib.parse.unquote(url))[:-len(".emo.zip")]
        zp = os.path.join(cart, nome + ".emo.zip")
        if not os.path.exists(zp) or os.path.getsize(zp) == 0:
            with urllib.request.urlopen(url.replace(" ", "%20"), timeout=1800) as r, open(zp + ".tmp", "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            os.replace(zp + ".tmp", zp)
        z = zipfile.ZipFile(zp)
        nc = [n for n in z.namelist() if n.endswith(".nc")]
        mz = [n for n in z.namelist() if n.startswith("SDN_CPRD") and n.endswith(".zip")]
        xml = ""
        if mz:
            inner = zipfile.ZipFile(io.BytesIO(z.read(mz[0])))
            xml = next((inner.read(n).decode("utf-8", errors="replace") for n in inner.namelist() if n.endswith("metadata/metadata.xml")), "")
        r = {"id": nome, "risoluzione": p.get("resolution"), "anno": p.get("release"), "scheda": p.get("metadata_url"), **licenza(xml)}
        if not nc or not any("Creative Commons Attribution" in l for l in r["licenza_cc"]):
            scartate.append((nome, "niente NetCDF" if not nc else "licenza: " + ("; ".join(r["licenza_cc"]) or "non scritta")))
            continue
        dest = os.path.join(cart, "nc", nome + ".nc")
        if not os.path.exists(dest):
            with open(dest, "wb") as f:
                f.write(z.read(nc[0]))
        righe.append(r)
        print(f"{nome:55s} 1/{p.get('resolution')}'  {'; '.join(r['produttori'])[:70]}", flush=True)
    json.dump(righe, open(os.path.join(cart, "licenze.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"pronte {len(righe)}, scartate {len(scartate)}", flush=True)
    for nome, perche in scartate:
        print(f"   scartata {nome}: {perche}", flush=True)


if __name__ == "__main__":
    principale()
