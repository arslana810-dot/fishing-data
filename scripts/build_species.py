"""Builds the worldwide species catalogue. Run manually about once a year (GitHub Actions).

  python scripts/build_species.py   -> static/species/catalog_v1.json

For every species in CURATED (scientific name, behaviour group, waters):
  - GBIF: checks the name and collects observation points (CC0 / CC-BY records only)
    -> range as 1-degree cells, plus the month of each observation
  - Water temperature preference: long-term monthly sea temperature (NOAA OISST) for
    sea species, or air temperature (NCEP reanalysis) + 1.5 C for fresh-water species,
    sampled at every observation point and month -> 5/25/50/75/95 percentiles
  - Wikidata (CC0): common names in the app's languages
Anything that fails for a species is reported, never invented.
"""
import datetime as dt
import json
import math
import os
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "OneDiveFishing/0.1 (github.com/arslana810-dot/fishing-data)"}
OUT = os.path.join("static", "species")
WORK = "speciesdata"
LANGS = ["en", "es", "pt", "fr", "de", "it", "nl", "pl", "ru", "uk", "tr", "el", "ar", "hi",
         "id", "ms", "vi", "th", "zh", "zh-hant", "ja", "ko", "sv", "nb"]
MAX_POINTS = 3000

# Groups: pelagic (open-water predators), coastal (shore predators), demersal (near the bottom),
# reef, flatfish, mullet, nocturnal, cephalopod, shark (sharks & rays), salmonid,
# fpredator (fresh-water predators), cyprinid (carp family), catfish, panfish.
# Waters: S = sea, F = fresh water, SF = both (estuaries, migrating species).
CURATED = [
    # Europe, Mediterranean, Black Sea, East Atlantic - sea
    ("Dicentrarchus labrax", "coastal", "S"), ("Dicentrarchus punctatus", "coastal", "S"),
    ("Sparus aurata", "demersal", "S"), ("Diplodus sargus", "demersal", "S"), ("Diplodus vulgaris", "demersal", "S"),
    ("Diplodus puntazzo", "demersal", "S"), ("Diplodus annularis", "demersal", "S"), ("Lithognathus mormyrus", "demersal", "S"),
    ("Pagrus pagrus", "demersal", "S"), ("Pagellus erythrinus", "demersal", "S"), ("Pagellus bogaraveo", "demersal", "S"),
    ("Dentex dentex", "coastal", "S"), ("Spondyliosoma cantharus", "demersal", "S"), ("Oblada melanura", "reef", "S"),
    ("Boops boops", "reef", "S"), ("Sarpa salpa", "reef", "S"), ("Pomatomus saltatrix", "pelagic", "S"),
    ("Sarda sarda", "pelagic", "S"), ("Scomber scombrus", "pelagic", "S"), ("Scomber colias", "pelagic", "S"),
    ("Trachurus trachurus", "pelagic", "S"), ("Trachurus mediterraneus", "pelagic", "S"), ("Seriola dumerili", "pelagic", "S"),
    ("Lichia amia", "pelagic", "S"), ("Trachinotus ovatus", "coastal", "S"), ("Thunnus thynnus", "pelagic", "S"),
    ("Thunnus alalunga", "pelagic", "S"), ("Euthynnus alletteratus", "pelagic", "S"), ("Auxis rochei", "pelagic", "S"),
    ("Xiphias gladius", "pelagic", "S"), ("Sphyraena sphyraena", "pelagic", "S"), ("Mugil cephalus", "mullet", "SF"),
    ("Chelon labrosus", "mullet", "SF"), ("Chelon auratus", "mullet", "S"), ("Chelon ramada", "mullet", "SF"),
    ("Merluccius merluccius", "demersal", "S"), ("Gadus morhua", "demersal", "S"), ("Pollachius pollachius", "coastal", "S"),
    ("Pollachius virens", "demersal", "S"), ("Melanogrammus aeglefinus", "demersal", "S"), ("Molva molva", "demersal", "S"),
    ("Conger conger", "nocturnal", "S"), ("Muraena helena", "nocturnal", "S"), ("Scorpaena scrofa", "reef", "S"),
    ("Epinephelus marginatus", "reef", "S"), ("Sciaena umbra", "nocturnal", "S"), ("Argyrosomus regius", "coastal", "S"),
    ("Umbrina cirrosa", "demersal", "S"), ("Mullus surmuletus", "demersal", "S"), ("Mullus barbatus", "demersal", "S"),
    ("Solea solea", "flatfish", "S"), ("Platichthys flesus", "flatfish", "SF"), ("Pleuronectes platessa", "flatfish", "S"),
    ("Scophthalmus maximus", "flatfish", "S"), ("Labrus bergylta", "reef", "S"), ("Belone belone", "pelagic", "S"),
    ("Loligo vulgaris", "cephalopod", "S"), ("Sepia officinalis", "cephalopod", "S"), ("Octopus vulgaris", "cephalopod", "S"),
    ("Todarodes sagittatus", "cephalopod", "S"), ("Squalus acanthias", "shark", "S"), ("Raja clavata", "shark", "S"),
    ("Galeorhinus galeus", "shark", "S"),
    # North-west Atlantic, Gulf of Mexico, Caribbean - sea
    ("Morone saxatilis", "coastal", "SF"), ("Pogonias cromis", "demersal", "S"), ("Sciaenops ocellatus", "coastal", "S"),
    ("Cynoscion nebulosus", "coastal", "S"), ("Cynoscion regalis", "coastal", "S"), ("Paralichthys dentatus", "flatfish", "S"),
    ("Paralichthys lethostigma", "flatfish", "S"), ("Centropomus undecimalis", "coastal", "SF"), ("Megalops atlanticus", "pelagic", "SF"),
    ("Albula vulpes", "coastal", "S"), ("Rachycentron canadum", "pelagic", "S"), ("Coryphaena hippurus", "pelagic", "S"),
    ("Scomberomorus maculatus", "pelagic", "S"), ("Scomberomorus cavalla", "pelagic", "S"), ("Acanthocybium solandri", "pelagic", "S"),
    ("Thunnus albacares", "pelagic", "S"), ("Thunnus obesus", "pelagic", "S"), ("Katsuwonus pelamis", "pelagic", "S"),
    ("Makaira nigricans", "pelagic", "S"), ("Kajikia albida", "pelagic", "S"), ("Istiophorus platypterus", "pelagic", "S"),
    ("Lutjanus campechanus", "reef", "S"), ("Lutjanus griseus", "reef", "S"), ("Ocyurus chrysurus", "reef", "S"),
    ("Epinephelus morio", "reef", "S"), ("Mycteroperca microlepis", "reef", "S"), ("Archosargus probatocephalus", "demersal", "S"),
    ("Tautoga onitis", "reef", "S"), ("Centropristis striata", "demersal", "S"), ("Caranx hippos", "pelagic", "S"),
    ("Sphyraena barracuda", "pelagic", "S"), ("Hippoglossus hippoglossus", "flatfish", "S"),
    # North-east Pacific - sea and migrating
    ("Hippoglossus stenolepis", "flatfish", "S"), ("Oncorhynchus tshawytscha", "salmonid", "SF"), ("Oncorhynchus kisutch", "salmonid", "SF"),
    ("Oncorhynchus nerka", "salmonid", "SF"), ("Oncorhynchus keta", "salmonid", "SF"), ("Oncorhynchus gorbuscha", "salmonid", "SF"),
    ("Ophiodon elongatus", "reef", "S"), ("Paralabrax clathratus", "reef", "S"), ("Seriola lalandi", "pelagic", "S"),
    ("Atractoscion nobilis", "coastal", "S"), ("Paralichthys californicus", "flatfish", "S"), ("Thunnus orientalis", "pelagic", "S"),
    # Indo-Pacific, Asia, Oceania - sea and estuaries
    ("Caranx ignobilis", "pelagic", "S"), ("Caranx melampygus", "pelagic", "S"), ("Lateolabrax japonicus", "coastal", "SF"),
    ("Pagrus major", "demersal", "S"), ("Acanthopagrus schlegelii", "demersal", "S"), ("Seriola quinqueradiata", "pelagic", "S"),
    ("Lates calcarifer", "coastal", "SF"), ("Epinephelus coioides", "reef", "S"), ("Lutjanus argentimaculatus", "reef", "SF"),
    ("Scomberomorus commerson", "pelagic", "S"), ("Rastrelliger kanagurta", "pelagic", "S"), ("Euthynnus affinis", "pelagic", "S"),
    ("Chrysophrys auratus", "demersal", "S"), ("Argyrosomus japonicus", "coastal", "S"), ("Arripis trutta", "coastal", "S"),
    ("Platycephalus fuscus", "flatfish", "S"), ("Sillago ciliata", "demersal", "S"),
    # Fresh water - Europe and Asia
    ("Esox lucius", "fpredator", "F"), ("Sander lucioperca", "fpredator", "F"), ("Perca fluviatilis", "panfish", "F"),
    ("Silurus glanis", "catfish", "F"), ("Cyprinus carpio", "cyprinid", "F"), ("Abramis brama", "cyprinid", "F"),
    ("Rutilus rutilus", "cyprinid", "F"), ("Scardinius erythrophthalmus", "cyprinid", "F"), ("Tinca tinca", "cyprinid", "F"),
    ("Barbus barbus", "cyprinid", "F"), ("Squalius cephalus", "cyprinid", "F"), ("Leuciscus aspius", "fpredator", "F"),
    ("Salmo trutta", "salmonid", "SF"), ("Salmo salar", "salmonid", "SF"), ("Oncorhynchus mykiss", "salmonid", "SF"),
    ("Salvelinus alpinus", "salmonid", "SF"), ("Salvelinus fontinalis", "salmonid", "F"), ("Coregonus lavaretus", "salmonid", "F"),
    ("Thymallus thymallus", "salmonid", "F"), ("Thymallus arcticus", "salmonid", "F"), ("Hucho hucho", "salmonid", "F"),
    ("Hucho taimen", "salmonid", "F"), ("Lota lota", "nocturnal", "F"), ("Anguilla anguilla", "nocturnal", "SF"),
    ("Carassius carassius", "cyprinid", "F"), ("Ctenopharyngodon idella", "cyprinid", "F"), ("Hypophthalmichthys molitrix", "cyprinid", "F"),
    ("Channa striata", "fpredator", "F"), ("Channa micropeltes", "fpredator", "F"), ("Tor putitora", "cyprinid", "F"),
    ("Siniperca chuatsi", "fpredator", "F"), ("Plecoglossus altivelis", "salmonid", "SF"),
    # Fresh water - North America
    ("Micropterus salmoides", "fpredator", "F"), ("Micropterus dolomieu", "fpredator", "F"), ("Micropterus punctulatus", "fpredator", "F"),
    ("Sander vitreus", "fpredator", "F"), ("Sander canadensis", "fpredator", "F"), ("Esox masquinongy", "fpredator", "F"),
    ("Esox niger", "fpredator", "F"), ("Perca flavescens", "panfish", "F"), ("Lepomis macrochirus", "panfish", "F"),
    ("Lepomis gibbosus", "panfish", "F"), ("Pomoxis nigromaculatus", "panfish", "F"), ("Pomoxis annularis", "panfish", "F"),
    ("Ictalurus punctatus", "catfish", "F"), ("Ictalurus furcatus", "catfish", "F"), ("Pylodictis olivaris", "catfish", "F"),
    ("Ameiurus nebulosus", "catfish", "F"), ("Morone chrysops", "fpredator", "F"), ("Salvelinus namaycush", "salmonid", "F"),
    ("Oncorhynchus clarkii", "salmonid", "SF"), ("Amia calva", "fpredator", "F"), ("Lepisosteus osseus", "fpredator", "F"),
    ("Aplodinotus grunniens", "demersal", "F"),
    # Fresh water - South America, Africa, Australia
    ("Salminus brasiliensis", "fpredator", "F"), ("Cichla ocellaris", "fpredator", "F"), ("Arapaima gigas", "fpredator", "F"),
    ("Pseudoplatystoma corruscans", "catfish", "F"), ("Hoplias malabaricus", "fpredator", "F"), ("Colossoma macropomum", "panfish", "F"),
    ("Odontesthes bonariensis", "panfish", "F"), ("Hydrocynus vittatus", "fpredator", "F"), ("Lates niloticus", "fpredator", "F"),
    ("Oreochromis niloticus", "panfish", "F"), ("Clarias gariepinus", "catfish", "F"), ("Maccullochella peelii", "fpredator", "F"),
    ("Macquaria ambigua", "fpredator", "F"),
]

SST_URLS = [
    "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.mon.ltm.1991-2020.nc",
    "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.mon.ltm.1982-2010.nc",
]
AIR_URLS = [
    "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/Monthlies/surface/air.mon.ltm.1991-2020.nc",
    "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/Monthlies/surface/air.mon.ltm.nc",
]


def get_json(url, data=None, headers=None):
    h = dict(UA)
    if headers:
        h.update(headers)
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, data=data, headers=h)
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print("    retry", attempt + 1, url[:80], e)
            time.sleep(4 * (attempt + 1))
    return None


def species_id(scientific):
    return scientific.lower().replace(" ", "_")


def cell_code(lat, lon):
    """1-degree cell number (pure function, tested)."""
    r = min(179, max(0, int(math.floor(lat + 90))))
    c = int(math.floor(lon + 180)) % 360
    return r * 360 + c


def dilate(cells):
    """Adds the 8 neighbours of every cell, to close small gaps between observations."""
    out = set(cells)
    for code in cells:
        r, c = divmod(code, 360)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                if 0 <= rr < 180:
                    out.add(rr * 360 + (c + dc) % 360)
    return out


def percentiles(values, ps=(5, 25, 50, 75, 95)):
    """Simple percentiles (pure function, tested)."""
    v = sorted(values)
    if not v:
        return None
    out = {}
    for p in ps:
        k = (len(v) - 1) * p / 100
        lo, hi = math.floor(k), math.ceil(k)
        out[f"p{p:02d}"] = round(v[lo] + (v[hi] - v[lo]) * (k - lo), 2)
    return out


def gbif_points(name):
    m = get_json("https://api.gbif.org/v1/species/match?" + urllib.parse.urlencode({"name": name, "strict": "true"}))
    if not m or m.get("matchType") not in ("EXACT",) or m.get("rank") != "SPECIES":
        return None, []
    key = m.get("usageKey")
    pts = []
    offset = 0
    while len(pts) < MAX_POINTS:
        q = urllib.parse.urlencode([
            ("taxonKey", key), ("hasCoordinate", "true"), ("hasGeospatialIssue", "false"),
            ("occurrenceStatus", "PRESENT"), ("license", "CC0_1_0"), ("license", "CC_BY_4_0"),
            ("limit", 300), ("offset", offset),
        ])
        res = get_json("https://api.gbif.org/v1/occurrence/search?" + q)
        if not res:
            break
        for o in res.get("results", []):
            lat, lon = o.get("decimalLatitude"), o.get("decimalLongitude")
            if lat is None or lon is None:
                continue
            pts.append((float(lat), float(lon), o.get("month")))
        if res.get("endOfRecords") or not res.get("results"):
            break
        offset += 300
        time.sleep(0.3)
    return key, pts


def wikidata_names(name):
    langs = ",".join(f'"{x}"' for x in LANGS)
    query = f"""
    SELECT ?lang ?name ?kind WHERE {{
      ?item wdt:P225 "{name}" .
      {{ ?item wdt:P1843 ?name . BIND("common" AS ?kind) }} UNION {{ ?item rdfs:label ?name . BIND("label" AS ?kind) }}
      BIND(LANG(?name) AS ?lang)
      FILTER(?lang IN ({langs}))
    }}"""
    res = get_json("https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": query, "format": "json"}),
                   headers={"Accept": "application/sparql-results+json"})
    names = {}
    if not res:
        return names
    best = {}
    for b in res.get("results", {}).get("bindings", []):
        lang = b["lang"]["value"]
        text = b["name"]["value"].strip()
        kind = b["kind"]["value"]
        if not text or text.lower() == name.lower():
            continue  # a scientific name is not a common name
        rank = 0 if kind == "common" else 1
        if lang not in best or rank < best[lang][0]:
            best[lang] = (rank, text[:60])
    for lang, (_, text) in best.items():
        names["zh_Hant" if lang == "zh-hant" else lang] = text
    return names


class Climate:
    """Long-term monthly temperature grid, sampled at a point and month."""

    def __init__(self, urls, var, kelvin=False):
        import netCDF4
        import numpy as np
        self.np = np
        self.ok = False
        for url in urls:
            try:
                path = os.path.join(WORK, os.path.basename(url))
                if not os.path.exists(path):
                    req = urllib.request.Request(url, headers=UA)
                    with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as f:
                        f.write(r.read())
                nc = netCDF4.Dataset(path)
                self.lat = np.asarray(nc.variables["lat"][:], dtype=float)
                self.lon = np.asarray(nc.variables["lon"][:], dtype=float)
                data = nc.variables[var][:]
                self.data = np.ma.filled(data.astype(float), np.nan)
                if kelvin and np.nanmean(self.data) > 100:
                    self.data = self.data - 273.15
                self.source = os.path.basename(url)
                self.ok = True
                print("climate loaded:", self.source, self.data.shape)
                break
            except Exception as e:
                print("climate source failed:", url, e)

    def at(self, lat, lon, month):
        np = self.np
        if not self.ok:
            return None
        lon360 = lon % 360
        i = int(np.abs(self.lat - lat).argmin())
        j = int(np.abs(self.lon - lon360).argmin())
        months = [month - 1] if month else range(12)
        vals = []
        for m in months:
            # nearest valid value in a small window (coasts often fall on land cells)
            for d in (0, 1, 2):
                block = self.data[m, max(0, i - d):i + d + 1, max(0, j - d):j + d + 1]
                if np.isfinite(block).any():
                    vals.append(float(np.nanmean(block)))
                    break
        return float(np.mean(vals)) if vals else None


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    sst = Climate(SST_URLS, "sst")
    air = Climate(AIR_URLS, "air", kelvin=True)
    species, problems = [], {}
    for sci, group, waters in CURATED:
        sid = species_id(sci)
        key, pts = gbif_points(sci)
        if key is None:
            problems[sid] = "name not found in GBIF"
            print(f"{sid}: NOT FOUND")
            continue
        if len(pts) < 30:
            problems[sid] = f"only {len(pts)} usable observations"
        cells = dilate({cell_code(la, lo) for la, lo, _ in pts}) if pts else set()
        temps = []
        source = None
        for la, lo, mo in pts[:1500]:
            t = None
            if "S" in waters:
                t = sst.at(la, lo, mo)
                source = sst.source if sst.ok else None
            if t is None and "F" in waters:
                a = air.at(la, lo, mo)
                if a is not None:
                    t = max(0.5, a + 1.5)
                    source = (source + "+" if source else "") + (air.source if air.ok else "")
            if t is not None and -3 < t < 40:
                temps.append(t)
        temp = percentiles(temps)
        if temp:
            temp["n"] = len(temps)
        names = wikidata_names(sci)
        time.sleep(0.5)
        species.append({
            "id": sid,
            "scientific": sci,
            "gbif_key": key,
            "group": group,
            "waters": waters,
            "observations": len(pts),
            "cells": sorted(cells),
            "temp": temp,
            "temp_source": source,
            "names": names,
        })
        print(f"{sid}: {len(pts)} points, {len(cells)} cells, temp {temp and temp.get('p50')}, names {len(names)}")
    catalog = {
        "version": 1,
        "built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"),
        "languages": LANGS,
        "cell": "code = floor(lat+90)*360 + floor(lon+180) mod 360 (1-degree cells)",
        "sources": {
            "ranges": "GBIF.org occurrence data (CC0 and CC BY 4.0 records)",
            "temperature": "NOAA OISST v2 and NCEP/NCAR Reanalysis long-term monthly means (NOAA PSL, public domain)",
            "names": "Wikidata (CC0)",
        },
        "species": species,
        "problems": problems,
    }
    path = os.path.join(OUT, "catalog_v1.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, separators=(",", ":"))
    print(f"SPECIES DONE: {len(species)} species, {len(problems)} with problems, {os.path.getsize(path) / 1e6:.1f} MB")
    for sid, why in sorted(problems.items()):
        print(f"  PROBLEM {sid}: {why}")


if __name__ == "__main__":
    main()
