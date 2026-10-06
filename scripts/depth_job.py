"""Static depth tiles from the GEBCO grid (public domain). Run manually about once a year.

  python scripts/depth_job.py probe   -> finds a working GEBCO source and prints sample depths
  python scripts/depth_job.py build   -> writes static/depth/{lat}_{lon}.dpt for every sea tile

Sources, tried in order:
  1) Cloud-optimised GeoTIFF copy (reads only the needed parts over the internet)
  2) Official GEBCO global netCDF file (about 4 GB download, unpacked on the runner)

File format "DPT1" (little-endian):
  "DPT1", int16 lat0, int16 lon0, uint16 n, then int16[n*n] heights in metres
  (row 0 = southern edge, columns west to east; negative = below sea level, positive = land).
Attribution: GEBCO Compilation Group (2026) GEBCO 2026 Grid. Not for navigation.
"""
import concurrent.futures as cf
import json
import os
import shutil
import struct
import sys
import threading
import urllib.request
import zipfile

import numpy as np

COOP = "https://data.source.coop/giswqs/gebco-bathymetry/"
COG_CANDIDATES = [
    COOP + "gebco_2026/gebco_2026.tif",  # confirmed by the folder listing (Oct 2026)
    COOP + "gebco_2026_geotiff/gebco_2026.tif",
    COOP + "gebco_2026/gebco_2026_geotiff/gebco_2026.tif",
    COOP + "gebco_2026_geotiff/GEBCO_2026.tif",
    COOP + "geotiff/gebco_2026.tif",
    COOP + "gebco_2026.tif",
]
LIST_URL = COOP + "?list-type=2&max-keys=200"
ZIP_URL = "https://www.bodc.ac.uk/data/open_download/gebco/gebco_2026/zip/"
WORK = "/tmp/gebco"
INDEX_URL = "https://arslana810-dot.github.io/fishing-data/index.json"
OUT = os.path.join("static", "depth")
CELLS_PER_DEG = 240  # GEBCO: 15 arc-seconds
N = 120              # stored: 30 arc-seconds (about 900 m)
ATTRIBUTION = "GEBCO Compilation Group (2026) GEBCO 2026 Grid. Not to be used for navigation."

_local = threading.local()


def http_status(url, method="HEAD"):
    try:
        req = urllib.request.Request(url, method=method, headers={"User-Agent": "OneDiveFishing/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.headers.get("Content-Length")
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:
        return f"{type(e).__name__}: {e}", None


def north_up_to_south_up(arr):
    # Rasters store the northern row first; our tiles store the southern row first.
    return np.flipud(arr)


def downsample(a):
    """240 x 240 -> 120 x 120 by averaging 2 x 2 blocks."""
    a = np.asarray(a, dtype=np.float32).reshape(N, 2, N, 2).mean(axis=(1, 3))
    return np.clip(np.round(a), -32768, 32767).astype("<i2")


def pack(lat0, lon0, arr):
    return b"DPT1" + struct.pack("<hhH", lat0, lon0, N) + np.asarray(arr, dtype="<i2").tobytes()


def unpack(blob):
    assert blob[:4] == b"DPT1"
    lat0, lon0, n = struct.unpack_from("<hhH", blob, 4)
    arr = np.frombuffer(blob, dtype="<i2", offset=10).reshape(n, n)
    return lat0, lon0, arr


# ---------------- Source 1: cloud-optimised GeoTIFF ----------------

class CogSource:
    def __init__(self, url):
        self.url = url

    def _ds(self):
        import rasterio
        ds = getattr(_local, "ds", None)
        if ds is None:
            ds = rasterio.open("/vsicurl/" + self.url)
            _local.ds = ds
        return ds

    def read_degree(self, lat0, lon0):
        from rasterio.windows import from_bounds
        ds = self._ds()
        win = from_bounds(lon0, lat0, lon0 + 1, lat0 + 1, transform=ds.transform)
        arr = ds.read(1, window=win, out_shape=(CELLS_PER_DEG, CELLS_PER_DEG), boundless=True, fill_value=0)
        return north_up_to_south_up(arr)

    def reset(self):
        _local.ds = None

    def describe(self):
        ds = self._ds()
        return f"COG {self.url} | {ds.width}x{ds.height} {ds.dtypes} {ds.crs}"


def listed_cogs():
    """Global GEBCO files found in the cloud folder listing (newest year first)."""
    try:
        req = urllib.request.Request(LIST_URL, headers={"User-Agent": "OneDiveFishing/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception:
        return []
    keys = [k.split("</Key>")[0] for k in text.split("<Key>")[1:]]
    found = []
    for k in keys:
        name = k.rsplit("/", 1)[-1]
        # the global grid only: "gebco_YYYY.tif" (not tiles, web-mercator, sub-ice or TID files)
        if name.startswith("gebco_") and name.endswith(".tif") and name[6:-4].isdigit():
            found.append(COOP + k.split("/", 1)[1])
    return sorted(found, reverse=True)


def find_cog():
    import rasterio
    for url in COG_CANDIDATES + [u for u in listed_cogs() if u not in COG_CANDIDATES]:
        try:
            with rasterio.open("/vsicurl/" + url) as ds:
                if ds.width >= 86000:
                    return url
        except Exception as e:
            print("  COG not at", url, "->", str(e)[:120])
    return None


# ---------------- Source 2: official netCDF file ----------------

class NetcdfSource:
    def __init__(self, path):
        import netCDF4
        self.nc = netCDF4.Dataset(path)
        names = self.nc.variables.keys()
        self.var = "elevation" if "elevation" in names else [n for n in names if n not in ("lat", "lon", "crs")][0]
        lat = self.nc.variables["lat"][:]
        self.south_first = bool(lat[0] < lat[-1])
        self.lock = threading.Lock()

    def read_degree(self, lat0, lon0):
        r0 = (lat0 + 90) * CELLS_PER_DEG if self.south_first else (90 - lat0 - 1) * CELLS_PER_DEG
        c0 = (lon0 + 180) * CELLS_PER_DEG
        with self.lock:  # netCDF reads are not thread safe
            arr = np.asarray(self.nc.variables[self.var][r0:r0 + CELLS_PER_DEG, c0:c0 + CELLS_PER_DEG])
        if arr.shape != (CELLS_PER_DEG, CELLS_PER_DEG):
            full = np.zeros((CELLS_PER_DEG, CELLS_PER_DEG), dtype=np.int16)
            full[:arr.shape[0], :arr.shape[1]] = arr
            arr = full
        return arr if self.south_first else north_up_to_south_up(arr)

    def reset(self):
        pass

    def describe(self):
        v = self.nc.variables[self.var]
        return f"netCDF variable '{self.var}' shape {v.shape} south_first={self.south_first}"


def download_netcdf():
    os.makedirs(WORK, exist_ok=True)
    free_gb = shutil.disk_usage(WORK).free / 1e9
    print(f"  free disk: {free_gb:.1f} GB")
    zpath = os.path.join(WORK, "gebco.zip")
    print("  downloading", ZIP_URL)
    req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "OneDiveFishing/0.1"})
    with urllib.request.urlopen(req, timeout=600) as r, open(zpath, "wb") as f:
        shutil.copyfileobj(r, f, length=16 * 1024 * 1024)
    with zipfile.ZipFile(zpath) as z:
        members = [m for m in z.namelist() if m.lower().endswith(".nc")]
        print("  zip contains:", members)
        z.extract(members[0], WORK)
        nc_path = os.path.join(WORK, members[0])
    os.remove(zpath)
    return nc_path


def get_source(allow_download):
    url = find_cog()
    if url:
        return CogSource(url)
    if not allow_download:
        return None
    return NetcdfSource(download_netcdf())


# ---------------- Commands ----------------

def probe():
    print("Listing the cloud folder:")
    try:
        req = urllib.request.Request(LIST_URL, headers={"User-Agent": "OneDiveFishing/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode("utf-8", "replace")
        keys = [k.split("</Key>")[0] for k in text.split("<Key>")[1:]]
        print("  files:", keys[:60] if keys else text[:1500])
    except Exception as e:
        print("  listing failed:", e)
    print("Official zip:", ZIP_URL, "->", http_status(ZIP_URL))
    src = get_source(allow_download=False)
    if src is None:
        print("No cloud copy found. The build will download the official file instead (about 4 GB).")
        print("PROBE DONE (official file route)")
        return
    print("Using:", src.describe())
    for name, lat, lon in [("Marmara (Tekirdag offshore)", 40.85, 27.6), ("Mid-Atlantic", 0.5, -29.5),
                           ("Samandag shore", 36.08, 35.93), ("Lake Van (lake surface)", 38.6, 42.9)]:
        a = src.read_degree(int(np.floor(lat)), int(np.floor(lon)))
        r = int((lat - np.floor(lat)) * CELLS_PER_DEG)
        c = int((lon - np.floor(lon)) * CELLS_PER_DEG)
        print(f"  {name}: {int(a[r, c])} m")
    print("PROBE OK (cloud route)")


def build():
    idx = json.load(urllib.request.urlopen(INDEX_URL, timeout=60))
    tiles = idx["tiles"]
    os.makedirs(OUT, exist_ok=True)
    src = get_source(allow_download=True)
    print("Using:", src.describe(), "| tiles to build:", len(tiles))

    def work(key):
        path = os.path.join(OUT, key + ".dpt")
        if os.path.exists(path):
            return key, "skip"
        lat0, lon0 = map(int, key.split("_"))
        err = ""
        for _ in range(3):
            try:
                arr = downsample(src.read_degree(lat0, lon0))
                with open(path, "wb") as f:
                    f.write(pack(lat0, lon0, arr))
                return key, "ok"
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                src.reset()
        return key, "fail " + err

    ok, failed = 0, {}
    workers = 8 if isinstance(src, CogSource) else 2
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (key, res) in enumerate(pool.map(work, tiles), 1):
            if res in ("ok", "skip"):
                ok += 1
            else:
                failed[key] = res
            if i % 500 == 0:
                print(f"  {i}/{len(tiles)} done")
    meta = {"format": "DPT1", "n": N, "source": src.describe(), "attribution": ATTRIBUTION,
            "tiles": ok, "failed": failed}
    with open(os.path.join(OUT, "index.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"DONE: {ok} depth tiles, {len(failed)} failed")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "probe":
        probe()
    elif cmd == "build":
        build()
    else:
        print(__doc__)
        sys.exit(2)
