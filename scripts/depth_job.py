"""Static depth tiles from the GEBCO grid (public domain). Run manually about once a year.

  python scripts/depth_job.py probe   -> checks the GEBCO source and prints sample depths
  python scripts/depth_job.py build   -> writes static/depth/{lat}_{lon}.dpt for every sea tile

File format "DPT1" (little-endian):
  "DPT1", int16 lat0, int16 lon0, uint16 n, then int16[n*n] heights in metres
  (row 0 = southern edge, columns west to east; negative = below sea level, positive = land).
Attribution: GEBCO Compilation Group (2026) GEBCO 2026 Grid. Not for navigation.
"""
import concurrent.futures as cf
import json
import os
import struct
import sys
import threading
import urllib.request

import numpy as np

SOURCES = [
    "https://data.source.coop/giswqs/gebco-bathymetry/gebco_2026_geotiff/gebco_2026.tif",
]
INDEX_URL = "https://arslana810-dot.github.io/fishing-data/index.json"
OUT = os.path.join("static", "depth")
CELLS_PER_DEG = 240  # GEBCO: 15 arc-seconds
N = 120              # stored: 30 arc-seconds (about 900 m)
ATTRIBUTION = "GEBCO Compilation Group (2026) GEBCO 2026 Grid. Not to be used for navigation."

_local = threading.local()


def open_source():
    import rasterio
    last = None
    for url in SOURCES:
        try:
            return rasterio.open("/vsicurl/" + url), url
        except Exception as e:
            last = e
            print("source failed:", url, e)
    raise RuntimeError(f"No GEBCO source reachable: {last}")


def dataset():
    ds = getattr(_local, "ds", None)
    if ds is None:
        ds, _ = open_source()
        _local.ds = ds
    return ds


def read_degree(ds, lat0, lon0):
    """Heights for [lat0, lat0+1) x [lon0, lon0+1), 240 x 240, row 0 = south."""
    from rasterio.windows import from_bounds
    win = from_bounds(lon0, lat0, lon0 + 1, lat0 + 1, transform=ds.transform)
    arr = ds.read(1, window=win, out_shape=(CELLS_PER_DEG, CELLS_PER_DEG), boundless=True, fill_value=0)
    return north_up_to_south_up(arr)


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


def probe():
    ds, url = open_source()
    print("source:", url)
    print("size:", ds.width, "x", ds.height, "| type:", ds.dtypes, "| crs:", ds.crs)
    print("transform:", ds.transform)
    for name, lat, lon in [("Marmara (Tekirdag offshore)", 40.85, 27.6), ("Mid-Atlantic", 0.5, -29.5),
                           ("Samandag shore", 36.08, 35.93), ("Lake Van (land/lake surface)", 38.6, 42.9)]:
        a = read_degree(ds, int(np.floor(lat)), int(np.floor(lon)))
        r = int((lat - np.floor(lat)) * CELLS_PER_DEG)
        c = int((lon - np.floor(lon)) * CELLS_PER_DEG)
        print(f"  {name}: {int(a[r, c])} m")
    print("PROBE OK")


def build():
    idx = json.load(urllib.request.urlopen(INDEX_URL, timeout=60))
    tiles = idx["tiles"]
    os.makedirs(OUT, exist_ok=True)
    ds, url = open_source()
    print("source:", url, "| tiles to build:", len(tiles))

    def work(key):
        path = os.path.join(OUT, key + ".dpt")
        if os.path.exists(path):
            return key, "skip"
        lat0, lon0 = map(int, key.split("_"))
        for attempt in range(3):
            try:
                arr = downsample(read_degree(dataset(), lat0, lon0))
                with open(path, "wb") as f:
                    f.write(pack(lat0, lon0, arr))
                return key, "ok"
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                _local.ds = None
        return key, "fail " + err

    ok, failed = 0, {}
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for i, (key, res) in enumerate(pool.map(work, tiles), 1):
            if res in ("ok", "skip"):
                ok += 1
            else:
                failed[key] = res
            if i % 500 == 0:
                print(f"  {i}/{len(tiles)} done")
    meta = {"format": "DPT1", "n": N, "source": url, "attribution": ATTRIBUTION,
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
