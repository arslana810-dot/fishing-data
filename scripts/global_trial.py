"""Trial: global Copernicus model for one large box, coastal cells only,
compact binary tiles. Measures time and size. Publishes nothing."""
import datetime as dt
import gzip
import io
import math
import os
import struct
import time

import numpy as np
import xarray as xr
import copernicusmarine as cm
from scipy.ndimage import binary_dilation

WORK = "work"
STEP_H = 3
DAYS = 7
COAST_CELLS = 4  # keep sea cells within ~4 grid cells (~35 km) of land
BBOX = (20.0, 45.0, 30.0, 48.0)  # lon_min, lon_max, lat_min, lat_max
VARS = ["hs", "sw", "tp", "dir", "sst", "cs", "cd"]
SCALES = {"hs": 100, "sw": 100, "tp": 10, "dir": 1, "sst": 10, "cs": 100, "cd": 1}
MISSING = -32768

NOW = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)
START = NOW - dt.timedelta(hours=NOW.hour % STEP_H)
END = START + dt.timedelta(days=DAYS)
TIMES = np.array(
    [np.datetime64(START + dt.timedelta(hours=STEP_H * i)) for i in range(DAYS * 24 // STEP_H + 1)]
).astype("datetime64[ns]")

TEST_POINTS = [
    ("Marmara (Tekirdag offshore)", 40.85, 27.60),
    ("Istanbul Bosphorus", 41.10, 29.05),
    ("Samandag", 36.08, 35.93),
    ("Izmir Bay", 38.45, 27.00),
    ("Trabzon", 41.00, 39.72),
    ("Antalya", 36.85, 30.70),
]


def fetch(dataset_id, variables, depth, name):
    path = os.path.join(WORK, name + ".nc")
    base = dict(
        dataset_id=dataset_id,
        variables=variables,
        minimum_longitude=BBOX[0],
        maximum_longitude=BBOX[1],
        minimum_latitude=BBOX[2],
        maximum_latitude=BBOX[3],
        start_datetime=START.strftime("%Y-%m-%dT%H:%M:%S"),
        end_datetime=END.strftime("%Y-%m-%dT%H:%M:%S"),
        output_directory=WORK,
        output_filename=name + ".nc",
    )
    attempts = [dict(base, minimum_depth=depth[0], maximum_depth=depth[1]), base] if depth else [base]
    t_start = time.time()
    last_error = None
    for kw in attempts:
        try:
            if os.path.exists(path):
                os.remove(path)
            result = cm.subset(**kw)
            path = str(getattr(result, "file_path", path) or path)
            last_error = None
            break
        except Exception as e:
            last_error = e
            print("    subset attempt failed:", type(e).__name__, e)
    if last_error is not None:
        raise last_error
    secs = time.time() - t_start
    mb = os.path.getsize(path) / 1e6
    print(f"  >>> {name}: {secs:.0f} s, file {mb:.0f} MB")

    ds = xr.open_dataset(path)
    if "depth" in ds.dims:
        ds = ds.isel(depth=0)
    ds = ds.sortby("latitude").sortby("longitude")
    ds = ds.reindex(time=TIMES, method="nearest", tolerance=np.timedelta64(90, "m"))
    return ds


def grid(da):
    return da.transpose("time", "latitude", "longitude").values.astype("float64")


def tile_bytes(layers):
    b = io.BytesIO()
    b.write(b"FSH1")
    t0 = int(START.replace(tzinfo=dt.timezone.utc).timestamp())
    b.write(struct.pack("<IHHBB", t0, STEP_H, len(TIMES), len(VARS), len(layers)))
    for L in layers:
        src = L["src"].encode("ascii")
        b.write(struct.pack("<B", len(src)))
        b.write(src)
        b.write(struct.pack("<ffffHHI", L["lat0"], L["lon0"], L["dlat"], L["dlon"],
                            L["nlat"], L["nlon"], len(L["idx"])))
        b.write(np.asarray(L["idx"], dtype="<u2").tobytes())
        for v in VARS:
            arr = L["vals"][v]
            q = np.where(np.isfinite(arr), np.round(arr * SCALES[v]), MISSING)
            q = np.clip(q, -32768, 32767).astype("<i2")
            b.write(q.tobytes())
    return b.getvalue()


def km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(a))


def probe(tiles, name, lat, lon):
    best = None
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            key = f"{int(math.floor(lat)) + dy}_{int(math.floor(lon)) + dx}"
            for L in tiles.get(key, []):
                idx = np.asarray(L["idx"])
                r = idx // L["nlon"]
                c = idx % L["nlon"]
                d = km(lat, lon, L["lat0"] + r * L["dlat"], L["lon0"] + c * L["dlon"])
                k = int(np.argmin(d))
                if best is None or d[k] < best[0]:
                    best = (float(d[k]), L, k)
    if best is None:
        print(f"  {name}: NO SEA DATA NEARBY")
        return
    d, L, k = best
    hs = L["vals"]["hs"][0, k]
    sst = L["vals"]["sst"][0, k]
    cs = L["vals"]["cs"][0, k]
    print(f"  {name}: nearest sea cell {d:.1f} km | wave {hs:.2f} m | sea temp {sst:.1f} C | current {cs:.2f} m/s")


def main():
    os.makedirs(WORK, exist_ok=True)
    t_all = time.time()
    print("Downloading global waves...")
    w = fetch("cmems_mod_glo_wav_anfc_0.083deg_PT3H-i", ["VHM0", "VMDR", "VTPK", "VHM0_SW1"], None, "glo_wave")
    print("Downloading global physics...")
    p = fetch("cmems_mod_glo_phy_anfc_0.083deg_PT1H-m", ["thetao", "uo", "vo"], (0.0, 1.0), "glo_phy")

    lats = w["latitude"].values.astype(float)
    lons = w["longitude"].values.astype(float)
    dlat = float(abs(lats[1] - lats[0]))
    dlon = float(abs(lons[1] - lons[0]))
    p = p.reindex(latitude=w["latitude"], longitude=w["longitude"], method="nearest",
                  tolerance=0.75 * max(dlat, dlon))

    u = grid(p["uo"])
    v = grid(p["vo"])
    data = {
        "hs": grid(w["VHM0"]),
        "sw": grid(w["VHM0_SW1"]),
        "tp": grid(w["VTPK"]),
        "dir": grid(w["VMDR"]),
        "sst": grid(p["thetao"]),
        "cs": np.sqrt(u * u + v * v),
        "cd": (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0,
    }
    sea = np.isfinite(data["hs"]).any(axis=0) | np.isfinite(data["sst"]).any(axis=0)
    coastal = sea & binary_dilation(~sea, iterations=COAST_CELLS)
    print(f"Cells: total {sea.size}, sea {int(sea.sum())}, coastal kept {int(coastal.sum())}")

    tiles = {}
    for ty in range(int(np.floor(lats.min())), int(np.ceil(lats.max()))):
        for tx in range(int(np.floor(lons.min())), int(np.ceil(lons.max()))):
            iy = np.where((lats >= ty) & (lats < ty + 1))[0]
            ix = np.where((lons >= tx) & (lons < tx + 1))[0]
            if len(iy) == 0 or len(ix) == 0:
                continue
            sub = coastal[np.ix_(iy, ix)]
            if not sub.any():
                continue
            yy, xx = np.nonzero(sub)
            sel_y = iy[yy]
            sel_x = ix[xx]
            tiles.setdefault(f"{ty}_{tx}", []).append({
                "src": "glo",
                "lat0": float(lats[iy[0]]),
                "lon0": float(lons[ix[0]]),
                "dlat": dlat,
                "dlon": dlon,
                "nlat": int(len(iy)),
                "nlon": int(len(ix)),
                "idx": (yy * len(ix) + xx).astype(int).tolist(),
                "vals": {name: arr[:, sel_y, sel_x] for name, arr in data.items()},
            })

    raw_total = 0
    gz_total = 0
    for layers in tiles.values():
        b = tile_bytes(layers)
        raw_total += len(b)
        gz_total += len(gzip.compress(b, 6))

    print("=" * 70)
    print(f"Tiles: {len(tiles)}")
    print(f"Binary size: {raw_total / 1e6:.2f} MB total, {raw_total / max(1, len(tiles)) / 1e3:.0f} KB per tile")
    print(f"Gzip size:   {gz_total / 1e6:.2f} MB total, {gz_total / max(1, len(tiles)) / 1e3:.0f} KB per tile")
    print(f"Total runtime: {time.time() - t_all:.0f} s")
    print("Test points:")
    for name, lat, lon in TEST_POINTS:
        probe(tiles, name, lat, lon)
    print("=" * 70)


if __name__ == "__main__":
    main()
