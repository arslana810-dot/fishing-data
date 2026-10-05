"""Builds compact sea-forecast tiles for Turkish seas from Copernicus Marine.

Output folder "site":
  index.json               run info, tile list, units, attribution
  tiles/{lat}_{lon}.json   one file per 1x1 degree tile that has sea cells
Each tile holds one or more "layers" (one per source model). The app picks
the nearest valid sea cell across layers.
"""
import datetime as dt
import json
import os
import sys

import numpy as np
import xarray as xr
import copernicusmarine as cm

OUT = "site"
WORK = "work"
STEP_H = 3
DAYS = 7

# stored integer = real value * scale
SCALES = {"hs": 100, "sw": 100, "tp": 10, "dir": 1, "sst": 10, "cs": 100, "cd": 1}
UNITS = {
    "hs": "wave height, cm",
    "sw": "primary swell height, cm",
    "tp": "peak wave period, 0.1 s",
    "dir": "mean wave direction, deg (coming from)",
    "sst": "sea surface temperature, 0.1 C",
    "cs": "surface current speed, cm/s",
    "cd": "surface current direction, deg (flowing towards)",
}

# bbox = (lon_min, lon_max, lat_min, lat_max)
REGIONS = {
    "med": {
        "bbox": (25.0, 36.6, 34.5, 41.5),
        "products": ["MEDSEA_ANALYSISFORECAST_WAV_006_017", "MEDSEA_ANALYSISFORECAST_PHY_006_013"],
        "wave": ("cmems_mod_med_wav_anfc_4.2km_PT1H-i", None),
        "temp": ("cmems_mod_med_phy-tem_anfc_4.2km-2D_PT1H-m", None),
        "cur": ("cmems_mod_med_phy-cur_anfc_4.2km-2D_PT1H-m", None),
    },
    "blk": {
        "bbox": (27.0, 42.0, 40.5, 43.0),
        "products": ["BLKSEA_ANALYSISFORECAST_WAV_007_003", "BLKSEA_ANALYSISFORECAST_PHY_007_001"],
        "wave": ("cmems_mod_blk_wav_anfc_2.5km_PT1H-i", None),
        "temp": ("cmems_mod_blk_phy-temp_anfc_2.5km_PT1H-m", (0.0, 5.0)),
        "cur": ("cmems_mod_blk_phy-cur_anfc_2.5km_PT1H-m", (0.0, 5.0)),
    },
}

NOW = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)
START = NOW - dt.timedelta(hours=NOW.hour % STEP_H)
END = START + dt.timedelta(days=DAYS)
TIMES = np.array(
    [np.datetime64(START + dt.timedelta(hours=STEP_H * i)) for i in range(DAYS * 24 // STEP_H + 1)]
).astype("datetime64[ns]")


def fetch(dataset_id, variables, bbox, depth, name):
    path = os.path.join(WORK, name + ".nc")
    base = dict(
        dataset_id=dataset_id,
        variables=variables,
        minimum_longitude=bbox[0],
        maximum_longitude=bbox[1],
        minimum_latitude=bbox[2],
        maximum_latitude=bbox[3],
        start_datetime=START.strftime("%Y-%m-%dT%H:%M:%S"),
        end_datetime=END.strftime("%Y-%m-%dT%H:%M:%S"),
        output_directory=WORK,
        output_filename=name + ".nc",
    )
    attempts = [dict(base, minimum_depth=depth[0], maximum_depth=depth[1]), base] if depth else [base]
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

    ds = xr.open_dataset(path)
    if "depth" in ds.dims:
        ds = ds.isel(depth=0)
    ds = ds.sortby("latitude").sortby("longitude")
    ds = ds.reindex(time=TIMES, method="nearest", tolerance=np.timedelta64(90, "m"))
    return ds


def grid(da):
    return da.transpose("time", "latitude", "longitude").values.astype("float64")


def pack(a, scale):
    return [[None if not np.isfinite(x) else int(round(float(x) * scale)) for x in row] for row in a]


def build_region(key, cfg, tiles):
    bbox = cfg["bbox"]
    print("  waves...")
    w = fetch(cfg["wave"][0], ["VHM0", "VMDR", "VTPK", "VHM0_SW1"], bbox, cfg["wave"][1], key + "_wave")
    print("  temperature...")
    t = fetch(cfg["temp"][0], ["thetao"], bbox, cfg["temp"][1], key + "_temp")
    print("  currents...")
    c = fetch(cfg["cur"][0], ["uo", "vo"], bbox, cfg["cur"][1], key + "_cur")

    lats = w["latitude"].values.astype(float)
    lons = w["longitude"].values.astype(float)
    dlat = float(abs(lats[1] - lats[0]))
    dlon = float(abs(lons[1] - lons[0]))
    tol = 0.75 * max(dlat, dlon)

    def on_wave_grid(ds):
        return ds.reindex(latitude=w["latitude"], longitude=w["longitude"], method="nearest", tolerance=tol)

    t = on_wave_grid(t)
    c = on_wave_grid(c)
    u = grid(c["uo"])
    v = grid(c["vo"])
    data = {
        "hs": grid(w["VHM0"]),
        "sw": grid(w["VHM0_SW1"]),
        "tp": grid(w["VTPK"]),
        "dir": grid(w["VMDR"]),
        "sst": grid(t["thetao"]),
        "cs": np.sqrt(u * u + v * v),
        "cd": (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0,
    }
    sea = np.isfinite(data["hs"]).any(axis=0) | np.isfinite(data["sst"]).any(axis=0)

    count = 0
    for ty in range(int(np.floor(lats.min())), int(np.ceil(lats.max()))):
        for tx in range(int(np.floor(lons.min())), int(np.ceil(lons.max()))):
            iy = np.where((lats >= ty) & (lats < ty + 1))[0]
            ix = np.where((lons >= tx) & (lons < tx + 1))[0]
            if len(iy) == 0 or len(ix) == 0:
                continue
            sub = sea[np.ix_(iy, ix)]
            if not sub.any():
                continue
            yy, xx = np.nonzero(sub)
            sel_y = iy[yy]
            sel_x = ix[xx]
            layer = {
                "src": key,
                "lat0": round(float(lats[iy[0]]), 5),
                "lon0": round(float(lons[ix[0]]), 5),
                "dlat": round(dlat, 6),
                "dlon": round(dlon, 6),
                "nlat": int(len(iy)),
                "nlon": int(len(ix)),
                "idx": (yy * len(ix) + xx).astype(int).tolist(),
                "vars": {name: pack(arr[:, sel_y, sel_x], SCALES[name]) for name, arr in data.items()},
            }
            tiles.setdefault(f"{ty}_{tx}", []).append(layer)
            count += 1
    print(f"  {count} tiles from region {key}")


def doi_of(product_id):
    try:
        cat = cm.describe(product_id=product_id)
        return getattr(cat.products[0], "digital_object_identifier", None)
    except Exception:
        return None


def main():
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(os.path.join(OUT, "tiles"), exist_ok=True)
    tiles, done, errors, dois = {}, [], {}, {}

    for key, cfg in REGIONS.items():
        print("Region", key)
        try:
            build_region(key, cfg, tiles)
            done.append(key)
            for pid in cfg["products"]:
                dois[pid] = doi_of(pid)
        except Exception as e:
            errors[key] = f"{type(e).__name__}: {e}"
            print("  REGION FAILED:", errors[key])

    if not tiles:
        print("No tiles produced - stopping without publishing.")
        sys.exit(1)

    head = {"v": 1, "t0": START.strftime("%Y-%m-%dT%H:%M:%SZ"), "step_h": STEP_H, "steps": int(len(TIMES))}
    total = 0
    for name, layers in tiles.items():
        path = os.path.join(OUT, "tiles", name + ".json")
        with open(path, "w") as f:
            json.dump(dict(head, layers=layers), f, separators=(",", ":"))
        total += os.path.getsize(path)

    index = dict(
        head,
        generated=NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),
        regions=done,
        errors=errors,
        tiles=sorted(tiles),
        scales=SCALES,
        units=UNITS,
        attribution="Generated using E.U. Copernicus Marine Service Information",
        dois=dois,
    )
    with open(os.path.join(OUT, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    open(os.path.join(OUT, ".nojekyll"), "w").close()

    print(f"DONE: {len(tiles)} tiles, {total / 1e6:.1f} MB, regions ok: {done}, errors: {errors}")


if __name__ == "__main__":
    main()
