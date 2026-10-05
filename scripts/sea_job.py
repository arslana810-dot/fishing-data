"""Daily sea-forecast pipeline: global coastal layer + high-resolution regional layers.

Commands:
  python scripts/sea_job.py list      -> JSON list of job names (for the matrix)
  python scripts/sea_job.py start     -> common forecast start time (UTC)
  python scripts/sea_job.py run NAME  -> builds tile layers into parts/
  python scripts/sea_job.py merge     -> merges parts/ into site/ (published)

Tile file format "FSH1" (little-endian):
  header: "FSH1", uint32 t0 (unix s), uint16 step_h, uint16 steps, uint8 nvars, uint8 nlayers
  layer:  uint8 len + src ascii, float32 lat0, lon0, dlat, dlon, uint16 nlat, nlon, uint32 ncells,
          uint16[ncells] cell index (row * nlon + col),
          then for each var in VARS: int16[steps * ncells] (step-major), MISSING = no data
  Apps must accept 7 or more variables (older tiles had no 'tide').
"""
import datetime as dt
import json
import math
import os
import struct
import sys

STEP_H = 3
DAYS = 7
STEPS = DAYS * 24 // STEP_H + 1
COAST_KM = 35   # keep sea cells within this distance of land
FILL_KM = 25    # fill model-coastline gaps from the nearest valid cell
MARGIN = 0.5    # extra degrees downloaded around each box
VARS = ["hs", "sw", "tp", "dir", "sst", "cs", "cd", "tide"]
SCALES = {"hs": 100, "sw": 100, "tp": 10, "dir": 1, "sst": 10, "cs": 100, "cd": 1, "tide": 100}
UNITS = {
    "hs": "wave height, cm",
    "sw": "primary swell height, cm",
    "tp": "peak wave period, 0.1 s",
    "dir": "mean wave direction, deg (coming from)",
    "sst": "sea surface temperature, 0.1 C",
    "cs": "surface current speed, cm/s",
    "cd": "surface current direction, deg (flowing towards)",
    "tide": "tide height relative to mean sea level, cm",
}
MISSING = -32768
PARTS = "parts"
SITE = "site"
WORK = "work"

GLO_WAVE = ("GLOBAL_ANALYSISFORECAST_WAV_001_027", "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i", None)
GLO_TEMP = ("GLOBAL_ANALYSISFORECAST_PHY_001_024", "cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i", (0.0, 1.0))
GLO_CUR = ("GLOBAL_ANALYSISFORECAST_PHY_001_024", "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i", (0.0, 1.0))
GLO_TIDE = ("GLOBAL_ANALYSISFORECAST_PHY_001_024", "cmems_mod_glo_phy_anfc_merged-sl_PT1H-i", None)

MED_WAVE = ("MEDSEA_ANALYSISFORECAST_WAV_006_017", "cmems_mod_med_wav_anfc_4.2km_PT1H-i", None)
MED_TEMP = ("MEDSEA_ANALYSISFORECAST_PHY_006_013", "cmems_mod_med_phy-tem_anfc_4.2km-2D_PT1H-m", None)
MED_CUR = ("MEDSEA_ANALYSISFORECAST_PHY_006_013", "cmems_mod_med_phy-cur_anfc_4.2km-2D_PT1H-m", None)

BLK_WAVE = ("BLKSEA_ANALYSISFORECAST_WAV_007_003", "cmems_mod_blk_wav_anfc_2.5km_PT1H-i", None)
BLK_TEMP = ("BLKSEA_ANALYSISFORECAST_PHY_007_001", "cmems_mod_blk_phy-temp_anfc_2.5km_PT1H-m", (0.0, 5.0))
BLK_CUR = ("BLKSEA_ANALYSISFORECAST_PHY_007_001", "cmems_mod_blk_phy-cur_anfc_2.5km_PT1H-m", (0.0, 5.0))


def tag(v):
    return f"m{-v}" if v < 0 else str(v)


def jobs():
    out = {}
    lat_edges = [-60, -30, 0, 30, 60, 80]
    for i in range(len(lat_edges) - 1):
        for lon in range(-180, 180, 30):
            out[f"g_{tag(lon)}_{tag(lat_edges[i])}"] = {
                "src": "glo",
                "bbox": (float(lon), float(lon + 30), float(lat_edges[i]), float(lat_edges[i + 1])),
                "wave": GLO_WAVE, "temp": GLO_TEMP, "cur": GLO_CUR, "tide": GLO_TIDE,
            }
    for name, lon0, lon1 in (("r_med_w", -6.0, 12.0), ("r_med_c", 12.0, 24.0), ("r_med_e", 24.0, 36.3)):
        out[name] = {"src": "med", "bbox": (lon0, lon1, 30.0, 46.0),
                     "wave": MED_WAVE, "temp": MED_TEMP, "cur": MED_CUR}
    out["r_blk"] = {"src": "blk", "bbox": (27.0, 42.0, 40.0, 47.0),
                    "wave": BLK_WAVE, "temp": BLK_TEMP, "cur": BLK_CUR}
    return out


def start_time():
    iso = os.environ.get("START_ISO", "").strip()
    if iso:
        return dt.datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ")
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    return now - dt.timedelta(hours=now.hour % STEP_H)


def layer_bytes(np, src, lat0, lon0, dlat, dlon, nlat, nlon, idx, vals):
    chunks = [
        struct.pack("<B", len(src)),
        src.encode("ascii"),
        struct.pack("<ffffHHI", lat0, lon0, dlat, dlon, nlat, nlon, len(idx)),
        np.asarray(idx, dtype="<u2").tobytes(),
    ]
    for v in VARS:
        arr = vals[v].astype("float64")
        q = np.where(np.isfinite(arr), np.round(arr * SCALES[v]), MISSING)
        chunks.append(np.clip(q, -32768, 32767).astype("<i2").tobytes())
    return b"".join(chunks)


def run(name):
    import numpy as np
    import xarray as xr
    import copernicusmarine as cm
    from scipy.ndimage import binary_dilation, distance_transform_edt

    cfg = jobs()[name]
    start = start_time()
    end = start + dt.timedelta(days=DAYS)
    times = np.array(
        [np.datetime64(start + dt.timedelta(hours=STEP_H * i)) for i in range(STEPS)]
    ).astype("datetime64[ns]")
    lon0, lon1, lat0, lat1 = cfg["bbox"]
    box = (max(-180.0, lon0 - MARGIN), min(180.0, lon1 + MARGIN),
           max(-90.0, lat0 - MARGIN), min(90.0, lat1 + MARGIN))

    def fetch(spec, variables, label):
        _, dataset_id, depth = spec
        fname = f"{name}_{label}.nc"
        path = os.path.join(WORK, fname)
        base = dict(
            dataset_id=dataset_id,
            variables=variables,
            minimum_longitude=box[0],
            maximum_longitude=box[1],
            minimum_latitude=box[2],
            maximum_latitude=box[3],
            start_datetime=(start - dt.timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%S"),
            end_datetime=(end + dt.timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%S"),
            output_directory=WORK,
            output_filename=fname,
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
        step_h = None
        if ds.sizes.get("time", 0) > 1:
            step_h = float((ds["time"].values[1] - ds["time"].values[0]) / np.timedelta64(1, "h"))
        if step_h is not None and step_h > STEP_H + 0.01:
            ds = ds.interp(time=times)
        else:
            ds = ds.reindex(time=times, method="nearest", tolerance=np.timedelta64(90, "m"))
        return ds

    print(f"Job {name}: bbox {cfg['bbox']}")
    w = fetch(cfg["wave"], ["VHM0", "VMDR", "VTPK", "VHM0_SW1"], "wave")
    t = fetch(cfg["temp"], ["thetao"], "temp")
    c = fetch(cfg["cur"], ["uo", "vo"], "cur")

    lats = w["latitude"].values.astype(float)
    lons = w["longitude"].values.astype(float)
    dlat = float(abs(lats[1] - lats[0]))
    dlon = float(abs(lons[1] - lons[0]))
    tol = 0.75 * max(dlat, dlon)
    t = t.reindex(latitude=w["latitude"], longitude=w["longitude"], method="nearest", tolerance=tol)
    c = c.reindex(latitude=w["latitude"], longitude=w["longitude"], method="nearest", tolerance=tol)

    def grid(da):
        return da.transpose("time", "latitude", "longitude").values.astype("float32")

    raw = {
        "hs": grid(w["VHM0"]),
        "sw": grid(w["VHM0_SW1"]),
        "tp": grid(w["VTPK"]),
        "dir": grid(w["VMDR"]),
        "sst": grid(t["thetao"]),
        "u": grid(c["uo"]),
        "v": grid(c["vo"]),
    }

    # Tide is optional: a problem with it must never stop the rest of the job.
    raw["tide"] = np.full_like(raw["hs"], np.nan)
    if cfg.get("tide"):
        try:
            td = fetch(cfg["tide"], ["ocean_tide"], "tide")
            td = td.reindex(latitude=w["latitude"], longitude=w["longitude"], method="nearest", tolerance=tol)
            raw["tide"] = grid(td["ocean_tide"])
        except Exception as e:
            print("  tide skipped:", type(e).__name__, e)

    sea = np.isfinite(raw["hs"]).any(axis=0) | np.isfinite(raw["sst"]).any(axis=0)
    if not sea.any():
        print("  no sea cells in this box")
        return 0

    cell_km = dlat * 111.0
    coast_cells = max(1, math.ceil(COAST_KM / cell_km))
    fill_cells = max(1, math.ceil(FILL_KM / cell_km))

    def fill(arr):
        valid = np.isfinite(arr).any(axis=0)
        if valid.all() or not valid.any():
            return arr
        dist, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
        target = (~valid) & sea & (dist <= fill_cells)
        out = arr.copy()
        out[:, target] = arr[:, iy[target], ix[target]]
        return out

    filled = {k: fill(v) for k, v in raw.items()}
    u = filled["u"]
    v = filled["v"]
    data = {
        "hs": filled["hs"],
        "sw": filled["sw"],
        "tp": filled["tp"],
        "dir": filled["dir"],
        "sst": filled["sst"],
        "cs": np.sqrt(u * u + v * v),
        "cd": (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0,
        "tide": filled["tide"],
    }
    coastal = sea & binary_dilation(~sea, iterations=coast_cells)

    count = 0
    for ty in range(int(math.floor(lat0)), int(math.ceil(lat1))):
        for tx in range(int(math.floor(lon0)), int(math.ceil(lon1))):
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
            blob = layer_bytes(
                np, cfg["src"], float(lats[iy[0]]), float(lons[ix[0]]), dlat, dlon,
                len(iy), len(ix), (yy * len(ix) + xx).astype(int),
                {k: arr[:, sel_y, sel_x] for k, arr in data.items()},
            )
            with open(os.path.join(PARTS, f"{ty}_{tx}__{name}.lyr"), "wb") as f:
                f.write(blob)
            count += 1
    print(f"  {count} tiles written")
    return count


def doi_map(cfg):
    import copernicusmarine as cm
    out = {}
    for key in ("wave", "temp", "cur", "tide"):
        spec = cfg.get(key)
        if not spec:
            continue
        pid = spec[0]
        if pid in out:
            continue
        try:
            out[pid] = getattr(cm.describe(product_id=pid).products[0], "digital_object_identifier", None)
        except Exception:
            out[pid] = None
    return out


def safe_run(name):
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(PARTS, exist_ok=True)
    meta = {"job": name}
    try:
        meta["tiles"] = run(name)
        meta["dois"] = doi_map(jobs()[name]) if meta["tiles"] else {}
    except Exception as e:
        meta["error"] = f"{type(e).__name__}: {e}"
        print("JOB FAILED:", meta["error"])
    with open(os.path.join(PARTS, f"meta__{name}.json"), "w") as f:
        json.dump(meta, f)


def merge():
    start = start_time()
    layers, dois, jobs_ok, errors = {}, {}, [], {}
    if os.path.isdir(PARTS):
        for fname in sorted(os.listdir(PARTS)):
            path = os.path.join(PARTS, fname)
            if fname.startswith("meta__") and fname.endswith(".json"):
                with open(path) as f:
                    m = json.load(f)
                if m.get("error"):
                    errors[m["job"]] = m["error"]
                else:
                    jobs_ok.append(m["job"])
                    dois.update({k: v for k, v in (m.get("dois") or {}).items() if v})
            elif fname.endswith(".lyr"):
                tile, job = fname[:-4].split("__", 1)
                with open(path, "rb") as f:
                    layers.setdefault(tile, []).append((0 if job.startswith("r_") else 1, job, f.read()))

    if not layers:
        print("No tiles - nothing to publish.")
        sys.exit(1)

    os.makedirs(os.path.join(SITE, "tiles"), exist_ok=True)
    t0 = int(start.replace(tzinfo=dt.timezone.utc).timestamp())
    total = 0
    for tile, items in layers.items():
        items.sort()
        head = b"FSH1" + struct.pack("<IHHBB", t0, STEP_H, STEPS, len(VARS), len(items))
        blob = head + b"".join(b for _, _, b in items)
        with open(os.path.join(SITE, "tiles", tile + ".bin"), "wb") as f:
            f.write(blob)
        total += len(blob)

    index = {
        "v": 3,
        "format": "FSH1",
        "t0": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "step_h": STEP_H,
        "steps": STEPS,
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "vars": VARS,
        "scales": SCALES,
        "units": UNITS,
        "missing": MISSING,
        "jobs_ok": sorted(jobs_ok),
        "errors": errors,
        "tiles": sorted(layers),
        "attribution": "Generated using E.U. Copernicus Marine Service Information",
        "dois": dois,
    }
    with open(os.path.join(SITE, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    open(os.path.join(SITE, ".nojekyll"), "w").close()
    print(f"DONE: {len(layers)} tiles, {total / 1e6:.1f} MB, jobs ok {len(jobs_ok)}, errors {len(errors)}")
    for job, err in sorted(errors.items()):
        print(f"  ERROR {job}: {err}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "list":
        print(json.dumps(sorted(jobs())))
    elif cmd == "start":
        print(start_time().strftime("%Y-%m-%dT%H:%M:%SZ"))
    elif cmd == "run" and len(sys.argv) > 2:
        safe_run(sys.argv[2])
    elif cmd == "merge":
        merge()
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
