"""Trains the on-phone fish identification model. Run manually (GitHub Actions).

  python scripts/train_fish_id.py collect   -> downloads training photos from iNaturalist
  python scripts/train_fish_id.py train     -> trains, tests and exports static/models/

Photos: iNaturalist research-grade observations, only CC0 and CC-BY licences
(commercial use allowed). Every photo's author is listed in the attributions file.
The base network (MobileNetV3, ImageNet weights) is Apache-2.0.
"""
import csv
import concurrent.futures as cf
import datetime as dt
import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request

# Our catalogue: app species id -> scientific name
SPECIES = {
    "dicentrarchus_labrax": "Dicentrarchus labrax",
    "sparus_aurata": "Sparus aurata",
    "pomatomus_saltatrix": "Pomatomus saltatrix",
    "trachurus_trachurus": "Trachurus trachurus",
    "scomber_scombrus": "Scomber scombrus",
    "loligo_vulgaris": "Loligo vulgaris",
    "sarda_sarda": "Sarda sarda",
    "mugil_cephalus": "Mugil cephalus",
    "gadus_morhua": "Gadus morhua",
    "morone_saxatilis": "Morone saxatilis",
    "sciaenops_ocellatus": "Sciaenops ocellatus",
    "centropomus_undecimalis": "Centropomus undecimalis",
    "lateolabrax_japonicus": "Lateolabrax japonicus",
    "chrysophrys_auratus": "Chrysophrys auratus",
    "micropterus_salmoides": "Micropterus salmoides",
    "oncorhynchus_mykiss": "Oncorhynchus mykiss",
    "salmo_trutta": "Salmo trutta",
    "cyprinus_carpio": "Cyprinus carpio",
    "esox_lucius": "Esox lucius",
    "sander_vitreus": "Sander vitreus",
    "sander_lucioperca": "Sander lucioperca",
    "silurus_glanis": "Silurus glanis",
    "ictalurus_punctatus": "Ictalurus punctatus",
    "perca_fluviatilis": "Perca fluviatilis",
}

API = "https://api.inaturalist.org/v1"
UA = {"User-Agent": "OneDiveFishing/0.1 (github.com/arslana810-dot/fishing-data)"}
DATA = "fishdata"
OUT = os.path.join("static", "models")
MAX_PER_SPECIES = 600
MIN_PER_SPECIES = 80
IMG = 224
VERSION = "fish_id_v1"


def get_json(url):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print("    retry", attempt + 1, url[:90], e)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("request failed: " + url)


def taxon_id(name):
    """iNaturalist taxon id for an exact species name, or None (the species is then skipped)."""
    searches = [
        f"{API}/taxa?" + urllib.parse.urlencode({"q": name, "rank": "species", "per_page": 30}),
        f"{API}/taxa/autocomplete?" + urllib.parse.urlencode({"q": name, "per_page": 30}),
    ]
    for url in searches:
        try:
            res = get_json(url)
        except Exception:
            continue
        for t in res.get("results", []):
            if (t.get("name") or "").lower() == name.lower() and t.get("rank") == "species":
                return t["id"]
        time.sleep(1.1)
    return None


def photos_from_observations(results):
    """First photo of each observation, with licence and author (pure function, tested)."""
    out = []
    for obs in results:
        p = None
        for candidate in obs.get("photos") or []:
            lic = (candidate.get("license_code") or "").lower()
            if lic in ("cc0", "cc-by") and "square" in (candidate.get("url") or ""):
                p = candidate
                break
        if p is None:
            continue
        lic = (p.get("license_code") or "").lower()
        url = p.get("url") or ""
        out.append({
            "photo_id": p.get("id"),
            "url": url.replace("square", "medium"),
            "license": lic,
            "attribution": p.get("attribution") or "",
            "observation": obs.get("id"),
        })
    return out


def collect():
    os.makedirs(DATA, exist_ok=True)
    manifest = []
    not_found = []
    for sid, name in SPECIES.items():
        tid = taxon_id(name)
        time.sleep(1.1)
        if tid is None:
            not_found.append(sid)
            print(f"{sid}: SKIPPED (name not found on iNaturalist)")
            continue
        found = []
        page = 1
        while len(found) < MAX_PER_SPECIES and page <= 8:
            q = urllib.parse.urlencode({
                "taxon_id": tid, "quality_grade": "research", "photos": "true",
                "photo_license": "cc0,cc-by", "per_page": 200, "page": page,
            })
            try:
                res = get_json(f"{API}/observations?{q}")
            except Exception as e:
                print(f"    {sid}: page {page} failed ({e}), keeping what we have")
                break
            batch = photos_from_observations(res.get("results", []))
            found.extend(batch)
            time.sleep(1.1)
            if len(res.get("results", [])) < 200:
                break
            page += 1
        found = found[:MAX_PER_SPECIES]
        folder = os.path.join(DATA, sid)
        os.makedirs(folder, exist_ok=True)

        def download(item):
            path = os.path.join(folder, f"{item['photo_id']}.jpg")
            if os.path.exists(path):
                return item
            try:
                req = urllib.request.Request(item["url"], headers=UA)
                with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
                    f.write(r.read())
                return item
            except Exception:
                return None

        with cf.ThreadPoolExecutor(max_workers=8) as pool:
            ok = [x for x in pool.map(download, found) if x]
        for x in ok:
            x["species"] = sid
        manifest.extend(ok)
        print(f"{sid}: {len(ok)} photos")
    with open(os.path.join(DATA, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    print("COLLECT DONE:", len(manifest), "photos | not found:", not_found)


def split_files(files, val_share=0.15, seed=7):
    """Deterministic train/validation split per species (pure function, tested)."""
    files = sorted(files)
    random.Random(seed).shuffle(files)
    n_val = max(1, int(len(files) * val_share))
    return files[n_val:], files[:n_val]


def train():
    import shutil
    import numpy as np
    import tensorflow as tf

    with open(os.path.join(DATA, "manifest.json")) as f:
        manifest = json.load(f)
    by_species = {}
    for m in manifest:
        by_species.setdefault(m["species"], []).append(m)
    labels = sorted(s for s, items in by_species.items() if len(items) >= MIN_PER_SPECIES)
    skipped = sorted(s for s in SPECIES if s not in labels)
    print("species used:", len(labels), "| skipped (too few photos):", skipped)

    split_root = "fishsplit"
    shutil.rmtree(split_root, ignore_errors=True)
    for sid in labels:
        files = [os.path.join(DATA, sid, f"{m['photo_id']}.jpg") for m in by_species[sid]]
        files = [p for p in files if os.path.exists(p)]
        tr, va = split_files(files)
        for part, items in (("train", tr), ("val", va)):
            d = os.path.join(split_root, part, sid)
            os.makedirs(d, exist_ok=True)
            for p in items:
                shutil.copy(p, d)

    def ds(part, shuffle):
        return tf.keras.utils.image_dataset_from_directory(
            os.path.join(split_root, part), labels="inferred", label_mode="int", class_names=labels,
            image_size=(IMG, IMG), batch_size=32, shuffle=shuffle, seed=7)

    train_ds = ds("train", True).prefetch(tf.data.AUTOTUNE)
    val_ds = ds("val", False).prefetch(tf.data.AUTOTUNE)

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.15),
    ])
    base = tf.keras.applications.MobileNetV3Large(
        input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet",
        pooling="avg", include_preprocessing=True)
    base.trainable = False
    inputs = tf.keras.Input(shape=(IMG, IMG, 3))
    x = augment(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(labels), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    metrics = ["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3")]
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=metrics)
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
    model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[stop], verbose=2)

    # Fine-tune the top of the base network a little.
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=metrics)
    model.fit(train_ds, validation_data=val_ds, epochs=6, callbacks=[stop], verbose=2)

    # Honest test numbers on photos the model never trained on.
    y_true, y_pred = [], []
    for xb, yb in val_ds:
        p = model.predict(xb, verbose=0)
        y_true.extend(yb.numpy().tolist())
        y_pred.extend(p.tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    top1 = float((y_pred.argmax(1) == y_true).mean())
    top3 = float(np.mean([t in np.argsort(-p)[:3] for t, p in zip(y_true, y_pred)]))
    per_class = {}
    for i, sid in enumerate(labels):
        mask = y_true == i
        if mask.any():
            per_class[sid] = round(float((y_pred[mask].argmax(1) == i).mean()), 3)
    print(f"TEST top-1 accuracy: {top1:.3f} | top-3 accuracy: {top3:.3f}")
    for sid, acc in sorted(per_class.items(), key=lambda kv: kv[1]):
        print(f"  {sid}: {acc:.2f}")

    # Export a small model for phones (float16).
    os.makedirs(OUT, exist_ok=True)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite = converter.convert()
    except Exception as e:
        print("direct conversion failed, using saved model:", e)
        model.export("fish_saved")
        converter = tf.lite.TFLiteConverter.from_saved_model("fish_saved")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite = converter.convert()
    with open(os.path.join(OUT, f"{VERSION}.tflite"), "wb") as f:
        f.write(tflite)
    meta = {
        "version": VERSION,
        "labels": labels,
        "input_size": IMG,
        "input": "RGB, 0-255 float32 (scaling is inside the model)",
        "test_top1": round(top1, 3),
        "test_top3": round(top3, 3),
        "per_species_top1": per_class,
        "trained": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"),
        "photos_per_species": {s: len(by_species[s]) for s in labels},
        "skipped_species": skipped,
        "training_photos": "iNaturalist research-grade observations, CC0 / CC-BY only",
        "base_model": "MobileNetV3Large, ImageNet weights (Apache-2.0)",
    }
    with open(os.path.join(OUT, f"{VERSION}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(OUT, f"{VERSION}_photo_credits.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["species", "photo_id", "license", "attribution", "observation"])
        for m in manifest:
            if m["species"] in labels:
                w.writerow([m["species"], m["photo_id"], m["license"], m["attribution"], m["observation"]])
    size_mb = os.path.getsize(os.path.join(OUT, f"{VERSION}.tflite")) / 1e6
    print(f"TRAIN DONE: {len(labels)} species, model {size_mb:.1f} MB")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "collect":
        collect()
    elif cmd == "train":
        train()
    else:
        print(__doc__)
        sys.exit(2)
