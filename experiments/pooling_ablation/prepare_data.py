"""Rebuild full processed shards with the original preprocessing version-2 recipe."""
from __future__ import annotations
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import ssl
import time
from urllib.parse import quote, urlencode
from urllib.request import urlopen

import numpy as np

HERE = Path(__file__).resolve().parent
SPLITS = ("train", "valid", "test")
SCALING = {"method": "max_bbox_dimension", "target": 1.0, "degenerate": "identity"}
EXPECTED_SCALE = 0.12035210854952101
TLS_CONTEXT = ssl.create_default_context()
for system_bundle in ("/etc/pki/tls/certs/ca-bundle.crt", "/etc/ssl/certs/ca-certificates.crt"):
    if Path(system_bundle).is_file():
        TLS_CONTEXT.load_verify_locations(cafile=system_bundle)
        break


def open_url(url, timeout):
    return urlopen(url, timeout=timeout, context=TLS_CONTEXT)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def pack_sequences(sequences):
    if len(sequences) == 0:
        raise ValueError("Empty split")
    if any(s.ndim != 2 or s.shape[1] != 3 or len(s) == 0 for s in sequences):
        raise ValueError("Expected nonempty [T, 3] stroke sequences")
    lengths = np.fromiter((len(s) for s in sequences), dtype=np.int64, count=len(sequences))
    points = np.concatenate(sequences).astype(np.float32)
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite coordinates")
    if not np.isin(points[:, 2], [0, 1]).all():
        raise ValueError("Pen flags must be 0 or 1")
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    return points, offsets, lengths


def uniform_scale(points, offsets):
    positions = np.cumsum(points[:, :2], axis=0, dtype=np.float64)
    lower = np.minimum.reduceat(positions, offsets[:-1], axis=0)
    upper = np.maximum.reduceat(positions, offsets[:-1], axis=0)
    dimensions = np.max(upper - lower, axis=1)
    divisors = np.where(dimensions > 0, dimensions, 1.0).astype(np.float32)
    scaled = points.copy()
    scaled[:, :2] /= np.repeat(divisors, np.diff(offsets))[:, None]
    return scaled


def inventory():
    entries, token = [], None
    while True:
        query = {"prefix": "sketchrnn/", "maxResults": 1000}
        if token:
            query["pageToken"] = token
        with open_url("https://storage.googleapis.com/storage/v1/b/quickdraw_dataset/o?" + urlencode(query), timeout=90) as response:
            page = json.load(response)
        entries.extend({"name": item["name"], "size": int(item["size"]), "md5Hash": item["md5Hash"]}
                       for item in page.get("items", []) if item["name"].endswith(".full.npz"))
        token = page.get("nextPageToken")
        if not token:
            break
    if len(entries) != 345:
        raise ValueError(f"Expected 345 official full archives, found {len(entries)}")
    return entries


def file_md5(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode()


def download(item, raw):
    target = raw / Path(item["name"]).name
    if target.is_file() and target.stat().st_size == item["size"] and file_md5(target) == item["md5Hash"]:
        return
    temporary = target.with_suffix(".npz.part")
    for attempt in range(3):
        try:
            with open_url("https://storage.googleapis.com/quickdraw_dataset/" + quote(item["name"]), timeout=180) as response, temporary.open("wb") as stream:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
            if temporary.stat().st_size != item["size"] or file_md5(temporary) != item["md5Hash"]:
                raise ValueError(f"Official archive checksum mismatch: {target.name}")
            temporary.replace(target)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def category_statistics(lengths, points):
    return {"samples": len(lengths), "points": len(points),
            "length_percentiles": dict(zip(["min", "median", "p90", "p95", "p99", "max"],
                np.percentile(lengths, [0, 50, 90, 95, 99, 100]).tolist()))}


def process_category(category, index, raw, output, scale, fingerprint):
    folder = output / category
    marker = folder / "preparation.json"
    if marker.is_file():
        saved = json.loads(marker.read_text())
        if saved.get("source_md5") == fingerprint and saved.get("scale") == scale:
            valid = all((folder / f"{s}.npz").is_file() and (folder / f"{s}.npz").stat().st_size == saved["file_bytes"][s] for s in SPLITS)
            if valid:
                return category, saved["statistics"]
    folder.mkdir(parents=True, exist_ok=True)
    statistics, file_bytes = {}, {}
    # Only archives from the official bucket with verified checksums are unpickled.
    with np.load(raw / (category + ".full.npz"), allow_pickle=True, encoding="latin1") as archive:
        for split in SPLITS:
            points, offsets, lengths = pack_sequences(archive[split])
            points = uniform_scale(points, offsets)
            points[:, :2] /= scale
            labels = np.full(len(lengths), index, dtype=np.int64)
            ids = np.asarray([f"{category}/full/{split}/{i}" for i in range(len(lengths))])
            target = folder / f"{split}.npz"
            temporary = target.with_suffix(".npz.part")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, points=points, offsets=offsets, labels=labels, ids=ids)
            temporary.replace(target)
            # Read every saved array before recording completion.
            with np.load(target, allow_pickle=False) as check:
                for name, expected in (("points", points), ("offsets", offsets), ("labels", labels), ("ids", ids)):
                    if not np.array_equal(check[name], expected):
                        raise ValueError(f"Saved shard differs: {target}, {name}")
            statistics[split] = category_statistics(lengths, points)
            file_bytes[split] = target.stat().st_size
    write_json(marker, {"source_md5": fingerprint, "scale": scale,
                        "statistics": statistics, "file_bytes": file_bytes})
    return category, statistics


def prepare(raw, output, workers=4):
    started = time.monotonic()
    raw, output = Path(raw).resolve(), Path(output).resolve()
    raw.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("Completed manifest exists; use a fresh output directory")
    entries = inventory()
    write_json(raw / "download_manifest.json", entries)
    by_category = {Path(item["name"]).name.removesuffix(".full.npz"): item for item in entries}
    reference = json.loads((HERE / "reference/baseline_config.json").read_text(encoding="utf-8-sig"))
    categories = reference["data"]["categories"]
    if set(categories) != set(by_category):
        raise ValueError("Official categories differ from baseline")
    print(f"DOWNLOAD: {len(entries)} official full archives; {sum(e['size'] for e in entries)/1024**3:.2f} GiB", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download, item, raw) for item in entries]
        for done, future in enumerate(as_completed(futures), 1):
            future.result()
            if done % 10 == 0 or done == 345:
                print(f"DOWNLOAD VERIFIED {done}/345", flush=True)
    print("FIT: all training coordinates after per-sketch geometric scaling", flush=True)
    count, total, square = 0, 0.0, 0.0
    sources = []
    for done, category in enumerate(categories, 1):
        path = raw / (category + ".full.npz")
        with np.load(path, allow_pickle=True, encoding="latin1") as archive:
            points, offsets, _ = pack_sequences(archive["train"])
            xy = uniform_scale(points, offsets)[:, :2]
            count += xy.size
            total += float(np.sum(xy, dtype=np.float64))
            square += float(np.einsum("ij,ij->", xy, xy, dtype=np.float64))
        sources.append({"category": category, "file": path.name, "bytes": path.stat().st_size})
        if done % 20 == 0 or done == 345:
            print(f"FIT {done}/345", flush=True)
    scale = float(np.sqrt(max(square / count - (total / count) ** 2, 0)))
    if not np.isclose(scale, EXPECTED_SCALE, rtol=0, atol=1e-12):
        raise ValueError(f"Training-fitted scale {scale} differs from preprocessing reference {EXPECTED_SCALE}")
    print(f"FIT VERIFIED scale={scale:.17g}", flush=True)
    statistics = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_category, category, index, raw, output, scale, by_category[category]["md5Hash"])
                   for index, category in enumerate(categories)]
        for done, future in enumerate(as_completed(futures), 1):
            category, stats = future.result()
            statistics[category] = stats
            if done % 10 == 0 or done == 345:
                print(f"PREPARED AND READ-BACK VERIFIED {done}/345", flush=True)
    totals = {split: sum(statistics[c][split]["samples"] for c in categories) for split in SPLITS}
    if totals != {"train": 43988874, "valid": 862500, "test": 862500}:
        raise ValueError(f"Unexpected dataset counts: {totals}")
    manifest = {"archive_version": "full", "class_to_index": {c: i for i, c in enumerate(categories)},
                "representation": "stroke-3", "preprocessing_version": 2, "per_sketch_scaling": SCALING,
                "normalization": {"fit_split": "train", "scale": scale}, "length_cap": None,
                "subsampling": False, "sources": sources, "split_counts": totals,
                "statistics": {c: statistics[c] for c in categories}}
    write_json(output / "manifest.json", manifest)
    write_json(output / "preparation_audit.json", {"complete": True, "official_source_checksums_verified": True,
        "all_shards_readback_verified": True, "normalization_recomputed_from_all_training_points": True,
        "manifest_sha256": hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
        "preprocessor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_archives": 345, "split_counts": totals, "scale": scale,
        "elapsed_seconds": time.monotonic() - started})
    print("PREPARATION COMPLETE", totals, str(output), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prepare(args.raw_dir, args.output_dir, max(1, args.workers))
