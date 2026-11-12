"""
assurance/datasets.py

Dataset ingestion. Every loader returns a `Dataset`: a list of
`ImageRecord`s (one image file each) plus the class names discovered in the
data. Nothing about the classes, contributors or batches is hardcoded — it
all comes from the folder layout, the label CSV or an optional metadata file.

Supported sources:
  1. Folder structure   root/<class_name>/<image>            (+ optional metadata CSV/JSONL)
  2. CSV + image files  labels.csv with a path column and a label column
  3. torchvision        CIFAR10, MNIST, … — materialised to PNG files once

COCO / YOLO loaders can be added later by producing the same `Dataset`.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import io
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import config

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

# Column names recognised automatically in CSV / JSONL files
PATH_COLUMNS        = ("filename", "file", "path", "filepath", "image", "image_path", "img")
LABEL_COLUMNS       = ("label", "class", "class_name", "category", "target")
CONTRIBUTOR_COLUMNS = ("contributor_id", "contributor", "source", "annotator", "uploader")
BATCH_COLUMNS       = ("batch_id", "batch", "upload_batch")

UNKNOWN = "unknown"


# ────────────────────────────────────────────────────────────────────────────
# Data structures
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class ImageRecord:
    filename: str                      # absolute path to the image file
    label: str
    class_idx: int
    contributor_id: str = UNKNOWN
    batch_id: str       = UNKNOWN
    extra: Dict         = field(default_factory=dict)   # any other metadata columns


@dataclass
class Dataset:
    name: str
    source: str                        # "folder" | "csv" | "torchvision"
    root: str
    class_names: List[str]
    records: List[ImageRecord]
    metadata_path: Optional[str] = None
    notes: List[str] = field(default_factory=list)      # warnings raised during ingestion

    def __len__(self) -> int:
        return len(self.records)

    @property
    def has_contributors(self) -> bool:
        return any(r.contributor_id != UNKNOWN for r in self.records)

    def class_counts(self) -> Dict[str, int]:
        c = Counter(r.label for r in self.records)
        return {name: c.get(name, 0) for name in self.class_names}

    def contributor_counts(self) -> Dict[str, int]:
        return dict(Counter(r.contributor_id for r in self.records))

    def fingerprint(self) -> str:
        """Cheap identity of the record list (paths + labels), for caching."""
        h = hashlib.sha256()
        for r in self.records:
            h.update(f"{r.filename}\0{r.label}\n".encode())
        return h.hexdigest()

    def summary(self) -> str:
        return f"{self.name} · {len(self)} images · {len(self.class_names)} classes"


# ────────────────────────────────────────────────────────────────────────────
# Metadata tables (CSV / JSONL)
# ────────────────────────────────────────────────────────────────────────────

def read_table(source) -> List[Dict]:
    """
    Read a CSV or JSONL table into a list of dicts.
    `source` may be a path or a file-like object / bytes (Streamlit upload).
    """
    name = getattr(source, "name", str(source))
    if isinstance(source, (str, Path)):
        text = Path(source).read_text(encoding="utf-8-sig")
    else:
        raw = source.getvalue() if hasattr(source, "getvalue") else source.read()
        text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw

    if name.lower().endswith((".jsonl", ".ndjson")) or text.lstrip().startswith("{"):
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    return list(csv.DictReader(io.StringIO(text)))


def table_columns(rows: Sequence[Dict]) -> List[str]:
    cols: List[str] = []
    for r in rows[:200]:
        for k in r:
            if k not in cols:
                cols.append(k)
    return cols


def guess_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _resolve(path_str: str, base: Path) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (base / p)


class _MetaIndex:
    """
    Looks up a metadata row for an image. Metadata files written on another
    machine often carry absolute paths that no longer exist, so rows are
    matched by absolute path, then path relative to the dataset root, then
    (if unambiguous) bare file name.
    """

    def __init__(self, rows: Sequence[Dict], key_col: str, root: Path):
        self.by_abs: Dict[str, Dict] = {}
        self.by_rel: Dict[str, Dict] = {}
        by_name: Dict[str, List[Dict]] = defaultdict(list)
        for row in rows:
            raw = str(row.get(key_col, "") or "").strip()
            if not raw:
                continue
            p = _resolve(raw, root)
            self.by_abs[str(p.resolve())] = row
            try:
                self.by_rel[p.resolve().relative_to(root.resolve()).as_posix()] = row
            except ValueError:
                # Absolute path from elsewhere: keep its last parts as a relative key
                parts = Path(raw).parts
                for n in (2, 1):
                    if len(parts) >= n:
                        self.by_rel.setdefault(Path(*parts[-n:]).as_posix(), row)
            by_name[Path(raw).name].append(row)
        self.by_name = {k: v[0] for k, v in by_name.items() if len(v) == 1}
        self.matched = 0

    def get(self, img_path: Path, root: Path) -> Dict:
        abs_key = str(img_path.resolve())
        row = self.by_abs.get(abs_key)
        if row is None:
            try:
                row = self.by_rel.get(img_path.resolve().relative_to(root.resolve()).as_posix())
            except ValueError:
                row = None
        if row is None:
            row = self.by_name.get(img_path.name)
        if row is not None:
            self.matched += 1
        return row or {}


def _meta_fields(row: Dict, contributor_col: Optional[str], batch_col: Optional[str],
                 skip: Iterable[Optional[str]]) -> tuple:
    contributor = str(row.get(contributor_col, "") or "").strip() if contributor_col else ""
    batch       = str(row.get(batch_col, "") or "").strip() if batch_col else ""
    skip_set = {s for s in skip if s} | {contributor_col, batch_col}
    extra = {k: v for k, v in row.items() if k not in skip_set}
    return contributor or UNKNOWN, batch or UNKNOWN, extra


def _find_sidecar_metadata(root: Path) -> Optional[Path]:
    for name in ("metadata.csv", "metadata.jsonl", "labels.csv"):
        p = root / name
        if p.exists():
            return p
    return None


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS


# ────────────────────────────────────────────────────────────────────────────
# 1. Folder structure
# ────────────────────────────────────────────────────────────────────────────

def load_folder(
    root: str,
    metadata=None,
    metadata_key_col: Optional[str] = None,
    contributor_col: Optional[str] = None,
    batch_col: Optional[str] = None,
    name: Optional[str] = None,
    auto_metadata: bool = True,
) -> Dataset:
    """
    Load `root/<class_name>/**/<image>`. Class names are the sub-folder names
    (sorted). Images directly under root are ignored.

    `metadata` is an optional CSV/JSONL (path or upload) with one row per image,
    used for contributor / batch / any extra columns. If not given and
    `auto_metadata` is set, root/metadata.csv or root/metadata.jsonl is used.
    """
    root_p = Path(root).expanduser()
    if not root_p.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root_p}")

    class_dirs = sorted(d for d in root_p.iterdir() if d.is_dir() and not d.name.startswith("."))
    class_files = {d.name: sorted(p for p in d.rglob("*") if _is_image(p)) for d in class_dirs}
    class_files = {k: v for k, v in class_files.items() if v}
    if not class_files:
        raise ValueError(
            f"No class sub-folders with images found in {root_p}. Expected "
            f"{root_p}/<class_name>/<image>.png")
    class_names = sorted(class_files)

    notes: List[str] = []
    meta_path_str = None
    meta_index = None
    if metadata is None and auto_metadata:
        auto = _find_sidecar_metadata(root_p)
        if auto is not None:
            metadata = auto
    if metadata is not None:
        rows = read_table(metadata)
        meta_path_str = str(getattr(metadata, "name", metadata))
        cols = table_columns(rows)
        key_col = metadata_key_col or guess_column(cols, PATH_COLUMNS)
        if key_col is None:
            raise ValueError(f"Metadata has no file-name column (looked for {PATH_COLUMNS}); "
                             f"columns are {cols}")
        contributor_col = contributor_col or guess_column(cols, CONTRIBUTOR_COLUMNS)
        batch_col       = batch_col or guess_column(cols, BATCH_COLUMNS)
        meta_index = _MetaIndex(rows, key_col, root_p)
        skip = (key_col,)
    else:
        skip = ()

    records: List[ImageRecord] = []
    for class_idx, cname in enumerate(class_names):
        for p in class_files[cname]:
            row = meta_index.get(p, root_p) if meta_index else {}
            contributor, batch, extra = _meta_fields(row, contributor_col, batch_col, skip)
            # The folder is the label of record; a conflicting metadata label is kept for review.
            extra.pop("class_idx", None)
            meta_label = extra.pop("label", None)
            if meta_label is not None and str(meta_label) != cname:
                extra["metadata_label"] = meta_label
            records.append(ImageRecord(
                filename=str(p.resolve()), label=cname, class_idx=class_idx,
                contributor_id=contributor, batch_id=batch, extra=extra,
            ))

    if meta_index is not None:
        unmatched = len(records) - meta_index.matched
        if unmatched:
            notes.append(f"{unmatched}/{len(records)} images had no metadata row "
                         f"(contributor/batch set to '{UNKNOWN}').")

    return Dataset(
        name=name or root_p.name, source="folder", root=str(root_p.resolve()),
        class_names=class_names, records=records, metadata_path=meta_path_str, notes=notes,
    )


# ────────────────────────────────────────────────────────────────────────────
# 2. CSV + image files
# ────────────────────────────────────────────────────────────────────────────

def load_csv(
    csv_source,
    image_root: Optional[str] = None,
    path_col: Optional[str] = None,
    label_col: Optional[str] = None,
    contributor_col: Optional[str] = None,
    batch_col: Optional[str] = None,
    name: Optional[str] = None,
) -> Dataset:
    """
    Load images listed in a CSV/JSONL table. Relative paths are resolved
    against `image_root` (default: the CSV's folder). Class names are the
    sorted unique values of the label column.
    """
    rows = read_table(csv_source)
    if not rows:
        raise ValueError("Label file is empty.")
    cols = table_columns(rows)
    path_col  = path_col or guess_column(cols, PATH_COLUMNS)
    label_col = label_col or guess_column(cols, LABEL_COLUMNS)
    if path_col is None or label_col is None:
        raise ValueError(f"Could not find path/label columns in {cols}. Choose them explicitly.")
    contributor_col = contributor_col or guess_column(cols, CONTRIBUTOR_COLUMNS)
    batch_col       = batch_col or guess_column(cols, BATCH_COLUMNS)

    if image_root:
        base = Path(image_root).expanduser()
    elif isinstance(csv_source, (str, Path)):
        base = Path(csv_source).expanduser().parent
    else:
        raise ValueError("An image root folder is required when the label file is uploaded.")

    labelled = [(row, str(row.get(label_col, "")).strip()) for row in rows]
    class_names = sorted({lbl for _, lbl in labelled if lbl})
    index = {c: i for i, c in enumerate(class_names)}

    records: List[ImageRecord] = []
    missing, unlabelled = 0, 0
    for row, lbl in labelled:
        if not lbl:
            unlabelled += 1
            continue
        p = _resolve(str(row.get(path_col, "")).strip(), base)
        if not p.is_file():
            missing += 1
            continue
        contributor, batch, extra = _meta_fields(row, contributor_col, batch_col, (path_col, label_col))
        records.append(ImageRecord(
            filename=str(p.resolve()), label=lbl, class_idx=index[lbl],
            contributor_id=contributor, batch_id=batch, extra=extra,
        ))

    notes = []
    if missing:
        notes.append(f"{missing} rows point to files that do not exist (skipped).")
    if unlabelled:
        notes.append(f"{unlabelled} rows have an empty label (skipped).")
    if not records:
        raise ValueError("No readable images found. " + " ".join(notes))

    src_name = getattr(csv_source, "name", str(csv_source))
    return Dataset(
        name=name or Path(src_name).stem, source="csv", root=str(base.resolve()),
        class_names=class_names, records=records, metadata_path=src_name, notes=notes,
    )


# ────────────────────────────────────────────────────────────────────────────
# 3. torchvision datasets
# ────────────────────────────────────────────────────────────────────────────

# Image-classification datasets whose constructor takes `train=` or `split=`.
TORCHVISION_DATASETS = {
    "CIFAR10":      "train",
    "CIFAR100":     "train",
    "MNIST":        "train",
    "FashionMNIST": "train",
    "KMNIST":       "train",
    "SVHN":         "split",
    "STL10":        "split",
    "GTSRB":        "split",
}


def _tv_targets(ds) -> List[int]:
    for attr in ("targets", "labels"):
        t = getattr(ds, attr, None)
        if t is not None:
            return [int(x) for x in (t.tolist() if hasattr(t, "tolist") else t)]
    samples = getattr(ds, "_samples", None) or getattr(ds, "samples", None)
    if samples:
        return [int(s[1]) for s in samples]
    return [int(ds[i][1]) for i in range(len(ds))]


def _safe_dirname(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_. " else "_" for ch in s).strip() or "class"


def _stratified_subset(targets: List[int], limit: Optional[int], seed: int) -> List[int]:
    idx = list(range(len(targets)))
    if not limit or limit >= len(idx):
        return idx
    by_cls: Dict[int, List[int]] = defaultdict(list)
    for i, t in enumerate(targets):
        by_cls[t].append(i)
    rng = random.Random(seed)
    for v in by_cls.values():
        rng.shuffle(v)
    chosen: List[int] = []
    queues = [by_cls[c] for c in sorted(by_cls)]
    while len(chosen) < limit:
        progressed = False
        for q in queues:
            if q and len(chosen) < limit:
                chosen.append(q.pop())
                progressed = True
        if not progressed:
            break
    return sorted(chosen)


def load_torchvision(
    dataset_name: str,
    split: str = "train",
    limit: Optional[int] = 2000,
    download: bool = True,
    seed: int = config.RANDOM_SEED,
    raw_root: str = config.TORCHVISION_ROOT,
    cache_root: str = config.DATASET_CACHE_DIR,
    progress: Optional[Callable[[str, float], None]] = None,
) -> Dataset:
    """
    Load a torchvision classification dataset and write a stratified subset
    (`limit` images; None = all) to PNG files under outputs/datasets/, so every
    scanner can treat it like any other on-disk dataset. Re-loading with the
    same arguments reuses the files.
    """
    import torchvision.datasets as tvd

    if dataset_name not in TORCHVISION_DATASETS:
        raise ValueError(f"Unsupported torchvision dataset {dataset_name!r}; "
                         f"choose from {sorted(TORCHVISION_DATASETS)}")

    tag = f"{dataset_name}_{split}_{limit or 'all'}_s{seed}"
    out_dir = Path(cache_root) / "torchvision" / tag
    manifest = out_dir / "records.json"

    if manifest.exists():
        data = json.loads(manifest.read_text())
        records = [ImageRecord(**r) for r in data["records"]]
        if all(Path(r.filename).exists() for r in records):
            return Dataset(name=f"{dataset_name} ({split})", source="torchvision",
                           root=str(out_dir), class_names=data["class_names"], records=records)

    cls = getattr(tvd, dataset_name)
    params = inspect.signature(cls.__init__).parameters
    kwargs = {"root": raw_root, "download": download}
    if "train" in params:
        kwargs["train"] = (split == "train")
    elif "split" in params:
        kwargs["split"] = split
    Path(raw_root).mkdir(parents=True, exist_ok=True)
    if progress:
        progress(f"Opening {dataset_name} ({'downloading if needed' if download else 'local only'})", 0.0)
    ds = cls(**kwargs)

    targets = _tv_targets(ds)
    n_cls = max(targets) + 1
    raw_names = list(getattr(ds, "classes", []) or [])
    class_names = [str(c) for c in raw_names] if len(raw_names) == n_cls else [str(i) for i in range(n_cls)]
    dir_names = [_safe_dirname(c) for c in class_names]

    chosen = _stratified_subset(targets, limit, seed)
    records: List[ImageRecord] = []
    for k, i in enumerate(chosen):
        img, t = ds[i]
        t = int(t)
        p = out_dir / dir_names[t] / f"{i:06d}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            img.save(p)
        records.append(ImageRecord(filename=str(p.resolve()), label=class_names[t], class_idx=t,
                                   extra={"source_index": i}))
        if progress and k % 100 == 0:
            progress(f"Writing images {k}/{len(chosen)}", k / max(len(chosen), 1))

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"class_names": class_names,
                                    "records": [asdict(r) for r in records]}))
    return Dataset(name=f"{dataset_name} ({split})", source="torchvision", root=str(out_dir),
                   class_names=class_names, records=records)
