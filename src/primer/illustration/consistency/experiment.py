"""Pick the embedder and set the consistency threshold on held-out picture books.

No generated books exist yet, so the experiment uses the closest real thing: the illustrated picture
books in ``storybook_pages``. Two pages of the same book usually show the same characters in the same
style (a positive pair); two pages from different books of the same source usually do not (a hard
negative, since a publisher's house style is shared). An embedder that separates those pairs can tell
"same character" from "different character", and the similarity that separates them best is the
starting threshold.

- **Split by book**, never by page: a stable hash of the book sends all its pages to ``val`` or
  ``test``. ``val`` picks the embedder and the threshold; ``test`` only reports.
- **Selection**: a candidate replaces the baseline only if it beats it on ``val`` ROC-AUC.
- **Bias**: every test metric is reported per ``source`` (picture-book origin) and ``language``, with
  slices under ``primer.data.bias.SMALL_SLICE`` pairs flagged as too small to judge.
- **Sensitivity**: a threshold sweep, and how much ROC-AUC drops when one side of each pair is
  perturbed (greyscale, downscale, crop, JPEG), i.e. what the score actually keys on.

Nothing is trained and nothing is written to the bucket; the report goes to ``--out``. An agent
session does not run this: it reads images from the bucket and downloads weights, so a person runs it.

    python -m primer.illustration.consistency.experiment --out .local-data/consistency --mlflow
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from primer.data.bias import SMALL_SLICE
from primer.illustration.consistency.scorer import ImageEmbedder

CONFIG = Path(__file__).with_name("configs") / "consistency.yaml"
SPLITS = ("val", "test")
Reader = Callable[[str], bytes]


@dataclass(frozen=True)
class ExperimentConfig:
    """``configs/consistency.yaml``. The whole file is logged as run params."""

    name: str
    data_version: str
    baseline: str
    candidates: tuple[str, ...]
    pairs_per_split: int = 1000
    val_fraction: float = 0.5
    seed: int = 0
    batch_size: int = 32
    revision: str = "main"
    slice_keys: tuple[str, ...] = ("source", "language")
    perturbations: tuple[str, ...] = field(default_factory=tuple)


def load_config(path: Path = CONFIG) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("candidates", "slice_keys", "perturbations"):
        if key in raw:
            raw[key] = tuple(raw[key])
    config = ExperimentConfig(**raw)
    if config.baseline not in config.candidates:
        raise ValueError(f"baseline {config.baseline!r} must be one of the candidates {config.candidates}")
    return config


# ------------------------------------------------------------------ pairs


def split_of(book_key: str, val_fraction: float) -> str:
    """``val`` or ``test`` for a book, from a hash of its key: stable across runs and machines."""
    fraction = int.from_bytes(hashlib.sha256(book_key.encode()).digest()[:8], "big") / 2**64
    return "val" if fraction < val_fraction else "test"


def illustrated_pages(pages: pd.DataFrame, val_fraction: float) -> pd.DataFrame:
    """Pages with an illustration, from books with at least two, each tagged with its book's split."""
    illustrated = pages[pages["image_uri"].notna()].copy()
    # book_id is only unique within a source.
    illustrated["book_key"] = illustrated["source"].astype(str) + "/" + illustrated["book_id"].astype(str)
    counts = illustrated.groupby("book_key")["image_uri"].transform("nunique")
    illustrated = illustrated[counts >= 2]
    illustrated["split"] = illustrated["book_key"].map(lambda key: split_of(key, val_fraction))
    return illustrated.reset_index(drop=True)


def build_pairs(pages: pd.DataFrame, n_pairs: int, seed: int) -> pd.DataFrame:
    """Half positive pairs (two pages of one book), half negative (pages of two books, same source).

    A negative falls back to any other book only when its source has no second book.
    """
    rng = np.random.default_rng(seed)
    books = pages.groupby("book_key")
    uris = {key: sorted(group["image_uri"].unique()) for key, group in books}
    # A book with no language still belongs in the slices, as "unknown" rather than silently dropped.
    info = books[["source", "language"]].first().fillna({"language": "unknown"})
    keys = sorted(uris)
    if len(keys) < 2:
        raise ValueError("need at least two illustrated books to build negative pairs")
    by_source = {source: sorted(group.index) for source, group in info.groupby("source")}

    rows = []
    for index in range(n_pairs):
        book = keys[rng.integers(len(keys))]
        if index % 2 == 0:
            first, second = rng.choice(len(uris[book]), size=2, replace=False)
            other, a, b, label = book, uris[book][first], uris[book][second], 1
        else:
            pool = [key for key in by_source[info.loc[book, "source"]] if key != book]
            pool = pool or [key for key in keys if key != book]
            other = pool[rng.integers(len(pool))]
            a = uris[book][rng.integers(len(uris[book]))]
            b = uris[other][rng.integers(len(uris[other]))]
            label = 0
        rows.append({"a": a, "b": b, "label": label, "book_a": book, "book_b": other,
                     "source": info.loc[book, "source"], "language": info.loc[book, "language"]})
    return pd.DataFrame(rows).drop_duplicates(subset=["a", "b", "label"], ignore_index=True)


# ------------------------------------------------------------------ similarity and metrics


def embed_uris(embedder: ImageEmbedder, uris: Sequence[str], read: Reader, batch_size: int,
               transform: Callable[[bytes], bytes] | None = None) -> dict[str, np.ndarray]:
    """One vector per distinct URI, embedded in batches."""
    unique = sorted(set(uris))
    vectors: dict[str, np.ndarray] = {}
    for start in range(0, len(unique), batch_size):
        batch = unique[start:start + batch_size]
        images = [read(uri) for uri in batch]
        if transform is not None:
            images = [transform(image) for image in images]
        vectors.update(zip(batch, embedder.embed(images), strict=True))
    return vectors


def similarities(pairs: pd.DataFrame, left: Mapping[str, np.ndarray],
                 right: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
    """Cosine similarity of each pair (vectors are unit length, so a dot product)."""
    right = left if right is None else right
    return np.array([float(left[a] @ right[b]) for a, b in zip(pairs["a"], pairs["b"], strict=True)])


def choose_threshold(similarity: np.ndarray, labels: np.ndarray) -> float:
    """The similarity that maximises Youden's J (recall minus false-positive rate)."""
    fpr, tpr, thresholds = roc_curve(labels, similarity)
    best = int(np.argmax(tpr - fpr))
    return round(float(min(thresholds[best], 1.0)), 4)


def classification(similarity: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    """How well ``similarity >= threshold`` predicts "same book". ROC-AUC is None for one class."""
    predicted = similarity >= threshold
    negatives = labels == 0
    return {
        "n": len(labels),
        "roc_auc": round(float(roc_auc_score(labels, similarity)), 4) if len(set(labels)) == 2 else None,
        "accuracy": round(float(np.mean(predicted == labels)), 4) if len(labels) else None,
        "precision": round(float(precision_score(labels, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, predicted, zero_division=0)), 4),
        "false_positive_rate": round(float(np.mean(predicted[negatives])), 4) if negatives.any() else None,
    }


def slice_metrics(pairs: pd.DataFrame, similarity: np.ndarray, threshold: float,
                  keys: Sequence[str]) -> list[dict[str, Any]]:
    """``classification`` per slice value. Under ``SMALL_SLICE`` pairs is reported but not judged."""
    labels = pairs["label"].to_numpy()
    rows = []
    for key in keys:
        for value, index in pairs.groupby(key).indices.items():
            rows.append({"key": key, "value": str(value), "n": len(index), "too_small": len(index) < SMALL_SLICE,
                         "metrics": classification(similarity[index], labels[index], threshold)})
    return rows


def slice_gaps(slices: Sequence[dict[str, Any]], metric: str = "roc_auc") -> dict[str, dict[str, Any]]:
    """Per slice key, the best and worst judged slice on ``metric``, and the gap between them."""
    gaps: dict[str, dict[str, Any]] = {}
    for key in dict.fromkeys(row["key"] for row in slices):
        judged = [row for row in slices
                  if row["key"] == key and not row["too_small"] and row["metrics"][metric] is not None]
        if len(judged) < 2:
            continue
        worst = min(judged, key=lambda row: row["metrics"][metric])
        best = max(judged, key=lambda row: row["metrics"][metric])
        gaps[key] = {"metric": metric, "worst": worst["value"], "best": best["value"],
                     "gap": round(best["metrics"][metric] - worst["metrics"][metric], 4)}
    return gaps


def threshold_sweep(similarity: np.ndarray, labels: np.ndarray,
                    thresholds: Sequence[float] = tuple(np.round(np.linspace(0.5, 1.0, 11), 2))) -> list[dict]:
    """Precision, recall and F1 across thresholds: what moving the gate trades for what."""
    return [{"threshold": float(t), **classification(similarity, labels, float(t))} for t in thresholds]


# ------------------------------------------------------------------ sensitivity


def _pil(image: bytes) -> Any:
    from PIL import Image

    with Image.open(io.BytesIO(image)) as opened:
        return opened.convert("RGB")


def _png(picture: Any) -> bytes:
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return buffer.getvalue()


def _greyscale(image: bytes) -> bytes:
    return _png(_pil(image).convert("L").convert("RGB"))


def _downscale(image: bytes) -> bytes:
    picture = _pil(image)
    return _png(picture.resize((64, 64)).resize(picture.size))


def _centre_crop(image: bytes) -> bytes:
    picture = _pil(image)
    width, height = picture.size
    return _png(picture.crop((width // 5, height // 5, width - width // 5, height - height // 5)))


def _jpeg(image: bytes) -> bytes:
    buffer = io.BytesIO()
    _pil(image).save(buffer, format="JPEG", quality=15)
    return buffer.getvalue()


#: What each perturbation asks: does the score depend on colour, fine detail, framing, artefacts?
PERTURBATIONS: dict[str, Callable[[bytes], bytes]] = {
    "greyscale": _greyscale,
    "downscale": _downscale,
    "centre_crop": _centre_crop,
    "jpeg": _jpeg,
}


def sensitivity(embedder: ImageEmbedder, pairs: pd.DataFrame, vectors: Mapping[str, np.ndarray], read: Reader,
                perturbations: Mapping[str, Callable[[bytes], bytes]], batch_size: int) -> dict[str, dict]:
    """ROC-AUC with the second image of every pair perturbed, against the clean ROC-AUC."""
    labels = pairs["label"].to_numpy()
    clean = float(roc_auc_score(labels, similarities(pairs, vectors)))
    report = {}
    for name, perturb in perturbations.items():
        perturbed = embed_uris(embedder, pairs["b"].tolist(), read, batch_size, transform=perturb)
        auc = float(roc_auc_score(labels, similarities(pairs, vectors, perturbed)))
        report[name] = {"roc_auc": round(auc, 4), "delta": round(auc - clean, 4)}
    return report


# ------------------------------------------------------------------ the experiment


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(config: ExperimentConfig, pages: pd.DataFrame, embedders: Mapping[str, ImageEmbedder], read: Reader,
        perturbations: Mapping[str, Callable[[bytes], bytes]] | None = None) -> dict[str, Any]:
    """Every candidate on the same pairs; the report ``main`` writes and MLflow logs."""
    illustrated = illustrated_pages(pages, config.val_fraction)
    pairs = {split: build_pairs(illustrated[illustrated["split"] == split], config.pairs_per_split,
                                config.seed + offset)
             for offset, split in enumerate(SPLITS)}
    labels = {split: pairs[split]["label"].to_numpy() for split in SPLITS}
    every_uri = [uri for split in SPLITS for column in ("a", "b") for uri in pairs[split][column]]

    candidates: dict[str, dict[str, Any]] = {}
    vectors: dict[str, dict[str, np.ndarray]] = {}
    for name in config.candidates:
        vectors[name] = embed_uris(embedders[name], every_uri, read, config.batch_size)
        similarity = {split: similarities(pairs[split], vectors[name]) for split in SPLITS}
        threshold = choose_threshold(similarity["val"], labels["val"])
        candidates[name] = {
            "threshold": threshold,
            **{split: classification(similarity[split], labels[split], threshold) for split in SPLITS},
            "_test_similarity": similarity["test"],
        }

    # Selected on val only. Ties and losses keep the baseline: a candidate has to beat it.
    baseline_auc = candidates[config.baseline]["val"]["roc_auc"] or 0.0
    challenger = max(config.candidates, key=lambda name: candidates[name]["val"]["roc_auc"] or 0.0)
    selected = challenger if (candidates[challenger]["val"]["roc_auc"] or 0.0) > baseline_auc else config.baseline
    chosen = candidates[selected]
    test_similarity = chosen["_test_similarity"]
    slices = slice_metrics(pairs["test"], test_similarity, chosen["threshold"], config.slice_keys)

    names = perturbations if perturbations is not None else {
        name: PERTURBATIONS[name] for name in config.perturbations}
    return {
        "stage": "illustration",
        "component": "consistency",
        "data_version": config.data_version,
        "git_sha": git_sha(),
        "config": asdict(config),
        "selected": selected,
        "baseline": config.baseline,
        "threshold": chosen["threshold"],
        "selected_test": chosen["test"],
        "baseline_test": candidates[config.baseline]["test"],
        "candidates": {name: {key: value for key, value in result.items() if not key.startswith("_")}
                       for name, result in candidates.items()},
        "pairs": {split: {"n": len(pairs[split]), "positives": int(labels[split].sum()),
                          "books": int(pd.concat([pairs[split]["book_a"], pairs[split]["book_b"]]).nunique())}
                  for split in SPLITS},
        "slices": slices,
        "slice_gaps": slice_gaps(slices),
        "sweep": threshold_sweep(test_similarity, labels["test"]),
        "sensitivity": sensitivity(embedders[selected], pairs["test"], vectors[selected], read, names,
                                   config.batch_size),
    }


_METRIC_NAME = re.compile(r"[^\w./\- :]")


def _metric_name(name: str) -> str:
    """MLflow refuses some characters in metric names, `=` among them."""
    return _METRIC_NAME.sub("_", name)


def log_to_mlflow(report: dict[str, Any], owner: str) -> list[str]:
    """One run per candidate in ``yip-illustration``, with the tags ``model-development.md`` lists."""
    import mlflow

    mlflow.set_experiment("yip-illustration")
    run_ids = []
    for name, result in report["candidates"].items():
        with mlflow.start_run(run_name=f"consistency-{name}") as active:
            mlflow.set_tags({
                "stage": "illustration", "component": "consistency", "owner": owner,
                "data_version": report["data_version"], "git_sha": report["git_sha"] or "unknown",
                "base_model": name, "gate_group": "illustration", "where": "local",
                "selected": str(name == report["selected"]).lower(),
            })
            mlflow.log_params({key: str(value) for key, value in report["config"].items()})
            metrics = {"threshold": result["threshold"]}
            for split in SPLITS:
                metrics |= {f"{split}_{key}": value for key, value in result[split].items() if value is not None}
            if name == report["selected"]:
                for row in report["slices"]:
                    for key, value in row["metrics"].items():
                        if value is not None:
                            metrics[f"slice/{row['key']}:{row['value']}/{key}"] = value
                for perturbation, values in report["sensitivity"].items():
                    metrics[f"sensitivity/{perturbation}/roc_auc"] = values["roc_auc"]
                mlflow.log_dict(report, "consistency_report.json")
            mlflow.log_metrics({_metric_name(key): float(value) for key, value in metrics.items()})
            run_ids.append(active.info.run_id)
    return run_ids


def _read_uri(uri: str) -> bytes:
    import fsspec

    with fsspec.open(uri, "rb") as handle:
        return handle.read()


class CachedReader:
    """Downloads each image once into a local folder; later reads come from disk.

    Every candidate embeds the same images and every perturbation re-reads the test images, so without
    this one image is fetched from the bucket up to seven times. The images are licensed picture-book
    illustrations, not child data, and the folder is under ``--out`` (gitignored ``.local-data``).
    """

    def __init__(self, folder: Path, read: Reader = _read_uri) -> None:
        self.folder = folder
        self.read = read
        folder.mkdir(parents=True, exist_ok=True)

    def __call__(self, uri: str) -> bytes:
        path = self.folder / hashlib.sha256(uri.encode()).hexdigest()
        if path.exists():
            return path.read_bytes()
        data = self.read(uri)
        partial = path.with_suffix(".partial")
        partial.write_bytes(data)
        partial.replace(path)  # never leave a half-written image where a later run would trust it
        return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--data-version", help="override the config's frozen data version")
    parser.add_argument("--out", type=Path, required=True, help="folder for consistency_report.json")
    parser.add_argument("--pairs", type=int, help="pairs per split; keep it small on a laptop")
    parser.add_argument("--candidates", nargs="+", help="a subset of the config's candidates")
    parser.add_argument("--mlflow", action="store_true", help="log one run per candidate to MLFLOW_TRACKING_URI")
    parser.add_argument("--owner", default=os.environ.get("USER", "unknown"))
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.data_version:
        config = replace(config, data_version=args.data_version)
    if args.pairs:
        config = replace(config, pairs_per_split=args.pairs)
    if args.candidates:
        baseline = config.baseline if config.baseline in args.candidates else args.candidates[0]
        if baseline != config.baseline:
            print(f"note: {config.baseline} is not among --candidates, so {baseline} is the baseline for this run")
        config = replace(config, candidates=tuple(args.candidates), baseline=baseline)

    from primer.data.catalog import read_dataset
    from primer.illustration.consistency.embedders import load_embedder

    pages = read_dataset("storybook_pages", version=config.data_version)
    embedders = {name: load_embedder(name, revision=config.revision, batch_size=config.batch_size)
                 for name in config.candidates}
    report = run(config, pages, embedders, CachedReader(args.out / "images"))
    report["revisions"] = {name: getattr(embedder, "resolved_revision", None) for name, embedder in embedders.items()}

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "consistency_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"selected {report['selected']} at threshold {report['threshold']}; "
          f"test ROC-AUC {report['selected_test']['roc_auc']} (baseline {report['baseline_test']['roc_auc']})")
    print(f"report: {path}")
    if args.mlflow:
        print("mlflow runs:", ", ".join(log_to_mlflow(report, args.owner)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
