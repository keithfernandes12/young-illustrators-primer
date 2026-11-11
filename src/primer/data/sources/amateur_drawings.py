"""Amateur Drawings (Meta FAIR, 2023): children's drawings of human figures, with annotations.

Over 178,000 drawings, each with a user-confirmed bounding box, segmentation polygon and 17 joint
locations (COCO keypoint format). Meta publishes two files:

    amateur_drawings_annotations.json   ~290 MB, COCO: "images" (id, file_name, width, height),
                                        "annotations" (image_id, bbox, segmentation, keypoints)
    amateur_drawings.tar                ~48 GB gzipped tar of PNGs at amateur_drawings/<x>/<uuid>.png,
                                        the same paths as the annotations' file_name

A snapshot source keyed to both files' ETags: a new batch is made only when Meta changes either
file, which a HEAD request checks without downloading anything. The archive is streamed member by
member into ``raw/amateur_drawings/<batch>/`` (images keep their own paths, unchanged), so it never
has to fit in memory or on disk. ``PRIMER_AMATEUR_DRAWINGS_DIR`` points at a folder holding local
copies of both files instead (downloaded first with ``curl -C -``, which resumes), so a failed upload
doesn't mean downloading the archive again.

Each batch produces ``character_drawings``: one row per annotated drawing, with ``image_uri``
pointing at the PNG in ``raw/``.

The published PNGs are not the uploaded drawings: Meta resized them to 512 px on the long side, and
the bounding boxes, joints and polygons are in that resized frame. The annotations' ``width`` and
``height`` are still the uploaded size. So each PNG's real size is read from its header (recorded in
``_files.jsonl.gz`` at ingest, or read back from the bucket for older batches), boxes are checked
against it, and it becomes ``width``/``height``; the uploaded size is kept as ``original_*``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import os
import struct
import tarfile
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import IO, Any

import pandas as pd

from primer.data import checks
from primer.data.manifest import Check, StageRun
from primer.data.schemas import CHARACTER_DRAWINGS, conform
from primer.data.sources.base import USER_AGENT, Source
from primer.data.text import record_id

log = logging.getLogger(__name__)

LOCAL_DIR_ENV = "PRIMER_AMATEUR_DRAWINGS_DIR"
ANNOTATIONS_FILE = "amateur_drawings_annotations.json"
FILES_LIST = "_files.jsonl.gz"
WRITE_CHUNK = 200       # images uploaded together (concurrently on GCS)
LOG_EVERY = 10_000
N_KEYPOINTS = 17
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_HEADER_BYTES = 24   # signature, IHDR chunk length and type, width, height
SIZE_CHUNK = 10_000     # PNG headers read from the bucket per request batch

Head = Callable[[str], dict[str, Any]]
Opener = Callable[[str], AbstractContextManager[IO[bytes]]]


def http_head(url: str) -> dict[str, Any]:
    """ETag and size of a remote file, without downloading it."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return {"etag": (response.headers.get("ETag") or "").strip('"'),
                "bytes": int(response.headers.get("Content-Length") or 0),
                "last_modified": response.headers.get("Last-Modified")}


def open_file(url: str) -> AbstractContextManager[IO[bytes]]:
    """Open a source file: the local copy in ``PRIMER_AMATEUR_DRAWINGS_DIR`` if there is one."""
    local_dir = os.environ.get(LOCAL_DIR_ENV)
    if local_dir:
        path = Path(local_dir) / url.rsplit("/", 1)[-1]
        if path.exists():
            log.info("Reading local copy %s", path)
            return open(path, "rb")
        log.warning("%s is set but %s is missing; downloading %s", LOCAL_DIR_ENV, path, url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=600)


def remote_version(url: str, info: dict[str, Any]) -> str:
    """What identifies the current version of a remote file: its ETag, else its size and date.

    Fails rather than guessing when the server sends none of them, since "changed" would mean
    downloading and uploading 48 GB again.
    """
    if info.get("etag"):
        return str(info["etag"])
    if info.get("bytes") and info.get("last_modified"):
        return f"{info['bytes']}@{info['last_modified']}"
    raise RuntimeError(f"Can't tell whether {url} changed: no ETag, Content-Length or Last-Modified")


def png_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from a PNG's header, or None if ``data`` isn't a PNG with a non-zero size."""
    if len(data) < PNG_HEADER_BYTES or data[:8] != PNG_SIGNATURE or data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:PNG_HEADER_BYTES])
    return (width, height) if width and height else None


def pixel_size(file: dict[str, Any] | None) -> tuple[int, int] | None:
    """An image file's recorded (width, height) when both are positive integers, else None.

    The one test of "readable size" that validate and preprocess both use.
    """
    if not file:
        return None
    width, height = file.get("width"), file.get("height")
    if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
        return width, height
    return None


def bbox_values(bbox: Any) -> tuple[float, float, float, float] | None:
    """A COCO [x, y, width, height] box as four finite numbers with a positive size, else None."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x, y, w, h = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        return None
    return x, y, w, h


def keypoints_ok(keypoints: Any) -> bool:
    """17 joints as COCO's flat list of 51 numbers: x, y and visibility for each."""
    return (isinstance(keypoints, list) and len(keypoints) == 3 * N_KEYPOINTS
            and all(isinstance(v, (int, float)) for v in keypoints))


def archive_images(stream: IO[bytes]) -> Iterator[tuple[str, bytes]]:
    """Yield (path, bytes) for each regular file in a gzipped tar stream, in archive order.

    Paths that could escape the batch folder (absolute, or containing "..") are skipped.
    """
    with tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                log.warning("Skipping unsafe archive path %r", member.name)
                continue
            content = archive.extractfile(member)
            if content is not None:
                yield str(path), content.read()


class AmateurDrawings(Source):
    key = "amateur_drawings"
    dataset = "character_drawings"
    snapshot = True

    def __init__(self, config, store, head: Head | None = None, opener: Opener | None = None):
        super().__init__(config, store)
        self.head = head or http_head
        self.opener = opener or open_file

    @property
    def urls(self) -> dict[str, str]:
        return {"annotations": self.options["annotations_url"], "archive": self.options["archive_url"]}

    # ------------------------------------------------------------------ ingest

    def ingest(self, run: StageRun, batch_size: int | None = None) -> bool:
        remote = {name: self.head(url) for name, url in self.urls.items()}
        # Keys keep their "_etag" names so existing state still matches; the value is the ETag when
        # the server sends one (see remote_version).
        etags = {f"{name}_etag": remote_version(url, remote[name]) for name, url in self.urls.items()}
        state = self.store.read_json(self.state_path, default={})
        run.records_in = 0
        if all(state.get(k) == v for k, v in etags.items()):
            latest = state.get("latest_batch")
            log.info("Amateur Drawings unchanged since batch %s; nothing new", latest)
            if latest and not self.store.exists(self.processed_path(latest)):
                # The files are already in raw/; a re-ingest would download the same 48 GB again.
                log.warning("Batch %s was ingested but never preprocessed (validation failed or was not "
                            "run). Fix the cause, then run `primer-data validate %s` and `primer-data "
                            "preprocess %s`; they reuse that batch, nothing is downloaded.",
                            latest, self.key, self.key)
            run.records_out = 0
            return False

        prefix = self.raw_prefix(run.batch_id)
        with self.opener(self.urls["annotations"]) as fh:
            annotations = fh.read()
        self.store.write_bytes(f"{prefix}/{ANNOTATIONS_FILE}", annotations)

        files: list[dict[str, Any]] = []
        chunk: dict[str, bytes] = {}
        with self.opener(self.urls["archive"]) as stream:
            for path, data in archive_images(stream):
                chunk[f"{prefix}/{path}"] = data
                size = png_size(data)
                files.append({"name": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                              "width": size[0] if size else None, "height": size[1] if size else None})
                if len(chunk) >= WRITE_CHUNK:
                    self.store.write_many(chunk)
                    chunk = {}
                if len(files) % LOG_EVERY == 0:
                    log.info("Uploaded %s images (%.1f GB)", f"{len(files):,}",
                             sum(f["bytes"] for f in files) / 1e9)
        self.store.write_many(chunk)

        self.store.write_jsonl_gz(f"{prefix}/{FILES_LIST}", files)
        self.store.write_json(f"{prefix}/_batch.json", {
            "batch_id": run.batch_id,
            "urls": self.urls,
            "remote": remote,
            "local_dir": os.environ.get(LOCAL_DIR_ENV),
            "annotations_sha256": hashlib.sha256(annotations).hexdigest(),
            "images": len(files),
            "image_bytes": sum(f["bytes"] for f in files),
            "fetched_at": datetime.now(UTC).isoformat(),
        })
        # Record the ETags only once every image is written, so an ingest interrupted part-way (a
        # dropped connection, a crash) starts a fresh batch next time. They are recorded before
        # validation on purpose: if validation fails, rerunning validate/preprocess on this batch
        # is the fix, not downloading the same files again (ingest says so when it sees that case).
        self.store.write_json(self.state_path, {**etags, "latest_batch": run.batch_id})
        run.records_out = run.records_total = len(files)
        run.output_uri = self.store.url(prefix)
        log.info("Ingested %s images into %s", f"{len(files):,}", run.output_uri)
        return True

    # ------------------------------------------------------------------ reading a raw batch

    def read_annotations(self, batch_id: str) -> dict[str, Any]:
        coco = json.loads(self.read_raw_file(batch_id, ANNOTATIONS_FILE))
        if not isinstance(coco, dict):
            raise TypeError(f"{ANNOTATIONS_FILE} is not a COCO object")
        return coco

    def read_files(self, batch_id: str) -> dict[str, dict[str, Any]]:
        """Each image file's size in bytes, SHA-256 and pixel size (None if not a readable PNG)."""
        lines = gzip.decompress(self.read_raw_file(batch_id, FILES_LIST)).decode("utf-8").splitlines()
        files = {f["name"]: f for f in map(json.loads, filter(None, lines))}
        self._add_pixel_sizes(batch_id, files)
        return files

    def _add_pixel_sizes(self, batch_id: str, files: dict[str, dict[str, Any]]) -> None:
        """Fill in width/height from the PNG headers in the bucket, for batches ingested before ingest
        recorded them (only the first batch, 20261110T055432Z). Batches ingested now already carry
        them, so this returns at once; ~5 minutes per stage otherwise."""
        missing = [name for name, f in files.items() if "width" not in f]
        if not missing:
            return
        log.info("Reading the pixel size of %s images from their PNG headers", f"{len(missing):,}")
        prefix = self.raw_prefix(batch_id)
        for start in range(0, len(missing), SIZE_CHUNK):
            names = missing[start:start + SIZE_CHUNK]
            heads = self.store.read_heads([f"{prefix}/{name}" for name in names], PNG_HEADER_BYTES)
            for name, head in zip(names, heads, strict=True):
                size = png_size(head) if head else None
                files[name]["width"], files[name]["height"] = size if size else (None, None)
            log.info("Read %s of %s PNG headers", f"{start + len(names):,}", f"{len(missing):,}")

    # ---------------------------------------------------------------- validate

    def validate(self, run: StageRun) -> None:
        coco = self.read_annotations(run.batch_id)
        files = self.read_files(run.batch_id)
        images = coco.get("images") or []
        annotations = coco.get("annotations") or []
        image_ids = {img.get("id") for img in images}
        by_id = {img.get("id"): img for img in images}

        def missing_file(img: dict) -> bool:
            return img.get("file_name") not in files

        def bbox_outside(ann: dict) -> bool:
            """A well-formed box outside the PNG it is drawn on (not the uploaded size, which is larger).

            Malformed or empty boxes, missing files and unreadable PNGs have their own checks.
            """
            img = by_id.get(ann.get("image_id"))
            box = bbox_values(ann.get("bbox"))
            size = pixel_size(files.get(img.get("file_name"))) if img else None
            if box is None or size is None:
                return False
            x, y, w, h = box
            width, height = size
            tol = 0.01 * max(width, height) + 1   # annotation tools round box corners outward
            return x < -tol or y < -tol or x + w > width + tol or y + h > height + tol

        orphans = sum(1 for ann in annotations if ann.get("image_id") not in image_ids)
        run.records_in = run.records_out = len(images)
        run.output_uri = self.store.url(self.raw_prefix(run.batch_id))
        # Annotations that fail the ratio checks below are dropped in preprocess; the limits stop a
        # batch where too many would be.
        run.checks += [
            checks.count_between("images", len(images), self.options.get("min_images", 1)),
            checks.required_fields(images, ["id", "file_name", "width", "height"], name="image_fields"),
            checks.required_fields(annotations, ["id", "image_id", "bbox", "keypoints", "segmentation", "area"],
                                   name="annotation_fields"),
            checks.unique(images, "id", name="unique_image_id"),
            checks.unique(annotations, "id", name="unique_annotation_id"),       # record_id comes from it
            checks.unique(annotations, "image_id", name="one_annotation_per_image"),  # dedupe assumes it
            Check("annotations_match_images", orphans == 0, f"{orphans:,} annotations for unknown images"),
            checks.max_ratio("image_file_missing", images, missing_file, self.options.get("max_missing", 0.005)),
            checks.max_ratio("empty_image_file", list(files.values()), lambda f: f["bytes"] == 0, 0.001),
            checks.max_ratio("image_size_unreadable", list(files.values()), lambda f: pixel_size(f) is None,
                             0.001),
            checks.max_ratio(f"keypoints_not_{N_KEYPOINTS}", annotations,
                             lambda ann: not keypoints_ok(ann.get("keypoints")), 0.01),
            checks.max_ratio("bbox_malformed_or_empty", annotations,
                             lambda ann: bbox_values(ann.get("bbox")) is None, 0.01),
            checks.max_ratio("bbox_outside_image", annotations, bbox_outside, 0.01),
        ]

    # -------------------------------------------------------------- preprocess

    def preprocess(self, run: StageRun) -> None:
        coco = self.read_annotations(run.batch_id)
        files = self.read_files(run.batch_id)
        images = pd.DataFrame.from_records(coco["images"])[["id", "file_name", "width", "height"]]
        anns = pd.DataFrame.from_records(coco["annotations"])
        df = anns.merge(images.rename(columns={"id": "image_id"}), on="image_id", how="inner")
        run.records_in = len(df)

        df["image_sha256"] = df["file_name"].map(lambda name: files.get(name, {}).get("sha256"))
        sizes = df["file_name"].map(lambda name: pixel_size(files.get(name)))
        boxes = df["bbox"].map(bbox_values)

        # Each dropped row gets the first reason that applies; the same predicates as validate.
        reasons = pd.Series(None, index=df.index, dtype=object)
        for why, mask in [("image file missing", df["image_sha256"].isna()),
                          ("not a readable PNG", sizes.isna()),
                          ("empty or malformed bounding box", boxes.isna()),
                          ("malformed keypoints", ~df["keypoints"].map(keypoints_ok))]:
            reasons = reasons.mask(reasons.isna() & mask, why)
        kept = reasons.isna()
        reasons = reasons.mask(kept & df["image_sha256"].where(kept).duplicated(), "duplicate image")
        drops = Counter(reasons.dropna())
        df = df[reasons.isna()].copy()
        if drops:
            log.info("Dropped %s", ", ".join(f"{n:,} ({why})" for why, n in drops.items()))

        prefix = self.raw_prefix(run.batch_id)
        df["record_id"] = df["id"].map(lambda ann_id: record_id(self.key, str(ann_id)))
        df["source"] = self.key
        df["source_id"] = df["image_id"].astype(str)
        df["image_uri"] = df["file_name"].map(lambda name: self.store.url(f"{prefix}/{name}"))
        df = df.rename(columns={"width": "original_width", "height": "original_height"})
        df[["width", "height"]] = pd.DataFrame(sizes[df.index].tolist(), index=df.index,
                                               columns=["width", "height"])
        bbox_columns = ["bbox_x", "bbox_y", "bbox_width", "bbox_height"]
        df[bbox_columns] = pd.DataFrame(boxes[df.index].tolist(), index=df.index, columns=bbox_columns,
                                        dtype="float64")
        df["keypoints"] = df["keypoints"].map(
            lambda kp: json.dumps([kp[i:i + 3] for i in range(0, len(kp), 3)], separators=(",", ":")))
        df["n_keypoints"] = df["keypoints"].map(lambda kp: sum(1 for p in json.loads(kp) if p[2] > 0))
        df["segmentation"] = df["segmentation"].map(lambda seg: json.dumps(seg, separators=(",", ":")))
        df["area"] = pd.to_numeric(df["area"], errors="coerce")   # required by validate
        df["license"] = self.config.license
        df["attribution"] = self.config.attribution
        df["batch_id"] = run.batch_id
        df["ingested_at"] = pd.Timestamp.now(tz="UTC")
        df = conform(df, CHARACTER_DRAWINGS)

        out_path = self.processed_path(run.batch_id)
        self.store.write_parquet(out_path, df)
        run.records_out = run.records_total = len(df)
        run.output_uri = self.store.url(out_path)
