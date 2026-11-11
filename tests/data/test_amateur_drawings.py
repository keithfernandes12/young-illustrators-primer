"""Amateur Drawings source, run against a fake archive and annotations in a temp store."""

import gzip
import io
import json
import logging
import struct
import tarfile

import pytest

from primer.data import pipeline
from primer.data.catalog import read_dataset
from primer.data.config import SourceConfig
from primer.data.manifest import ValidationFailed
from primer.data.sources import amateur_drawings
from primer.data.sources.amateur_drawings import AmateurDrawings, archive_images, png_size

ANN_URL = "https://example.test/amateur_drawings_annotations.json"
ARC_URL = "https://example.test/amateur_drawings.tar"


def keypoints(visible=17):
    return [v for i in range(17) for v in (10 + i, 20 + i, 2 if i < visible else 0)]


def png(width=50, height=100, body=b""):
    """The start of a PNG: signature and IHDR header (enough for its size), then ``body``."""
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I4sII", 13, b"IHDR", width, height) + body


def coco(n_images=4, **overrides):
    images = [{"id": i, "width": 100, "height": 200, "file_name": f"amateur_drawings/{i % 2}/{i}.png", "license": 1}
              for i in range(n_images)]
    annotations = [{"id": i, "image_id": i, "category_id": 1, "segmentation": [[1, 2, 3, 4, 5, 6]],
                    "area": 50.5, "bbox": [1.5, 2.0, 30.0, 40.0], "iscrowd": 0,
                    "keypoints": keypoints(16 if i == 1 else 17), "num_keypoints": 17}
                   for i in range(n_images)]
    data = {"info": {}, "categories": [], "images": images, "annotations": annotations,
            "licenses": [{"id": 1, "name": "Attribution-NonCommercial License"}]}
    data.update(overrides)
    return data


def tar_gz(files: dict[str, bytes], extra_dir=True) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        if extra_dir:
            folder = tarfile.TarInfo("amateur_drawings")
            folder.type = tarfile.DIRTYPE
            tar.addfile(folder)
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def default_files():
    # Uploaded at 100x200 (the annotations), published resized to 50x100 (the PNGs), like Meta's.
    # Image 2 is a byte-for-byte copy of image 0; image 3 is annotated but not in the archive.
    return {"amateur_drawings/0/0.png": png(body=b"zero"), "amateur_drawings/1/1.png": png(body=b"one"),
            "amateur_drawings/0/2.png": png(body=b"zero"), "../escape.png": b"nope"}


def all_files(n=4):
    """A distinct 50x100 PNG for every annotated image."""
    return {f"amateur_drawings/{i % 2}/{i}.png": png(body=bytes([i])) for i in range(n)}


def make_source(store, annotations=None, files=None, max_missing=0.3, fail_after=None, min_images=2):
    remote = {ANN_URL: {"etag": "a1", "bytes": 1}, ARC_URL: {"etag": "t1", "bytes": 1}}
    blobs = {ANN_URL: json.dumps(annotations or coco()).encode(),
             ARC_URL: tar_gz(default_files() if files is None else files)}

    def head(url):
        return remote[url]

    def opener(url):
        data = blobs[url]
        if fail_after is not None and url == ARC_URL:
            data = data[:fail_after]          # connection drops mid-archive
        return io.BytesIO(data)

    cfg = SourceConfig(key="amateur_drawings", name="Amateur Drawings", license="CC-BY-NC-4.0",
                       attribution="AD", ingest={"annotations_url": ANN_URL, "archive_url": ARC_URL,
                                                 "min_images": min_images, "max_missing": max_missing})
    return AmateurDrawings(cfg, store, head=head, opener=opener), remote


@pytest.fixture
def batch_ids(monkeypatch):
    ids = iter(f"2026110{i}T000000Z" for i in range(10))
    monkeypatch.setattr(pipeline, "new_batch_id", lambda: next(ids))


def test_end_to_end(store, batch_ids):
    source, remote = make_source(store)
    batch = pipeline.run_all(source)

    raw = f"raw/amateur_drawings/{batch}"
    assert store.read_bytes(f"{raw}/amateur_drawings/1/1.png") == png(body=b"one")  # original path, unchanged
    assert store.exists(f"{raw}/amateur_drawings_annotations.json")
    assert not store.glob("raw/amateur_drawings/escape.png") and not store.glob("raw/escape.png")
    assert store.read_json(f"{raw}/_batch.json")["images"] == 3

    df = read_dataset("character_drawings", store)
    assert df["source_id"].tolist() == ["0", "1"]          # 2 duplicates 0's bytes, 3 has no file
    first = df.iloc[0]
    assert first["image_uri"] == store.url(f"{raw}/amateur_drawings/0/0.png")
    assert (first["width"], first["height"]) == (50, 100)                    # the PNG's own size
    assert (first["original_width"], first["original_height"]) == (100, 200)  # as uploaded
    assert (first["bbox_x"], first["bbox_y"], first["bbox_width"], first["bbox_height"]) == (1.5, 2.0, 30.0, 40.0)
    assert json.loads(first["keypoints"])[0] == [10, 20, 2] and len(json.loads(first["keypoints"])) == 17
    assert df["n_keypoints"].tolist() == [17, 16]
    assert json.loads(first["segmentation"]) == [[1, 2, 3, 4, 5, 6]]
    assert (first["area"], first["license"], first["attribution"]) == (50.5, "CC-BY-NC-4.0", "AD")

    assert pipeline.ingest(source) is None                 # both ETags unchanged: nothing downloaded
    remote[ARC_URL] = {"etag": "t2", "bytes": 1}           # Meta replaced the archive
    assert pipeline.run_all(source) is not None
    assert len(read_dataset("character_drawings", store)) == 2   # latest snapshot only


def test_validation_fails_when_annotated_images_are_missing(store, batch_ids):
    source, _ = make_source(store, max_missing=0.005)
    batch = pipeline.ingest(source)
    with pytest.raises(ValidationFailed, match="image_file_missing"):
        pipeline.validate(source, batch)


def test_validation_fails_on_boxes_outside_the_image_and_orphan_annotations(store, batch_ids):
    bad = coco(4)
    # Fits the 100 px uploaded width but runs 10 px past the 50 px PNG the box is drawn on
    bad["annotations"][0]["bbox"] = [30, 0, 30, 40]
    bad["annotations"].append({**bad["annotations"][1], "id": 99, "image_id": 1234})
    files = {f"amateur_drawings/{i % 2}/{i}.png": png(body=bytes([i])) for i in range(4)}
    source, _ = make_source(store, annotations=bad, files=files)
    batch = pipeline.ingest(source)
    with pytest.raises(ValidationFailed, match="bbox_outside_image") as failed:
        pipeline.validate(source, batch)
    assert "annotations_match_images" in str(failed.value)


def test_an_interrupted_archive_download_is_retried_next_time(store, batch_ids):
    source, _ = make_source(store, fail_after=60)
    with pytest.raises((EOFError, tarfile.ReadError, OSError)):
        pipeline.ingest(source)
    assert not store.exists("metadata/state/amateur_drawings.json")   # ETags not recorded

    source, _ = make_source(store)
    assert pipeline.ingest(source) is not None


def test_archive_images_skips_directories_and_unsafe_paths():
    data = tar_gz({"amateur_drawings/a.png": b"a", "/abs.png": b"x", "x/../../up.png": b"y"})
    assert list(archive_images(io.BytesIO(data))) == [("amateur_drawings/a.png", b"a")]


def test_open_file_prefers_a_local_copy(tmp_path, monkeypatch):
    (tmp_path / "amateur_drawings.tar").write_bytes(b"local bytes")
    monkeypatch.setenv(amateur_drawings.LOCAL_DIR_ENV, str(tmp_path))
    with amateur_drawings.open_file(ARC_URL) as fh:
        assert fh.read() == b"local bytes"


def test_png_size_reads_the_header_and_rejects_other_files():
    assert png_size(png(384, 512, b"rest")) == (384, 512)
    assert png_size(b"GIF89a" + bytes(30)) is None
    assert png_size(png()[:20]) is None
    assert png_size(png(0, 100)) is None                     # a zero size is not a usable size


def test_sizes_are_read_from_the_bucket_for_batches_ingested_without_them(store, batch_ids):
    files = {f"amateur_drawings/{i % 2}/{i}.png": png(40 + i, 80, bytes([i])) for i in range(4)}
    source, _ = make_source(store, files=files)
    batch = pipeline.ingest(source)
    listing = f"raw/amateur_drawings/{batch}/_files.jsonl.gz"
    old = [{k: v for k, v in f.items() if k not in ("width", "height")} for f in store.read_jsonl_gz(listing)]
    store.write_bytes(listing, gzip.compress("\n".join(map(json.dumps, old)).encode()))

    pipeline.validate(source, batch)
    pipeline.preprocess(source, batch)
    df = read_dataset("character_drawings", store)
    assert df["width"].tolist() == [40, 41, 42, 43] and set(df["original_width"]) == {100}


def test_images_that_are_not_pngs_fail_validation_and_are_dropped(store, batch_ids):
    files = {f"amateur_drawings/{i % 2}/{i}.png": png(body=bytes([i])) for i in range(3)}
    files["amateur_drawings/1/3.png"] = b"not a png"
    source, _ = make_source(store, files=files)
    batch = pipeline.ingest(source)
    with pytest.raises(ValidationFailed, match="image_size_unreadable"):
        pipeline.validate(source, batch)
    pipeline.preprocess(source, batch)
    assert read_dataset("character_drawings", store)["source_id"].tolist() == ["0", "1", "2"]


def test_images_are_uploaded_in_chunks(store, batch_ids, monkeypatch):
    monkeypatch.setattr(amateur_drawings, "WRITE_CHUNK", 2)
    sizes = []
    write_many = store.write_many
    monkeypatch.setattr(store, "write_many", lambda files: (sizes.append(len(files)), write_many(files)))
    files = all_files(5)
    source, _ = make_source(store, annotations=coco(5), files=files)
    batch = pipeline.ingest(source)
    assert sizes == [2, 2, 1]                                  # streamed: never the whole archive at once
    for name, data in files.items():
        assert store.read_bytes(f"raw/amateur_drawings/{batch}/{name}") == data


def test_a_clean_batch_passes_including_a_box_within_the_rounding_slack(store, batch_ids):
    data = coco(4)
    data["annotations"][0]["bbox"] = [0, 0, 51, 100]           # 1 px past a 50 px PNG: within slack
    source, _ = make_source(store, annotations=data, files=all_files())
    pipeline.validate(source, pipeline.ingest(source))


@pytest.mark.parametrize(("check", "mutate", "options"), [
    ("images", lambda c: None, {"min_images": 5}),
    ("image_fields", lambda c: c["images"][0].pop("width"), {}),
    ("annotation_fields", lambda c: c["annotations"][0].pop("id"), {}),
    ("annotation_fields", lambda c: c["annotations"][0].pop("area"), {}),
    ("unique_image_id", lambda c: c["images"][1].update(id=0), {}),
    ("unique_annotation_id", lambda c: c["annotations"][1].update(id=0), {}),
    ("one_annotation_per_image", lambda c: c["annotations"].append({**c["annotations"][0], "id": 50}), {}),
    ("keypoints_not_17", lambda c: c["annotations"][0].update(keypoints=keypoints()[:-1]), {}),
    ("bbox_malformed_or_empty", lambda c: c["annotations"][0].update(bbox=[1, 2, 30, 0]), {}),
    ("bbox_outside_image", lambda c: c["annotations"][0].update(bbox=[0, 0, 53, 100]), {}),
])
def test_each_validation_check_catches_its_problem(store, batch_ids, check, mutate, options):
    data = coco(4)
    mutate(data)
    source, _ = make_source(store, annotations=data, files=all_files(), **options)
    batch = pipeline.ingest(source)
    with pytest.raises(ValidationFailed, match=check):
        pipeline.validate(source, batch)


def test_validation_fails_on_empty_image_files(store, batch_ids):
    files = all_files()
    files["amateur_drawings/0/0.png"] = b""
    source, _ = make_source(store, files=files)
    batch = pipeline.ingest(source)
    with pytest.raises(ValidationFailed, match="empty_image_file"):
        pipeline.validate(source, batch)


def test_preprocess_drops_empty_boxes_and_malformed_annotations_instead_of_crashing(store, batch_ids):
    data = coco(4)
    data["annotations"][0]["bbox"] = [1, 2, 30, 0]            # zero height, like 3 boxes in Meta's data
    data["annotations"][1]["bbox"] = [1, 2, 30]               # wrong length
    data["annotations"][2]["keypoints"] = keypoints()[:-1]    # 50 numbers, not 51
    source, _ = make_source(store, annotations=data, files=all_files())
    pipeline.preprocess(source, pipeline.ingest(source))
    assert read_dataset("character_drawings", store)["source_id"].tolist() == ["3"]


def test_a_failed_validation_points_at_validate_instead_of_reingesting(store, batch_ids, caplog):
    source, _ = make_source(store, max_missing=0.005)
    batch = pipeline.ingest(source)
    with pytest.raises(ValidationFailed):
        pipeline.validate(source, batch)
    with caplog.at_level(logging.WARNING):
        assert pipeline.ingest(source) is None                # same files upstream: no re-download
    assert f"Batch {batch} was ingested but never preprocessed" in caplog.text


def test_without_etags_size_and_date_identify_the_version(store, batch_ids):
    source, remote = make_source(store)
    for url in remote:
        remote[url] = {"etag": "", "bytes": 10, "last_modified": "Mon, 01 Jan 2024 00:00:00 GMT"}
    assert pipeline.ingest(source) is not None
    assert pipeline.ingest(source) is None                    # same size and date: unchanged
    remote[ARC_URL] = {"etag": "", "bytes": 11, "last_modified": "Tue, 02 Jan 2024 00:00:00 GMT"}
    assert pipeline.ingest(source) is not None


def test_with_no_way_to_tell_the_version_ingest_fails_instead_of_downloading(store, batch_ids):
    source, remote = make_source(store)
    remote[ARC_URL] = {"etag": "", "bytes": 0}
    with pytest.raises(RuntimeError, match="Can't tell whether"):
        pipeline.ingest(source)
    assert not store.glob("raw/amateur_drawings/*")
