"""Command line for the data pipeline.

    primer-data sources                      list registered sources
    primer-data run tinystories              ingest + validate + preprocess one new batch
    primer-data ingest tinystories           fetch the next batch into raw/
    primer-data validate tinystories         check a raw batch (default: latest)
    primer-data preprocess tinystories       clean a raw batch into processed/
    primer-data version v1 -m "notes"        freeze all processed datasets as version v1
    primer-data versions                     list dataset versions
    primer-data dvc-pointers                 refresh data/*.dvc to the current object versions

Data goes to PRIMER_DATA_URI (default gs://young-illustrators-primer-data);
override with --data-uri.
"""

from __future__ import annotations

import argparse
import logging
import sys

from primer.data import pipeline
from primer.data.config import load_sources
from primer.data.manifest import ValidationFailed
from primer.data.sources import SOURCES, get_source
from primer.data.storage import Store
from primer.data.versioning import VersionExists, create_version, list_versions, refresh_dvc_pointers

log = logging.getLogger("primer.data")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="primer-data", description="Young Illustrator's Primer data pipeline")
    parser.add_argument("--data-uri", help="Data root (gs://bucket or a local folder for testing)")
    parser.add_argument("--sources-file", help="Path to sources.yaml (default: configs/sources.yaml)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("sources", help="List registered sources and which have a pipeline")
    p = sub.add_parser("version", help="Freeze all processed datasets as a named version")
    p.add_argument("tag", help="Version name, e.g. v1")
    p.add_argument("-m", "--message", default="", help="What this version contains or why it was made")
    p.add_argument("--no-dvc", action="store_true", help="Skip refreshing the data/*.dvc pointer files")
    sub.add_parser("versions", help="List dataset versions")
    sub.add_parser("dvc-pointers", help="Refresh data/*.dvc to the current object versions (no download)")
    for name, text in (("run", "Ingest, validate, and preprocess one new batch"),
                       ("ingest", "Fetch the next batch into raw/")):
        p = sub.add_parser(name, help=text)
        p.add_argument("source")
        p.add_argument("--batch-size", type=int, help="Batch size: records, or categories for quickdraw (default from sources.yaml)")
    for name, text in (("validate", "Validate a raw batch"),
                       ("preprocess", "Clean a raw batch into processed/")):
        p = sub.add_parser(name, help=text)
        p.add_argument("source")
        p.add_argument("--batch", help="Batch id (default: latest raw batch)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    noisy_loggers = ("fsspec", "gcsfs", "google", "urllib3", "httpx", "httpx2",
                     "datasets", "huggingface_hub", "filelock")
    for noisy in noisy_loggers:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Harmless: our bucket paths are explicit, so no default project is needed.
    logging.getLogger("google.auth._default").setLevel(logging.ERROR)

    configs = load_sources(args.sources_file)

    if args.command == "sources":
        for key, cfg in sorted(configs.items(), key=lambda kv: kv[1].priority):
            ready = "pipeline ready" if key in SOURCES else "not implemented"
            print(f"{cfg.priority:>2}  {key:<20} {ready:<16} {cfg.name}")
        return 0

    if args.command == "dvc-pointers":
        for path in refresh_dvc_pointers(Store(args.data_uri)):
            log.info("Updated %s", path)
        return 0

    if args.command in ("version", "versions"):
        return _versions(args, Store(args.data_uri))

    if args.source not in configs:
        log.error("Unknown source %r. Known: %s", args.source, ", ".join(configs))
        return 2
    config = configs[args.source]

    store = Store(args.data_uri)
    source = get_source(config, store)
    log.info("Data root: %s", store.uri)

    try:
        if args.command == "run":
            batch = pipeline.run_all(source, args.batch_size)
            log.info(f"Batch {batch} done" if batch else "Nothing new to ingest")
        elif args.command == "ingest":
            batch = pipeline.ingest(source, args.batch_size)
            log.info(f"Ingested batch {batch}" if batch else "Nothing new to ingest")
        elif args.command == "validate":
            pipeline.validate(source, args.batch)
        elif args.command == "preprocess":
            pipeline.preprocess(source, args.batch)
    except ValidationFailed:
        return 1
    return 0


def _versions(args: argparse.Namespace, store: Store) -> int:
    if args.command == "versions":
        for v in sorted(list_versions(store), key=lambda v: v["created_at"]):
            sizes = ", ".join(f"{name} {d['rows']:,}" for name, d in v["datasets"].items())
            print(f"{v['tag']:<10} {v['created_at'][:16]}  {sizes}" + (f"  - {v['message']}" if v["message"] else ""))
        return 0
    try:
        version = create_version(store, args.tag, args.message, dvc=not args.no_dvc)
    except VersionExists as exc:
        log.error(str(exc))
        return 1
    for name, d in version["datasets"].items():
        log.info("  %-16s %s rows", name, f"{d['rows']:,}")
    log.info("Version %s recorded at %s", args.tag, store.url(f"metadata/versions/{args.tag}.json"))
    if version["dvc_files"]:
        log.info("Next: git add data/*.dvc && git commit -m 'chore(data): data version %s (KAN-<n>)' && git tag data-%s",
                 args.tag, args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
