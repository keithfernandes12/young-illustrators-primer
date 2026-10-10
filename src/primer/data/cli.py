"""Command line for the data pipeline.

    primer-data sources                      list registered sources
    primer-data run tinystories              ingest + validate + preprocess one new batch
    primer-data ingest tinystories           fetch the next batch into raw/
    primer-data validate tinystories         check a raw batch (default: latest)
    primer-data preprocess tinystories       clean a raw batch into processed/

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

log = logging.getLogger("primer.data")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="primer-data", description="Young Illustrator's Primer data pipeline")
    parser.add_argument("--data-uri", help="Data root (gs://bucket or a local folder for testing)")
    parser.add_argument("--sources-file", help="Path to sources.yaml (default: configs/sources.yaml)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("sources", help="List registered sources and which have a pipeline")
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


if __name__ == "__main__":
    sys.exit(main())
