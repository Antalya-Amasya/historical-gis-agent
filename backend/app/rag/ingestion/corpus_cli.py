from __future__ import annotations

import argparse
from pathlib import Path

from .corpus_dry_run import dry_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Historical corpus ingestion planning")
    parser.add_argument("--dry-run", action="store_true", help="Parse and validate only; never embed or write Chroma")
    parser.add_argument("--ingest", action="store_true", help="Incrementally embed into the isolated Roman Republic collection")
    parser.add_argument("--incoming", type=Path, default=Path("data/historical_sources/incoming"))
    parser.add_argument("--registry", type=Path, default=Path("data/historical_sources/corpus.json"))
    parser.add_argument("--report", type=Path, default=Path("data/historical_sources/processed/corpus_ingestion_report.json"))
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-documents", type=int)
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument("--document-id", action="append", default=[])
    selection.add_argument("--all-documents", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--collection")
    parser.add_argument("--state-prefix", default="v2")
    return parser


def lifecycle_command(handler, args, chunks=None, *, batch_size=128):
    if args.status: return handler.status()
    if args.stop: handler.stop(); return {"stop_requested": True}
    if args.resume:
        if chunks is None and not getattr(handler,'accepts_cli_resume',False): raise ValueError("resume requires injected chunks/backend")
        return handler.resume(chunks, batch_size=batch_size, max_batches=args.max_batches, max_documents=args.max_documents)
    return None


def production_config(args, *, state_dir: Path, default_collection: str) -> object:
    """Pure CLI-to-config mapping; it never creates a client or loads a model."""
    from .production_lifecycle import CorpusIngestionConfig
    document_ids=tuple(args.document_id)
    if args.all_documents: raise ValueError("all_documents_not_enabled_for_smoke")
    if not document_ids: raise ValueError("document_selection_required")
    return CorpusIngestionConfig(args.host, args.port, args.collection or default_collection, state_dir, args.state_prefix, document_ids, batch_size=128, max_batches=args.max_batches, max_documents=args.max_documents)


def main(argv=None, *, backend_factory=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.status or args.stop or args.resume:
        try:
            config=production_config(args,state_dir=Path("data/historical_sources/processed"),default_collection="roman_republic_primary_sources_v2")
        except ValueError as error: parser.error(str(error))
        if backend_factory is None:
            from .production_lifecycle import default_backend_factory
            backend_factory=default_backend_factory
        backend=backend_factory(config,args)
        backend.accepts_cli_resume=True
        lifecycle_command(backend,args)
        return 0
    if args.dry_run == args.ingest:
        parser.error("Choose exactly one of --dry-run or --ingest.")
    if args.dry_run:
        report = dry_run(args.incoming, args.registry, args.report)
        print(f"Dry run complete: {report['documents_registered']} registered documents, {report['totals']['expected_chunk_count']} expected chunks")
    else:
        parser.error("Legacy --ingest is disabled; use lifecycle --resume with an explicit document selection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
