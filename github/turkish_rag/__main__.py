"""Run with python -m turkish_rag --help."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def evaluate(rows, k):
    from .core import retrieval_metrics
    if not rows:
        raise ValueError('Evaluation input is empty.')
    values = []
    for i, row in enumerate(rows, 1):
        if row.get('error'):
            raise ValueError(f'Row {i} failed generation. Resolve errors before evaluation.')
        values.append(retrieval_metrics(row['retrieved_ids'], row['relevant_ids'], k))
    return {'n': len(rows), 'k': k, 'averaging': 'macro',
            'metrics': {key: sum(v[key] for v in values) / len(values) for key in values[0]}}


def main():
    parser = argparse.ArgumentParser(description='Reconstructed Turkish hybrid RAG reference implementation')
    sub = parser.add_subparsers(dest='command', required=True)
    ingest_parser = sub.add_parser('ingest', help='Index PDFs in an isolated reference schema')
    ingest_parser.add_argument('--pdf-dir', required=True)
    ask = sub.add_parser('ask', help='Answer a single question')
    ask.add_argument('question')
    ask.add_argument('--model', required=True, help='Available Gemini model identifier')
    ask.add_argument('--mode', choices=['hybrid', 'dense', 'bm25'], default='hybrid')
    search = sub.add_parser('search', help='Retrieve without a paid generation request')
    search.add_argument('question')
    search.add_argument('--mode', choices=['hybrid', 'dense', 'bm25'], default='hybrid')
    batch = sub.add_parser('batch', help='Answer a JSONL question set; records each failure')
    batch.add_argument('--input', required=True)
    batch.add_argument('--output', required=True)
    batch.add_argument('--model', required=True)
    batch.add_argument('--mode', choices=['hybrid', 'dense', 'bm25'], default='hybrid')
    metric = sub.add_parser('evaluate', help='Offline retrieval metrics from labelled chunk IDs')
    metric.add_argument('--input', required=True)
    metric.add_argument('--k', type=int, default=5)
    sub.add_parser('config', help='Print reconstruction defaults without external dependencies')
    args = parser.parse_args()
    from .config import Settings
    if args.command == 'config':
        print(json.dumps(asdict(Settings()), ensure_ascii=False, indent=2))
        return
    if args.command == 'evaluate':
        print(json.dumps(evaluate(read_jsonl(args.input), args.k), indent=2))
        return
    from .pipeline import RAG, ingest
    from . import store
    if args.command == 'ingest':
        ingest(args.pdf_dir, Settings())
        return
    if args.command == 'batch':
        rows = read_jsonl(args.input)
        if not rows or any(not isinstance(r.get('question'), str) or not r['question'].strip() for r in rows):
            raise ValueError('Each JSONL row must have a non-empty question.')
        # Exclusive creation avoids overwriting a previous experiment.
        with Path(args.output).open('x', encoding='utf-8') as output, store.connect() as conn:
            rag = RAG(conn)
            failed = 0
            for row in rows:
                try:
                    result = rag.answer(row['question'], args.model, args.mode)
                except Exception as exc:
                    conn.rollback()
                    failed += 1
                    result = {'question': row['question'], 'error': type(exc).__name__,
                              'model': args.model, 'mode': args.mode}
                result['created_at'] = datetime.now(timezone.utc).isoformat()
                for key in ('id', 'reference', 'relevant_ids'):
                    if key in row:
                        result[key] = row[key]
                output.write(json.dumps(result, ensure_ascii=False) + '\n')
                output.flush()
            print(f'Completed {len(rows)} questions; failed: {failed}.', file=sys.stderr)
            if failed:
                raise SystemExit(1)
        return
    with store.connect() as conn:
        rag = RAG(conn)
        result = (rag.retrieve(args.question, args.mode) if args.command == 'search'
                  else rag.answer(args.question, args.model, args.mode))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
