"""Measure HNSW recall against an exact scan for the eval queries.

For each query, the exact top-k (sequential scan, ``enable_indexscan = off``) defines the
ground truth; HNSW recall@k is the share of HNSW results whose cosine distance is within the
exact k-th distance (ties count as hits, since many HTS lines share identical text).

    uv run python scripts/hnsw_recall.py --database-url postgresql://... \
        --ef-search 40 100 200 400 --rebuild 16/64 16/200

``--rebuild m/ef_construction`` temporarily rebuilds ``hts_lines_embedding_hnsw`` with other
parameters; the original index definition is restored afterwards.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector_async
from psycopg.rows import TupleRow

from landedcost.config import Settings
from landedcost.container import build_embedder
from landedcost.embeddings.base import Vector
from landedcost.evals.dataset import load_eval_items

INDEX = "hts_lines_embedding_hnsw"
Conn = psycopg.AsyncConnection[TupleRow]


async def exact_kth_distance(conn: Conn, vector: Vector, k: int) -> float:
    async with conn.transaction():
        await conn.execute("SET LOCAL enable_indexscan = off")
        cur = await conn.execute(
            "SELECT embedding <=> %s AS d FROM hts_lines ORDER BY d LIMIT %s", (vector, k)
        )
        rows = await cur.fetchall()
    return float(rows[-1][0])


async def hnsw_recall(
    conn: Conn, vectors: list[Vector], kth: list[float], k: int, ef: int
) -> tuple[float, float]:
    hits = 0
    started = time.perf_counter()
    for vector, threshold in zip(vectors, kth, strict=True):
        async with conn.transaction():
            await conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef),))
            cur = await conn.execute(
                "SELECT embedding <=> %s FROM hts_lines ORDER BY embedding <=> %s LIMIT %s",
                (vector, vector, k),
            )
            hits += sum(1 for (d,) in await cur.fetchall() if float(d) <= threshold + 1e-6)
    elapsed_ms = (time.perf_counter() - started) * 1000 / len(vectors)
    return hits / (k * len(vectors)), elapsed_ms


async def report(
    conn: Conn, label: str, vectors: list[Vector], kth: list[float], args: argparse.Namespace
) -> None:
    for ef in args.ef_search:
        recall, ms = await hnsw_recall(conn, vectors, kth, args.k, ef)
        print(f"| {label} | {ef} | {recall:.3f} | {ms:.1f} |")


async def main(args: argparse.Namespace) -> None:
    settings = Settings()
    embedder = build_embedder(settings)
    queries = [i.description for i in load_eval_items(args.dataset)]
    vectors = [np.asarray(await embedder.embed_query(q), dtype=np.float32) for q in queries]
    async with await psycopg.AsyncConnection.connect(args.database_url, autocommit=True) as conn:
        await register_vector_async(conn)
        kth = [await exact_kth_distance(conn, v, args.k) for v in vectors]
        cur = await conn.execute("SELECT pg_get_indexdef(to_regclass(%s))", (INDEX,))
        original = (await cur.fetchone() or [None])[0]
        if not original:
            raise SystemExit(f"{INDEX} not found; run `landedcost migrate` and `ingest-hts` first")
        print(f"{len(queries)} queries, embedder {embedder.name}, recall@{args.k} vs exact scan\n")
        print("| index | ef_search | recall | ms/query |")
        print("|---|---|---|---|")
        await report(conn, "as migrated", vectors, kth, args)
        try:
            for spec in args.rebuild:
                m, efc = (int(x) for x in spec.split("/"))
                await conn.execute(f"DROP INDEX {INDEX}")
                await conn.execute("SET maintenance_work_mem = '256MB'")
                started = time.perf_counter()
                await conn.execute(
                    f"CREATE INDEX {INDEX} ON hts_lines USING hnsw (embedding vector_cosine_ops) "
                    f"WITH (m = {m}, ef_construction = {efc})".encode()
                )
                build_s = time.perf_counter() - started
                await report(
                    conn, f"m={m}, ef_construction={efc} (build {build_s:.1f}s)", vectors, kth, args
                )
        finally:
            if args.rebuild:
                await conn.execute(f"DROP INDEX IF EXISTS {INDEX}")
                await conn.execute(str(original).encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--database-url", default=Settings().database_url)
    parser.add_argument("--dataset", type=Path, default=Path("data/eval/starter_eval.jsonl"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, nargs="+", default=[40, 100, 200, 400])
    parser.add_argument("--rebuild", nargs="*", default=[], metavar="M/EF_CONSTRUCTION")
    asyncio.run(main(parser.parse_args()))
