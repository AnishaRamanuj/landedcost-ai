"""``landedcost`` command-line interface."""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, Any, NoReturn

import psycopg
import typer
from pydantic import ValidationError

from landedcost import DISCLAIMER
from landedcost.config import DEFAULT_EVAL_DATASET, Settings, StoreBackend, resolve_data_path
from landedcost.container import (
    ConfigurationError,
    Services,
    build_embedder,
    build_services,
    connect_postgres,
)
from landedcost.evals.dataset import items_from_rulings, load_eval_items
from landedcost.evals.runner import render_markdown, run_answer_eval, run_eval
from landedcost.hts.loader import load_lines
from landedcost.ingest import import_rulings_file, ingest_hts
from landedcost.landed.calculator import LandedCostError
from landedcost.landed.freight import FreightQuoteError, TransportMode
from landedcost.landed.models import LandedCostRequest
from landedcost.llm.base import aclose_all
from landedcost.rates.parser import RateKind, parse_rate
from landedcost.retrieval.hybrid import RetrievalMode
from landedcost.store.base import EmbedderMismatchError, StoreUnavailableError

app = typer.Typer(
    help=f"HTS classification, retrieval evals and landed-cost estimates. {DISCLAIMER}",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(2)


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro``, turning the errors a user can fix into one-line hints (exit code 2)."""
    try:
        return asyncio.run(coro)
    except (ConfigurationError, StoreUnavailableError, EmbedderMismatchError) as exc:
        _fail(str(exc))
    except psycopg.errors.UndefinedTable:
        _fail("the landedcost schema is missing; run `landedcost migrate` first")
    except psycopg.OperationalError as exc:
        _fail(f"PostgreSQL error (check DATABASE_URL): {exc}")
    except FileNotFoundError as exc:
        _fail(f"file not found: {exc.filename} (run from the repository root or set its path)")


def _settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        _fail("invalid configuration: " + "; ".join(_describe(e) for e in exc.errors()))


def _describe(error: Any, names: dict[str, str] | None = None) -> str:
    loc = [str(part) for part in error.get("loc", ())]
    field = loc[0] if loc else ""
    label = (names or {}).get(field, field.upper() if names is None else field)
    suffix = "".join(f"[{part}]" for part in loc[1:])
    message = str(error.get("msg", "invalid value"))
    return f"{label}{suffix}: {message}" if label else message


async def _with_services[T](settings: Settings, fn: Callable[[Services], Awaitable[T]]) -> T:
    services = await build_services(settings)
    try:
        return await fn(services)
    finally:
        await services.aclose()


def _postgres_only(settings: Settings, command: str) -> None:
    if settings.store_backend is not StoreBackend.POSTGRES:
        typer.echo(f"`{command}` writes to PostgreSQL; set STORE_BACKEND=postgres", err=True)
        raise typer.Exit(2)


def _split_list(value: str) -> list[str]:
    return [v for v in re.split(r"[,\s]+", value.strip()) if v]


@app.command()
def migrate(
    drop_existing: Annotated[
        bool, typer.Option("--drop-existing", help="Drop all landedcost tables first")
    ] = False,
) -> None:
    """Create or upgrade the PostgreSQL schema (idempotent)."""
    from landedcost.store.migrate import apply_migrations

    settings = _settings()
    _postgres_only(settings, "migrate")
    applied = _run(
        apply_migrations(
            settings.database_url, embedding_dim=settings.embedding_dim, drop_existing=drop_existing
        )
    )
    typer.echo(f"applied: {', '.join(applied)}" if applied else "schema is up to date")


@app.command("ingest-hts")
def ingest_hts_cmd(
    snapshot: Annotated[Path | None, typer.Option(help="HTS snapshot (.json or .json.gz)")] = None,
    chapters: Annotated[
        str, typer.Option(help="Only these chapters, e.g. '61,85' (default: all but 98/99)")
    ] = "",
    if_empty: Annotated[
        bool, typer.Option("--if-empty", help="Skip when the index already holds HTS lines")
    ] = False,
) -> None:
    """Parse, embed and load the HTS into PostgreSQL (replaces the existing index atomically)."""
    settings = _settings()
    _postgres_only(settings, "ingest-hts")
    path = resolve_data_path(snapshot) if snapshot else settings.hts_snapshot_path

    async def go() -> None:
        embedder = build_embedder(settings)
        try:
            store = await connect_postgres(settings)
            try:
                info = await store.index_info()
                if if_empty and info.hts_lines:
                    typer.echo(
                        f"index already holds {info.hts_lines} HTS lines ({info.hts_version}); "
                        "skipped"
                    )
                    return
                report = await ingest_hts(
                    store, embedder, path, chapters=_split_list(chapters) or None
                )
            finally:
                await store.close()
        finally:
            await aclose_all(embedder)
        skipped = ", ".join(f"{k}={v}" for k, v in sorted(report.skipped.items()))
        typer.echo(
            f"ingested {report.lines} HTS lines from {report.hts_version} with {embedder.name} "
            f"(embed {report.embed_seconds}s, load {report.load_seconds}s); skipped: {skipped}"
        )
        if report.rulings_reembedded:
            typer.echo(
                f"re-embedded {report.rulings_reembedded} stored rulings with {embedder.name}"
            )

    _run(go())


@app.command("import-rulings")
def import_rulings_cmd(path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    """Import CBP rulings from CSV or JSONL (see data/rulings/README.md)."""
    settings = _settings()
    _postgres_only(settings, "import-rulings")

    async def go() -> int:
        embedder = build_embedder(settings)
        try:
            store = await connect_postgres(settings)
            try:
                report = await import_rulings_file(store, embedder, path)
            finally:
                await store.close()
        finally:
            await aclose_all(embedder)
        typer.echo(
            f"imported {report.imported} rulings ({report.duplicates} duplicate rows merged)"
        )
        for err in report.errors:
            typer.echo(f"  skipped {err}", err=True)
        return 1 if report.errors and not report.imported else 0

    try:
        code = _run(go())
    except ValueError as exc:  # unreadable file or missing columns
        _fail(str(exc))
    raise typer.Exit(code)


@app.command()
def classify(
    description: Annotated[
        str, typer.Argument(help="Product description, as an importer would write it")
    ],
    top_k: Annotated[int, typer.Option(min=1, max=10)] = 3,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full JSON result")] = False,
) -> None:
    """Suggest HTS codes with citations (or abstain)."""
    result = _run(
        _with_services(_settings(), lambda s: s.classifier.classify(description, top_k=top_k))
    )
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return
    ev = result.evidence
    typer.echo(f"query: {result.query}")
    typer.echo(
        f"mode: {result.answer_mode}  confidence: {result.confidence:.2f}  "
        f"support: {ev.support:.2f}  heading consensus: {ev.heading_consensus:.2f}"
    )
    if result.abstained:
        typer.echo(f"ABSTAINED: {result.abstain_reason}")
    for i, cand in enumerate(result.candidates, start=1):
        typer.echo(f"{i}. {cand.hts_code}  general rate: {cand.general_rate or 'n/a'}")
        typer.echo(f"   {cand.breadcrumb}")
        typer.echo(f"   {cand.rationale}")
        typer.echo(f"   citations: {', '.join(cand.citations)}")
    for warning in result.warnings:
        typer.echo(f"warning: {warning}")
    typer.echo(result.disclaimer)


def _decimal(value: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise typer.BadParameter(f"not a number: {value!r}") from exc


def _parse_quantities(values: list[str]) -> dict[str, Decimal]:
    out: dict[str, Decimal] = {}
    for item in values:
        unit, _, amount = item.partition("=")
        try:
            out[unit.strip()] = Decimal(amount)
        except InvalidOperation as exc:
            raise typer.BadParameter(f"expected UNIT=AMOUNT, got {item!r}") from exc
    return out


_LANDED_COST_OPTIONS = {
    "hts_code": "--hts",
    "customs_value_usd": "--value",
    "freight_usd": "--freight",
    "insurance_usd": "--insurance",
    "weight_kg": "--weight-kg",
    "volume_cbm": "--volume-cbm",
    "quantities": "--qty",
}


@app.command("landed-cost")
def landed_cost(
    hts: Annotated[str, typer.Option(help="8- or 10-digit HTS code")],
    value: Annotated[
        Decimal, typer.Option(parser=_decimal, help="Customs (transaction) value in USD")
    ],
    mode: Annotated[TransportMode, typer.Option()] = TransportMode.OCEAN,
    freight: Annotated[
        Decimal | None, typer.Option(parser=_decimal, help="Known freight in USD (else quoted)")
    ] = None,
    insurance: Annotated[
        Decimal | None, typer.Option(parser=_decimal, help="Insurance in USD")
    ] = None,
    weight_kg: Annotated[
        Decimal | None, typer.Option(parser=_decimal, help="Gross weight in kg")
    ] = None,
    volume_cbm: Annotated[
        Decimal | None, typer.Option(parser=_decimal, help="Volume in cubic metres")
    ] = None,
    qty: Annotated[
        list[str] | None, typer.Option(help="Quantity for specific rates, e.g. kg=120")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Estimate landed cost for one shipment line."""
    try:
        request = LandedCostRequest(
            hts_code=hts,
            customs_value_usd=value,
            mode=mode,
            freight_usd=freight,
            insurance_usd=insurance,
            weight_kg=weight_kg,
            volume_cbm=volume_cbm,
            quantities=_parse_quantities(qty or []),
        )
    except ValidationError as exc:
        _fail("; ".join(_describe(e, _LANDED_COST_OPTIONS) for e in exc.errors()))
    try:
        result = _run(_with_services(_settings(), lambda s: s.landed_cost.estimate(request)))
    except (LandedCostError, FreightQuoteError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return
    typer.echo(f"{result.hts_code}  {result.breadcrumb}")
    rows = [
        ("customs value", result.customs_value_usd),
        (f"freight ({result.freight.source})", result.freight.amount_usd),
        ("insurance", result.insurance_usd),
        (f"duty @ {result.general_rate}", result.duty.amount_usd),
        ("MPF", result.mpf.amount_usd),
    ]
    if result.hmf:
        rows.append(("HMF", result.hmf.amount_usd))
    rows.append(("TOTAL landed cost", result.total_landed_cost_usd))
    width = max(len(label) for label, _ in rows)
    for label, amount in rows:
        shown = f"${amount:,.2f}" if amount is not None else "n/a"
        typer.echo(f"  {label:<{width}}  {shown:>12}")
    typer.echo(f"  duty: {result.duty.explanation}")
    for warning in result.warnings:
        typer.echo(f"  ! {warning}")
    typer.echo(result.disclaimer)


def _parse_modes(value: str) -> list[RetrievalMode]:
    try:
        return [RetrievalMode(m) for m in _split_list(value)]
    except ValueError as exc:
        allowed = ", ".join(m.value for m in RetrievalMode)
        raise typer.BadParameter(f"{exc}; choose from {allowed}") from exc


def _parse_ks(value: str) -> list[int]:
    try:
        ks = sorted({int(x) for x in _split_list(value)})
    except ValueError as exc:
        raise typer.BadParameter(f"cut-offs must be integers: {value!r}") from exc
    if not ks or ks[0] < 1:
        raise typer.BadParameter("cut-offs must be positive integers")
    return ks


def _parse_floors(values: list[str], modes: list[RetrievalMode]) -> dict[RetrievalMode, float]:
    floors: dict[RetrievalMode, float] = {}
    for value in values:
        name, _, number = value.partition("=")
        try:
            mode, floor = RetrievalMode(name.strip()), float(number)
        except ValueError as exc:
            raise typer.BadParameter(f"expected MODE=MRR (e.g. hybrid=0.5), got {value!r}") from exc
        if mode not in modes:
            raise typer.BadParameter(f"--fail-under {mode.value}: that mode is not evaluated")
        floors[mode] = floor
    return floors


@app.command("eval")
def eval_cmd(
    dataset: Annotated[Path, typer.Option(dir_okay=False)] = DEFAULT_EVAL_DATASET,
    mode: Annotated[
        str, typer.Option(help="Comma/space separated: vector,fts,hybrid")
    ] = "vector,fts,hybrid",
    k: Annotated[
        str, typer.Option(help="Comma/space separated cut-offs, e.g. '1 3 5 10'")
    ] = "1,3,5,10",
    rulings: Annotated[
        bool, typer.Option("--rulings", help="Also evaluate items derived from imported rulings")
    ] = False,
    answers: Annotated[
        bool,
        typer.Option(
            "--answers",
            help="Also score the classifier's final answers (LLM_PROVIDER, extractive if none)",
        ),
    ] = False,
    output: Annotated[Path | None, typer.Option(help="Write the markdown report here")] = None,
    details: Annotated[
        Path | None, typer.Option(help="Write per-item outcomes (JSONL) here")
    ] = None,
    fail_under: Annotated[
        list[str] | None,
        typer.Option(
            help="Exit 1 if a mode's heading MRR falls below a floor, e.g. 'hybrid=0.50' "
            "(repeatable; used by CI to catch retrieval regressions)"
        ),
    ] = None,
) -> None:
    """Recall@k and MRR at heading and subheading level, per retrieval mode."""
    settings = _settings()
    modes = _parse_modes(mode)
    ks = _parse_ks(k)
    floors = _parse_floors(fail_under or [], modes)
    dataset = resolve_data_path(dataset)
    if not dataset.is_file():
        _fail(f"eval dataset not found: {dataset}")
    items = load_eval_items(dataset)

    async def go(services: Services) -> str:
        info = await services.store.index_info()
        all_items = list(items)
        if rulings:
            all_items += items_from_rulings(await services.store.list_rulings())
        gate = services.gate
        report = await run_eval(services.retriever, gate, all_items, modes=modes, ks=ks)
        if answers:
            report = replace(report, answers=await run_answer_eval(services.classifier, all_items))
        weights = settings.rrf_weights
        meta = {
            "Dataset": f"{dataset.name} ({len(items)} items"
            + (f" + {len(all_items) - len(items)} ruling-derived" if rulings else "")
            + ")",
            "HTS data": f"{info.hts_version} ({info.hts_lines} indexed lines)",
            "Embedder": services.embedder.name,
            "Store": settings.store_backend.value
            + (
                f" (full-text ranking: {settings.fts_ranking}, "
                f"hnsw.ef_search: {settings.hnsw_ef_search})"
                if settings.store_backend is StoreBackend.POSTGRES
                else " (BM25 in Python, exact vector search)"
            ),
            "RRF": f"k={settings.rrf_k}, weights "
            + ", ".join(f"{name}={w:g}" for name, w in weights.items()),
            "Candidate pool per retriever": str(settings.retrieval_candidate_pool),
            "Run at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        }
        if details:
            _write_details(details, report)
        heading_mrr.update({m.mode: m.heading.mrr for m in report.modes})
        return render_markdown(report, title="Retrieval eval", meta=meta)

    heading_mrr: dict[RetrievalMode, float] = {}
    markdown = _run(_with_services(settings, go))
    typer.echo(markdown)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
    failures = [
        f"{m.value} heading MRR@{max(ks)} {heading_mrr[m]:.3f} < {floor:.3f}"
        for m, floor in floors.items()
        if heading_mrr[m] < floor
    ]
    if failures:
        typer.echo("eval below floor: " + "; ".join(failures), err=True)
        raise typer.Exit(1)


def _write_details(path: Path, report: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for mode_report in report.modes:
            for o in mode_report.outcomes:
                record = {
                    "id": o.item_id,
                    "mode": o.mode.value,
                    "heading_rank": o.heading_rank,
                    "subheading_rank": o.subheading_rank,
                    "abstained": o.abstained,
                    "abstain_reason": o.abstain_reason,
                    "top_codes": list(o.top_codes),
                }
                fh.write(json.dumps(record) + "\n")
        if report.answers is not None:
            for a in report.answers.outcomes:
                record = {
                    "id": a.item_id,
                    "mode": "answer",
                    "answer": a.top_code,
                    "heading_correct": a.heading_correct,
                    "subheading_correct": a.subheading_correct,
                    "abstained": a.abstained,
                    "abstain_reason": a.abstain_reason,
                    "llm_fallback": a.llm_fallback,
                }
                fh.write(json.dumps(record) + "\n")


@app.command()
def stats(snapshot: Annotated[Path | None, typer.Option()] = None) -> None:
    """Offline statistics for the HTS snapshot: lines per level and rate-parser coverage."""
    path = resolve_data_path(snapshot) if snapshot else _settings().hts_snapshot_path
    if not path.is_file():
        _fail(f"snapshot not found: {path}")
    snap, parsed = load_lines(path)
    levels = Counter(line.level.value for line in parsed.lines)
    rated = [line for line in parsed.lines if line.general_rate]
    kinds = Counter(parse_rate(line.general_rate).kind for line in rated)
    structured = sum(v for kind, v in kinds.items() if kind is not RateKind.UNKNOWN)
    typer.echo(
        f"snapshot: {path} ({snap.version}, {len(snap.rows)} raw rows, source {snap.source})"
    )
    typer.echo(
        f"indexed lines: {len(parsed.lines)}  "
        + "  ".join(f"{k}={v}" for k, v in sorted(levels.items()))
    )
    typer.echo("skipped rows: " + "  ".join(f"{k}={v}" for k, v in sorted(parsed.skipped.items())))
    inherited = sum(1 for line in parsed.lines if line.rate_inherited)
    typer.echo(
        f"lines with a general rate: {len(rated)} (inherited from a parent line: {inherited})"
    )
    typer.echo("general-rate kinds: " + "  ".join(f"{k.value}={v}" for k, v in kinds.most_common()))
    typer.echo(
        f"structured (computable) share: {structured}/{len(rated)} = {structured / len(rated):.1%}"
    )


@app.command()
def serve(
    host: Annotated[str | None, typer.Option()] = None,
    port: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Run the HTTP API (uvicorn)."""
    import uvicorn

    from landedcost.api.app import create_app

    settings = _settings()
    uvicorn.run(
        create_app(settings=settings),
        host=host or settings.api_host,
        port=port or settings.api_port,
    )


if __name__ == "__main__":  # pragma: no cover
    app()
