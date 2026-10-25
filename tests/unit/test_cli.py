from pathlib import Path

import pytest
from typer.testing import CliRunner

from landedcost.cli import app
from tests.conftest import SAMPLE_SNAPSHOT

runner = CliRunner()


@pytest.fixture
def memory_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STORE_BACKEND", "memory")
    monkeypatch.setenv("HTS_SNAPSHOT_PATH", str(SAMPLE_SNAPSHOT))


def test_stats_reports_parser_coverage() -> None:
    result = runner.invoke(app, ["stats", "--snapshot", str(SAMPLE_SNAPSHOT)])
    assert result.exit_code == 0, result.output
    assert "indexed lines:" in result.output
    assert "structured (computable) share:" in result.output


@pytest.mark.usefixtures("memory_env")
def test_classify_prints_candidates_and_disclaimer() -> None:
    result = runner.invoke(app, ["classify", "natural honey in glass jars", "--top-k", "2"])
    assert result.exit_code == 0, result.output
    assert "Not legal or customs advice" in result.output
    json_result = runner.invoke(app, ["classify", "natural honey", "--json"])
    assert '"disclaimer"' in json_result.output


@pytest.mark.usefixtures("memory_env")
def test_landed_cost_command() -> None:
    result = runner.invoke(
        app,
        [
            "landed-cost",
            "--hts",
            "8507.60.00.90",
            "--value",
            "12000",
            "--weight-kg",
            "850",
            "--insurance",
            "60",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "TOTAL landed cost" in result.output
    assert "Chapter 99" in result.output
    bad = runner.invoke(app, ["landed-cost", "--hts", "8507", "--value", "100", "--freight", "1"])
    assert bad.exit_code == 2
    assert "carries no duty rate" in bad.output
    with_qty = runner.invoke(
        app,
        [
            "landed-cost",
            "--hts",
            "8213.00.30.00",
            "--value",
            "1000",
            "--freight",
            "5",
            "--qty",
            "each=10",
            "--json",
        ],
    )
    assert with_qty.exit_code == 0, with_qty.output
    assert '"status": "computed"' in with_qty.output


@pytest.mark.usefixtures("memory_env")
def test_eval_writes_markdown(tmp_path: Path) -> None:
    dataset = tmp_path / "mini.jsonl"
    dataset.write_text(
        '{"id": "h", "description": "natural honey", "gold_headings": ["0409"]}\n', encoding="utf-8"
    )
    out = tmp_path / "eval.md"
    details = tmp_path / "details.jsonl"
    result = runner.invoke(
        app,
        [
            "eval",
            "--dataset",
            str(dataset),
            "--mode",
            "fts hybrid",
            "--k",
            "1 3",
            "--output",
            str(out),
            "--details",
            str(details),
        ],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8")
    assert "| fts | heading (4) | 1 | 1.000 | 1.000 | 1.000 |" in text
    assert len(details.read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.usefixtures("memory_env")
@pytest.mark.parametrize("command", [["migrate"], ["ingest-hts"]])
def test_postgres_only_commands_refuse_memory_backend(command: list[str]) -> None:
    result = runner.invoke(app, command)
    assert result.exit_code == 2
    assert "STORE_BACKEND=postgres" in result.output


@pytest.mark.usefixtures("memory_env")
@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--value", "-5"], "--value: Input should be greater than 0"),
        (["--value", "1e30"], "--value: Decimal input should have no more than 18 digits"),
        (["--value", "10", "--qty", "furlongs=3"], "--qty: Value error, unknown unit"),
        (["--value", "10", "--qty", "kg=-1"], "--qty[kg]: Input should be greater than or equal"),
    ],
)
def test_landed_cost_rejects_bad_numbers_without_a_traceback(
    extra: list[str], message: str
) -> None:
    result = runner.invoke(app, ["landed-cost", "--hts", "8507.60.00.90", "--freight", "1", *extra])
    assert result.exit_code == 2
    assert f"error: {message}" in result.output
    assert "Traceback" not in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


@pytest.mark.usefixtures("memory_env")
@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--mode", "semantic"], "is not a valid RetrievalMode"),
        (["--k", "one"], "cut-offs must be integers"),
        (["--k", "0"], "cut-offs must be positive"),
        (["--dataset", "nope.jsonl"], "eval dataset not found"),
    ],
)
def test_eval_validates_its_options(args: list[str], message: str) -> None:
    result = runner.invoke(app, ["eval", *args])
    assert result.exit_code == 2
    assert message in result.output


def test_unreachable_database_is_a_one_line_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody:wrong@127.0.0.1:1/none")
    monkeypatch.setenv("DB_CONNECT_TIMEOUT_S", "1")
    result = runner.invoke(app, ["classify", "natural honey"])
    assert result.exit_code == 2
    assert "error: cannot connect to PostgreSQL" in result.output
    assert "Traceback" not in result.output


def test_invalid_configuration_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_POOL_MAX_SIZE", "zero")
    result = runner.invoke(app, ["classify", "natural honey"])
    assert result.exit_code == 2
    assert "error: invalid configuration: DB_POOL_MAX_SIZE" in result.output


def test_missing_llm_settings_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STORE_BACKEND", "memory")
    monkeypatch.setenv("LLM_PROVIDER", "azure_openai")
    result = runner.invoke(app, ["classify", "natural honey"])
    assert result.exit_code == 2
    assert "error: LLM_PROVIDER=azure_openai requires AZURE_OPENAI_CHAT_DEPLOYMENT" in result.output


def test_default_paths_resolve_outside_the_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0, result.output
    assert "hts_2026-09-04.json.gz" in result.output


@pytest.mark.usefixtures("memory_env")
def test_eval_answers_section(tmp_path: Path) -> None:
    dataset = tmp_path / "mini.jsonl"
    dataset.write_text(
        '{"id": "h", "description": "natural honey", "gold_headings": ["0409"]}\n'
        '{"id": "x", "description": "zzqx plugh", "expect_abstain": true}\n',
        encoding="utf-8",
    )
    details = tmp_path / "details.jsonl"
    result = runner.invoke(
        app,
        [
            "eval",
            "--dataset",
            str(dataset),
            "--mode",
            "hybrid",
            "--answers",
            "--details",
            str(details),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "### Final answers (answer mode: extractive (no LLM))" in result.output
    assert "| Out-of-scope items abstained | 1/1 (1.000) |" in result.output
    assert sum('"mode": "answer"' in line for line in details.read_text().splitlines()) == 2


@pytest.mark.usefixtures("memory_env")
def test_eval_fail_under_guards_against_regressions(tmp_path: Path) -> None:
    dataset = tmp_path / "mini.jsonl"
    dataset.write_text(
        '{"id": "h", "description": "natural honey", "gold_headings": ["0409"]}\n',
        encoding="utf-8",
    )
    base = ["eval", "--dataset", str(dataset), "--mode", "fts", "--k", "1"]
    assert runner.invoke(app, [*base, "--fail-under", "fts=0.9"]).exit_code == 0
    failed = runner.invoke(app, [*base, "--fail-under", "fts=1.01"])
    assert failed.exit_code == 1
    assert "eval below floor: fts heading MRR@1 1.000 < 1.010" in failed.output
    bad = runner.invoke(app, [*base, "--fail-under", "hybrid=0.5"])
    assert bad.exit_code == 2
