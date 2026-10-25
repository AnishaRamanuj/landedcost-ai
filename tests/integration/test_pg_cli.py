import pytest
from typer.testing import CliRunner

from landedcost.cli import app
from tests.conftest import SAMPLE_SNAPSHOT, SYNTHETIC_RULINGS

pytestmark = pytest.mark.integration
runner = CliRunner()


def test_cli_migrate_ingest_import_and_classify(
    pg_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", pg_url)
    monkeypatch.setenv("STORE_BACKEND", "postgres")

    result = runner.invoke(app, ["migrate", "--drop-existing"])
    assert result.exit_code == 0, result.output
    assert "applied: 0001_init, 0002_bm25_stats" in result.output

    result = runner.invoke(
        app, ["ingest-hts", "--snapshot", str(SAMPLE_SNAPSHOT), "--chapters", "85,4"]
    )
    assert result.exit_code == 0, result.output
    assert "with hash-v2-c35" in result.output

    result = runner.invoke(app, ["ingest-hts", "--snapshot", str(SAMPLE_SNAPSHOT), "--if-empty"])
    assert "skipped" in result.output

    result = runner.invoke(app, ["import-rulings", str(SYNTHETIC_RULINGS)])
    assert result.exit_code == 0, result.output
    assert "imported 4 rulings" in result.output

    result = runner.invoke(app, ["classify", "lithium-ion battery module for kick scooters"])
    assert result.exit_code == 0, result.output
    assert "8507.60" in result.output
