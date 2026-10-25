from datetime import date
from pathlib import Path

import pytest

from landedcost.rulings.importer import REQUIRED_COLUMNS, parse_ruling, read_rulings
from tests.conftest import FIXTURES


def test_csv_import_parses_and_normalises() -> None:
    report = read_rulings(FIXTURES / "rulings_synthetic.csv")
    assert report.errors == []
    by_number = {r.ruling_number: r for r in report.rulings}
    assert set(by_number) == {"TEST-0001", "TEST-0002", "TEST-0003", "TEST-0004"}
    tee = by_number["TEST-0004"]
    assert tee.hts_codes == ("6109.10.00.12", "6109.10.00.04")
    assert tee.ruling_date == date(2025, 4, 22)
    assert tee.headings == ("6109",)
    assert tee.subheadings == ("6109.10",)
    assert tee.chunk_id == "ruling:TEST-0004"


def test_jsonl_import_accepts_lists_and_strings() -> None:
    report = read_rulings(FIXTURES / "rulings_synthetic.jsonl")
    assert report.errors == []
    olive, jars = report.rulings
    assert olive.hts_codes == ("1509.20.20.20",)
    assert jars.hts_codes == ("7010.90.50.09",)
    assert jars.ruling_date is None
    assert jars.search_text == "Glass jars for preserving food"


def test_invalid_rows_are_reported_with_line_numbers_and_duplicates_merged() -> None:
    report = read_rulings(FIXTURES / "rulings_invalid.csv")
    assert [r.ruling_number for r in report.rulings] == ["TEST-0201"]
    assert report.rulings[0].hts_codes == ("8507.60.00.10",)  # later duplicate wins
    assert report.duplicates == 1
    joined = "\n".join(report.errors)
    assert len(report.errors) == 5
    assert "line 3: ruling_number is empty" in joined
    assert "line 4: product_description is empty" in joined
    assert "not an HTS code" in joined
    assert "not ISO-8601" in joined
    assert "hts_codes is empty" in joined


def test_missing_columns_fail_fast(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("ruling_number,hts_codes\nX,8507\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        read_rulings(bad)


def test_template_has_the_expected_header_and_no_rows() -> None:
    template = Path("data/rulings/rulings_template.csv").read_text(encoding="utf-8").splitlines()
    assert template == [",".join(REQUIRED_COLUMNS)]


def test_parse_ruling_strips_whitespace() -> None:
    ruling = parse_ruling(
        {"ruling_number": " TEST-1 ", "product_description": "  a\n b ", "hts_codes": "8507 8506"}
    )
    assert ruling.ruling_number == "TEST-1"
    assert ruling.product_description == "a b"
    assert ruling.hts_codes == ("8507", "8506")


def test_bad_jsonl_lines_are_reported_and_the_rest_imported(tmp_path: Path) -> None:
    path = tmp_path / "rulings.jsonl"
    good = (
        '{"ruling_number": "TEST-0301", "product_description": "Clover honey", '
        '"hts_codes": ["0409.00.00"]}'
    )
    path.write_text(
        "\n".join(
            [
                good,
                '{"ruling_number": "TEST-0302", "product_',
                '["not", "an object"]',
                good.replace("0301", "0303"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report = read_rulings(path)
    assert [r.ruling_number for r in report.rulings] == ["TEST-0301", "TEST-0303"]
    assert len(report.errors) == 2
    assert report.errors[0].startswith("line 2: invalid JSON")
    assert report.errors[1] == "line 3: expected a JSON object"


def test_csv_with_a_byte_order_mark_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "excel.csv"
    source = (FIXTURES / "rulings_synthetic.csv").read_text(encoding="utf-8")
    path.write_text(source, encoding="utf-8-sig")  # Excel "CSV UTF-8" adds a BOM
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    report = read_rulings(path)
    assert report.errors == []
    assert len(report.rulings) == 4
