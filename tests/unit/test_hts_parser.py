from collections import Counter
from pathlib import Path

from landedcost.hts.codes import CodeLevel
from landedcost.hts.loader import load_lines, load_snapshot
from landedcost.hts.models import HtsLine
from landedcost.hts.parser import clean_text, parse_rows

SHIPPED_SNAPSHOT = Path("data/hts/hts_2026-09-04.json.gz")


def row(
    htsno: str, indent: int | str, description: str, general: str = "", **extra: object
) -> dict[str, object]:
    return {
        "htsno": htsno,
        "indent": str(indent),
        "description": description,
        "superior": None if htsno else "true",
        "units": extra.get("units", []),
        "general": general,
        "special": extra.get("special", ""),
        "other": extra.get("other", ""),
        "footnotes": extra.get("footnotes", []),
    }


def test_leaf_other_keeps_full_hierarchy(lines_by_code: dict[str, HtsLine]) -> None:
    leaf = lines_by_code["8507.60.00.90"]
    assert leaf.description == "Other"
    assert leaf.path[0].startswith("Electric storage batteries")
    assert leaf.path[1:] == ("Lithium-ion batteries", "Other")
    assert leaf.breadcrumb.endswith("Lithium-ion batteries > Other > Other")
    assert leaf.chunk_id == "hts:8507.60.00.90"


def test_statistical_lines_inherit_rate_from_parent_tariff_line(
    lines_by_code: dict[str, HtsLine],
) -> None:
    stat = lines_by_code["8507.60.00.90"]
    assert stat.general_rate == "3.4%"
    assert stat.rate_source_code == "8507.60.00"
    assert stat.rate_inherited
    # special/other columns travel with the general rate
    assert stat.other_rate == lines_by_code["8507.60.00"].other_rate


def test_statistical_line_with_its_own_rate_is_not_inherited(
    lines_by_code: dict[str, HtsLine],
) -> None:
    asses = lines_by_code["0101.30.00.00"]
    assert asses.general_rate == "6.8%"
    assert asses.rate_source_code == "0101.30.00.00"
    assert not asses.rate_inherited


def test_headings_and_subheadings_have_no_rate(lines_by_code: dict[str, HtsLine]) -> None:
    assert lines_by_code["8507"].general_rate == ""
    assert lines_by_code["8507"].rate_source_code is None
    assert lines_by_code["0101.90"].level is CodeLevel.SUBHEADING
    assert lines_by_code["0101.90"].general_rate == ""


def test_context_rows_appear_in_path_but_are_not_lines(lines_by_code: dict[str, HtsLine]) -> None:
    dairy_male = lines_by_code["0102.21.00.10"]
    assert dairy_male.path[-2:] == ("Purebred breeding animals", "Dairy")
    assert all(line.code for line in lines_by_code.values())


def test_chapters_98_and_99_are_excluded_and_counted() -> None:
    _, parsed = load_lines(Path("tests/fixtures/hts_sample.json"))
    assert not [line for line in parsed.lines if line.chapter in {"98", "99"}]
    assert parsed.skipped["excluded_chapter_98"] > 0
    assert parsed.skipped["excluded_chapter_99"] > 0
    assert parsed.skipped["context_row"] > 0


def test_excluded_chapters_are_configurable() -> None:
    snapshot = load_snapshot(Path("tests/fixtures/hts_sample.json"))
    parsed = parse_rows(snapshot.rows, excluded_chapters=frozenset())
    assert any(line.chapter == "98" for line in parsed.lines)


def test_siblings_pop_the_stack_and_rates_do_not_leak_across_headings() -> None:
    rows = [
        row("1111", 0, "Widgets:"),
        row("1111.10.00", 1, "Blue", "5%"),
        row("1111.10.00.10", 2, "Small"),
        row("1111.20.00", 1, "Red:", "Free"),
        row("2222", 0, "Gadgets"),
        row("2222.00.00.10", 1, "Stat line under a heading without a rate"),
    ]
    by_code = {line.code: line for line in parse_rows(rows).lines}
    assert by_code["1111.10.00.10"].path == ("Widgets", "Blue")
    assert by_code["1111.20.00"].path == ("Widgets",)
    assert by_code["1111.20.00"].description == "Red"
    assert by_code["2222.00.00.10"].general_rate == ""
    assert by_code["2222.00.00.10"].rate_source_code is None


def test_invalid_codes_are_skipped_and_odd_indents_parsed() -> None:
    rows = [
        row("1111", "0", "Widgets"),
        row("1111.1", "1", "Broken code"),
        row("1111.10.00", "03", "Deep", "2%", footnotes=[{"value": " See note 1. "}, "junk"]),
    ]
    parsed = parse_rows(rows)
    assert parsed.skipped["invalid_code"] == 1
    deep = parsed.lines[-1]
    assert deep.indent == 3
    assert deep.footnotes == ("See note 1.",)


def test_clean_text_normalises_whitespace_and_trailing_colon() -> None:
    assert clean_text("  Lithium-ion  batteries:  ") == "Lithium-ion batteries"
    assert clean_text(None) == ""


def test_shipped_snapshot_parses_completely() -> None:
    snapshot, parsed = load_lines(SHIPPED_SNAPSHOT)
    assert snapshot.fetched_at == "2026-09-04"
    assert snapshot.version == "hts-2026-09-04"
    codes = [line.code for line in parsed.lines]
    assert len(codes) == len(set(codes)), "codes must be unique"
    assert len(codes) > 25_000
    levels = Counter(line.level for line in parsed.lines)
    assert levels[CodeLevel.HEADING] > 900
    stat_without_rate = [
        line.code
        for line in parsed.lines
        if line.level is CodeLevel.STATISTICAL and not line.general_rate
    ]
    assert stat_without_rate == []
