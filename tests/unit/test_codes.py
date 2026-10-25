import pytest

from landedcost.hts.codes import CodeLevel, HtsCode, InvalidHtsCodeError, normalize_code


@pytest.mark.parametrize(
    ("raw", "dotted", "level"),
    [
        ("8507", "8507", CodeLevel.HEADING),
        ("850760", "8507.60", CodeLevel.SUBHEADING),
        ("8507.60.00", "8507.60.00", CodeLevel.TARIFF_LINE),
        ("8507600090", "8507.60.00.90", CodeLevel.STATISTICAL),
        ("8507.60.0090", "8507.60.00.90", CodeLevel.STATISTICAL),
        (" 8507-60-00-90 ", "8507.60.00.90", CodeLevel.STATISTICAL),
    ],
)
def test_parse_accepts_common_spellings(raw: str, dotted: str, level: CodeLevel) -> None:
    code = HtsCode.parse(raw)
    assert code.dotted == dotted
    assert code.level is level
    assert normalize_code(raw) == dotted


@pytest.mark.parametrize("raw", ["", "85", "85076", "8507.6", "ABCD", "8507600090123", "85O7"])
def test_parse_rejects_invalid_codes(raw: str) -> None:
    with pytest.raises(InvalidHtsCodeError):
        HtsCode.parse(raw)


def test_hierarchy_accessors() -> None:
    code = HtsCode.parse("8507.60.00.90")
    assert code.chapter == "85"
    assert code.heading == "8507"
    assert code.subheading == "8507.60"
    assert HtsCode.parse("8507").subheading is None
    assert code.truncate(CodeLevel.SUBHEADING).dotted == "8507.60"
    assert code.truncate(CodeLevel.TARIFF_LINE).dotted == "8507.60.00"
    assert HtsCode.parse("8507").truncate(CodeLevel.STATISTICAL).dotted == "8507"


def test_is_within() -> None:
    leaf = HtsCode.parse("8507.60.00.90")
    assert leaf.is_within(HtsCode.parse("8507"))
    assert leaf.is_within(leaf)
    assert not leaf.is_within(HtsCode.parse("8506"))
    assert not HtsCode.parse("8507").is_within(leaf)
