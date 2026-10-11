import math

import pytest

from embryoscan.blastocyst.data import decode_amh, decode_count, decode_endo, gardner_label, image_key


def test_image_key_ignores_padding():
    assert image_key("187_1.png") == image_key("0187_01.png") == (187, 1)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Jän.26", 1.26),  # "1.26": month 26 invalid, read as January 1926
        ("Jul.13", 7.13),
        ("Dez.54", 12.54),
        ("02.Nov", 2.11),  # "2.11": 2 November
        ("05.Dez", 5.12),
        ("0.72", 0.72),  # day 0 is invalid, so the number survived
        ("16.73", 16.73),
        ("2", 2.0),
    ],
)
def test_amh_unambiguous(raw, expected):
    d = decode_amh(raw)
    assert d.value == pytest.approx(expected)
    assert not d.ambiguous


def test_amh_day_month_is_ambiguous():
    d = decode_amh("01.Apr")  # "1.4" or "1.04"
    assert (d.lo, d.hi) == pytest.approx((1.04, 1.4))
    assert d.value == pytest.approx(1.22)
    assert d.ambiguous


def test_amh_missing():
    d = decode_amh("")
    assert math.isnan(d.value) and d.how == "missing" and not d.ambiguous


def test_endo():
    assert decode_endo("08.Mai").value == pytest.approx(8.5)
    assert decode_endo("11").value == 11.0
    assert math.isnan(decode_endo("").value)


def test_counts():
    assert decode_count("10+1") == 11
    assert decode_count("12+5") == 17
    assert decode_count("7") == 7
    assert math.isnan(decode_count(""))


def test_gardner_label():
    assert gardner_label(3, 0, 1) == "4AB"
    assert gardner_label(1, 3, 3) == "2"
    assert gardner_label(float("nan"), 0, 0) == "n/a"
