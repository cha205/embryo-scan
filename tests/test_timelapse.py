import numpy as np

from embryoscan.timelapse import P, decode_monotonic, direct_cleavage, meseguer_category, morphokinetics, onsets, patient_of


def test_patient_of():
    assert patient_of("ADM715-1-2") == "ADM715"
    assert patient_of("AA83-7") == "AA83"


def test_decode_never_goes_backwards_and_fixes_flicker():
    truth = np.array([0, 0, 1, 1, 1, 3, 3, 3, 4, 4])
    logp = np.full((10, 6), -5.0)
    logp[np.arange(10), truth] = 0.0
    logp[6] = -5.0
    logp[6, 1] = 0.0  # a noisy frame that jumps back to phase 1
    path = decode_monotonic(logp)
    assert (np.diff(path) >= 0).all()
    assert path.tolist() == truth.tolist()


def test_decode_can_skip_a_phase():
    logp = np.log(np.array([[0.9, 0.05, 0.05], [0.05, 0.05, 0.9], [0.05, 0.05, 0.9]]))
    assert decode_monotonic(logp).tolist() == [0, 2, 2]


def test_onsets_and_morphokinetics():
    phase = np.array([P["t2"]] * 3 + [P["t3"]] * 2 + [P["t4"]] * 4 + [P["t5"]] * 2)
    hours = np.arange(len(phase)) * 2.0 + 25
    on = onsets(phase, hours)
    assert on["t2"] == 25 and on["t3"] == 31 and on["t4"] == 35 and on["t5"] == 43
    m = morphokinetics(on)
    assert m["cc2"] == 6 and m["s2"] == 4


def test_meseguer_categories():
    assert meseguer_category({"t5": 52.0, "s2": 0.5, "cc2": 11.0}) == "A+"
    assert meseguer_category({"t5": 52.0, "s2": 2.0, "cc2": 13.0}) == "B-"
    assert meseguer_category({"t5": 60.0, "s2": 0.5, "cc2": 11.0}) == "C+"
    assert meseguer_category({"t5": 45.0, "s2": 3.0, "cc2": 9.0}) == "D+"
    assert meseguer_category({"t5": float("nan"), "s2": 0.5, "cc2": 11.0}) == "C+"  # never reached 5 cells
    assert meseguer_category({"t5": 52.0, "s2": float("nan"), "cc2": 11.0}) == "X"
    assert direct_cleavage({"cc2": 3.0}) and not direct_cleavage({"cc2": 11.0})


def test_skipped_phase_takes_the_next_onset():
    phase = np.array([P["t2"]] * 3 + [P["t4"]] * 3)  # 2 cells straight to 4
    on = onsets(phase, np.arange(6.0))
    assert on["t3"] == on["t4"] == 3.0 and "t5" not in on
