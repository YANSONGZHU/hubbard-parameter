"""Regression and physics tests for hubbard-parameter.

Two ways to run them:

    pytest -q                      # if pytest is installed
    python tests/test_hubbard.py   # dependency-free fallback runner
"""

import numpy as np

from hubbard_parameter import compute_hubbard_parameters, mass_of
from hubbard_parameter.constants import HPLANCK, A0
from hubbard_parameter.lattice import recoil_energy

LI6_1064 = dict(species="6Li", lattice_constant_nm=532.0)


def test_recoil_energy_6li_1064nm():
    """E_R/h = 29.30 kHz and hbar*omega = 262.1 kHz at 20 E_R."""
    m = mass_of(species="6Li")
    ER, _k = recoil_energy(532e-9, m)
    assert abs(ER / HPLANCK - 29298.8) < 5.0
    assert abs(2 * np.sqrt(20.0) * ER / HPLANCK - 262060) < 100


def test_hopping_and_lengths_at_20Er():
    """t/h = 72.8 Hz and a_ho = 2140 a0 for 6Li at V0 = 20 E_R."""
    r = compute_hubbard_parameters(depths_Er=(20, 20, 20), **LI6_1064)
    assert abs(r["t_Hz"][0] - 72.8) < 0.5
    assert abs(r["a_ho_a0"] - 2140) < 3


def test_isotropic_lattice_gives_equal_hopping():
    r = compute_hubbard_parameters(depths_Er=(20, 20, 20), **LI6_1064)
    assert np.allclose(r["t_Hz"], r["t_Hz"][0], rtol=1e-10)


def test_deeper_lattice_means_less_hopping():
    r = compute_hubbard_parameters(depths_Er=(10, 20, 30), **LI6_1064)
    tx, ty, tz = r["t_Hz"]
    assert tx > ty > tz


def test_small_scattering_length_recovers_first_order():
    """U must approach the standard (delta + lowest band) result as a -> 0."""
    r = compute_hubbard_parameters(depths_Er=(20, 20, 20),
                                   scattering_lengths_a0=5.0, **LI6_1064)
    assert abs(r["interaction"][0]["U_over_Ustandard"] - 1.0) < 2e-3


def test_strong_scattering_length_is_suppressed():
    """For |a| well beyond a_ho the exact U grows *sublinearly* in a.

    (Note U/a is not monotonic everywhere: the exact curve has a shallow
    maximum near x = a/a_ho ~ 0.15, which is a genuine feature, not noise.)
    """
    r = compute_hubbard_parameters(
        depths_Er=(20, 20, 20),
        scattering_lengths_a0=[400.0, 800.0, 1200.0], **LI6_1064)
    U = [it["U_Hz"] for it in r["interaction"]]
    ratio = [u / a for u, a in zip(U, [400.0, 800.0, 1200.0])]
    assert all(np.diff(ratio) < 0), "U/a must decrease for large |a|"
    assert U[1] < 2.0 * U[0] and U[2] < 3.0 * U[0]


def test_regression_values_at_20Er():
    """Fixed reference numbers, validated against an experimental data set."""
    r = compute_hubbard_parameters(
        depths_Er=(20, 20, 20),
        scattering_lengths_a0=[171.41, -738.04], **LI6_1064)
    assert abs(r["interaction"][0]["U_Hz"] - 20249) < 5
    assert abs(r["interaction"][1]["U_Hz"] + 71907) < 10


def test_anisotropy_is_weak_perturbation():
    """A 3% depth anisotropy leaves U almost unchanged but splits t strongly."""
    a_iso = compute_hubbard_parameters(depths_Er=(20, 20, 20),
                                       scattering_lengths_a0=-500.0, **LI6_1064)
    a_ani = compute_hubbard_parameters(depths_Er=(20.6, 20.0, 19.4),
                                       scattering_lengths_a0=-500.0, **LI6_1064)
    u_iso = a_iso["interaction"][0]["U_Hz"]
    u_ani = a_ani["interaction"][0]["U_Hz"]
    assert abs(u_ani / u_iso - 1) < 0.005
    assert a_ani["t_Hz"][0] < a_ani["t_Hz"][1] < a_ani["t_Hz"][2]


def test_scalar_scattering_length_accepted():
    r = compute_hubbard_parameters(depths_Er=(20, 20, 20),
                                   scattering_lengths_a0=300.0, **LI6_1064)
    assert len(r["interaction"]) == 1
    assert r["interaction"][0]["scattering_length_a0"] == 300.0


def test_unknown_species_raises():
    try:
        compute_hubbard_parameters(species="Unobtainium", depths_Er=(20, 20, 20))
    except KeyError:
        return
    raise AssertionError("expected a KeyError for an unknown species")


if __name__ == "__main__":
    import sys
    import time

    failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        t0 = time.time()
        try:
            fn()
            print(f"  PASS  {name}   ({time.time() - t0:.1f} s)")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}   {exc}")
    print(f"\n{failed} failure(s)")
    sys.exit(1 if failed else 0)
