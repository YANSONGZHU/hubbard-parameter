#!/usr/bin/env python3
"""
Worked examples for the hubbard-parameter package.

Run from the package root:

    python examples/basic_usage.py
"""

import sys

import numpy as np

sys.path.insert(0, ".")

from hubbard_parameter import compute_hubbard_parameters  # noqa: E402


def example_1_basic():
    """6Li in a 1064 nm lattice (period 532 nm), cubic, 20 E_R."""
    print("=" * 72)
    print("Example 1: 6Li, 1064 nm lattice, V0 = 20 E_R (cubic)")
    print("=" * 72)

    res = compute_hubbard_parameters(
        lattice_constant_nm=532.0,
        depths_Er=(20.0, 20.0, 20.0),
        scattering_lengths_a0=[300.0, 600.0, 1000.0, -1900.0],
        species="6Li",
    )

    print(f"  E_R/h = {res['recoil_Hz'] / 1e3:.3f} kHz")
    print(f"  a_ho  = {res['a_ho_a0']:.0f} a0")
    print(f"  t/h   = {res['t_Hz'][0] / 1e3:.4f} kHz")
    print()
    print(f"  {'a [a0]':>9} {'a/a_ho':>8} {'U/h [kHz]':>12} {'U/t':>9} "
          f"{'U/U_std':>9}")
    for it in res["interaction"]:
        print(f"  {it['scattering_length_a0']:9.1f} {it['x_over_a_ho']:8.3f} "
              f"{it['U_Hz'] / 1e3:12.3f} {it['U_over_t'][0]:9.2f} "
              f"{it['U_over_Ustandard']:9.4f}")
    print()
    print("  U/U_std < 1 means that the standard formula")
    print("  U = 4 pi hbar^2 a/m * int|w|^4 overestimates the interaction.\n")


def example_2_anisotropic():
    """Same atom, slightly anisotropic lattice (beam imbalance)."""
    print("=" * 72)
    print("Example 2: anisotropic lattice  V = (20.6, 20.0, 19.4) E_R")
    print("=" * 72)

    res = compute_hubbard_parameters(
        lattice_constant_nm=532.0,
        depths_Er=(20.6, 20.0, 19.4),
        scattering_lengths_a0=500.0,
        species="6Li",
    )
    for c, t in zip("xyz", res["t_Hz"]):
        print(f"  t_{c}/h = {t / 1e3:8.4f} kHz")
    it = res["interaction"][0]
    print(f"  U/h   = {it['U_Hz'] / 1e3:.3f} kHz")
    print()


def example_3_scan():
    """Scan the lattice depth at fixed scattering length."""
    print("=" * 72)
    print("Example 3: U/t versus lattice depth, a = 500 a0")
    print("=" * 72)

    print(f"  {'V0 [E_R]':>9} {'t/h [kHz]':>11} {'U/h [kHz]':>11} {'U/t':>9}")
    for V in (4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0):
        res = compute_hubbard_parameters(
            lattice_constant_nm=532.0,
            depths_Er=(V, V, V),
            scattering_lengths_a0=500.0,
            species="6Li",
            basis="fast",
        )
        it = res["interaction"][0]
        print(f"  {V:9.1f} {res['t_Hz'][0] / 1e3:11.4f} {it['U_Hz'] / 1e3:11.3f} "
              f"{it['U_over_t'][0]:9.2f}")
    print()


def example_4_cli_equivalent():
    """What the CLI does, from Python."""
    print("=" * 72)
    print("Example 4: machine-readable output")
    print("=" * 72)
    import json

    res = compute_hubbard_parameters(
        lattice_constant_nm=532.0,
        depths_Er=(20, 20, 20),
        scattering_lengths_a0=[171.41],
        species="6Li",
    )
    print(json.dumps({k: res[k] for k in ("t_Hz", "a_ho_a0", "recoil_Hz")},
                     indent=2))


if __name__ == "__main__":
    example_1_basic()
    example_2_anisotropic()
    example_3_scan()
    example_4_cli_equivalent()
