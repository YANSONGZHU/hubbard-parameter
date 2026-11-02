"""Physical constants and atomic masses (CODATA 2018 / AME 2020)."""

import numpy as np

HBAR = 1.054571817e-34          # J s
HPLANCK = 6.62607015e-34        # J s
M_U = 1.66053906660e-27         # kg
A0 = 5.29177210903e-11          # m
KB = 1.380649e-23               # J / K

#: neutral-atom masses in atomic mass units
ATOMIC_MASSES = {
    "1H": 1.00782503207,
    "6Li": 6.015122795,
    "7Li": 7.01600455,
    "23Na": 22.9897692809,
    "39K": 38.96370668,
    "40K": 39.96399848,
    "41K": 40.96182576,
    "85Rb": 84.911789738,
    "87Rb": 86.909180527,
    "133Cs": 132.905451933,
    "87Sr": 86.9088774970,
    "173Yb": 172.9382108,
    "174Yb": 173.9388621,
}


def mass_of(species=None, mass_u=None):
    """Mass in kg from a species label ("6Li") or a mass in atomic mass units."""
    if mass_u is not None:
        return float(mass_u) * M_U
    if species is None:
        raise ValueError("give either `species` or `mass_u`")
    key = str(species).strip()
    if key in ATOMIC_MASSES:
        return ATOMIC_MASSES[key] * M_U
    # tolerate "Li6" / "li6" / "6li"
    norm = key.replace(" ", "")
    for k, v in ATOMIC_MASSES.items():
        if norm.lower() in (k.lower(), (k[::-1]).lower()):
            return v * M_U
    raise KeyError(f"unknown species {species!r}; known: {sorted(ATOMIC_MASSES)}")
