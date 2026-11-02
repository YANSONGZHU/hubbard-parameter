"""
hubbard-parameter
=================

Ab-initio tunnelling and on-site interaction parameters of the (Fermi-)
Hubbard model for ultracold atoms in a three-dimensional optical lattice,
computed from the *exact* six-dimensional two-body problem.

    >>> from hubbard_parameter import compute_hubbard_parameters
    >>> r = compute_hubbard_parameters(depths_Er=(20,20,20),
    ...                                scattering_lengths_a0=300)
    >>> r["t_Hz"], r["interaction"][0]["U_Hz"]

MIT licensed.  See README.md for the method and references.
"""

from .constants import ATOMIC_MASSES, A0, HBAR, HPLANCK, mass_of
from .core import BASIS_PRESETS, compute_hubbard_parameters
from .lattice import harmonic_lengths, recoil_energy, wannier_1d
from .twobody import LatticeSite

__version__ = "1.0.0"

__all__ = [
    "compute_hubbard_parameters",
    "BASIS_PRESETS",
    "LatticeSite",
    "wannier_1d",
    "harmonic_lengths",
    "recoil_energy",
    "mass_of",
    "ATOMIC_MASSES",
    "__version__",
]
