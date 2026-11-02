"""Top-level API: from lattice parameters to Hubbard parameters."""

import numpy as np

from .constants import A0, HBAR, HPLANCK, mass_of
from .lattice import harmonic_lengths, recoil_energy, wannier_1d
from .twobody import LatticeSite

BASIS_PRESETS = {
    "fast": dict(ncm=3, lmax=4, rel_emax=8.0),
    "default": dict(ncm=4, lmax=4, rel_emax=9.0),
    "accurate": dict(ncm=5, lmax=6, rel_emax=10.0),
}


def compute_hubbard_parameters(
    lattice_constant_nm=532.0,
    depths_Er=(20.0, 20.0, 20.0),
    scattering_lengths_a0=None,
    species="6Li",
    mass_u=None,
    basis="default",
    ncm=None,
    lmax=None,
    rel_emax=None,
):
    """Compute the Hubbard parameters of a 3D optical lattice.

    Parameters
    ----------
    lattice_constant_nm : float
        Spatial period of the lattice potential, i.e. half the laser
        wavelength (532 nm for a 1064 nm lattice).
    depths_Er : (Vx, Vy, Vz)
        Lattice depths in units of the recoil energy E_R of the lattice photon.
    scattering_lengths_a0 : float or sequence of floats
        Scattering length(s) in Bohr radii.  One on-site interaction energy is
        returned per value.
    species : str
        Atomic species label, e.g. ``"6Li"``, ``"40K"``, ``"87Rb"``.
    mass_u : float, optional
        Alternative to `species`: mass in atomic mass units.
    basis : {"fast", "default", "accurate"}
        Basis cut-off preset for the 6D calculation.

    Returns
    -------
    dict
        Keys (all SI unless the name says otherwise):
        ``lattice_constant_nm``, ``species``, ``mass_u``, ``recoil_Hz``,
        ``omega_Hz``, ``a_ho_a0``, ``t_Hz``, ``t_over_ER``,
        ``interaction`` (list of dicts, one per scattering length).
    """
    preset = dict(BASIS_PRESETS.get(basis, BASIS_PRESETS["default"]))
    if ncm is not None:
        preset["ncm"] = ncm
    if lmax is not None:
        preset["lmax"] = lmax
    if rel_emax is not None:
        preset["rel_emax"] = rel_emax

    mass = mass_of(species=species, mass_u=mass_u)
    depths = tuple(float(v) for v in depths_Er)
    if len(depths) != 3:
        raise ValueError("depths_Er needs exactly three values (x, y, z)")

    a_lat = lattice_constant_nm * 1e-9
    ER, k = recoil_energy(a_lat, mass)
    omega, l_single, l_cm, l_rel = harmonic_lengths(depths, k, mass)

    # ---- single-particle: Wannier functions and hopping -----------------
    wann = [wannier_1d(V, k, mass) for V in depths]
    I4 = np.array([w["I4"] for w in wann])          # int|w_c|^4  [m^-1]
    t = np.array([w["t"] for w in wann])            # hopping      [J]
    A1 = np.sqrt(2 * np.pi) * l_single * I4         # per-direction A factor
    A3 = float(np.prod(A1))                         # 3D enhancement factor

    # ---- two-body: on-site interaction ----------------------------------
    site = LatticeSite(depths, k, mass, **preset)
    if scattering_lengths_a0 is None:
        scattering_lengths_a0 = []
    if np.isscalar(scattering_lengths_a0):
        scattering_lengths_a0 = [scattering_lengths_a0]

    E0 = float(site.spectrum(1e-6)[0][0]) if scattering_lengths_a0 else 0.0

    interactions = []
    for a_a0 in scattering_lengths_a0:
        a = float(a_a0) * A0
        x = a / site.a_ho
        if abs(x) < 1e-9:
            e_gs = E0
        else:
            e_gs = float(site.spectrum(x)[0][0])
        U = (e_gs - E0) * site.hw0
        U_std = 4 * np.pi * HBAR**2 * a / mass * float(np.prod(I4))
        interactions.append(
            dict(
                scattering_length_a0=float(a_a0),
                x_over_a_ho=float(x),
                U_Hz=float(U / HPLANCK),
                U_over_ER=float(U / ER),
                U_standard_Hz=float(U_std / HPLANCK),
                U_standard_over_ER=float(U_std / ER),
                U_over_Ustandard=float(U / U_std) if U_std != 0 else np.nan,
                U_over_t=[float(U / tc) if tc != 0 else np.nan for tc in t],
            )
        )

    return dict(
        lattice_constant_nm=float(lattice_constant_nm),
        wavelength_nm=2 * float(lattice_constant_nm),
        species=species,
        mass_u=float(mass / (1.66053906660e-27)),
        depths_Er=list(depths),
        recoil_Hz=float(ER / HPLANCK),
        recoil_uK=float(ER / 1.380649e-23 * 1e6),
        omega_Hz=[float(o / (2 * np.pi)) for o in omega],
        a_ho_a0=float(site.a_ho / A0),
        t_Hz=[float(v / HPLANCK) for v in t],
        t_over_ER=[float(v / ER) for v in t],
        A_1D=[float(v) for v in A1],
        A3=A3,
        int_w4_m3=[float(v) for v in I4],
        basis=dict(ncm=site.ncm, lmax=site.lmax, rel_emax=site.rel_emax),
        interaction=interactions,
    )
