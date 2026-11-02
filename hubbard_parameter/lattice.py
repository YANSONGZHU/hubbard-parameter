"""
One-dimensional lattice problem: band structure, Wannier functions, tunneling.

The 3D lattice produced by three orthogonal standing waves,

    V(r) = sum_c V_c sin^2(k x_c),      k = pi / a_lat,

is *separable*, so every single-particle quantity factorises into three 1D
problems.  In particular the Wannier function is a product,
w(r) = wx(x) wy(y) wz(z), and the hopping amplitude along c is the 1D
tight-binding hopping of the c-th 1D lattice (the overlap in the two
transverse directions is exactly unity).
"""

import numpy as np

from .constants import HBAR


def recoil_energy(lattice_constant_m, mass):
    """Recoil energy of the *lattice photon*, E_R = hbar^2 k^2 / (2 m).

    `lattice_constant` is the period of the lattice potential, i.e. half the
    laser wavelength:  a_lat = lambda/2  ->  k = pi/a_lat.
    """
    k = np.pi / lattice_constant_m
    return HBAR**2 * k**2 / (2 * mass), k


def _lowest_band(V_Er, k, mass, nPW=30, Nk=41):
    """Lowest-band Bloch states |u_k> in a plane-wave basis G = 2 pi n / a_lat.

    Returns (cvecs, evals, G).  V_Er is the depth in units of E_R.
    """
    ER = HBAR**2 * k**2 / (2 * mass)
    V0 = V_Er * ER
    a_lat = np.pi / k
    n = np.arange(-nPW, nPW + 1)
    G = n * 2 * np.pi / a_lat
    T = HBAR**2 * G**2 / (2 * mass)
    # V = V0 sin^2(kx) = V0/2 - (V0/4)(e^{2ikx} + e^{-2ikx})
    H = np.diag(T + V0 / 2)
    idx = np.arange(len(G))
    H[idx[:-1], idx[1:]] += -V0 / 4
    H[idx[1:], idx[:-1]] += -V0 / 4
    H = 0.5 * (H + H.T)

    ks = -np.pi / a_lat + 2 * np.pi * np.arange(Nk) / (Nk * a_lat)
    cvecs, evals = [], []
    for kk in ks:
        Hb = H.copy()
        Hb[np.diag_indices(len(G))] += HBAR**2 * (2 * kk * G + kk**2) / (2 * mass)
        e, v = np.linalg.eigh(Hb)
        cvecs.append(v[:, 0])
        evals.append(e[0])
    return np.array(cvecs), np.array(evals), G


def wannier_1d(V_Er, k, mass, nPW=30, Nk=31, nx_per_cell=2128):
    """Maximally localised (Kohn) Wannier function of the lowest band.

    Returns a dict with
        I4        : int |w(x)|^4 dx   [m^-1]
        t         : nearest-neighbour hopping [J]
        bandwidth : lowest-band width [J]
        band_bottom : energy of the lowest band minimum, measured from V=0 [J]
        x, w      : the Wannier function on the supercell grid
    """
    ER = HBAR**2 * k**2 / (2 * mass)
    a_lat = np.pi / k
    cvecs, evals, G = _lowest_band(V_Er, k, mass, nPW=nPW, Nk=Nk)
    L = Nk * a_lat

    # smooth (parallel-transport) gauge along the k-sampling
    phases = np.zeros(Nk)
    ks = -np.pi / a_lat + 2 * np.pi * np.arange(Nk) / L
    for j in range(1, Nk):
        phases[j] = phases[j - 1] - np.angle(np.vdot(cvecs[j - 1], cvecs[j]))

    nx = nx_per_cell * Nk
    x = np.arange(nx) * (L / nx)
    dx = L / nx
    w = np.zeros(nx, dtype=complex)
    for j, kk in enumerate(ks):
        pw = np.exp(1j * (kk + G)[:, None] * x[None, :]) / np.sqrt(L)
        w += np.exp(1j * phases[j]) * (cvecs[j] @ pw)
    w /= np.sqrt(Nk)
    w /= np.sqrt(np.sum(np.abs(w) ** 2) * dx)
    w = np.roll(w, int(round((L / 2 - x[np.argmax(np.abs(w) ** 2)]) / dx)))

    I4 = float(np.sum(np.abs(w) ** 4) * dx)
    t = float((evals.max() - evals.min()) / 4.0)
    return dict(I4=I4, t=t, bandwidth=float(np.ptp(evals)),
                band_bottom=float(evals.min()), x=x, w=w)


def harmonic_lengths(V_Er, k, mass):
    """Harmonic-oscillator lengths of one lattice site.

    omega_c = k sqrt(2 V_c / m),  l_c = sqrt(hbar/(m omega_c)).
    Returns (omega, l_single, l_cm, l_rel):  l_cm = l/sqrt2, l_rel = 2 l_cm.
    """
    ER = HBAR**2 * k**2 / (2 * mass)
    V = np.asarray(V_Er, dtype=float) * ER
    omega = k * np.sqrt(2 * V / mass)
    l_single = np.sqrt(HBAR / (mass * omega))
    l_cm = l_single / np.sqrt(2.0)
    l_rel = 2.0 * l_cm
    return omega, l_single, l_cm, l_rel
