"""
Exact six-dimensional two-body problem for two atoms in one lattice site.

This is the core of the package.  Two atoms of mass m interact through a
zero-range s-wave contact of scattering length a and are trapped in a single
site of an orthorhombic optical lattice,

    V(r) = sum_c V_c sin^2(k x_c),     k = pi / a_lat.

Strategy
--------
The exact potential is written in centre-of-mass / relative coordinates.
With  X_c = (x1c + x2c)/2  and  r_c = x1c - x2c,

    V_c sin^2(k(X_c + r_c/2)) + V_c sin^2(k(X_c - r_c/2))
        = V_c [ 1 - cos(2 k X_c) cos(k r_c) ],

which is *exact* (no Taylor expansion).  The two-body Hamiltonian is then
expanded in the eigenbasis of the corresponding **harmonic** problem,

    H_ref = T_cm + T_rel + V_harm + contact,

because that basis is known analytically: the centre of mass is a 3D harmonic
oscillator, and for l = 0 the relative motion is the exact contact-interacting
state of a 3D harmonic trap (Busch states), while l >= 2 states are untouched
by the contact (the pseudopotential acts only in the s-wave).  The zero-range
interaction is therefore *built in* and needs no regularisation.

The residual  W = V_exact - V_harm  is then diagonalised in this basis
(configuration interaction).  Because V_exact is bounded, the truncated CI is
variational.

For an anisotropic lattice a single reference frequency w0 (geometric mean) is
used for the basis; the per-direction depths enter the CI matrix through the
coefficients V_c / (hbar w0).  For a cubic lattice the basis is exact.

Units inside this module: energies in hbar*w0, lengths in the reference
oscillator lengths, and x = a / a_ho with a_ho = 2 l_cm (relative-motion
length of two equal masses).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import sph_harm_y

from .constants import HBAR

# radial grid used for all relative-motion functions
_YMAX, _NY = 15.0, 2500
_YGRID = np.linspace(1e-7, _YMAX, _NY)

# angular quadrature
_NQ, _NPHI = 120, 120


# --------------------------------------------------------------------------
#                          1D harmonic oscillator
# --------------------------------------------------------------------------
def ho_x2(nmax):
    """<n|x^2|m> in oscillator units (length = 1)."""
    M = np.zeros((nmax + 1, nmax + 1))
    for n in range(nmax + 1):
        M[n, n] = n + 0.5
        if n + 2 <= nmax:
            v = 0.5 * np.sqrt((n + 1) * (n + 2))
            M[n, n + 2] = M[n + 2, n] = v
    return M


def ho_x4(nmax):
    return ho_x2(nmax) @ ho_x2(nmax)


def ho_exp_matrix(nmax, lam):
    """<m|exp(i lam x)|n> for 1D oscillator states (real phi -> symmetric)."""
    from math import factorial

    from scipy.special import eval_genlaguerre

    M = np.zeros((nmax + 1, nmax + 1), dtype=complex)
    for m in range(nmax + 1):
        for n in range(m + 1):
            v = (
                (1j) ** (n - m)
                * np.sqrt(factorial(n) / factorial(m))
                * (lam / np.sqrt(2)) ** (m - n)
                * np.exp(-lam**2 / 4)
                * eval_genlaguerre(n, m - n, lam**2 / 2)
            )
            M[m, n] = v
            M[n, m] = v
    return M


# --------------------------------------------------------------------------
#                        relative-motion radial functions
# --------------------------------------------------------------------------
def busch_eps(x, nmax=5):
    """eps_k = 2E/(hbar w) of the l=0 relative states, from the exact relation

        x = (1/2) Gamma(1/4 - nu/2) / Gamma(3/4 - nu/2),   nu = E/(hbar w).

    k = 0 : ground state.  Negative x -> branch nu in (2k+1/2, 2k+3/2);
    positive x -> nu in (2k+3/2, 2k+5/2) (the scattering branch; the
    molecular branch is deliberately excluded).
    """
    from scipy.special import gamma as G

    def fun(nu):
        return 0.5 * G(0.25 - nu / 2) / G(0.75 - nu / 2) - x

    out = []
    for k in range(nmax + 1):
        lo, hi = (2 * k + 0.5, 2 * k + 1.5) if x < 0 else (2 * k + 1.5, 2 * k + 2.5)
        out.append(2.0 * brentq(fun, lo + 1e-9, hi - 1e-9, xtol=1e-13, rtol=1e-14))
    return out


def busch_radial(x, eps, y=None):
    """u(y) = y R(y) of an l=0 relative state, normalised to 1.

    Integrated inwards from y_max with the decaying asymptotic, because
    outward integration of a decaying solution is exponentially unstable.
    u = exp(-y^2/2) v keeps the integrator well conditioned.
    """
    y = _YGRID if y is None else y
    ym = y[-1]
    nu = 0.5 * (eps - 1.0)
    sol = solve_ivp(
        lambda yy, u: [u[1], (yy * yy - eps) * u[0]],
        [ym, 1e-7],
        [1.0, -ym + nu / ym],
        t_eval=y[::-1],
        rtol=1e-10,
        atol=1e-14,
        method="DOP853",
    )
    u = sol.y[0][::-1]
    u /= np.sqrt(np.trapezoid(u * u, y))
    return y, u


def ho_radial(n, l, y=None):
    """u(y) of a 3D oscillator state (n,l), normalised."""
    from scipy.special import eval_genlaguerre, gammaln

    y = _YGRID if y is None else y
    p = np.sqrt(2.0 * np.exp(gammaln(n + 1) - gammaln(n + l + 1.5)))
    u = p * y ** (l + 1) * np.exp(-y * y / 2) * eval_genlaguerre(n, l + 0.5, y * y)
    u /= np.sqrt(np.trapezoid(u * u, y))
    return y, u


# --------------------------------------------------------------------------
#                            angular matrix elements
# --------------------------------------------------------------------------
def _angular_setup(lmax):
    xc, wq = np.polynomial.legendre.leggauss(_NQ)
    theta = np.arccos(xc)
    phi = 2 * np.pi * np.arange(_NPHI) / _NPHI
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    W = np.outer(wq, 2 * np.pi / _NPHI)
    lab, Y = [], []
    for l in range(lmax + 1):
        for mm in range(-l, l + 1):
            lab.append((l, mm))
            Y.append(sph_harm_y(l, mm, TH, PH))
    return lab, np.array(Y), W, np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)


def _ang_int(Y, Wf):
    Yf = Y.reshape(len(Y), -1)
    return (np.conj(Yf) * Wf.ravel()) @ Yf.T


def angular_matrices(lmax=4):
    """C[n][axis][i,j] = <l m| cos^n(theta_axis) |l' m'> for n = 2, 4."""
    lab, Y, W, cx, cy, cz = _angular_setup(lmax)
    out = {}
    for n in (2, 4):
        out[n] = []
        for f in (cx, cy, cz):
            out[n].append(_ang_int(Y, W * f**n))
    return out, lab


def legendre_matrices(lmax=4, jmax=8):
    """Pl[j][axis][i,j] = <l m| P_j(cos theta_axis) |l' m'> for even j."""
    from numpy.polynomial.legendre import legval

    lab, Y, W, cx, cy, cz = _angular_setup(lmax)
    out = {}
    for j in range(0, jmax + 1, 2):
        out[j] = []
        for f in (cx, cy, cz):
            out[j].append(_ang_int(Y, W * legval(f, [0] * j + [1])))
    return out, lab


# --------------------------------------------------------------------------
#                                  site
# --------------------------------------------------------------------------
class LatticeSite:
    """Harmonic-reference basis for one lattice site.

    Parameters
    ----------
    depths_Er : sequence of three floats
        Lattice depths along x, y, z in units of the recoil energy E_R.
    k : float
        Lattice wave vector, k = pi / a_lat  [m^-1].
    mass : float
        Atomic mass [kg].
    ncm, lmax, rel_emax : basis cut-offs
        Maximum centre-of-mass quanta, maximum relative angular momentum and
        maximum relative-motion energy (in units of hbar w0).
    """

    def __init__(self, depths_Er, k, mass, ncm=4, lmax=4, rel_emax=9.0):
        depths = np.asarray(depths_Er, dtype=float)
        if depths.shape != (3,):
            raise ValueError("depths_Er must have three entries (x, y, z)")
        self.k = float(k)
        self.mass = float(mass)
        self.depths = depths
        self.ER = HBAR**2 * k**2 / (2 * mass)

        self.omega = k * np.sqrt(2 * depths * self.ER / mass)      # per direction
        self.omega0 = float(np.prod(self.omega) ** (1.0 / 3.0))     # reference
        self.hw0 = HBAR * self.omega0
        self.lcm = np.sqrt(HBAR / (2 * mass * self.omega0))         # cm length
        self.lrel = 2.0 * self.lcm                                 # relative length
        self.a_ho = self.lrel                                       # = 2 l_cm

        self.ncm = int(ncm)
        self.lmax = int(lmax)
        self.rel_emax = float(rel_emax)

        self.x2 = ho_x2(self.ncm)
        self.x4 = ho_x4(self.ncm)
        self.cm_states = [
            (a, b, c)
            for a in range(self.ncm + 1)
            for b in range(self.ncm + 1)
            for c in range(self.ncm + 1)
            if a + b + c <= self.ncm
        ]
        self.cm_index = {s: i for i, s in enumerate(self.cm_states)}

        self.Cang, self.lab = angular_matrices(self.lmax)
        self.lab_index = {s: i for i, s in enumerate(self.lab)}

    # ---- one-particle hopping ------------------------------------------
    @property
    def hopping(self):
        """1D tight-binding hopping along each direction [J]."""
        from .lattice import wannier_1d

        return np.array(
            [wannier_1d(V, self.k, self.mass)["t"] for V in self.depths]
        )

    # ---- relative-motion spectrum --------------------------------------
    def relative_states(self, x, jmax=8):
        """Relative-motion basis states at x = a/a_ho.

        Returns a list of dicts with l, m, n, E (hbar w0 units), y, u.
        """
        from scipy.special import spherical_jn  # noqa: F401  (documentation)

        st = []
        for nu, eps in enumerate(busch_eps(x, nmax=4)):
            if eps / 2.0 > self.rel_emax:
                break
            y, u = busch_radial(x, eps)
            st.append(dict(l=0, m=0, n=nu, E=eps / 2.0, y=y, u=u))
        for l in range(2, self.lmax + 1, 2):
            n = 0
            while 2 * n + l + 1.5 <= self.rel_emax:
                y, u = ho_radial(n, l)
                for mm in range(-l, l + 1):
                    st.append(dict(l=l, m=mm, n=n, E=2 * n + l + 1.5, y=y, u=u))
                n += 1
        return st

    # ---- operator helpers ----------------------------------------------
    def _cm_op(self, power):
        M = self.x4 if power == 4 else self.x2
        nc = len(self.cm_states)
        ops = [np.zeros((nc, nc)) for _ in range(3)]
        for i, st in enumerate(self.cm_states):
            for ci in range(3):
                for n2 in range(self.ncm + 1):
                    v = M[n2, st[ci]]
                    if abs(v) < 1e-14:
                        continue
                    m = list(st)
                    m[ci] = n2
                    if tuple(m) in self.cm_index:
                        ops[ci][self.cm_index[tuple(m)], i] += v
        return ops

    def _cm_op_from(self, M1):
        nc = len(self.cm_states)
        ops = [np.zeros((nc, nc), dtype=complex) for _ in range(3)]
        for i, st in enumerate(self.cm_states):
            for ci in range(3):
                for n2 in range(self.ncm + 1):
                    v = M1[n2, st[ci]]
                    if abs(v) < 1e-14:
                        continue
                    m = list(st)
                    m[ci] = n2
                    if tuple(m) in self.cm_index:
                        ops[ci][self.cm_index[tuple(m)], i] += v
        return ops

    def _rel_ops(self, rel, power):
        nr = len(rel)
        ops = [np.zeros((nr, nr), dtype=complex) for _ in range(3)]
        Rr = np.zeros((nr, nr))
        for i, a in enumerate(rel):
            for j, b in enumerate(rel):
                Rr[i, j] = np.trapezoid(a["u"] * b["u"] * a["y"] ** power, a["y"])
        for i, a in enumerate(rel):
            for j, b in enumerate(rel):
                if abs(Rr[i, j]) < 1e-13:
                    continue
                for ci in range(3):
                    ops[ci][i, j] = (
                        Rr[i, j]
                        * self.Cang[power][ci][
                            self.lab_index[(a["l"], a["m"])],
                            self.lab_index[(b["l"], b["m"])],
                        ]
                    )
        return ops

    # ---- the 6D Hamiltonian --------------------------------------------
    def spectrum(self, x, nstates=1, jmax=8):
        """Lowest eigenvalues (in units of hbar w0) of the full 6D problem."""
        from scipy.special import spherical_jn

        rel = self.relative_states(x)
        nc, nr = len(self.cm_states), len(rel)
        Ecm = np.array([sum(s) + 1.5 for s in self.cm_states])
        Erel = np.array([r["E"] for r in rel])
        dim = nc * nr
        H = np.diag(np.repeat(Ecm, nr) + np.tile(Erel, nc)).astype(complex)

        lam = 2.0 * self.k * self.lcm                 # 2k l_cm  = k l_rel
        M = ho_exp_matrix(self.ncm, lam)
        C1 = (M + np.conj(M).T) / 2.0                  # cos(2k X_c)
        Ccm = self._cm_op_from(C1)

        # relative: cos(k r_c) through the Rayleigh expansion
        Pl, _lab2 = legendre_matrices(self.lmax, jmax=jmax)
        eta = _YGRID
        uarr = np.array([r["u"] for r in rel])
        labi = [self.lab_index[(r["l"], r["m"])] for r in rel]
        Rc = [np.zeros((nr, nr), dtype=complex) for _ in range(3)]
        for jj in range(0, jmax + 1, 2):
            bj = spherical_jn(jj, lam * eta)
            Ij = np.trapezoid(uarr[:, None, :] * uarr[None, :, :] * bj, eta, axis=-1)
            fac = (2 * jj + 1) * ((-1) ** (jj // 2))
            for ci in range(3):
                Rc[ci] += fac * Ij * Pl[jj][ci][np.ix_(labi, labi)]

        A2 = self._cm_op(2)
        B2 = self._rel_ops(rel, 2)
        I_cm, I_rel = np.eye(nc), np.eye(nr)
        vhw = self.depths * self.ER / self.hw0          # V_c / (hbar w0)
        for ci in range(3):
            H += vhw[ci] * (np.kron(I_cm, I_rel) - np.kron(Ccm[ci], Rc[ci]))
            H -= 0.5 * np.kron(A2[ci], I_rel)
            H -= 0.5 * np.kron(I_cm, B2[ci])
        H = 0.5 * (H + H.conj().T)
        w = np.linalg.eigvalsh(H)
        return w[:nstates], rel

    # ---- on-site interaction -------------------------------------------
    def interaction_energy(self, scattering_length_m):
        """On-site interaction U for a pair with the given scattering length.

        U = E_2(a) - 2 E_1, computed as  hbar*w0 * [E_gs(a) - E_gs(0)]
        (the non-interacting reference cancels exactly).  Returns J.
        """
        x = scattering_length_m / self.a_ho
        if abs(x) < 1e-9:
            return 0.0
        w, _ = self.spectrum(x)
        w0, _ = self.spectrum(1e-6 * np.sign(x) if x != 0 else 1e-6)
        return float(w[0] - w0[0]) * self.hw0
