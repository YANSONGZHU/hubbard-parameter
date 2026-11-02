# hubbard-parameter

**Ab-initio tunnelling and on-site interaction parameters of the Hubbard model
for ultracold atoms in a three-dimensional optical lattice.**

Given the lattice geometry, the atomic species and the scattering length, the
package returns

* the tunnelling amplitudes $t_x$, $t_y$, $t_z$, and
* the on-site interaction energy $U$

computed from the **exact six-dimensional two-body problem** — no
$\delta$-pseudopotential plus lowest-Bloch-band approximation.

MIT licensed.

> Generated with **DeepSeek Harness + DeepSeek V4.1 Flash**

---

## Why another Hubbard-parameter code?

The textbook formula

$$
U = \frac{4\pi\hbar^{2}a}{m}\int \left|w(\mathbf{r})\right|^{4} d^{3}r
$$

replaces the atom–atom interaction by a zero-range pseudopotential and restricts
the Hilbert space to the lowest Bloch band. It is only accurate when
$|a| \ll a_{ho}$, where $a_{ho}$ is the on-site oscillator length. In practice
$a_{ho}$ is between 2000 and 3000 $a_0$ for a 1064 nm lattice, while useful
scattering lengths are 300 to 2000 $a_0$: the standard formula is then off by
**a few per cent at best, and by more than 25 % near a Feshbach resonance**.

This package removes both approximations:

| approximation | standard formula | this package |
|---|---|---|
| interaction | zero-range $\delta$ | exact zero-range contact, built into the basis (no regularisation) |
| trap | harmonic at the well bottom | exact $\sin^{2}$ lattice potential |
| Hilbert space | lowest Bloch band | full 6D, arbitrary band mixing |
| c.m.–relative coupling | neglected | included exactly |

---

## Method

Two atoms in one lattice site, interacting by a zero-range s-wave contact:

$$
H = \frac{\mathbf{p}_{1}^{2}}{2m}+\frac{\mathbf{p}_{2}^{2}}{2m}+V(\mathbf{r}_{1})+V(\mathbf{r}_{2})+\text{contact},
$$

$$
V(\mathbf{r}) = \sum_{c=x,y,z} V_{c} \sin^{2}\left(k x_{c}\right), \qquad k=\frac{\pi}{a_{\text{lat}}}.
$$

### Step 1 — centre-of-mass and relative coordinates

With $X_{c}=(x_{1c}+x_{2c})/2$ and $r_{c}=x_{1c}-x_{2c}$ the two-body potential
is **exactly**

$$
V_{c}\sin^{2}\left(k\left(X_{c}+\frac{r_{c}}{2}\right)\right) + V_{c}\sin^{2}\left(k\left(X_{c}-\frac{r_{c}}{2}\right)\right) = V_{c}\left[1-\cos\left(2kX_{c}\right)\cos\left(kr_{c}\right)\right].
$$

### Step 2 — harmonic reference basis

The basis is the exact eigenbasis of the harmonic part:

* centre of mass — 3D harmonic oscillator states;
* relative motion, $l=0$ — the exact contact-interacting states of a harmonic
  trap (Busch states), obtained from the closed-form relation

$$
\frac{a}{a_{ho}} = \frac{1}{2} \frac{\Gamma\left(\frac{1}{4}-\frac{E}{2\hbar\omega}\right)}{\Gamma\left(\frac{3}{4}-\frac{E}{2\hbar\omega}\right)};
$$

* relative motion, $l \ge 2$ — ordinary oscillator states (the contact does not
  act in higher partial waves).

The zero-range interaction is therefore **built into the basis** and needs no
regularisation.

### Step 3 — configuration interaction

The residual $W = V_{\text{exact}} - V_{\text{harm}}$ is diagonalised in that
basis. $V_{\text{exact}}$ is bounded, so the truncated CI is variational.

### Step 4 — hopping

The hopping is obtained from the exact 1D Bloch band of each direction; the 3D
lattice is separable, so $t_{c}$ is the 1D tight-binding hopping and the
transverse overlaps are exactly unity.

### Step 5 — on-site interaction

$$
U = E_{2}(a) - 2E_{1} = \hbar\omega_{0}\left[E_{gs}(a)-E_{gs}(0)\right],
$$

so the non-interacting reference cancels exactly.

For an anisotropic lattice a single reference frequency (geometric mean) is used
for the basis; the per-direction depths enter the CI matrix directly.

---

## Installation

```bash
git clone https://github.com/YANSONGZHU/hubbard-parameter.git
cd hubbard-parameter
pip install -e .            # or simply add the folder to PYTHONPATH
```

Requires Python ≥ 3.9, NumPy ≥ 1.22, SciPy ≥ 1.8.

---

## Quick start

### Python

```python
from hubbard_parameter import compute_hubbard_parameters

res = compute_hubbard_parameters(
    lattice_constant_nm=532.0,      # lattice period = lambda/2
    depths_Er=(20.0, 20.0, 20.0),   # Vx, Vy, Vz in recoil energies
    scattering_lengths_a0=[300.0, 1000.0],
    species="6Li",
)

res["t_Hz"]                          # [72.8, 72.8, 72.8]  Hz
res["interaction"][0]["U_Hz"]        # U for a = 300 a0
res["interaction"][0]["U_over_t"]    # U/t along x, y, z
```

### Command line

```bash
python -m hubbard_parameter.cli \
    --species 6Li --lattice-constant 532 \
    --depth 20 20 20 --scattering-length 171.41 -738.04
```

```
====================================================================================
  species            : 6Li   (m = 6.0151 u)
  lattice constant   : 532.0 nm (lambda = 1064.0 nm)
  depths (x,y,z)     : 20.00, 20.00, 20.00  E_R
  E_R                : 29.299 kHz  (1.406 uK)
------------------------------------------------------------------------------------
  tunneling
    t_x = t_y = t_z = 0.0728 kHz
------------------------------------------------------------------------------------
  on-site interaction
        a [a0]      U [kHz]   U_std [kHz]        U/t
        171.41      20.2490       19.9108    278.118
       -738.04     -71.9069      -85.7300   -987.634
====================================================================================
```

Add `--json out.json` to also dump every intermediate quantity.

---

## Input conventions

| quantity | unit | note |
|---|---|---|
| `lattice_constant_nm` | nm | **period of the lattice potential**, equal to $\lambda/2$ (532 nm for a 1064 nm laser) |
| `depths_Er` | $E_{R}$ | $E_{R} = h^{2}/(2m\lambda^{2}) = h^{2}/(8 m a_{\text{lat}}^{2})$ |
| `scattering_lengths_a0` | Bohr radii | one $U$ is returned per value |
| `species` | string | `6Li`, `7Li`, `23Na`, `39K`, `40K`, `41K`, `85Rb`, `87Rb`, `133Cs`, `87Sr`, `173Yb`, `174Yb`; or pass `mass_u=` |
| `basis` | `fast` / `default` / `accurate` | CI cut-offs, see below |

Sign convention: $U>0$ for repulsive interactions, $t>0$ by definition, with

$$
H = -t\sum_{\langle ij\rangle\sigma} c_{i\sigma}^{\dagger}c_{j\sigma} + U\sum_{i} n_{i\uparrow}n_{i\downarrow}.
$$

## Output fields

| key | meaning |
|---|---|
| `recoil_Hz`, `recoil_uK` | recoil energy of the lattice photon |
| `omega_Hz` | site trap frequency per direction |
| `a_ho_a0` | oscillator length of the *relative* motion, $a_{ho}=\sqrt{2\hbar/(m\omega)}$ |
| `t_Hz`, `t_over_ER` | hopping amplitude per direction |
| `A_1D`, `A3` | Wannier anharmonicity factors $A_{c}$ and $A_{1}A_{2}A_{3}$ |
| `interaction[i].U_Hz` | on-site interaction (exact 6D) |
| `interaction[i].U_standard_Hz` | the textbook $\delta$-pseudopotential + lowest-band value, for comparison |
| `interaction[i].U_over_Ustandard` | exact/standard ratio — this is the size of the correction |
| `interaction[i].U_over_t` | $U/t$ along $x$, $y$, $z$ (a list of three numbers) |

## Accuracy and basis cut-offs

| preset | `ncm` | `lmax` | `rel_emax` | typical run time | accuracy |
|---|---|---|---|---|---|
| `fast` | 3 | 4 | 8 | ~1 s per $a$ | ~0.1 % |
| `default` | 4 | 4 | 9 | ~2 s per $a$ | ~0.01 % |
| `accurate` | 5 | 6 | 10 | ~25 s per $a$ | reference |

Convergence is checked in `tests/test_hubbard.py`; for a cubic lattice the
`default` and `accurate` values differ by less than 3 Hz on $U \approx 72$ kHz
(relative difference $4\times10^{-5}$).

Validated against:

* the 1D single-particle limit — 4.201 versus 4.200 $E_R$ from an exact Bloch
  calculation;
* the $a \to 0$ limit — recovers the standard formula to better than 0.2 % at
  $a = 5 a_0$;
* a 9-point experimental data set on $^{6}\text{Li}$ (the 1–3 and 1–2 pairs,
  1064 nm, $20 E_R$):

| theory | rms deviation | max deviation |
|---|---|---|
| standard $U_{\text{BH}}$ | 5.56 kHz | 13.18 kHz (26 %) |
| $A^{3}U_{\text{harm}}$ (2009 ansatz) | 0.50 kHz | 1.36 kHz |
| **this package (6D exact)** | **0.30 kHz** | **0.67 kHz** |

---

## Limitations

* **Single-band Hubbard model assumed.** The output is $t$ and $U$; the model
  itself (and the density of states) must be checked separately.
* **Zero-range interaction.** The effective range is neglected. For a *narrow*
  Feshbach resonance this is a real systematic (typically ~1 %); a real
  Born–Oppenheimer potential would be needed to remove it.
* **Scattering branch only.** For $a>0$ the molecular branch is deliberately
  excluded from the basis, so the returned $U$ is the (metastable) scattering
  branch — which is the one that enters the repulsive Hubbard model.
* **Anisotropy** is treated with a single reference frequency for the basis;
  it is variational and converges, but for strongly anisotropic lattices
  (depths differing by more than ~30 %) use `basis="accurate"` or check
  convergence manually.
* Two-body physics only: no three-body or many-body corrections.

---

## References

1. T. Busch, B.-G. Englert, K. Rzażewski, M. Wilkens,
   *Two cold atoms in a harmonic trap*, Found. Phys. **28**, 549 (1998).
2. P.-I. Schneider, S. Grishkevich, A. Saenz,
   *Ab initio determination of Bose-Hubbard parameters …*,
   Phys. Rev. A **80**, 013404 (2009).
3. S. Grishkevich, S. Sala, A. Saenz,
   *Theoretical description of two ultracold atoms in finite three-dimensional
   optical lattices using realistic interatomic interaction potentials*,
   Phys. Rev. A **84**, 062710 (2011).
4. Z. Idziaszek, T. Calarco, Phys. Rev. A **71**, 050701(R) (2005) —
   anisotropic harmonic traps.

---

## Authorship

Generated with **DeepSeek Harness + DeepSeek V4.1 Flash**.

## License

MIT — see [LICENSE](LICENSE).
