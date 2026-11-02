"""Command-line interface.

Examples
--------
    hubbard-parameter --species 6Li --lattice-constant 532 \
        --depth 20 20 20 --scattering-length 171.4 -738.0

    hubbard-parameter --species 6Li --lattice-constant 532 \
        --depth 20 20 20 --scattering-length 300 --json out.json
"""

import argparse
import json
import sys

from .core import BASIS_PRESETS, compute_hubbard_parameters


def build_parser():
    p = argparse.ArgumentParser(
        prog="hubbard-parameter",
        description="Ab-initio Hubbard parameters (t, U) for atoms in a 3D "
        "optical lattice, from the exact six-dimensional two-body problem.",
    )
    p.add_argument("--lattice-constant", type=float, default=532.0,
                   help="lattice period in nm = half the laser wavelength "
                        "(default 532, i.e. a 1064 nm lattice)")
    p.add_argument("--species", default="6Li",
                   help="atomic species, e.g. 6Li, 40K, 87Rb (default 6Li)")
    p.add_argument("--mass-u", type=float, default=None,
                   help="use this mass (atomic mass units) instead of --species")
    p.add_argument("--depth", type=float, nargs=3, default=[20.0, 20.0, 20.0],
                   metavar=("VX", "VY", "VZ"),
                   help="lattice depths along x, y, z in units of E_R")
    p.add_argument("--scattering-length", type=float, nargs="+", default=None,
                   metavar="A0",
                   help="scattering length(s) in Bohr radii")
    p.add_argument("--basis", choices=sorted(BASIS_PRESETS), default="default",
                   help="basis cut-off preset (default: default)")
    p.add_argument("--json", nargs="?", const="-", default=None,
                   metavar="FILE", help="write results as JSON ('-' = stdout)")
    return p


def _groups(values, names=("x", "y", "z"), rtol=1e-9):
    """Group entries that are numerically identical, preserving order.

    Returns a list of (list_of_names, value).
    """
    out = []
    for v, n in zip(values, names):
        for grp in out:
            if abs(v - grp[1]) <= rtol * max(abs(v), abs(grp[1]), 1e-300):
                grp[0].append(n)
                break
        else:
            out.append([[n], v])
    return [(g[0], g[1]) for g in out]


def _print_report(res):
    """Human-readable summary.  All frequencies are printed in kHz."""
    line = "=" * 84
    print(line)
    print(f"  species            : {res['species']}   "
          f"(m = {res['mass_u']:.4f} u)")
    print(f"  lattice constant   : {res['lattice_constant_nm']:.1f} nm "
          f"(lambda = {res['wavelength_nm']:.1f} nm)")
    print(f"  depths (x,y,z)     : {res['depths_Er'][0]:.2f}, "
          f"{res['depths_Er'][1]:.2f}, {res['depths_Er'][2]:.2f}  E_R")
    print(f"  E_R                : {res['recoil_Hz'] / 1e3:.3f} kHz  "
          f"({res['recoil_uK']:.3f} uK)")
    print("-" * 84)

    # ---- tunneling (identical directions are collapsed) ------------------
    print("  tunneling")
    t_khz = [v / 1e3 for v in res["t_Hz"]]
    grps = _groups(t_khz)
    txt = "   ".join(
        (" = ".join(f"t_{n}" for n in names)) + f" = {val:.4f} kHz"
        for names, val in grps
    )
    print(f"    {txt}")
    print("-" * 84)

    # ---- on-site interaction --------------------------------------------
    print("  on-site interaction")
    if not res["interaction"]:
        print("    (no scattering length given)")
    else:
        ut = [it["U_over_t"] for it in res["interaction"]]
        # collapse directions whose U/t agree for *every* scattering length
        names = ["x", "y", "z"]
        cols = []
        for c in range(3):
            for col in cols:
                if all(
                    abs(u[c] - u[col[0]]) <= 1e-9 * max(abs(u[c]), abs(u[col[0]]), 1e-300)
                    for u in ut
                ):
                    col.append(c)
                    break
            else:
                cols.append([c])
        if len(cols) == 1:
            heads = ["U/t"]
        else:
            heads = [
                "U/t_" + ("".join(names[c] for c in col) if len(col) > 1 else names[col[0]])
                for col in cols
            ]
        head = (f"    {'a [a0]':>10} {'U [kHz]':>12} {'U_std [kHz]':>13}"
                + "".join(f" {h:>10}" for h in heads))
        print(head)
        for it, u in zip(res["interaction"], ut):
            row = (f"    {it['scattering_length_a0']:10.2f} "
                   f"{it['U_Hz'] / 1e3:12.4f} "
                   f"{it['U_standard_Hz'] / 1e3:13.4f}")
            row += "".join(f" {u[col[0]]:10.3f}" for col in cols)
            print(row)
    print(line)


def main(argv=None):
    args = build_parser().parse_args(argv)
    res = compute_hubbard_parameters(
        lattice_constant_nm=args.lattice_constant,
        depths_Er=args.depth,
        scattering_lengths_a0=args.scattering_length,
        species=args.species,
        mass_u=args.mass_u,
        basis=args.basis,
    )
    if args.json is None:
        _print_report(res)
    elif args.json == "-":
        json.dump(res, sys.stdout, indent=2)
        print()
    else:
        _print_report(res)
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"JSON written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
