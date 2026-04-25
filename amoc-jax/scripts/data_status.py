#!/usr/bin/env python3
"""Read data_manifest.yaml, walk the filesystem, and report status.

For each catalogued field:
- Is the file present?
- Does shape match the manifest's expectation?
- Are values within the sanity bounds?
- (Optional) compute a sha256 for provenance.

Usage:
    python scripts/data_status.py [--by-role] [--missing-only] [--sha]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).resolve().parents[1] / "data_manifest.yaml"

OK = "\033[32m"
WARN = "\033[33m"
ERR = "\033[31m"
DIM = "\033[2m"
END = "\033[0m"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()[:12]


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def _check_field(entry: dict, args) -> tuple[str, list[str]]:
    """Return (status_color, lines)."""
    file_path = _resolve(entry["file"])
    expected_shape = entry.get("shape")
    sanity = entry.get("sanity")
    field_key = entry.get("field_key")
    encoding = entry.get("encoding")

    if not file_path.exists():
        # Try fallback if listed.
        fallback = entry.get("fallback")
        if fallback:
            fp = _resolve(fallback)
            if fp.exists():
                return WARN, [f"FALLBACK ({fallback})  {fp.stat().st_size:>10,} B"]
        return ERR, ["MISSING"]

    size = file_path.stat().st_size
    lines = [f"present       {size:>10,} B"]

    if args.sha:
        lines[-1] += f"   sha={_sha256(file_path)}"

    # Try to parse and check shape/values for JSON files.
    if file_path.suffix == ".json":
        try:
            with open(file_path) as f:
                d = json.load(f)
        except json.JSONDecodeError as e:
            return ERR, [f"INVALID JSON: {e}"]

        if isinstance(d, dict):
            nx, ny = d.get("nx"), d.get("ny")
            if expected_shape and nx and ny and len(expected_shape) == 2:
                exp_ny, exp_nx = expected_shape
                if nx != exp_nx or ny != exp_ny:
                    lines.append(f"{ERR}shape mismatch{END}: file {nx}x{ny}, manifest {exp_nx}x{exp_ny}")
                    return ERR, lines

            # Sanity-check values for the simple-array case.
            if sanity and field_key and isinstance(field_key, str) and field_key in d:
                arr = d[field_key]
                if isinstance(arr, list):
                    mn, mx = min(arr), max(arr)
                    if mn < sanity["min"] or mx > sanity["max"]:
                        lines.append(
                            f"{WARN}values out of bounds{END}: [{mn:.3g}, {mx:.3g}] "
                            f"vs manifest [{sanity['min']:.3g}, {sanity['max']:.3g}]"
                        )
                    else:
                        lines.append(f"values OK     [{mn:.3g}, {mx:.3g}]")
            elif encoding == "hex_packed_bits" and "hex" in d:
                hex_len = len(d["hex"])
                expected_bits = nx * ny if nx and ny else None
                expected_chars = expected_bits // 4 if expected_bits else None
                if expected_chars and hex_len != expected_chars:
                    lines.append(f"{WARN}mask hex length {hex_len} != expected {expected_chars}{END}")
                else:
                    lines.append(f"hex mask      {hex_len} chars  ({expected_bits} bits)")

    return OK, lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--by-role", action="store_true", help="Group by role")
    p.add_argument("--missing-only", action="store_true", help="Show only missing/error rows")
    p.add_argument("--sha", action="store_true", help="Compute SHA256 for each file")
    p.add_argument("--manifest", default=str(MANIFEST))
    args = p.parse_args()

    with open(args.manifest) as f:
        m = yaml.safe_load(f)

    fields = m["fields"]
    if args.by_role:
        # Sort by primary role (first element if list).
        def primary_role(e):
            r = e.get("role", "?")
            return r[0] if isinstance(r, list) else r
        fields = sorted(fields, key=primary_role)

    n_ok = n_warn = n_err = 0
    current_role = None
    for entry in fields:
        status, lines = _check_field(entry, args)
        if status == OK:
            n_ok += 1
        elif status == WARN:
            n_warn += 1
        else:
            n_err += 1
        if args.missing_only and status == OK:
            continue

        if args.by_role:
            r = entry.get("role", "?")
            primary = r[0] if isinstance(r, list) else r
            if primary != current_role:
                print(f"\n{DIM}--- {primary} ---{END}")
                current_role = primary

        used = entry.get("used_in", [])
        used_str = ",".join(used) if used else "—"
        print(f"{status}{entry['id']:<30}{END}  {DIM}{used_str:<22}{END}  {lines[0]}")
        for extra in lines[1:]:
            print(f"  {DIM}└─{END} {extra}")

    print()
    total = n_ok + n_warn + n_err
    print(
        f"{OK}{n_ok} OK{END}  "
        f"{WARN}{n_warn} fallback/warning{END}  "
        f"{ERR}{n_err} missing/error{END}  "
        f"({total} total)"
    )

    if n_err:
        print(f"\nTo refetch missing fields, see {DIM}docs/data.md{END} for instructions.")
        sys.exit(1)


if __name__ == "__main__":
    main()
