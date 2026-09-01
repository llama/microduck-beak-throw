#!/usr/bin/env python3
"""Prepare a Header checkpoint as a safe BeakThrow initialization."""

from __future__ import annotations

import argparse
from pathlib import Path

from mjlab_microduck.beak_bootstrap import prepare_beak_bootstrap_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="trained Header .pt checkpoint")
    parser.add_argument("output", type=Path, help="output BeakThrow .pt checkpoint")
    parser.add_argument("--exploration-std", type=float, default=0.35)
    args = parser.parse_args()

    output = prepare_beak_bootstrap_file(
        args.input, args.output, exploration_std=args.exploration_std
    )
    print(f"Prepared BeakThrow bootstrap: {output}")


if __name__ == "__main__":
    main()
