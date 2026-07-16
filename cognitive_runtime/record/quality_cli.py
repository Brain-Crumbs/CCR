"""JSON bridge exposing :mod:`record.quality` to the read-only clinic.

This deliberately accepts only a Record session directory.  It gives the Node
service the authoritative Python verdict without teaching the service about a
World, Program, or brain implementation.
"""

from __future__ import annotations

import argparse
import json

from cognitive_runtime.record.quality import verdict_for_session


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir")
    args = parser.parse_args()
    print(json.dumps(verdict_for_session(args.session_dir).as_dict()))


if __name__ == "__main__":
    main()
