"""Fail if a file exceeds a size budget in KB. Used by the CI smoke test."""

import pathlib
import sys

path = pathlib.Path(sys.argv[1])
limit = float(sys.argv[2]) * 1024
size = path.stat().st_size
if size > limit:
    sys.exit(f"{path} is {size} bytes, over the {limit:.0f} byte budget")
print(f"{path}: {size} bytes, within {limit:.0f}")
