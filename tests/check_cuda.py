"""Run the numerical contracts on CUDA in a fresh process."""

import sys

from check_metal import main

if __name__ == "__main__":
    raise SystemExit(main("cuda", sys.argv[1:]))
