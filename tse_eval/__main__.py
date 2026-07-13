"""Enable ``python -m tse_eval``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
