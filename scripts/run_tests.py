from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    try:
        import pytest
    except ImportError as exc:
        raise SystemExit("pytest is required to run this test suite.") from exc

    return pytest.main([str(repo_root / "tests"), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
