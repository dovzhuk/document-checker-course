from __future__ import annotations

import sys
from pathlib import Path

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_checker.api.fastapi_app import app  # noqa: E402


def main() -> None:
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()