"""Databricks Apps entry point for the Sehat-e-Aam FastAPI server.

Resolves the ``sehat`` package via, in order:
  1. ``$PYTHONPATH`` (set by ``app.yaml``)
  2. ``$SEHAT_PROJECT_ROOT/src``
  3. Walking up from ``__file__`` looking for ``src/sehat/__init__.py``
  4. ``/Workspace/Users/<user>/sehat-e-aam/src`` (default if uploaded as a
     Git folder under the user's home)

It then loads pipeline configuration from ``$SEHAT_ENV_FILE`` (defaults to the
Volume sidecar written by the setup notebook) and exposes the FastAPI ``app``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _try_path(candidate: Path) -> bool:
    if (candidate / "sehat" / "__init__.py").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        return True
    return False


def _resolve_src() -> bool:
    if root := os.environ.get("SEHAT_PROJECT_ROOT"):
        if _try_path(Path(root) / "src"):
            return True

    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        if _try_path(parent / "src"):
            return True

    user = os.environ.get("DATABRICKS_USER_NAME") or os.environ.get("USER")
    if user:
        if _try_path(Path(f"/Workspace/Users/{user}/sehat-e-aam/src")):
            return True

    return False


def _load_env_file() -> None:
    default_env = "/Volumes/workspace/sehat/data/sehat.env"
    env_file = os.environ.get("SEHAT_ENV_FILE", default_env)
    if not Path(env_file).exists():
        return
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


if not _resolve_src():
    raise RuntimeError(
        "Could not locate the `sehat` source package. Set SEHAT_PROJECT_ROOT "
        "or PYTHONPATH to <repo>/src."
    )
_load_env_file()

from sehat.api.server import app  # noqa: E402

__all__ = ["app"]


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
