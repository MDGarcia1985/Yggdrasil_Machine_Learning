"""
Run the app through the current project workflow.

Workflow:
1. Execute `main.py` so training artifacts are refreshed.
2. Launch the Streamlit app.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from utils.paths import MAIN_ENTRYPOINT, PROJECT_ROOT, STREAMLIT_APP_PATH


def ensure_streamlit_available() -> None:
    """
    Fail early if the active Python environment cannot launch Streamlit.
    """
    if importlib.util.find_spec("streamlit") is not None:
        return

    install_command = f'"{sys.executable}" -m pip install -r requirements.txt'
    raise RuntimeError(
        "Streamlit is not installed in the active Python environment.\n"
        f"Python executable: {sys.executable}\n"
        f"From the project root, run:\n  {install_command}"
    )


def run_pipeline() -> None:
    """
    Execute main.py from the project root.
    """
    subprocess.run(
        [sys.executable, str(MAIN_ENTRYPOINT)],
        cwd=str(PROJECT_ROOT),
        check=True,
    )


def run_streamlit() -> None:
    """
    Launch the Streamlit application.
    """
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(STREAMLIT_APP_PATH)],
        cwd=str(PROJECT_ROOT),
        check=True,
    )


def main() -> None:
    """
    Optional flag:
      --skip-pipeline: launch Streamlit without running main.py first.
    """
    parser = argparse.ArgumentParser(description="Run pipeline and launch app.")
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Skip running main.py before launching Streamlit.",
    )
    args = parser.parse_args()

    ensure_streamlit_available()

    if not args.skip_pipeline:
        run_pipeline()

    run_streamlit()


if __name__ == "__main__":
    main()
