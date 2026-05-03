# launcher.py
"""
Local launcher for the Yggdrasil Circuit ML Streamlit app.

Purpose:
Starts the Streamlit UI from a normal Python command.

Expected app target:
ui/app_streamlit.py

Run:
python launcher.py
"""

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from utils.paths import PROJECT_ROOT, STREAMLIT_APP_PATH


def main() -> int:
    """
    Launch the Circuit ML Streamlit UI.
    """

    if not STREAMLIT_APP_PATH.exists():
        raise FileNotFoundError(
            f"Could not find Streamlit app at: {STREAMLIT_APP_PATH}. "
            "Expected ui/app_streamlit.py"
        )

    port = 8501
    url = f"http://localhost:{port}"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(STREAMLIT_APP_PATH),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--client.toolbarMode",
        "minimal",
    ]

    creationflags = 0

    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

    time.sleep(1.5)
    webbrowser.open(url)

    try:
        while proc.poll() is None:
            time.sleep(0.5)

    except KeyboardInterrupt:
        pass

    finally:
        try:
            proc.terminate()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
