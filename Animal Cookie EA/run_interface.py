"""
run_interface.py - start the Animal Cookie Lab interface and open it in your browser.

Press Run on this file in PyCharm (or: python run_interface.py).
It installs missing packages, starts Streamlit without its first-run email
question, waits until the page is ready and opens it in your default browser.
Stop it with the red stop button in PyCharm.
"""

import importlib.util
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGES = {"deap": "deap", "numpy": "numpy", "pandas": "pandas", "streamlit": "streamlit"}


def ensure_packages():
    missing = [pip for mod, pip in PACKAGES.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print(f"Installing missing packages: {', '.join(missing)} ...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


def free_port(start=8501):
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port found")


def wait_until_up(port, proc, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


def main():
    ensure_packages()
    port = free_port()
    url = f"http://localhost:{port}"
    print(f"Starting Animal Cookie Lab on {url} ...", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(HERE / "app.py"),
         "--server.headless", "true",            # no email question, we open the browser ourselves
         "--server.port", str(port),
         "--browser.gatherUsageStats", "false"],
        cwd=HERE)
    try:
        if wait_until_up(port, proc):
            print(f"\nOpening {url} in your browser. If nothing opens, click the link.\n", flush=True)
            webbrowser.open(url)
        else:
            print("Streamlit did not start - see the messages above.", flush=True)
        proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if proc.poll() is None:
            proc.terminate()


if __name__ == "__main__":
    main()
