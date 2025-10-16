"""
A script to run yolo-service, molmo-service, and orchestrator all at once. 
Waits for molmo-service to start up before running the other two.
"""

import os
import sys
import time
import threading
import subprocess
import signal
from pathlib import Path
import urllib.request

IS_WIN = os.name == "nt"
ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
PYTHON = sys.executable  # <- use the current venv's python

def sp(path: Path) -> str:
    return str(path.resolve())


# Define services by script path (no cwd needed)
SERVICES = [
    ("molmo", [PYTHON, "-u", sp(ROOT / "molmo-service" / "app.py")]),
    ("yolo",  [PYTHON, "-u", sp(ROOT / "yolo-service"  / "app.py")]),
    ("orchestrator", [PYTHON, "-u", sp(ROOT / "orchestrator" / "modified_main.py")]),
]


# sanity check so we fail early with a clear message
missing = [name for name, cmd in SERVICES if not Path(cmd[-1]).is_file()]
if missing:
    print("[fatal] These scripts were not found relative to project root:")
    for name in missing:
        print(f"  - {name}: expected at {SERVICES[[n for n,_ in SERVICES].index(name)][1][-1]}")
    sys.exit(1)

BASE_ENV = os.environ.copy()
BASE_ENV.setdefault("PYTHONUNBUFFERED", "1")

procs = []
files = []

def wait_for_http(url, name="?", timeout=60.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if 200 <= r.status < 300:
                    print(f"[{name}] healthy at {url}")
                    return True
        except Exception:
            time.sleep(1)
    print(f"[{name}] health check failed after {timeout}s")
    return False

def pump(stream, fh, prefix):
    for line in iter(stream.readline, ''):
        if not line:
            break
        fh.write(line); fh.flush()
        sys.stdout.write(f"[{prefix}] {line}"); sys.stdout.flush()
    stream.close()

def start_service(name, cmd):
    if IS_WIN:
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # own console group
        p = subprocess.Popen(
            cmd,
            env=BASE_ENV,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
            creationflags=creationflags
        )
    else:
        p = subprocess.Popen(
            cmd,
            env=BASE_ENV,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
            start_new_session=True  # posix process group
        )

#logging errors and outputs into logs folder, replaces old logs every time
    fout = open(LOG_DIR / f"{name}.out.log", "w", buffering=1)
    ferr = open(LOG_DIR / f"{name}.err.log", "w", buffering=1)
    files.append((fout, ferr))

    threading.Thread(target=pump, args=(p.stdout, fout, f"{name}:OUT"), daemon=True).start()
    threading.Thread(target=pump, args=(p.stderr, ferr, f"{name}:ERR"), daemon=True).start()
    return p

def graceful_stop(p):
    try:
        if IS_WIN:
            # Send CTRL+BREAK to the process group, then fall back
            p.send_signal(signal.CTRL_BREAK_EVENT)
            return True
        else:
            # Send SIGTERM to the group
            os.killpg(p.pid, signal.SIGTERM)
            return True
    except Exception:
        return False

def hard_kill(p):
    try:
        if IS_WIN:
            p.kill()
        else:
            os.killpg(p.pid, signal.SIGKILL)
    except Exception:
        pass

def shutdown():
    print("\nShutting down all services…")
    for p in procs:
        graceful_stop(p)
    deadline = time.time() + 8
    for p in procs:
        while p.poll() is None and time.time() < deadline:
            time.sleep(0.2)
        if p.poll() is None:
            hard_kill(p)
    for fout, ferr in files:
        try: fout.close()
        except: pass
        try: ferr.close()
        except: pass
    print("All services stopped.")


if __name__ == "__main__":
    try:
        for name, cmd in SERVICES:
            print(f"Starting {name}: {' '.join(cmd)}")
            procs.append(start_service(name, cmd))

            # Wait until molmo’s /health is ready before YOLO
            if name == "molmo":
                print("[launcher] Waiting for molmo /health to respond...")
                wait_for_http("http://127.0.0.1:8000/health", "molmo")

            time.sleep(0.8)
        print("All services started. Logs in ./logs/*.log")
        while any(p.poll() is None for p in procs):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()


