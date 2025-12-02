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
PYTHON = sys.executable  # use the current venv's python

def sp(path: Path) -> str:
    return str(path.resolve())

# ────────────────────────────────────────────────────────────────────────────
# Config (override via env if needed)
MOLMO_PORT = int(os.getenv("MOLMO_PORT", "8000"))
YOLO_PORT  = int(os.getenv("YOLO_PORT",  "9000"))  # if your yolo differs, change here
ORCH_PORT  = int(os.getenv("ORCH_PORT",  "7000"))
SKIP_HEALTH = os.getenv("RUNALL_SKIP_HEALTH", "0") == "1"

# Service table: (name, cmd, cwd, health_url|None)
SERVICES = [
    ("molmo",
     [PYTHON, "-u", sp(ROOT / "molmo-service" / "app.py")],
     sp(ROOT / "molmo-service"),
     f"http://127.0.0.1:{MOLMO_PORT}/health"),
    ("yolo",
     [PYTHON, "-u", sp(ROOT / "yolo-service" / "app.py")],
     sp(ROOT / "yolo-service"),
     f"http://127.0.0.1:{YOLO_PORT}/health"),
    ("orchestrator",
     [PYTHON, "-u", sp(ROOT / "orchestrator" / "modified_main.py")],
     sp(ROOT / "orchestrator"),
     f"http://127.0.0.1:{ORCH_PORT}/health"),
]

# Sanity check so we fail early with a clear message
missing = [name for (name, cmd, cwd, _) in SERVICES if not Path(cmd[-1]).is_file()]
if missing:
    print("[fatal] These scripts were not found relative to project root:")
    for name in missing:
        entry = next(s for s in SERVICES if s[0] == name)
        print(f"  - {name}: expected at {entry[1][-1]}")
    sys.exit(1)

BASE_ENV = os.environ.copy()
BASE_ENV.setdefault("PYTHONUNBUFFERED", "1")

procs = []  # [(name, Popen)]
files = []  # [(fout, ferr)]

def wait_for_http(url, name="?", timeout=60.0):
    if SKIP_HEALTH or not url:
        print(f"[{name}] health check skipped")
        return True
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if 200 <= r.status < 300:
                    print(f"[{name}] healthy at {url}")
                    return True
        except Exception:
            time.sleep(1)
    print(f"[{name}] health check failed after {timeout}s ({url})")
    return False

def pump(stream, fh, prefix):
    for line in iter(stream.readline, ''):
        if not line:
            break
        fh.write(line); fh.flush()
        sys.stdout.write(f"[{prefix}] {line}"); sys.stdout.flush()
    stream.close()

def start_service(name, cmd, cwd):
    # Windows: own console group so CTRL_BREAK reaches children
    if IS_WIN:
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        p = subprocess.Popen(
            cmd,
            env=BASE_ENV,
            cwd=cwd,
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
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
            start_new_session=True  # posix process group
        )

    # logging (overwritten on each run)
    fout = open(LOG_DIR / f"{name}.out.log", "w", buffering=1)
    ferr = open(LOG_DIR / f"{name}.err.log", "w", buffering=1)
    files.append((fout, ferr))

    threading.Thread(target=pump, args=(p.stdout, fout, f"{name}:OUT"), daemon=True).start()
    threading.Thread(target=pump, args=(p.stderr, ferr, f"{name}:ERR"), daemon=True).start()
    return p

def graceful_stop(p):
    try:
        if IS_WIN:
            p.send_signal(signal.CTRL_BREAK_EVENT)  # to process group
            return True
        else:
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
    for _, p in procs:
        graceful_stop(p)
    deadline = time.time() + 8
    for _, p in procs:
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
        for idx, (name, cmd, cwd, health) in enumerate(SERVICES):
            print(f"Starting {name}: {' '.join(cmd)}  (cwd={cwd})")
            p = start_service(name, cmd, cwd)
            procs.append((name, p))

            # Wait until molmo’s /health is ready before starting the NEXT service(s)
            if name == "molmo":
                print("[launcher] Waiting for molmo /health to respond...")
                ok = wait_for_http(health, "molmo", timeout=120.0)
                if not ok:
                    print("[launcher] molmo failed to become healthy; aborting.")
                    raise SystemExit(2)

            # small stagger to make logs clearer
            time.sleep(0.8)

        print("All services started. Logs in ./logs/*.log")

        # main loop: if any service dies, tear down the rest
        while True:
            alive = [(n, p) for (n, p) in procs if p.poll() is None]
            if len(alive) != len(procs):
                dead = [(n, p.returncode) for (n, p) in procs if p.poll() is not None]
                if dead:
                    print(f"\n[launcher] Detected service exit: {dead}. Initiating shutdown…")
                break
            time.sleep(0.5)

    except KeyboardInterrupt:
        pass
    finally:
        shutdown()
