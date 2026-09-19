#!/usr/bin/env python3
"""PixelOpt launcher: one command from a fresh clone to a running app.

    python launch.py                      # set up if needed, self-test, launch, open browser
    python launch.py --check              # run the full test suites before launching
    python launch.py --once --no-browser  # launch, confirm healthy, shut down (smoke/CI)
    python launch.py --install-autostart  # start PixelOpt when you log in

The pipeline runs in five stages and stops at the first one that fails, with
a message saying what went wrong and where the log is:

  1. preflight     Python version, project files, an already-running instance
  2. environment   create or reuse .venv
  3. dependencies  install only when requirements changed or an import is missing
  4. self-test     compress a synthetic image through the real engine
  5. launch        pick a free port, start Streamlit, wait until it answers its
                   health check, open the browser, restart it if it crashes

Standard library only, on purpose: this has to run before any dependency is
installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import venv
import webbrowser
from pathlib import Path
from typing import List, Optional, Sequence

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
LOG_DIR = ROOT / "logs"
INSTANCE_FILE = LOG_DIR / "instance.json"
STAMP_NAME = ".pixelopt-deps.sha256"

MIN_PYTHON = (3, 9)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8501
HEALTH_PATH = "/_stcore/health"
DEPENDENCY_FILES = ("requirements.txt", "pyproject.toml")
REQUIRED_MODULES = ("numpy", "PIL", "cv2", "skimage", "streamlit", "altair", "pandas")
STAGES = 5

# Runs inside the target environment. It exercises the real engine end to end,
# so a broken install fails here with a clear message instead of as a
# traceback in the browser.
SMOKE_TEST = """
import time
import numpy as np
from pixelopt.forms import FORM_PRESETS, encode_for_form
from pixelopt.pipeline import process

rng = np.random.default_rng(1)
y, x = np.mgrid[0:360, 0:480]
base = 128 + 60 * np.sin(x / 23.0) * np.cos(y / 19.0) + rng.normal(0, 8, (360, 480))
image = np.dstack([np.clip(base, 0, 255)] * 3).astype(np.uint8)
started = time.monotonic()
result = process(image, 30)
assert result["actual_size_kb"] <= 30, "budget exceeded"
form = encode_for_form(image, FORM_PRESETS["photo"])
assert form["meets_min"] and form["meets_max"], "form range missed"
print(
    "compressed to %.1f KB (%s, SSIM %.3f); form photo %.1f KB; %.2fs"
    % (result["actual_size_kb"], result["format"], result["fidelity_ssim"],
       form["size_kb"], time.monotonic() - started)
)
"""


class LaunchError(RuntimeError):
    """A stage failed. The message is written for the person running it."""


# ------------------------------------------------------------------ output


def stage(number: int, title: str) -> None:
    print(f"[{number}/{STAGES}] {title}", flush=True)


def detail(message: str) -> None:
    for line in str(message).splitlines() or [""]:
        print(f"      {line}", flush=True)


# ---------------------------------------------------------------- helpers


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def dependency_hash(root: Path, files: Sequence[str] = DEPENDENCY_FILES) -> str:
    """Fingerprint of everything that decides what gets installed."""
    digest = hashlib.sha256()
    for name in files:
        path = root / name
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes() if path.exists() else b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def read_stamp(venv_dir: Path) -> Optional[str]:
    try:
        return (venv_dir / STAMP_NAME).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def write_stamp(venv_dir: Path, value: str) -> None:
    (venv_dir / STAMP_NAME).write_text(value + "\n", encoding="utf-8")


def missing_modules(python: Path, modules: Sequence[str] = REQUIRED_MODULES) -> List[str]:
    """Which modules fail to import in the target interpreter."""
    probe = (
        "import importlib, json, sys\n"
        "missing = []\n"
        "for name in sys.argv[1:]:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except Exception:\n"
        "        missing.append(name)\n"
        "print(json.dumps(missing))\n"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", probe, *modules], capture_output=True, text=True,
            cwd=ROOT, timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired):
        return list(modules)
    if result.returncode != 0 or not result.stdout.strip():
        return list(modules)
    return json.loads(result.stdout.strip().splitlines()[-1])


def port_is_free(port: int, host: str = DEFAULT_HOST) -> bool:
    # No SO_REUSEADDR: on Windows it lets a bind succeed on a port that is
    # already in use, which would make every port look free.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def choose_port(preferred: int, host: str = DEFAULT_HOST, attempts: int = 20) -> int:
    for port in range(preferred, preferred + attempts):
        if 0 < port < 65536 and port_is_free(port, host):
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return probe.getsockname()[1]


def is_healthy(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200 and response.read().strip() == b"ok"
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, ValueError):
        return False


def wait_for_health(url: str, timeout: float, process=None, interval: float = 0.5) -> bool:
    """Poll the health endpoint. Gives up at once if the process has died."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            return False
        if is_healthy(url):
            return True
        time.sleep(interval)
    return False


def tail(path: Path, lines: int = 40) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(no log output)"
    return "\n".join(content[-lines:]) or "(log is empty)"


def read_instance(path: Path = INSTANCE_FILE) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_instance(record: dict, path: Path = INSTANCE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def clear_instance(path: Path = INSTANCE_FILE) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def running_instance(path: Path = INSTANCE_FILE) -> Optional[str]:
    """URL of a PixelOpt this launcher already started and that still answers."""
    record = read_instance(path)
    if not record or "url" not in record:
        return None
    if is_healthy(record["url"].rstrip("/") + HEALTH_PATH):
        return record["url"]
    clear_instance(path)  # stale: the process is gone
    return None


# ----------------------------------------------------------------- stages


def preflight() -> None:
    stage(1, "Preflight")
    if sys.version_info < MIN_PYTHON:
        raise LaunchError(
            "PixelOpt needs Python %d.%d or newer; this is %s."
            % (*MIN_PYTHON, sys.version.split()[0])
        )
    for required in ("app.py", "requirements.txt", "pixelopt/__init__.py"):
        if not (ROOT / required).exists():
            raise LaunchError(f"{required} is missing. Run the launcher from a full PixelOpt checkout.")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    detail(f"Python {sys.version.split()[0]} at {sys.executable}")


def prepare_environment(use_venv: bool) -> Path:
    stage(2, "Environment")
    if not use_venv:
        detail(f"using the current interpreter: {sys.executable}")
        return Path(sys.executable)
    python = venv_python(VENV_DIR)
    if python.exists():
        detail(f"reusing {VENV_DIR.name}")
        return python
    detail(f"creating {VENV_DIR.name} (first run only)")
    venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    if not python.exists():
        raise LaunchError(f"Created {VENV_DIR} but found no interpreter at {python}.")
    return python


def ensure_dependencies(python: Path, use_venv: bool, skip: bool, force: bool) -> None:
    stage(3, "Dependencies")
    wanted = dependency_hash(ROOT)
    missing = missing_modules(python)

    if skip or not use_venv:
        if missing:
            raise LaunchError(
                "Missing modules: " + ", ".join(missing) + ". "
                + ("Run without --skip-install." if use_venv
                   else "Install requirements.txt into this interpreter first.")
            )
        detail("all required modules import")
        return

    if not force and not missing and read_stamp(VENV_DIR) == wanted:
        detail("up to date (requirements unchanged since the last install)")
        return

    reason = ("reinstall requested" if force
              else "missing: " + ", ".join(missing) if missing
              else "requirements changed since the last install")
    detail(f"installing ({reason}) - this can take a few minutes the first time")
    log_path = LOG_DIR / "install.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== install {time.strftime('%Y-%m-%d %H:%M:%S')} ({reason}) =====\n")
        log.flush()
        completed = subprocess.run(
            [str(python), "-m", "pip", "install", "--disable-pip-version-check",
             "-r", str(ROOT / "requirements.txt")],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        raise LaunchError(
            f"pip install failed (exit {completed.returncode}). Last lines of {log_path}:\n"
            + tail(log_path, 25)
        )
    still_missing = missing_modules(python)
    if still_missing:
        raise LaunchError("Installed, but these still fail to import: " + ", ".join(still_missing))
    write_stamp(VENV_DIR, wanted)
    detail("installed")


def self_test(python: Path, mode: str) -> None:
    stage(4, "Self-test")
    if mode == "skip":
        detail("skipped")
        return
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8")

    smoke = subprocess.run([str(python), "-c", SMOKE_TEST], capture_output=True,
                           text=True, cwd=ROOT, env=env, timeout=300)
    if smoke.returncode != 0:
        raise LaunchError("The engine smoke test failed:\n" + (smoke.stderr or smoke.stdout)[-2000:])
    detail(smoke.stdout.strip())

    if mode != "full":
        return
    failures = []
    for suite in sorted((ROOT / "tests").glob("test_*.py")):
        started = time.monotonic()
        run = subprocess.run([str(python), str(suite)], capture_output=True, text=True,
                             cwd=ROOT, env=env, timeout=1800)
        summary = [line for line in run.stderr.splitlines() if line.startswith(("Ran ", "OK", "FAILED"))]
        detail(f"{suite.name:<24} {' '.join(summary) or 'no summary'} ({time.monotonic() - started:.0f}s)")
        if run.returncode != 0:
            failures.append((suite.name, run.stderr[-1500:]))
    if failures:
        name, output = failures[0]
        raise LaunchError(f"{len(failures)} test suite(s) failed; first was {name}:\n{output}")


def start_app(python: Path, host: str, port: int, log_path: Path, app: str):
    command = [
        str(python), "-m", "streamlit", "run", str(ROOT / app),
        "--server.headless", "true",
        "--server.address", host,
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false",
    ]
    log = log_path.open("a", encoding="utf-8")
    log.write(f"\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')} on {host}:{port} =====\n")
    log.flush()
    options = {}
    if os.name == "nt":
        # Own process group, so Ctrl+C in this console reaches the launcher
        # only -- otherwise Streamlit would die first and the supervisor would
        # mistake a deliberate stop for a crash and restart it.
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        env=dict(os.environ, PYTHONIOENCODING="utf-8"), **options,
    )
    process._pixelopt_log = log  # closed in stop_app
    return process


def stop_app(process, grace: float = 10.0) -> None:
    if process.poll() is None:
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.send_signal(signal.SIGINT)
            process.wait(grace)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            process.kill()
            try:
                process.wait(5)
            except subprocess.TimeoutExpired:
                pass
    log = getattr(process, "_pixelopt_log", None)
    if log is not None and not log.closed:
        log.close()


def launch_and_supervise(python: Path, args) -> int:
    stage(5, "Launch")
    log_path = LOG_DIR / f"pixelopt-{time.strftime('%Y%m%d')}.log"
    port = choose_port(args.port, args.host)
    if port != args.port:
        detail(f"port {args.port} is busy, using {port}")
    restarts = 0

    while True:
        url = f"http://{args.host}:{port}"
        process = start_app(python, args.host, port, log_path, args.app)
        write_instance({"url": url, "port": port, "app_pid": process.pid,
                        "launcher_pid": os.getpid(),
                        "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
        detail(f"starting Streamlit (pid {process.pid}), logging to {log_path.relative_to(ROOT)}")

        try:
            healthy = wait_for_health(url + HEALTH_PATH, args.timeout, process)
        except KeyboardInterrupt:
            stop_app(process)
            clear_instance()
            raise
        if not healthy:
            code = process.poll()
            stop_app(process)
            clear_instance()
            why = (f"exited with code {code}" if code is not None
                   else f"did not answer its health check within {args.timeout:.0f}s")
            raise LaunchError(f"PixelOpt {why}. Last lines of {log_path}:\n" + tail(log_path))

        if restarts == 0:
            print(f"\n  PixelOpt is running at {url}\n", flush=True)
            if not args.no_browser:
                webbrowser.open(url)
        else:
            detail(f"back up at {url}")

        if args.once:
            stop_app(process)
            clear_instance()
            detail("--once: healthy, shut down cleanly")
            return 0

        print("  Press Ctrl+C to stop.", flush=True)
        try:
            code = process.wait()
        except KeyboardInterrupt:
            print("\n  Stopping PixelOpt...", flush=True)
            stop_app(process)
            clear_instance()
            return 0

        stop_app(process)  # closes the log handle
        clear_instance()
        if restarts >= args.max_restarts:
            raise LaunchError(
                f"PixelOpt exited with code {code} and was restarted {restarts} time(s) "
                f"already, so the launcher is giving up. Last lines of {log_path}:\n" + tail(log_path)
            )
        restarts += 1
        delay = min(30, 2 ** restarts)
        detail(f"PixelOpt exited unexpectedly (code {code}); restart {restarts} of "
               f"{args.max_restarts} in {delay}s")
        try:
            time.sleep(delay)
        except KeyboardInterrupt:
            return 0
        if not port_is_free(port, args.host):
            port = choose_port(port + 1, args.host)


# -------------------------------------------------------------- autostart


def autostart_command(python: Optional[str] = None) -> List[str]:
    """What runs at login: no browser pop-up, no self-test, logs to file."""
    return [python or sys.executable, str(ROOT / "launch.py"), "--no-browser", "--skip-selftest"]


def autostart_path() -> Optional[Path]:
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if not appdata:
            return None
        return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "PixelOpt.cmd"
    if sys.platform.startswith("linux"):
        return Path.home() / ".config" / "autostart" / "pixelopt.desktop"
    return None


def autostart_contents(command: Sequence[str]) -> str:
    quoted = " ".join(f'"{part}"' for part in command)
    if os.name == "nt":
        # pythonw would hide the window entirely; a minimised console keeps a
        # visible way to stop it.
        return ("@echo off\r\n"
                f'cd /d "{ROOT}"\r\n'
                f'start "PixelOpt" /min {quoted}\r\n')
    return ("[Desktop Entry]\nType=Application\nName=PixelOpt\n"
            f"Exec={quoted}\nPath={ROOT}\nTerminal=false\nX-GNOME-Autostart-enabled=true\n")


def install_autostart() -> int:
    path = autostart_path()
    if path is None:
        print("Autostart is supported on Windows and Linux. On macOS, add a Login Item that runs:")
        print("  " + " ".join(autostart_command()))
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(autostart_contents(autostart_command()), encoding="utf-8")
    print(f"PixelOpt will start when you log in.\n  {path}\nRemove it with: python launch.py --remove-autostart")
    return 0


def remove_autostart() -> int:
    path = autostart_path()
    if path is None or not path.exists():
        print("No PixelOpt autostart entry found.")
        return 0
    path.unlink()
    print(f"Removed {path}")
    return 0


# -------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="launch.py", description="Set up and launch PixelOpt in one step.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="preferred port; the next free one is used if busy (default 8501)")
    parser.add_argument("--host", default=DEFAULT_HOST, help="address to bind (default 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    parser.add_argument("--once", action="store_true",
                        help="exit after the app is confirmed healthy (for smoke tests and CI)")
    tests = parser.add_mutually_exclusive_group()
    tests.add_argument("--check", action="store_true", help="run the full test suites before launching")
    tests.add_argument("--skip-selftest", action="store_true", help="skip the engine smoke test")
    parser.add_argument("--skip-install", action="store_true",
                        help="never install; fail if something is missing")
    parser.add_argument("--reinstall", action="store_true", help="reinstall requirements even if unchanged")
    parser.add_argument("--no-venv", action="store_true",
                        help="use the current interpreter instead of .venv (never installs)")
    parser.add_argument("--max-restarts", type=int, default=3,
                        help="restart attempts after a crash before giving up (default 3)")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="seconds to wait for the health check (default 120)")
    parser.add_argument("--app", default="app.py", help=argparse.SUPPRESS)
    autostart = parser.add_mutually_exclusive_group()
    autostart.add_argument("--install-autostart", action="store_true",
                           help="start PixelOpt automatically when you log in")
    autostart.add_argument("--remove-autostart", action="store_true",
                           help="undo --install-autostart")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.install_autostart:
        return install_autostart()
    if args.remove_autostart:
        return remove_autostart()

    try:
        preflight()
        if not args.once:
            existing = running_instance()
            if existing:
                detail(f"PixelOpt is already running at {existing}")
                if not args.no_browser:
                    webbrowser.open(existing)
                return 0
        python = prepare_environment(use_venv=not args.no_venv)
        ensure_dependencies(python, use_venv=not args.no_venv,
                            skip=args.skip_install, force=args.reinstall)
        self_test(python, "skip" if args.skip_selftest else "full" if args.check else "quick")
        return launch_and_supervise(python, args)
    except LaunchError as error:
        print("\n  PixelOpt could not start.", file=sys.stderr)
        for line in str(error).splitlines():
            print(f"  {line}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n  Cancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
