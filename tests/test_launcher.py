"""Tests for launch.py helpers. Nothing here starts Streamlit."""

import http.server
import os
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import launch


class _Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - http.server naming
        body = b"ok" if self.path == launch.HEALTH_PATH else b"nope"
        self.send_response(200 if self.path == launch.HEALTH_PATH else 404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class HealthServer:
    def __enter__(self):
        self.server = http.server.HTTPServer(("127.0.0.1", 0), _Health)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


class FakeProcess:
    def __init__(self, code):
        self.code = code

    def poll(self):
        return self.code


class PathTests(unittest.TestCase):
    def test_venv_python_matches_platform_layout(self):
        path = launch.venv_python(Path("env"))
        if os.name == "nt":
            self.assertEqual(path, Path("env") / "Scripts" / "python.exe")
        else:
            self.assertEqual(path, Path("env") / "bin" / "python")


class DependencyStampTests(unittest.TestCase):
    def test_hash_changes_only_when_requirements_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "requirements.txt").write_text("numpy\n")
            first = launch.dependency_hash(root, ("requirements.txt",))
            self.assertEqual(first, launch.dependency_hash(root, ("requirements.txt",)))
            (root / "requirements.txt").write_text("numpy\npillow\n")
            self.assertNotEqual(first, launch.dependency_hash(root, ("requirements.txt",)))

    def test_missing_file_is_part_of_the_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            absent = launch.dependency_hash(root, ("pyproject.toml",))
            (root / "pyproject.toml").write_text("")
            self.assertNotEqual(absent, launch.dependency_hash(root, ("pyproject.toml",)))

    def test_stamp_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            venv_dir = Path(folder)
            self.assertIsNone(launch.read_stamp(venv_dir))
            launch.write_stamp(venv_dir, "abc123")
            self.assertEqual(launch.read_stamp(venv_dir), "abc123")


class ModuleProbeTests(unittest.TestCase):
    def test_reports_only_what_fails_to_import(self):
        missing = launch.missing_modules(Path(sys.executable),
                                         ("json", "definitely_not_a_real_module_xyz"))
        self.assertEqual(missing, ["definitely_not_a_real_module_xyz"])

    def test_broken_interpreter_reports_everything_missing(self):
        self.assertEqual(launch.missing_modules(Path("no/such/python"), ("json",)), ["json"])


class PortTests(unittest.TestCase):
    def test_busy_port_is_detected_and_skipped(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
            holder.bind(("127.0.0.1", 0))
            holder.listen(1)
            busy = holder.getsockname()[1]
            self.assertFalse(launch.port_is_free(busy))
            chosen = launch.choose_port(busy)
            self.assertNotEqual(chosen, busy)
            self.assertTrue(launch.port_is_free(chosen))


class HealthTests(unittest.TestCase):
    def test_healthy_server_is_recognised(self):
        with HealthServer() as server:
            self.assertTrue(launch.wait_for_health(server.url + launch.HEALTH_PATH, timeout=5))

    def test_wrong_path_is_not_healthy(self):
        with HealthServer() as server:
            self.assertFalse(launch.is_healthy(server.url + "/elsewhere"))

    def test_nothing_listening_times_out(self):
        port = launch.choose_port(47000)
        self.assertFalse(launch.wait_for_health(f"http://127.0.0.1:{port}/", timeout=1.2,
                                                interval=0.3))

    def test_dead_process_stops_the_wait_immediately(self):
        port = launch.choose_port(47100)
        started = __import__("time").monotonic()
        self.assertFalse(launch.wait_for_health(f"http://127.0.0.1:{port}/", timeout=30,
                                                process=FakeProcess(1)))
        self.assertLess(__import__("time").monotonic() - started, 2)


class InstanceTests(unittest.TestCase):
    def test_running_instance_is_reused_while_healthy(self):
        with tempfile.TemporaryDirectory() as folder, HealthServer() as server:
            record = Path(folder) / "instance.json"
            launch.write_instance({"url": server.url, "port": 0}, record)
            self.assertEqual(launch.running_instance(record), server.url)

    def test_stale_instance_is_cleared(self):
        with tempfile.TemporaryDirectory() as folder:
            record = Path(folder) / "instance.json"
            port = launch.choose_port(47200)
            launch.write_instance({"url": f"http://127.0.0.1:{port}"}, record)
            self.assertIsNone(launch.running_instance(record))
            self.assertFalse(record.exists())

    def test_tail_returns_the_last_lines(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "app.log"
            log.write_text("\n".join(f"line {i}" for i in range(100)))
            self.assertEqual(launch.tail(log, 3), "line 97\nline 98\nline 99")


class AutostartTests(unittest.TestCase):
    def test_login_command_is_quiet(self):
        command = launch.autostart_command("python")
        self.assertIn("--no-browser", command)
        self.assertIn("--skip-selftest", command)
        self.assertTrue(command[1].endswith("launch.py"))

    def test_entry_runs_the_launcher_from_the_project(self):
        contents = launch.autostart_contents(launch.autostart_command("python"))
        self.assertIn("launch.py", contents)
        self.assertIn(str(launch.ROOT), contents)


class ParserTests(unittest.TestCase):
    def test_check_and_skip_selftest_are_exclusive(self):
        with self.assertRaises(SystemExit):
            launch.build_parser().parse_args(["--check", "--skip-selftest"])

    def test_defaults(self):
        args = launch.build_parser().parse_args([])
        self.assertEqual((args.port, args.host, args.max_restarts), (8501, "127.0.0.1", 3))
        self.assertFalse(args.once or args.no_browser)


if __name__ == "__main__":
    unittest.main()
