"""Local HTTP and synthetic payload tests; never contacts dataset services."""

from dataclasses import replace
import hashlib
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch

import requests
from requests.adapters import BaseAdapter

from openhisto.default_datasets.models import AcquisitionPlan, Asset
from openhisto.default_datasets.state import StateStore
from openhisto.default_datasets.transfer import download, verify


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError):
            # Rejected ranges intentionally close a response before its body.
            pass

    def do_GET(self):
        route = self.server.routes[self.path]
        route["requests"].append((self.headers.get("Range"), self.headers.get("If-Range")))
        if route.get("failures", 0):
            route["failures"] -= 1
            self.send_response(route.get("error_status", 503))
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        data = route["data"]
        start = 0
        range_header = self.headers.get("Range")
        if range_header and not route.get("ignore_range"):
            start = int(range_header.removeprefix("bytes=").removesuffix("-"))
        if start >= len(data) and start:
            self.send_response(416)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = data[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        if start:
            value = f"bytes {start}-{len(data) - 1}/{len(data)}"
            self.send_header("Content-Range", "bytes 0-1/2" if route.get("bad_range") else value)
        if route.get("etag"):
            self.send_header("ETag", route["etag"])
        if route.get("last_modified"):
            self.send_header("Last-Modified", route["last_modified"])
        self.end_headers()
        if route.get("cut_once"):
            route["cut_once"] = False
            self.wfile.write(body[: len(body) // 2])
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return
        self.wfile.write(body)


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server.routes = {}
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "acquisition"
        self.server.routes = {}
        self.data = bytes(range(251)) * 17
        self.sleep_patch = patch("openhisto.default_datasets.transfer.time.sleep")
        self.sleep_patch.start()
        self.addCleanup(self.sleep_patch.stop)

    def asset(self, endpoint="/slide", *, checksums=True, size=True, **route_options):
        self.server.routes[endpoint] = {
            "data": self.data,
            "etag": '"version-1"',
            "requests": [],
            **route_options,
        }
        return Asset(
            asset_id=endpoint.removeprefix("/"),
            dataset="TCGA",
            kind="slide",
            filename="slide.svs",
            relative_path="slides/slide.svs",
            url=f"http://127.0.0.1:{self.server.server_port}{endpoint}",
            size=len(self.data) if size else None,
            checksums={"md5": hashlib.md5(self.data).hexdigest()} if checksums else {},
        )

    def plan(self, asset, dataset="TCGA"):
        return AcquisitionPlan(
            dataset=dataset, selection={"asset_ids": [asset.asset_id]}, assets=[asset]
        )

    def seed_partial(self, asset, length=100, *, etag='"version-1"', last_modified=None):
        store = StateStore(self.root)
        store.register(self.plan(asset))
        part = self.root / (asset.relative_path + ".part")
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(self.data[:length])
        store.update(
            "TCGA", asset.asset_id, status="failed", etag=etag, last_modified=last_modified
        )
        return part

    def test_download_hash_skip_and_offline_verify(self):
        asset = self.asset()
        plan = self.plan(asset)
        report = download(plan, self.root, retries=0, chunk_size=31)
        self.assertEqual((report.downloaded, report.skipped, report.failed), (1, 0, 0))
        self.assertEqual((self.root / asset.relative_path).read_bytes(), self.data)
        row = StateStore(self.root).get("TCGA", asset.asset_id)
        self.assertEqual(row["status"], "complete")
        self.assertEqual(row["integrity"], "checksum_verified")
        self.assertEqual(row["sha256"], hashlib.sha256(self.data).hexdigest())
        self.assertTrue((self.root / "acquisition_plan.json").is_file())
        self.assertFalse((self.root / (asset.relative_path + ".part")).exists())
        asset.labels = {"task": "synthetic"}
        report = download(plan, self.root, retries=0)
        self.assertEqual((report.downloaded, report.skipped), (0, 1))
        self.assertEqual(len(self.server.routes["/slide"]["requests"]), 1)
        with patch(
            "requests.Session.get", side_effect=AssertionError("verification must be offline")
        ):
            result = verify(self.root)
        self.assertEqual(
            (result.checked, result.valid, result.failed, result.missing), (1, 1, 0, 0)
        )

    def test_interrupted_transfer_resumes_with_validated_range(self):
        asset = self.asset(cut_once=True)
        report = download(self.plan(asset), self.root, retries=1, chunk_size=31)
        self.assertEqual(report.downloaded, 1)
        requests_seen = self.server.routes["/slide"]["requests"]
        self.assertEqual(requests_seen[0], (None, None))
        self.assertTrue(requests_seen[1][0].startswith("bytes="))
        self.assertEqual(requests_seen[1][1], '"version-1"')
        self.assertEqual((self.root / asset.relative_path).read_bytes(), self.data)

    def test_seeded_resume_and_if_range(self):
        asset = self.asset()
        self.seed_partial(asset, length=123)
        report = download(self.plan(asset), self.root, retries=0)
        self.assertEqual(report.downloaded, 1)
        self.assertEqual(self.server.routes["/slide"]["requests"], [("bytes=123-", '"version-1"')])

    def test_server_ignoring_range_restarts_without_appending(self):
        asset = self.asset(ignore_range=True)
        self.seed_partial(asset)
        report = download(self.plan(asset), self.root, retries=0)
        self.assertEqual(report.downloaded, 1)
        self.assertEqual((self.root / asset.relative_path).read_bytes(), self.data)

    def test_unknown_or_weak_validator_restarts(self):
        for etag in (None, 'W/"version-1"'):
            with self.subTest(etag=etag):
                asset = self.asset()
                self.seed_partial(asset, etag=etag)
                report = download(self.plan(asset), self.root, retries=0)
                self.assertEqual(report.downloaded, 1)
                self.assertEqual(self.server.routes["/slide"]["requests"], [(None, None)])
                (self.root / asset.relative_path).unlink()
                store = StateStore(self.root)
                store.update("TCGA", asset.asset_id, sha256=None)

    def test_last_modified_can_validate_resume(self):
        modified = "Mon, 26 Oct 2026 12:00:00 GMT"
        asset = self.asset(etag=None, last_modified=modified)
        self.seed_partial(asset, etag=None, last_modified=modified)
        report = download(self.plan(asset), self.root, retries=0)
        self.assertEqual(report.downloaded, 1)
        self.assertEqual(self.server.routes["/slide"]["requests"], [("bytes=100-", modified)])

    def test_changed_remote_identity_discards_partial(self):
        asset = self.asset(etag='"version-2"')
        self.seed_partial(asset, etag='"version-1"')
        report = download(self.plan(asset), self.root, retries=1)
        self.assertEqual(report.downloaded, 1)
        self.assertEqual(
            self.server.routes["/slide"]["requests"], [("bytes=100-", '"version-1"'), (None, None)]
        )

    def test_bad_content_range_cannot_be_published(self):
        asset = self.asset(bad_range=True)
        part = self.seed_partial(asset)
        report = download(self.plan(asset), self.root, retries=0)
        self.assertEqual(report.failed, 1)
        self.assertFalse(part.exists())
        self.assertFalse((self.root / asset.relative_path).exists())
        self.assertEqual(
            StateStore(self.root).get("TCGA", asset.asset_id)["error"], "InvalidContentRange"
        )

    def test_checksum_failure_retries_but_never_publishes(self):
        asset = self.asset()
        self.server.routes["/slide"]["data"] = b"X" * len(self.data)
        report = download(self.plan(asset), self.root, retries=1)
        self.assertEqual((report.downloaded, report.failed), (0, 1))
        self.assertEqual(len(self.server.routes["/slide"]["requests"]), 2)
        self.assertFalse((self.root / asset.relative_path).exists())
        self.assertFalse((self.root / (asset.relative_path + ".part")).exists())
        self.assertEqual(
            StateStore(self.root).get("TCGA", asset.asset_id)["error"], "ChecksumMismatch"
        )

    def test_transient_http_retries_and_permanent_http_does_not(self):
        asset = self.asset(failures=1)
        report = download(self.plan(asset), self.root, retries=1)
        self.assertEqual(report.downloaded, 1)
        (self.root / asset.relative_path).unlink()
        asset = self.asset(failures=2, error_status=404)
        report = download(self.plan(asset), self.root, retries=2)
        self.assertEqual(report.failed, 1)
        self.assertEqual(len(self.server.routes["/slide"]["requests"]), 1)

    def test_state_identity_change_invalidates_status_and_partials(self):
        asset = self.asset()
        download(self.plan(asset), self.root, retries=0)
        old_part = self.root / (asset.relative_path + ".part")
        old_part.write_bytes(b"stale partial")
        changed = replace(asset, relative_path="slides/revised.svs", url=asset.url + "?revision=2")
        new_part = self.root / (changed.relative_path + ".part")
        new_part.write_bytes(b"another stale partial")
        store = StateStore(self.root)
        store.register(self.plan(changed))
        row = store.get("TCGA", asset.asset_id)
        self.assertEqual(row["status"], "pending")
        self.assertIsNone(row["sha256"])
        self.assertIsNone(row["etag"])
        self.assertFalse(old_part.exists())
        self.assertFalse(new_part.exists())
        self.assertTrue((self.root / asset.relative_path).exists())

    def test_corruption_is_detected_and_previous_sha_is_preserved(self):
        asset = self.asset(checksums=False)
        download(self.plan(asset), self.root, retries=0)
        baseline = StateStore(self.root).get("TCGA", asset.asset_id)["sha256"]
        (self.root / asset.relative_path).write_bytes(b"X" * len(self.data))
        report = verify(self.root)
        self.assertEqual(report.failed, 1)
        row = StateStore(self.root).get("TCGA", asset.asset_id)
        self.assertEqual(row["sha256"], baseline)
        self.assertEqual(row["error"], "LocalHashMismatch")
        report = download(self.plan(asset), self.root, retries=0)
        self.assertEqual(report.downloaded, 1)
        self.assertEqual((self.root / asset.relative_path).read_bytes(), self.data)

    def test_local_import_missing_and_hash_only_baseline(self):
        local = Asset("local", "TCGA", "slide", "local.svs", "slides/local.svs")
        missing = replace(
            local, asset_id="missing", filename="missing.svs", relative_path="slides/missing.svs"
        )
        path = self.root / local.relative_path
        path.parent.mkdir(parents=True)
        path.write_bytes(self.data)
        plan = AcquisitionPlan("TCGA", {"local": True}, [local, missing])
        with patch(
            "requests.Session.get", side_effect=AssertionError("local imports must be offline")
        ):
            report = download(plan, self.root, retries=0)
        self.assertEqual((report.skipped, report.failed), (1, 1))
        store = StateStore(self.root)
        self.assertEqual(store.get("TCGA", "local")["integrity"], "local_hash_only")
        self.assertEqual(store.get("TCGA", "missing")["status"], "missing")
        result = verify(self.root)
        self.assertEqual((result.valid, result.missing, result.failed), (1, 1, 0))
        path.write_bytes(b"Y" * len(self.data))
        self.assertEqual(verify(self.root).failed, 1)

    def test_unknown_size_hash_only_and_empty_payload(self):
        asset = self.asset(checksums=False, size=False)
        report = download(self.plan(asset), self.root, retries=0)
        self.assertEqual(report.downloaded, 1)
        self.assertEqual(
            StateStore(self.root).get("TCGA", asset.asset_id)["integrity"], "size_verified"
        )
        empty = self.asset("/empty")
        empty = replace(
            empty,
            relative_path="slides/empty.svs",
            size=0,
            checksums={"sha256": hashlib.sha256(b"").hexdigest()},
        )
        self.server.routes["/empty"]["data"] = b""
        report = download(self.plan(empty), self.root, retries=0)
        self.assertEqual(report.downloaded, 1)
        self.assertEqual((self.root / empty.relative_path).stat().st_size, 0)
        self.assertEqual(verify(self.root).checked, 1)  # current plan, not all historical rows

    def test_root_dataset_and_control_path_protection(self):
        store = StateStore(self.root)
        asset = self.asset()
        store.register(self.plan(asset))
        other = replace(asset, dataset="KPMP")
        with self.assertRaises(ValueError):
            store.register(self.plan(other, "KPMP"))
        for relative in (
            "acquisition.sqlite",
            "acquisition.sqlite-wal",
            "acquisition_plan.json",
            "manifest.jsonl",
            "slides.csv",
            "clinical.jsonl",
            "provenance.json",
            "source_metadata/records.jsonl",
            "manifest.jsonl.part",
        ):
            with self.subTest(relative=relative):
                bad = replace(asset, relative_path=relative)
                with self.assertRaises(ValueError):
                    store.register(self.plan(bad))

    def test_symlink_escape_and_partial_collisions_are_rejected(self):
        asset = self.asset()
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        self.root.mkdir()
        (self.root / "slides").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            download(self.plan(asset), self.root, retries=0)
        self.assertEqual(list(outside.iterdir()), [])
        (self.root / "slides").unlink()
        alias = replace(asset, asset_id="collision", relative_path=asset.relative_path + ".part")
        with self.assertRaises(ValueError):
            StateStore(self.root).register(AcquisitionPlan("TCGA", {}, [asset, alias]))

    def test_errors_are_redacted_and_session_credentials_are_refused(self):
        asset = self.asset()

        class FailedSession:
            trust_env = True

            def get(self, *args, **kwargs):
                raise requests.ConnectionError("https://private.invalid?token=DO_NOT_PERSIST")

        session = FailedSession()
        report = download(self.plan(asset), self.root, retries=0, session=session)
        self.assertEqual(report.failed, 1)
        self.assertTrue(session.trust_env)
        row = StateStore(self.root).get("TCGA", asset.asset_id)
        self.assertEqual(row["error"], "NetworkError")
        self.assertNotIn("DO_NOT_PERSIST", str(row))
        with requests.Session() as unsafe:
            unsafe.headers["Authorization"] = "Bearer DO_NOT_PERSIST"
            with self.assertRaises(ValueError):
                download(self.plan(asset), self.root, session=unsafe)

    def test_transport_environment_survives_without_netrc_on_redirects(self):
        payload = self.data

        class RecordingAdapter(BaseAdapter):
            def __init__(self):
                self.calls = []
                self.closed = False

            def send(self, request, **options):
                self.calls.append((request, options))
                response = requests.Response()
                response.request = request
                response.url = request.url
                response.status_code = 302 if request.url.endswith("/redirect") else 200
                response.headers["Content-Length"] = str(
                    0 if response.status_code == 302 else len(payload)
                )
                if response.status_code == 302:
                    response.headers["Location"] = "https://cdn.example.invalid/slide"
                response._content = b"" if response.status_code == 302 else payload
                response._content_consumed = True
                return response

            def close(self):
                self.closed = True

        for no_proxy, expected_proxies in (
            ("", {"https": "http://proxy.example.invalid:3128"}),
            ("*", {}),
        ):
            with self.subTest(no_proxy=no_proxy), requests.Session() as session:
                adapter = RecordingAdapter()
                session.mount("https://", adapter)
                original_headers = session.headers.copy()
                original_hooks = session.hooks
                asset = replace(self.asset(), url="https://origin.example.invalid/redirect")
                environment = {
                    "HTTPS_PROXY": "http://proxy.example.invalid:3128",
                    "NO_PROXY": no_proxy,
                    "REQUESTS_CA_BUNDLE": "/configured/company-ca.pem",
                }
                root = self.root / ("bypass" if no_proxy else "proxy")
                with (
                    patch.dict(os.environ, environment, clear=True),
                    patch(
                        "requests.sessions.get_netrc_auth",
                        side_effect=AssertionError("implicit netrc lookup"),
                    ),
                ):
                    report = download(self.plan(asset), root, retries=0, session=session)
                self.assertEqual((report.downloaded, report.failed), (1, 0))
                self.assertEqual(len(adapter.calls), 2)
                for request, options in adapter.calls:
                    self.assertNotIn("Authorization", request.headers)
                    self.assertEqual(dict(options["proxies"]), expected_proxies)
                    self.assertEqual(options["verify"], environment["REQUESTS_CA_BUNDLE"])
                self.assertTrue(session.trust_env)
                self.assertIsNone(session.auth)
                self.assertEqual(session.headers, original_headers)
                self.assertIs(session.hooks, original_hooks)
                self.assertFalse(adapter.closed)


if __name__ == "__main__":
    unittest.main()
