import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from autoeval_planner.probe import run_probe


class _Handler(BaseHTTPRequestHandler):
    reject_stream = False
    omit_usage = False
    requests: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802 - http.server 约定
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests.append(
            {"path": self.path, "headers": dict(self.headers), "body": body}
        )

        if self.reject_stream and body.get("stream"):
            self._json(400, {"error": {"message": "stream is not supported"}})
            return

        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                {"choices": [{"delta": {"role": "assistant"}}]},
                {"choices": [{"delta": {"content": "hello"}}]},
                {"choices": [{"delta": {"content": " world"}}]},
            ]
            if not self.omit_usage:
                chunks.append({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2}})
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        payload = {"choices": [{"message": {"content": "hello world"}}]}
        if not self.omit_usage:
            payload["usage"] = {"prompt_tokens": 10, "completion_tokens": 2}
        self._json(200, payload)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        pass


@pytest.fixture
def fake_server():
    _Handler.reject_stream = False
    _Handler.omit_usage = False
    _Handler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _probe(url: str, **kwargs):
    params = dict(
        base_url=url,
        model="test-model",
        api_key="sk-test",
        concurrency_levels=[1],
        requests_per_level=1,
        prompt_tokens=16,
        max_tokens=8,
        timeout=10,
        retries=0,
    )
    params.update(kwargs)
    return run_probe(**params)


def test_stream_probe_uses_usage(fake_server):
    profile = _probe(fake_server)

    assert len(profile.levels) == 1
    level = profile.levels[0]
    assert level.decode_tps > 0
    assert level.ttft_sec is not None
    assert level.usage_estimated is False
    assert profile.request["stream"] is True
    assert _Handler.requests[0]["headers"]["Authorization"] == "Bearer sk-test"
    assert _Handler.requests[0]["body"]["stream_options"] == {"include_usage": True}


def test_non_stream_probe_has_no_ttft(fake_server):
    profile = _probe(fake_server, stream=False)

    level = profile.levels[0]
    assert level.ttft_sec is None
    assert level.decode_tps > 0
    assert "stream_options" not in _Handler.requests[0]["body"]
    assert profile.request["stream"] is False


def test_stream_rejected_falls_back_to_non_stream(fake_server):
    _Handler.reject_stream = True

    profile = _probe(fake_server, stream=True)

    assert profile.request["stream"] is False
    assert profile.levels[0].decode_tps > 0
    assert len(_Handler.requests) == 2
    assert _Handler.requests[0]["body"]["stream"] is True
    assert "stream" not in _Handler.requests[1]["body"]


def test_custom_auth_header_and_body_overrides(fake_server):
    profile = _probe(
        fake_server,
        api_key_header="api-key",
        api_key_scheme="",
        headers={"X-Test": "1"},
        body_overrides={"temperature": 0.7, "stream_options": None},
    )

    request = _Handler.requests[0]
    normalized = {key.lower(): value for key, value in request["headers"].items()}
    assert normalized["api-key"] == "sk-test"
    assert normalized["x-test"] == "1"
    assert request["body"]["temperature"] == 0.7
    assert "stream_options" not in request["body"]
    assert profile.request["api_key_header"] == "api-key"
    assert profile.request["headers"] == ["X-Test"]


def test_missing_usage_marks_estimated_tokens(fake_server):
    _Handler.omit_usage = True

    profile = _probe(fake_server)

    level = profile.levels[0]
    assert level.usage_estimated is True
    assert level.decode_tps > 0
    assert "usage" in (profile.notes or "")


def test_endpoint_override(fake_server):
    profile = _probe(fake_server, endpoint=fake_server + "/custom/chat/completions")

    assert _Handler.requests[0]["path"] == "/v1/custom/chat/completions"
    assert profile.request["endpoint"].endswith("/custom/chat/completions")
