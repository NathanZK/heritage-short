import base64
import http.client
import io
import json
import socket
import urllib.error
import wave

import pytest
from jsonschema import Draft202012Validator

from pipeline import mock, providers, smoke
from pipeline.common import ProviderError, call_model, check_film, read_runs, validate


def schnell_schema():
    from tests.conftest import FIXTURES
    return json.loads((FIXTURES / "flux-1-schnell-input.json").read_text())


def wav_b64(seconds=0.1):
    return base64.b64encode(mock.sine_wav(seconds)).decode()


@pytest.fixture
def capture(monkeypatch):
    calls = []

    def install(response):
        def fake(url, headers, body, timeout=120):
            calls.append((url, headers, body))
            return response
        monkeypatch.setattr(providers, "http_post", fake)
    return calls, install


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key-1234")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-1234")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token-1234")


def test_model_ids_are_the_prescribed_constants():
    assert providers.TEXT_MODEL == "gemini-3.8-flash"
    assert providers.TTS_MODEL == "gemini-3.8-flash-tts"
    assert providers.IMAGE_MODEL == "@cf/black-forest-labs/flux-1-schnell"


def test_cloudflare_request_has_only_documented_fields(capture):
    calls, install = capture
    install({"result": {"image": base64.b64encode(b"jpegbytes").decode()}, "success": True})
    assert providers.cloudflare_image("a prompt", 4) == b"jpegbytes"
    (url, headers, body), = calls
    assert url == "https://api.cloudflare.com/client/v4/accounts/acct-1234/ai/run/@cf/black-forest-labs/flux-1-schnell"
    assert headers["Authorization"] == "Bearer cf-token-1234"
    assert body == {"prompt": "a prompt", "steps": 4}
    validator = Draft202012Validator(schnell_schema())
    assert validator.is_valid(body)
    assert not validator.is_valid({**body, "seed": 1}), "the documented schema rejects seed"


def test_cloudflare_accepts_top_level_image_and_rejects_other_shapes(capture):
    calls, install = capture
    install({"image": base64.b64encode(b"x").decode()})
    assert providers.cloudflare_image("p", 4) == b"x"
    install({"result": {}, "success": True})
    with pytest.raises(ProviderError):
        providers.cloudflare_image("p", 4)
    install({"success": False, "errors": [{"code": 1}], "result": None})
    with pytest.raises(ProviderError):
        providers.cloudflare_image("p", 4)


def test_gemini_text_request_and_strict_parse(capture):
    calls, install = capture
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    install({"steps": [{"type": "model_output", "content": [{"type": "text", "text": '{"a": "b"}'}]}]})
    assert providers.gemini_json("write", schema) == {"a": "b"}
    (url, headers, body), = calls
    assert url == "https://generativelanguage.googleapis.com/v1beta/interactions"
    assert headers["x-goog-api-key"] == "gem-key-1234" and headers["Api-Revision"] == "2026-05-20"
    assert body == {"model": "gemini-3.8-flash", "input": "write",
                    "response_format": {"type": "text", "mime_type": "application/json", "schema": schema}}
    install({"steps": []})
    with pytest.raises(ProviderError):
        providers.gemini_json("write", schema)
    install({"steps": [{"type": "model_output", "content": [{"type": "text", "text": "not json"}]}]})
    with pytest.raises(ProviderError):
        providers.gemini_json("write", schema)


def test_gemini_tts_request_and_parse(capture):
    calls, install = capture
    install({"steps": [{"type": "user_input", "content": []},
                       {"type": "model_output", "content": [{"type": "audio", "data": wav_b64()}]}]})
    audio = providers.gemini_tts("hello there", "Kore")
    with wave.open(io.BytesIO(audio)) as w:
        assert w.getframerate() == 24000
    (_, _, body), = calls
    assert body == {"model": "gemini-3.8-flash-tts",
                    "input": [{"type": "user_input", "content": [{"type": "text", "text": "hello there"}]}],
                    "response_format": {"type": "audio"},
                    "generation_config": {"speech_config": [{"voice": "Kore"}]}}
    install({"steps": [{"type": "model_output", "content": [{"type": "text", "text": "x"}]}]})
    with pytest.raises(ProviderError):
        providers.gemini_tts("hi", "Kore")
    install({"steps": [{"type": "model_output", "content": [{"type": "audio",
                                                             "data": base64.b64encode(b"not wav").decode()}]}]})
    with pytest.raises(ProviderError):
        providers.gemini_tts("hi", "Kore")


@pytest.mark.parametrize("name", ["GEMINI_API_KEY", "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"])
def test_missing_secret_is_named(monkeypatch, name):
    monkeypatch.delenv(name)
    with pytest.raises(ProviderError) as exc:
        providers.gemini_json("p", {}) if name == "GEMINI_API_KEY" else providers.cloudflare_image("p", 4)
    assert name in str(exc.value) and ".env.example" in str(exc.value)


def fake_urlopen(monkeypatch, status, body=b"{}"):
    def opener(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, status, "err", {}, io.BytesIO(body))
    monkeypatch.setattr(providers.urllib.request, "urlopen", opener)


@pytest.mark.parametrize("status,body,retryable,quota", [
    (429, b'{"errors":[{"code":3036}]}', False, True),
    (429, b"slow down", True, False),
    (503, b"x", True, False),
    (400, b"x", False, False),
])
def test_http_errors_are_classified(monkeypatch, status, body, retryable, quota):
    fake_urlopen(monkeypatch, status, body)
    with pytest.raises(ProviderError) as exc:
        providers.http_post("https://example.invalid", {}, {})
    assert (exc.value.retryable, exc.value.quota, exc.value.status) == (retryable, quota, status)


def test_mock_media_is_deterministic_and_valid():
    assert mock.cloudflare_image("p", 4) == mock.cloudflare_image("p", 4)
    assert mock.cloudflare_image("p", 4) != mock.cloudflare_image("q", 4)
    assert mock.gemini_tts("one two three", "Kore") != mock.gemini_tts("one two four", "Kore")
    assert mock.gemini_tts("one two three", "Kore") == mock.gemini_tts("one two three", "Kore")
    with wave.open(io.BytesIO(mock.gemini_tts("word " * 40, "Kore"))) as w:
        assert w.getnframes() / w.getframerate() <= 3.0


def test_mock_script_is_a_valid_film(storyboard):
    sb = dict(storyboard, shots=mock.gemini_json("p", {})["shots"])
    validate(sb)
    check_film(sb)


def test_smoke_makes_one_call_per_provider(ctx):
    smoke.run(ctx, {"voice": "Kore"}, None)
    runs = read_runs(ctx.runs_path)
    assert [(r["stage"], r["provider"]) for r in runs] == [("smoke", "gemini"), ("smoke", "gemini"),
                                                          ("smoke", "cloudflare")]


@pytest.mark.parametrize("error", [
    socket.timeout("timed out"),
    http.client.RemoteDisconnected("closed"),
    http.client.IncompleteRead(b"x"),
    ConnectionResetError("reset"),
    urllib.error.URLError("no route"),
])
def test_network_failures_are_retried_and_logged(ctx, monkeypatch, error):
    delays, attempts = [], []
    ctx.sleep = delays.append

    def opener(req, timeout=None):
        attempts.append(1)
        raise error

    monkeypatch.setattr(providers.urllib.request, "urlopen", opener)
    with pytest.raises(ProviderError):
        call_model(ctx, stage="keyframes", shot="s01", provider="cloudflare", model="m", params={}, seed=None,
                   fn=lambda: providers.cloudflare_image("p", 4))
    assert len(attempts) == 3 and delays == [2, 4]
    assert [r["status"] for r in read_runs(ctx.runs_path)] == ["error"] * 3


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.body


def test_http_post_returns_the_parsed_body(monkeypatch):
    seen = {}

    def opener(req, timeout=None):
        seen["url"], seen["body"], seen["timeout"] = req.full_url, json.loads(req.data), timeout
        return FakeResponse(b'{"ok": true}')

    monkeypatch.setattr(providers.urllib.request, "urlopen", opener)
    assert providers.http_post("https://example.invalid/x", {"A": "b"}, {"k": 1}) == {"ok": True}
    assert seen == {"url": "https://example.invalid/x", "body": {"k": 1}, "timeout": 120}


def test_an_http_error_whose_body_cannot_be_read_is_still_retryable(monkeypatch):
    class Unreadable(io.BytesIO):
        def read(self, *args):
            raise socket.timeout("timed out reading the error body")

    def opener(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 503, "unavailable", {}, Unreadable())

    monkeypatch.setattr(providers.urllib.request, "urlopen", opener)
    with pytest.raises(ProviderError) as exc:
        providers.http_post("https://example.invalid/x", {}, {})
    assert exc.value.retryable and exc.value.status == 503


@pytest.mark.parametrize("body", [b"not json", b"[1, 2]", b'"text"', b"\xff\xfe"])
def test_http_post_rejects_non_object_bodies(monkeypatch, body):
    monkeypatch.setattr(providers.urllib.request, "urlopen", lambda req, timeout=None: FakeResponse(body))
    with pytest.raises(ProviderError):
        providers.http_post("https://example.invalid/x", {}, {})


@pytest.mark.parametrize("response", [
    {"result": {"image": "***not base64***"}},
    {"result": "text"},
    {"result": ["image"]},
])
def test_malformed_cloudflare_responses_raise_provider_errors(capture, response):
    _, install = capture
    install(response)
    with pytest.raises(ProviderError):
        providers.cloudflare_image("p", 4)


@pytest.mark.parametrize("response", [
    {"steps": "oops"},
    {"steps": [None]},
    {"steps": [{"type": "model_output", "content": "oops"}]},
    {"steps": [{"type": "model_output", "content": [{"type": "text", "text": '["a list"]'}]}]},
    {"steps": [{"type": "model_output", "content": [{"type": "text", "text": "[]"}]}]},
])
def test_malformed_gemini_text_responses_raise_provider_errors(capture, response):
    _, install = capture
    install(response)
    with pytest.raises(ProviderError):
        providers.gemini_json("p", {})


def test_malformed_gemini_audio_raises_a_provider_error(capture):
    _, install = capture
    install({"steps": [{"type": "model_output", "content": [{"type": "audio", "data": "***"}]}]})
    with pytest.raises(ProviderError):
        providers.gemini_tts("hi", "Kore")
