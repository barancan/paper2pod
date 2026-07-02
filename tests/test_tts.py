import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

import paper2pod.tts.edge as edge_module
import paper2pod.tts.elevenlabs as el_module
from paper2pod.logging_setup import TTSError
from paper2pod.tts import get_provider
from paper2pod.tts.edge import EdgeTTSProvider
from paper2pod.tts.elevenlabs import ElevenLabsTTSProvider


class FakeCommunicate:
    def __init__(self, text, voice, rate):
        self.text = text
        self.voice = voice
        self.rate = rate

    async def save(self, path):
        Path(path).write_bytes(b"FAKE_MP3_BYTES")


def test_edge_synthesize_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(edge_module.edge_tts, "Communicate", FakeCommunicate)
    provider = EdgeTTSProvider(voice="en-US-GuyNeural", rate="+8%")
    out = tmp_path / "out.mp3"
    result = provider.synthesize("hello world", out)
    assert result == out
    assert out.read_bytes() == b"FAKE_MP3_BYTES"


def test_edge_synthesize_retries_on_connection_error(tmp_path, monkeypatch):
    attempts = {"count": 0}

    class FlakyCommunicate:
        def __init__(self, text, voice, rate):
            pass

        async def save(self, path):
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise ConnectionError("boom")
            Path(path).write_bytes(b"OK")

    monkeypatch.setattr(edge_module.edge_tts, "Communicate", FlakyCommunicate)
    provider = EdgeTTSProvider()
    provider.synthesize("hi", tmp_path / "out.mp3")
    assert attempts["count"] == 2


def test_edge_synthesize_wraps_persistent_failure_as_tts_error(tmp_path, monkeypatch):
    class AlwaysFailCommunicate:
        def __init__(self, text, voice, rate):
            pass

        async def save(self, path):
            raise RuntimeError("network down")

    monkeypatch.setattr(edge_module.edge_tts, "Communicate", AlwaysFailCommunicate)
    provider = EdgeTTSProvider()
    with pytest.raises(TTSError):
        provider.synthesize("hi", tmp_path / "out.mp3")


class FakeResponse:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_elevenlabs_synthesize_writes_file(tmp_path, monkeypatch):
    def fake_urlopen(request, timeout=30):
        return FakeResponse(b"FAKE_AUDIO")

    monkeypatch.setattr(el_module.urllib.request, "urlopen", fake_urlopen)
    provider = ElevenLabsTTSProvider(api_key="sk-test", voice="voice123")
    out = tmp_path / "out.mp3"
    result = provider.synthesize("hello", out)
    assert result == out
    assert out.read_bytes() == b"FAKE_AUDIO"


def test_elevenlabs_401_raises_tts_error_without_retry(tmp_path, monkeypatch):
    attempts = {"count": 0}

    def fake_urlopen(request, timeout=30):
        attempts["count"] += 1
        raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", hdrs=None, fp=None)

    monkeypatch.setattr(el_module.urllib.request, "urlopen", fake_urlopen)
    provider = ElevenLabsTTSProvider(api_key="bad-key")
    with pytest.raises(TTSError, match="authentication failed"):
        provider.synthesize("hello", tmp_path / "out.mp3")
    assert attempts["count"] == 1


def test_elevenlabs_retries_on_500_then_succeeds(tmp_path, monkeypatch):
    attempts = {"count": 0}

    def fake_urlopen(request, timeout=30):
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise urllib.error.HTTPError(request.full_url, 503, "unavailable", hdrs=None, fp=None)
        return FakeResponse(b"OK")

    monkeypatch.setattr(el_module.urllib.request, "urlopen", fake_urlopen)
    provider = ElevenLabsTTSProvider(api_key="sk-test")
    provider.synthesize("hello", tmp_path / "out.mp3")
    assert attempts["count"] == 2


def test_get_provider_returns_edge_by_default():
    tts_config = SimpleNamespace(provider="edge", voice="en-US-GuyNeural", rate="+8%")
    provider = get_provider(tts_config, secrets=SimpleNamespace(elevenlabs_api_key=None))
    assert isinstance(provider, EdgeTTSProvider)


def test_get_provider_returns_elevenlabs_when_keyed():
    tts_config = SimpleNamespace(provider="elevenlabs", voice="voice123", rate="+8%")
    provider = get_provider(tts_config, secrets=SimpleNamespace(elevenlabs_api_key="sk-test"))
    assert isinstance(provider, ElevenLabsTTSProvider)


def test_get_provider_raises_when_elevenlabs_key_missing():
    tts_config = SimpleNamespace(provider="elevenlabs", voice="voice123", rate="+8%")
    with pytest.raises(TTSError, match="ELEVENLABS_API_KEY"):
        get_provider(tts_config, secrets=SimpleNamespace(elevenlabs_api_key=None))


def test_get_provider_raises_on_unknown_provider():
    tts_config = SimpleNamespace(provider="bogus", voice="x", rate="+8%")
    with pytest.raises(TTSError, match="Unknown TTS provider"):
        get_provider(tts_config, secrets=SimpleNamespace(elevenlabs_api_key=None))
