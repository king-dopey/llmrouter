import ast
import asyncio
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock
import logging

logging.getLogger("router").setLevel(logging.WARNING)

ROUTER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUTER_DIR))


def _install_dependency_stubs() -> None:
    if "httpx" not in sys.modules:
        httpx = types.ModuleType("httpx")

        class Timeout:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class AsyncClient:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class HTTPError(Exception):
            pass

        class HTTPStatusError(HTTPError):
            def __init__(self, message="", request=None, response=None):
                super().__init__(message)
                self.response = response or types.SimpleNamespace(status_code=500)
                self.request = request

        class ConnectError(HTTPError):
            pass

        class ConnectTimeout(HTTPError):
            pass

        class Response:
            def __init__(self, status_code=200, content=b"", headers=None):
                self.status_code = status_code
                self.content = content
                self.headers = headers or {}

        httpx.Timeout = Timeout
        httpx.AsyncClient = AsyncClient
        httpx.HTTPError = HTTPError
        httpx.HTTPStatusError = HTTPStatusError
        httpx.ConnectError = ConnectError
        httpx.ConnectTimeout = ConnectTimeout
        httpx.Response = Response
        sys.modules["httpx"] = httpx

    if "fastapi" not in sys.modules:
        fastapi = types.ModuleType("fastapi")

        class FastAPI:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def get(self, *args, **kwargs):
                def decorator(func):
                    return func

                return decorator

            def post(self, *args, **kwargs):
                def decorator(func):
                    return func

                return decorator

        class HTTPException(Exception):
            def __init__(self, status_code: int, detail: str):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        class Request:
            pass

        fastapi.FastAPI = FastAPI
        fastapi.HTTPException = HTTPException
        fastapi.Request = Request
        sys.modules["fastapi"] = fastapi

    if "fastapi.responses" not in sys.modules:
        responses = types.ModuleType("fastapi.responses")

        class JSONResponse:
            def __init__(self, content, status_code: int = 200):
                self.content = content
                self.status_code = status_code

        class StreamingResponse:
            def __init__(self, content, media_type: str | None = None):
                self.content = content
                self.media_type = media_type

        responses.JSONResponse = JSONResponse
        responses.StreamingResponse = StreamingResponse
        sys.modules["fastapi.responses"] = responses


_install_dependency_stubs()


def _decorator_paths(module_path: Path, method_name: str) -> set[str]:
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))
    paths: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        for deco in node.decorator_list:
            if not isinstance(deco, ast.Call):
                continue
            if not isinstance(deco.func, ast.Attribute):
                continue
            if deco.func.attr != method_name or not deco.args:
                continue
            arg0 = deco.args[0]
            if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                paths.add(arg0.value)
    return paths


class _FakeRequest:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


class _FakeURL:
    def __init__(self, hostname):
        self.hostname = hostname


class _FakeRequestWithURL:
    def __init__(self, hostname):
        self.url = _FakeURL(hostname)


class _FakeEndpointRequest:
    def __init__(self, hostname, payload):
        self.url = _FakeURL(hostname)
        self.headers = {"content-type": "application/json"}
        self._payload = payload

    async def json(self):
        return self._payload


class _FakeUpload:
    def __init__(self, filename: str, content_type: str, data: bytes):
        self.filename = filename
        self.content_type = content_type
        self._data = data

    async def read(self) -> bytes:
        return self._data


class _FakeMultipartRequest:
    def __init__(self, hostname: str, form_payload: dict):
        self.url = _FakeURL(hostname)
        self.headers = {"content-type": "multipart/form-data; boundary=abc"}
        self._form_payload = form_payload

    async def form(self):
        return self._form_payload


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


class RouterPayloadTests(unittest.TestCase):
    """Test app.py functionality."""

    def test_router_registers_alignment_routes(self):
        post_paths = _decorator_paths(ROUTER_DIR / "app.py", "post")
        self.assertIn("/align", post_paths)
        self.assertIn("/v1/audio/align", post_paths)
        self.assertIn("/v1/audio/transcriptions", post_paths)
        self.assertIn("/v1/embeddings", post_paths)

    def test_resolve_asr_base_url_uses_explicit_override(self):
        import app as app_module

        original_base = app_module.ASR_BASE_URL_ENV
        original_port = app_module.ASR_PORT
        original_scheme = app_module.ASR_SCHEME
        try:
            app_module.ASR_BASE_URL_ENV = "http://asr-override:18000"
            app_module.ASR_PORT = 8000
            app_module.ASR_SCHEME = "http"
            resolved = app_module._resolve_asr_base_url(_FakeRequestWithURL("router-host"))
            self.assertEqual("http://asr-override:18000", resolved)
        finally:
            app_module.ASR_BASE_URL_ENV = original_base
            app_module.ASR_PORT = original_port
            app_module.ASR_SCHEME = original_scheme

    def test_resolve_asr_base_url_infers_from_request_host_when_unset(self):
        import app as app_module

        original_base = app_module.ASR_BASE_URL_ENV
        original_port = app_module.ASR_PORT
        original_scheme = app_module.ASR_SCHEME
        try:
            original_ollama = app_module.OLLAMA_BASE_URL
            app_module.ASR_BASE_URL_ENV = ""
            app_module.ASR_PORT = 18000
            app_module.ASR_SCHEME = "http"
            app_module.OLLAMA_BASE_URL = "http://orin-model-host:11434"
            resolved = app_module._resolve_asr_base_url(_FakeRequestWithURL("router-host"))
            self.assertEqual("http://orin-model-host:18000", resolved)
        finally:
            app_module.ASR_BASE_URL_ENV = original_base
            app_module.ASR_PORT = original_port
            app_module.ASR_SCHEME = original_scheme
            app_module.OLLAMA_BASE_URL = original_ollama

    def test_v1_audio_align_rejects_json_path_only_payloads(self):
        import app as app_module

        original_admission = app_module._ensure_asr_admission
        original_asr_post_json = app_module._asr_post_json
        try:
            async def _allow_admission():
                return True

            async def _fake_asr_post(base_url, path, payload):
                raise AssertionError("_asr_post_json should not be called for cross-host alignment")

            app_module._ensure_asr_admission = _allow_admission
            app_module._asr_post_json = _fake_asr_post

            with self.assertRaises(Exception) as ctx:
                asyncio.run(
                    app_module.v1_audio_align(
                        _FakeEndpointRequest(
                            hostname="router-host",
                            payload={
                                "audio_path": "/work/_audio.wav",
                                "media_path": "/work/input.mp4",
                                "model": "whisper-large-v3-turbo",
                                "model_accuracy": "whisper-large-v3",
                                "return_word_timestamps": True,
                                "prefer_forced_alignment": True,
                            },
                        )
                    )
                )
            self.assertEqual(400, getattr(ctx.exception, "status_code", None))
            self.assertIn(
                "cross_host_alignment_requires_multipart_upload",
                str(getattr(ctx.exception, "detail", "")),
            )
        finally:
            app_module._ensure_asr_admission = original_admission
            app_module._asr_post_json = original_asr_post_json

    def test_v1_audio_align_forwards_multipart_upload(self):
        import app as app_module

        original_admission = app_module._ensure_asr_admission
        original_asr_post_multipart = app_module._asr_post_multipart
        try:
            async def _allow_admission():
                return True

            async def _fake_asr_post_multipart(base_url, path, *, fields, file_field, file_name, file_bytes, file_content_type):
                self.assertEqual("http://ollama:8000", base_url)
                self.assertEqual("/align", path)
                self.assertEqual("media_file", file_field)
                self.assertEqual("_audio.wav", file_name)
                self.assertEqual(b"RIFF....WAVEfmt ", file_bytes)
                self.assertEqual("audio/wav", file_content_type)
                self.assertEqual("whisper-large-v3-turbo", fields["model"])
                self.assertEqual("true", fields["prefer_forced_alignment"])
                self.assertNotIn("audio_path", fields)
                self.assertNotIn("media_path", fields)
                return _FakeResponse(status_code=200, payload={"forced_alignment_used": True, "words": []})

            app_module._ensure_asr_admission = _allow_admission
            app_module._asr_post_multipart = _fake_asr_post_multipart

            req = _FakeMultipartRequest(
                hostname="router-host",
                form_payload={
                    "media_file": _FakeUpload("_audio.wav", "audio/wav", b"RIFF....WAVEfmt "),
                    "media_path": "/work/input.mp4",
                    "audio_path": "/work/_audio.wav",
                    "model": "whisper-large-v3-turbo",
                    "prefer_forced_alignment": "true",
                    "return_word_timestamps": "true",
                },
            )
            response = asyncio.run(app_module.v1_audio_align(req))
            self.assertEqual(200, response.status_code)
            self.assertTrue(response.content["forced_alignment_used"])
        finally:
            app_module._ensure_asr_admission = original_admission
            app_module._asr_post_multipart = original_asr_post_multipart

    def test_build_payload_forwards_unlocked_options_format_and_keep_alive(self):
        """Test with Thor profile model (qwen3-coder-next:q4_K_M)."""
        from unittest.mock import patch
        import importlib
        
        with patch.dict(os.environ, {"MODEL_POLICY_FILE": str(ROUTER_DIR / "profiles" / "thor" / "models.yaml")}):
            import app as app_module
            importlib.reload(app_module)
            
            body = {
                "model": "qwen3-coder-next:q4_K_M",
                "messages": [{"role": "user", "content": "hi"}],
                "keep_alive": "5m",
                "format": "json",
                "options": {
                    "num_batch": 1024,          # not policy-locked: client should win
                    "repeat_penalty": 1.2,       # not policy-locked: client should win
                },
                "temperature": 0.4,
                "max_tokens": 256,
            }
            
            payload = app_module._build_ollama_payload(body, think=False)

            self.assertEqual(payload["keep_alive"], "5m")
            self.assertEqual(payload["format"], "json")
            self.assertEqual(payload["options"]["num_batch"], 1024)
            self.assertEqual(payload["options"]["repeat_penalty"], 1.2)
            self.assertEqual(payload["options"]["temperature"], 0.4)
            self.assertEqual(payload["options"]["num_predict"], 256)
            # Policy default still present because client did not override it.
            self.assertEqual(payload["options"]["num_ctx"], 131072)

    def test_build_payload_policy_num_ctx_overrides_client(self):
        """Test with Thor profile model (qwen3-coder-next:q4_K_M)."""
        from unittest.mock import patch
        import importlib
        
        with patch.dict(os.environ, {"MODEL_POLICY_FILE": str(ROUTER_DIR / "profiles" / "thor" / "models.yaml")}):
            import app as app_module
            importlib.reload(app_module)
            
            body = {
                "model": "qwen3-coder-next:q4_K_M",
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"num_ctx": 16384},
            }
            
            with self.assertLogs("router", level="INFO") as cm:
                payload = app_module._build_ollama_payload(body, think=False)
            self.assertEqual(payload["options"]["num_ctx"], 131072)
            self.assertTrue(any("overriding client num_ctx" in m for m in cm.output))

    def test_build_payload_uses_policy_defaults_when_client_silent(self):
        """Test with Thor profile model (qwen3-coder-next:q4_K_M)."""
        from unittest.mock import patch
        import importlib
        
        with patch.dict(os.environ, {"MODEL_POLICY_FILE": str(ROUTER_DIR / "profiles" / "thor" / "models.yaml")}):
            import app as app_module
            importlib.reload(app_module)
            
            body = {
                "model": "qwen3-coder-next:q4_K_M",
                "messages": [{"role": "user", "content": "hi"}],
            }
            
            payload = app_module._build_ollama_payload(body, think=False)
            opts = payload["options"]
            self.assertEqual(opts["num_ctx"], 131072)
            self.assertEqual(opts["num_batch"], 256)
            self.assertEqual(opts["temperature"], 0.12)
            self.assertEqual(payload["keep_alive"], "45m")  # from policy

    def test_build_payload_uses_model_keep_alive_when_request_omits_it(self):
        """Test with Orin profile model (nemotron-cascade-2:30b-a3b-q4_K_M)."""
        from unittest.mock import patch
        import importlib
        
        # Save original state
        original_policy_file = os.environ.get("MODEL_POLICY_FILE")
        import app as app_module
        
        try:
            with patch.dict(os.environ, {"MODEL_POLICY_FILE": str(ROUTER_DIR / "profiles" / "orin" / "models.yaml")}):
                # Reload app module to pick up the new environment variable
                importlib.reload(app_module)
                
                body = {
                    "model": "nemotron-cascade-2:30b-a3b-q4_K_M",
                    "messages": [{"role": "user", "content": "Verify this answer."}],
                }
                
                payload = app_module._build_ollama_payload(body, think=True)

                self.assertEqual(payload["keep_alive"], "10m")
        finally:
            # Restore original state
            if original_policy_file is not None:
                os.environ["MODEL_POLICY_FILE"] = original_policy_file
            elif "MODEL_POLICY_FILE" in os.environ:
                del os.environ["MODEL_POLICY_FILE"]
            # Reload the module to restore original state
            importlib.reload(app_module)


class TestWarmupAutoPull(unittest.IsolatedAsyncioTestCase):
    """Tests for _warmup_model() auto-pull behavior on 404 responses."""

    def setUp(self):
        import importlib
        import app as app_module
        self.app = app_module

    def _make_entry(self):
        return {
            "keep_alive": "-1",
            "options": {"num_ctx": 65536},
        }

    async def test_warmup_success_no_pull(self):
        """Warmup succeeds on first try — no pull triggered."""
        from unittest.mock import AsyncMock, patch

        entry = self._make_entry()
        mock_response = mock.MagicMock()
        mock_response.status_code = 200

        with (
            patch.object(self.app, "_send_warmup_request", new=AsyncMock(return_value=mock_response)),
            patch.object(self.app, "pull_model", new=AsyncMock()) as mock_pull,
        ):
            await self.app._warmup_model("test-model", entry)
            mock_pull.assert_not_called()

    async def test_warmup_404_triggers_pull_when_auto_pull_enabled(self):
        """404 triggers pull_model when AUTO_PULL_MISSING_MODELS is True."""
        from unittest.mock import AsyncMock, patch
        import httpx

        entry = self._make_entry()
        mock_404_response = unittest.mock.MagicMock(status_code=404)
        mock_200_response = mock.MagicMock(status_code=200)
        status_error = httpx.HTTPStatusError(
            "Not Found",
            request=mock.MagicMock(),
            response=mock_404_response,
        )

        # First call raises 404, second call succeeds
        send_mock = AsyncMock(side_effect=[status_error, mock_200_response])

        with (
            patch.object(self.app, "AUTO_PULL_MISSING_MODELS", True),
            patch.object(self.app, "_send_warmup_request", send_mock),
            patch.object(self.app, "pull_model", new=AsyncMock(return_value=True)) as mock_pull,
        ):
            await self.app._warmup_model("test-model", entry)
            mock_pull.assert_called_once_with("test-model")
            self.assertEqual(send_mock.call_count, 2)

    async def test_warmup_404_no_pull_when_auto_pull_disabled(self):
        """404 does NOT trigger pull when AUTO_PULL_MISSING_MODELS is False."""
        from unittest.mock import AsyncMock, patch
        import httpx

        entry = self._make_entry()
        mock_404_response = mock.MagicMock(status_code=404)
        status_error = httpx.HTTPStatusError(
            "Not Found",
            request=mock.MagicMock(),
            response=mock_404_response,
        )

        with (
            patch.object(self.app, "AUTO_PULL_MISSING_MODELS", False),
            patch.object(self.app, "_send_warmup_request", new=AsyncMock(side_effect=status_error)),
            patch.object(self.app, "pull_model", new=AsyncMock()) as mock_pull,
        ):
            await self.app._warmup_model("test-model", entry)
            mock_pull.assert_not_called()

    async def test_warmup_404_pull_fails_logs_warning(self):
        """404 + pull returns False — warning logged, no crash."""
        from unittest.mock import AsyncMock, patch
        import httpx

        entry = self._make_entry()
        mock_404_response = mock.MagicMock(status_code=404)
        status_error = httpx.HTTPStatusError(
            "Not Found",
            request=mock.MagicMock(),
            response=mock_404_response,
        )

        with (
            patch.object(self.app, "AUTO_PULL_MISSING_MODELS", True),
            patch.object(self.app, "_send_warmup_request", new=AsyncMock(side_effect=status_error)),
            patch.object(self.app, "pull_model", new=AsyncMock(return_value=False)) as mock_pull,
        ):
            # Should not raise
            await self.app._warmup_model("test-model", entry)
            mock_pull.assert_called_once_with("test-model")

    async def test_warmup_404_pull_succeeds_retry_fails(self):
        """404 + pull succeeds + retry fails — warning logged, no crash."""
        from unittest.mock import AsyncMock, patch
        import httpx

        entry = self._make_entry()
        mock_404_response = mock.MagicMock(status_code=404)
        status_error_404 = httpx.HTTPStatusError(
            "Not Found",
            request=mock.MagicMock(),
            response=mock_404_response,
        )
        generic_error = Exception("retry failed")

        # First call: 404. Second call: generic exception.
        send_mock = AsyncMock(side_effect=[status_error_404, generic_error])

        with (
            patch.object(self.app, "AUTO_PULL_MISSING_MODELS", True),
            patch.object(self.app, "_send_warmup_request", send_mock),
            patch.object(self.app, "pull_model", new=AsyncMock(return_value=True)) as mock_pull,
        ):
            # Should not raise
            await self.app._warmup_model("test-model", entry)
            mock_pull.assert_called_once_with("test-model")
            self.assertEqual(send_mock.call_count, 2)

    async def test_warmup_non_404_no_pull(self):
        """Non-404 HTTP error does NOT trigger pull."""
        from unittest.mock import AsyncMock, patch
        import httpx

        entry = self._make_entry()
        mock_500_response = mock.MagicMock(status_code=500)
        status_error = httpx.HTTPStatusError(
            "Internal Server Error",
            request=mock.MagicMock(),
            response=mock_500_response,
        )

        with (
            patch.object(self.app, "AUTO_PULL_MISSING_MODELS", True),
            patch.object(self.app, "_send_warmup_request", new=AsyncMock(side_effect=status_error)),
            patch.object(self.app, "pull_model", new=AsyncMock()) as mock_pull,
        ):
            await self.app._warmup_model("test-model", entry)
            mock_pull.assert_not_called()

    async def test_warmup_connection_error_no_pull(self):
        """ConnectError does NOT trigger pull — Ollama host unreachable."""
        from unittest.mock import AsyncMock, patch
        import httpx

        entry = self._make_entry()

        with (
            patch.object(self.app, "AUTO_PULL_MISSING_MODELS", True),
            patch.object(self.app, "_send_warmup_request", new=AsyncMock(side_effect=httpx.ConnectError("connection refused"))),
            patch.object(self.app, "pull_model", new=AsyncMock()) as mock_pull,
        ):
            await self.app._warmup_model("test-model", entry)
            mock_pull.assert_not_called()


class TestEmbeddingModelDetection(unittest.TestCase):
    """Tests for _is_embedding_model() detection logic."""

    def setUp(self):
        import importlib
        import app as app_module
        self.app = app_module

    def test_embedding_model_detected_by_name(self):
        """Models with 'embed' in name are detected as embedding models."""
        self.assertTrue(self.app._is_embedding_model("qwen3-embedding:8b"))
        self.assertTrue(self.app._is_embedding_model("nomic-embed-text"))
        self.assertTrue(self.app._is_embedding_model("EMBEDDING-model"))
        self.assertTrue(self.app._is_embedding_model("model-embed-v1"))

    def test_chat_model_not_detected_as_embedding(self):
        """Chat models without 'embed' are not detected as embedding models."""
        self.assertFalse(self.app._is_embedding_model("qwen3.6:35b-a3b-q8_0"))
        self.assertFalse(self.app._is_embedding_model("laguna-xs-2.1:q4_K_M"))
        self.assertFalse(self.app._is_embedding_model("llama3:8b"))
        self.assertFalse(self.app._is_embedding_model("gpt-4"))


class TestWarmupEndpointSelection(unittest.IsolatedAsyncioTestCase):
    """Tests for _send_warmup_request() endpoint selection based on model type."""

    def setUp(self):
        import importlib
        import app as app_module
        self.app = app_module

    def _make_entry(self):
        return {
            "keep_alive": "-1",
            "options": {"num_ctx": 65536},
        }

    async def test_embedding_model_uses_embeddings_endpoint(self):
        """Embedding models use /api/embeddings endpoint."""
        from unittest.mock import AsyncMock, patch, MagicMock

        entry = self._make_entry()
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await self.app._send_warmup_request("qwen3-embedding:8b", entry)

            # Verify /api/embeddings endpoint was called
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            self.assertIn("/api/embeddings", call_args[0][0])
            # Verify payload has 'prompt' field for embeddings
            payload = call_args[1]["json"]
            self.assertIn("prompt", payload)
            self.assertEqual(payload["model"], "qwen3-embedding:8b")

    async def test_chat_model_uses_chat_endpoint(self):
        """Chat models use /api/chat endpoint."""
        from unittest.mock import AsyncMock, patch, MagicMock

        entry = self._make_entry()
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await self.app._send_warmup_request("qwen3.6:35b-a3b-q8_0", entry)

            # Verify /api/chat endpoint was called
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            self.assertIn("/api/chat", call_args[0][0])
            # Verify payload has 'messages' field for chat
            payload = call_args[1]["json"]
            self.assertIn("messages", payload)
            self.assertEqual(payload["model"], "qwen3.6:35b-a3b-q8_0")


if __name__ == "__main__":
    unittest.main()
