import ast
import asyncio
import os
from pathlib import Path
import sys
import types
import unittest
import logging

logging.getLogger("router").setLevel(logging.WARNING)

ROUTER_DIR = Path(__file__).resolve().parents[1]
os.environ["MODEL_POLICY_FILE"] = str(ROUTER_DIR / "model_policy.yml")
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

        httpx.Timeout = Timeout
        httpx.AsyncClient = AsyncClient
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

from app import MODEL_POLICY, _build_ollama_payload, _resolve_asr_base_url, v1_audio_align  # noqa: E402


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
    def test_model_policy_includes_all_served_models(self):
        coder = MODEL_POLICY["qwen3-coder-next:q4_K_M"]
        self.assertEqual(coder["keep_alive"], -1)
        self.assertEqual(coder["think"], False)
        self.assertEqual(coder["options"]["num_ctx"], 262144)
        self.assertTrue(coder["warmup"])

        thinker = MODEL_POLICY["qwen3.6:35b-a3b-q8_0"]
        self.assertEqual(thinker["keep_alive"], -1)
        self.assertEqual(thinker["think"], True)
        self.assertEqual(thinker["options"]["num_ctx"], 262144)
        self.assertEqual(thinker["warmup"], True)

        chat_small = MODEL_POLICY["qwen3:4b"]
        self.assertEqual(chat_small["keep_alive"], "30m")
        self.assertEqual(chat_small["think"], True)
        self.assertEqual(chat_small["options"]["num_ctx"], 65536)
        self.assertEqual(chat_small["warmup"], False)



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
            resolved = _resolve_asr_base_url(_FakeRequestWithURL("router-host"))
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
            resolved = _resolve_asr_base_url(_FakeRequestWithURL("router-host"))
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
                    v1_audio_align(
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
            response = asyncio.run(v1_audio_align(req))
            self.assertEqual(200, response.status_code)
            self.assertTrue(response.content["forced_alignment_used"])
        finally:
            app_module._ensure_asr_admission = original_admission
            app_module._asr_post_multipart = original_asr_post_multipart

    def test_build_payload_forwards_unlocked_options_format_and_keep_alive(self):
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
        payload = _build_ollama_payload(body, think=False)

        self.assertEqual(payload["keep_alive"], "5m")
        self.assertEqual(payload["format"], "json")
        self.assertEqual(payload["options"]["num_batch"], 1024)
        self.assertEqual(payload["options"]["repeat_penalty"], 1.2)
        self.assertEqual(payload["options"]["temperature"], 0.4)
        self.assertEqual(payload["options"]["num_predict"], 256)
        # Policy default still present because client did not override it.
        self.assertEqual(payload["options"]["num_ctx"], 262144)


    def test_build_payload_policy_num_ctx_overrides_client(self):
        body = {
            "model": "qwen3-coder-next:q4_K_M",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"num_ctx": 16384},
        }
        with self.assertLogs("router", level="INFO") as cm:
            payload = _build_ollama_payload(body, think=False)
        self.assertEqual(payload["options"]["num_ctx"], 262144)
        self.assertTrue(any("overriding client num_ctx" in m for m in cm.output))


    def test_build_payload_uses_policy_defaults_when_client_silent(self):
        body = {
            "model": "qwen3-coder-next:q4_K_M",
            "messages": [{"role": "user", "content": "hi"}],
        }
        payload = _build_ollama_payload(body, think=False)
        opts = payload["options"]
        self.assertEqual(opts["num_ctx"], 262144)
        self.assertEqual(opts["num_batch"], 512)
        self.assertEqual(opts["temperature"], 0.15)
        self.assertEqual(payload["keep_alive"], -1)  # from policy


    def test_build_payload_uses_model_keep_alive_when_request_omits_it(self):
        body = {
            "model": "nemotron-cascade-2:30b",
            "messages": [{"role": "user", "content": "Verify this answer."}],
        }

        payload = _build_ollama_payload(body, think=True)

        self.assertEqual(payload["keep_alive"], "10m")


if __name__ == "__main__":
    unittest.main()
