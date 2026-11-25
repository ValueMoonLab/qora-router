"""GPU-free unit tests for qora-router scheduler.

Run: python -m unittest discover -s tests -v
These exercise routing, prompt rendering, config merging and the HTTP surface
with fake backends — no GPU, no engine binaries, no network downloads.
"""
import json
import os
import socket
import subprocess
import sys
import threading
import unittest
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scheduler  # noqa: E402


class FakeRS:
    """Stand-in for RSBackend: records swaps, returns canned completions."""

    def __init__(self):
        self.current_model = None
        self.swaps = []
        self.started = []
        self.stopped = 0
        self.proc = None

    def start(self, gguf):
        self.started.append(gguf)
        self.current_model = gguf.replace(".gguf", "")
        return True

    def swap(self, gguf):
        self.swaps.append(gguf)
        self.current_model = gguf.replace(".gguf", "")
        return True

    def stop(self):
        self.stopped += 1
        self.current_model = None

    def chat(self, model_id, messages, max_tokens=512, temperature=0.7, stream=False):
        return {
            "id": "chatcmpl-fake", "object": "chat.completion", "created": 0,
            "model": model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


class FakeInk:
    def __init__(self):
        self.started = 0
        self.stopped = 0
        self.calls = []

    def start(self):
        self.started += 1
        return True

    def stop(self):
        self.stopped += 1

    def generate(self, messages, max_tokens=512, temperature=0.7, effort="0"):
        self.calls.append(messages)
        return "ink says hi", 4


def make_config():
    cfg = dict(scheduler.DEFAULT_CONFIG)
    cfg["models"] = {
        "m-a": {"backend": "rs", "gguf": "a.gguf", "description": "A"},
        "m-b": {"backend": "rs", "gguf": "b.gguf", "description": "B"},
        "m-ink": {"backend": "inkling", "description": "Ink"},
    }
    cfg["default_model"] = "m-a"
    cfg["rs_exe"] = ""   # keep _init_backends from building real ones
    cfg["ink_exe"] = ""
    return cfg


def make_scheduler():
    s = scheduler.Scheduler(make_config())
    s.rs = FakeRS()
    s.inkling = FakeInk()
    return s


class TestRouting(unittest.TestCase):
    def test_first_request_starts_backend(self):
        s = make_scheduler()
        r = s.chat("m-a", [{"role": "user", "content": "hi"}])
        self.assertEqual(r["choices"][0]["message"]["content"], "ok")
        self.assertEqual(s.rs.started, ["a.gguf"])
        self.assertEqual(s.active_backend, "rs")
        self.assertEqual(s.swap_count, 1)

    def test_same_model_no_swap(self):
        s = make_scheduler()
        s.chat("m-a", [])
        s.chat("m-a", [])
        self.assertEqual(s.rs.swaps, [])
        self.assertEqual(s.swap_count, 1)

    def test_same_engine_hot_swap(self):
        s = make_scheduler()
        s.chat("m-a", [])
        s.chat("m-b", [])
        self.assertEqual(s.rs.swaps, ["b.gguf"])
        self.assertEqual(s.rs.stopped, 0)  # hot-swap must NOT restart process
        self.assertEqual(s.swap_count, 2)

    def test_cross_engine_stops_old(self):
        s = make_scheduler()
        s.chat("m-a", [])
        r = s.chat("m-ink", [{"role": "user", "content": "hi"}])
        self.assertEqual(r["choices"][0]["message"]["content"], "ink says hi")
        self.assertEqual(r["model"], "m-ink")
        self.assertEqual(s.rs.stopped, 1)
        self.assertEqual(s.inkling.started, 1)
        self.assertEqual(s.active_backend, "inkling")

    def test_cross_engine_back(self):
        s = make_scheduler()
        s.chat("m-ink", [])
        s.chat("m-a", [])
        self.assertEqual(s.inkling.stopped, 1)
        self.assertEqual(s.active_backend, "rs")

    def test_unknown_model_raises(self):
        s = make_scheduler()
        with self.assertRaises(ValueError):
            s.chat("nope", [])

    def test_status_shape(self):
        s = make_scheduler()
        st = s.status()
        self.assertEqual(st["status"], "ok")
        self.assertEqual(set(st["models"]), {"m-a", "m-b", "m-ink"})


class TestPromptRendering(unittest.TestCase):
    """InklingEngine.render_prompt is pure string work — no tokenizer needed."""

    def setUp(self):
        # bypass __init__ (which loads a tokenizer file)
        self.eng = scheduler.InklingEngine.__new__(scheduler.InklingEngine)

    def test_system_user_assistant(self):
        msgs = [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ]
        p = self.eng.render_prompt(msgs, effort="0")
        self.assertIn("Thinking effort level: 0", p)
        self.assertIn("<|message_system|><|content_text|>be brief<|end_message|>", p)
        self.assertIn("<|message_user|><|content_text|>hi<|end_message|>", p)
        self.assertIn("<|message_model|><|content_text|>hello<|end_message|>", p)
        self.assertTrue(p.endswith("<|message_model|>"))

    def test_effort_high(self):
        p = self.eng.render_prompt([{"role": "user", "content": "x"}], effort="0.9")
        self.assertIn("Thinking effort level: 0.9", p)


class TestHTTPSurface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sched = make_scheduler()
        scheduler.Handler.scheduler = cls.sched
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), scheduler.Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _get(self, path):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def _post(self, path, obj):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(obj).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_health(self):
        code, body = self._get("/health")
        self.assertEqual(code, 200)
        self.assertEqual(body["status"], "ok")

    def test_models_list(self):
        code, body = self._get("/v1/models")
        self.assertEqual(code, 200)
        ids = {m["id"] for m in body["data"]}
        self.assertEqual(ids, {"m-a", "m-b", "m-ink"})

    def test_engines(self):
        code, body = self._get("/engines")
        self.assertEqual(code, 200)
        engines = body["engines"]
        for name in ("llama.cpp", "ollama", "vllm", "sglang", "lmdeploy",
                     "lm-studio", "koboldcpp", "tgi", "tensorrt-llm", "mlx",
                     "localai", "jan", "gpt4all", "mistral.rs",
                     "rustingstrata", "inkling"):
            self.assertIn(name, engines)
        names = {item["name"] for item in body["data"]}
        self.assertIn("llama.cpp", names)

    def test_chat_default_model(self):
        code, body = self._post("/v1/chat/completions",
                                {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(code, 200)
        self.assertEqual(body["choices"][0]["message"]["content"], "ok")

    def test_chat_unknown_model_500(self):
        code, body = self._post("/v1/chat/completions",
                                {"model": "ghost", "messages": []})
        self.assertEqual(code, 500)
        self.assertIn("Unknown model", body["error"]["message"])

    def test_404(self):
        code, _ = self._get("/nope")
        self.assertEqual(code, 404)


class TestConfig(unittest.TestCase):
    def test_example_config_valid(self):
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "config.example.json")
        cfg = json.load(open(p, encoding="utf-8"))
        for k in ("port", "host", "models", "default_model"):
            self.assertIn(k, cfg)
        self.assertIn(cfg["default_model"], cfg["models"])
        for name, info in cfg["models"].items():
            self.assertIn(info["backend"], ("rs", "inkling"))
            if info["backend"] == "rs":
                self.assertIn("gguf", info)

    def test_default_port(self):
        self.assertEqual(scheduler.DEFAULT_CONFIG["port"], 8090)


def free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class FakeOpenAIServer:
    """Tiny OpenAI-compatible server used to exercise HTTPBackend."""

    def __init__(self):
        self.requests = []
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def _send(self, obj, code=200):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path in ("/v1/models", "/health"):
                    self._send({"object": "list",
                                "data": [{"id": "fake-server-model"}]})
                else:
                    self._send({"error": "not found"}, 404)

            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    self._send({"error": "not found"}, 404)
                    return
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append((self.path, body))
                self._send({
                    "id": "chatcmpl-http", "object": "chat.completion",
                    "created": 0, "model": body.get("model", ""),
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "http ok"},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                              "total_tokens": 2},
                })

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def healthy(self):
        try:
            with urllib.request.urlopen(self.url + "/v1/models", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def close(self):
        self.server.shutdown()
        self.server.server_close()


MANAGED_SERVER = """
import json
import sys
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        data = json.dumps({"object": "list",
                           "data": [{"id": "managed-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
"""


class TestHTTPBackend(unittest.TestCase):
    def setUp(self):
        self.fake = FakeOpenAIServer()

    def tearDown(self):
        self.fake.close()

    def test_attach_external_start_and_stop(self):
        backend = scheduler.HTTPBackend("llamacpp", {"base_url": self.fake.url})
        self.assertTrue(backend.external)
        self.assertTrue(backend.start())
        self.assertTrue(backend.alive)
        backend.stop()
        # External servers are never killed by the router.
        self.assertTrue(self.fake.healthy())

    def test_external_start_fails_when_unreachable(self):
        port = free_port()
        backend = scheduler.HTTPBackend(
            "dead", {"base_url": f"http://127.0.0.1:{port}"})
        self.assertFalse(backend.start())
        self.assertFalse(backend.alive)

    def test_managed_config_attaches_to_existing_server_without_killing_it(self):
        backend = scheduler.HTTPBackend(
            "managed", {"exe": "engine-that-is-already-running",
                        "base_url": self.fake.url})
        self.assertFalse(backend.external)
        self.assertTrue(backend.start())
        self.assertFalse(backend.owned)
        self.assertTrue(backend.alive)
        backend.stop()
        self.assertTrue(self.fake.healthy())

    def test_chat_request_body_merge_overrides_model(self):
        backend = scheduler.HTTPBackend("llamacpp", {"base_url": self.fake.url})
        self.assertTrue(backend.start())
        messages = [{"role": "user", "content": "hi"}]
        result = backend.chat(
            {"request": {"model": "server-override", "top_p": 0.8}},
            messages, max_tokens=9, temperature=0.2,
            model_id="public-model")
        self.assertEqual(result["choices"][0]["message"]["content"], "http ok")
        path, body = self.fake.requests[-1]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(body["model"], "server-override")
        self.assertEqual(body["top_p"], 0.8)
        self.assertEqual(body["max_tokens"], 9)
        self.assertEqual(body["temperature"], 0.2)
        self.assertEqual(body["messages"], messages)

    def test_chat_without_request_mapping_uses_public_model(self):
        backend = scheduler.HTTPBackend("llamacpp", {"base_url": self.fake.url})
        self.assertTrue(backend.start())
        backend.chat({}, [], model_id="public-model")
        self.assertEqual(self.fake.requests[-1][1]["model"], "public-model")

    def test_base_url_ending_in_v1_is_not_duplicated(self):
        backend = scheduler.HTTPBackend(
            "v1", {"base_url": self.fake.url + "/v1"})
        self.assertTrue(backend.start())
        backend.chat({}, [], model_id="public-model")
        self.assertEqual(self.fake.requests[-1][0], "/v1/chat/completions")

    def test_managed_start_and_stop(self):
        port = free_port()
        log_path = os.devnull
        backend = scheduler.HTTPBackend("managed", {
            "exe": sys.executable,
            "args": ["-c", MANAGED_SERVER, str(port)],
            "base_url": f"http://127.0.0.1:{port}",
            "health_path": "/v1/models",
            "startup_timeout": 60,
        }, log_path=log_path)
        self.assertFalse(backend.external)
        try:
            self.assertTrue(backend.start())
            self.assertTrue(backend.alive)
        finally:
            backend.stop()
        self.assertFalse(backend.alive)


class TestHTTPRouting(unittest.TestCase):
    def setUp(self):
        self.fake = FakeOpenAIServer()

    def tearDown(self):
        self.fake.close()

    def _scheduler(self):
        cfg = dict(scheduler.DEFAULT_CONFIG)
        cfg["models"] = {
            "m-a": {"backend": "rs", "gguf": "a.gguf", "description": "A"},
            "m-http": {"backend": "llamacpp",
                       "request": {"model": "server-model"}},
            "m-http2": {"backend": "llamacpp",
                        "request": {"model": "server-model-2"}},
        }
        cfg["default_model"] = "m-a"
        cfg["rs_exe"] = ""
        cfg["ink_exe"] = ""
        cfg["backends"] = {
            "llamacpp": {"type": "http", "base_url": self.fake.url},
        }
        s = scheduler.Scheduler(cfg)
        s.rs = FakeRS()  # the HTTP backend is real and GPU-free
        return s

    def test_engine_switch_rs_http_rs(self):
        s = self._scheduler()
        first = s.chat("m-a", [])
        self.assertEqual(first["choices"][0]["message"]["content"], "ok")
        self.assertEqual(s.active_backend, "rs")
        self.assertEqual(s.swap_count, 1)

        second = s.chat("m-http", [])
        self.assertEqual(second["choices"][0]["message"]["content"], "http ok")
        self.assertEqual(s.active_backend, "llamacpp")
        self.assertEqual(s.active_model, "m-http")
        self.assertEqual(s.rs.stopped, 1)
        self.assertEqual(s.swap_count, 2)
        self.assertEqual(self.fake.requests[-1][1]["model"], "server-model")
        self.assertEqual(
            s.status()["backends"]["llamacpp"]["current_model"], "server-model")

        third = s.chat("m-a", [])
        self.assertEqual(third["choices"][0]["message"]["content"], "ok")
        self.assertEqual(s.active_backend, "rs")
        self.assertEqual(s.rs.started, ["a.gguf", "a.gguf"])
        self.assertEqual(s.swap_count, 3)
        # Switching away from an external HTTP server must not stop it.
        self.assertTrue(self.fake.healthy())

    def test_same_http_engine_model_switch_counts_only_when_id_differs(self):
        s = self._scheduler()
        s.chat("m-http", [])
        self.assertEqual(s.swap_count, 1)
        s.chat("m-http", [])
        self.assertEqual(s.swap_count, 1)
        s.chat("m-http2", [])
        self.assertEqual(s.swap_count, 2)
        self.assertEqual(self.fake.requests[-1][1]["model"], "server-model-2")
        self.assertEqual(
            s.status()["backends"]["llamacpp"]["current_model"], "server-model-2")

    def test_generic_http_backend_alias_resolves(self):
        cfg = dict(scheduler.DEFAULT_CONFIG)
        cfg["models"] = {
            "m-http": {"backend": "http", "request": {"model": "server-model"}},
        }
        cfg["default_model"] = "m-http"
        cfg["rs_exe"] = ""
        cfg["ink_exe"] = ""
        cfg["backends"] = {
            "llamacpp": {"type": "http", "base_url": self.fake.url},
        }
        s = scheduler.Scheduler(cfg)
        backend_name, _ = s._resolve_model("m-http")
        self.assertEqual(backend_name, "llamacpp")


class TestBackendConfig(unittest.TestCase):
    def test_unknown_backend_type_raises(self):
        cfg = dict(scheduler.DEFAULT_CONFIG)
        cfg["models"] = {}
        cfg["default_model"] = "x"
        cfg["backends"] = {"weird": {"type": "quantum"}}
        with self.assertRaises(ValueError) as ctx:
            scheduler.Scheduler(cfg)
        self.assertIn("Unknown backend type", str(ctx.exception))

    def test_infer_backend_type_from_fields(self):
        s = scheduler.Scheduler.__new__(scheduler.Scheduler)
        self.assertEqual(s._infer_backend_type({"gguf": "a.gguf"}), "rs")
        self.assertEqual(s._infer_backend_type({"bridge_port": 1234}), "inkling")
        self.assertEqual(
            s._infer_backend_type({"base_url": "http://127.0.0.1:1"}), "http")

    def test_model_resolution_by_gguf_and_bridge_port(self):
        cfg = dict(scheduler.DEFAULT_CONFIG)
        cfg["models"] = {
            "gguf-model": {"gguf": "a.gguf"},
            "bridge-model": {"bridge_port": 1234},
        }
        cfg["default_model"] = "gguf-model"
        cfg["rs_exe"] = ""
        cfg["ink_exe"] = ""
        s = scheduler.Scheduler(cfg)
        self.assertEqual(s._resolve_model("gguf-model")[0], "rs")
        self.assertEqual(s._resolve_model("bridge-model")[0], "inkling")

    def test_top_level_single_backend_config(self):
        cfg = dict(scheduler.DEFAULT_CONFIG)
        cfg["models"] = {"m": {"request": {"model": "server-model"}}}
        cfg["default_model"] = "m"
        cfg["rs_exe"] = ""
        cfg["ink_exe"] = ""
        cfg["type"] = "http"
        cfg["base_url"] = "http://127.0.0.1:9"
        s = scheduler.Scheduler(cfg)
        self.assertEqual(s.backend_types["http"], "http")
        self.assertEqual(s._resolve_model("m")[0], "http")


class TestCLI(unittest.TestCase):
    def test_list_engines(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        proc = subprocess.run(
            [sys.executable, os.path.join(root, "scheduler.py"), "--list-engines"],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for name in ("llama.cpp", "ollama", "vllm", "sglang", "lmdeploy",
                     "lm-studio", "koboldcpp", "tgi", "tensorrt-llm", "mlx",
                     "localai", "jan", "gpt4all", "mistral.rs",
                     "rustingstrata", "inkling"):
            self.assertIn(name, proc.stdout)


if __name__ == "__main__":
    unittest.main()
