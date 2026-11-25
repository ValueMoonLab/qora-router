"""GPU-free unit tests for qora-router scheduler.

Run: python -m unittest discover -s tests -v
These exercise routing, prompt rendering, config merging and the HTTP surface
with fake backends — no GPU, no engine binaries, no network downloads.
"""
import json
import os
import sys
import threading
import unittest
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer

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


if __name__ == "__main__":
    unittest.main()
