"""
qora-router scheduler bridge
==================================
Single OpenAI-compatible endpoint that routes between multiple local LLM backends
on a single GPU. Only one backend is resident at a time; switching is automatic
and transparent to the client (request queues during swap).

Backends:
  - RustingStrata (GGUF models, native HTTP, hot-swap via /api/load ~40s)
  - Inkling (strata-inkling, GEN stdin/stdout protocol, cold start ~4min)

Usage:
  python scheduler.py --config config.json
  # or CLI flags:
  python scheduler.py --rs-exe /path/to/strata \
                      --rs-models /path/to/gguf/models \
                      --port 8090

Client just picks model name:
  POST /v1/chat/completions {"model": "qwen3.6-uncensored", ...}
  POST /v1/chat/completions {"model": "inkling-2-mini", ...}
  GET /v1/models  -> lists all available models
  GET /health     -> current backend + status
"""

import argparse, json, os, re, subprocess, sys, threading, time, queue, ctypes
import urllib.request, urllib.error
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "port": 8090,
    "host": "0.0.0.0",
    "rs_exe": "",
    "rs_models_dir": "",
    "rs_addr": "127.0.0.1:8084",
    "rs_ctx": 4096,
    "rs_parallel": 2,
    "ink_exe": "",
    "ink_model": "",
    "ink_vision": "",
    "ink_tokenizer": "",
    "ink_chat_template": "",
    "ink_max_context": 8192,
    "models": {
        "qwen3.6-uncensored": {
            "backend": "rs",
            "gguf": "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf",
            "description": "Qwen3.6 35B MoE uncensored (IQ2_M, ~100 tok/s)"
        },
        "qwen3.6": {
            "backend": "rs",
            "gguf": "Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf",
            "description": "Qwen3.6 35B MoE official (Q2_K_XL, ~30 tok/s)"
        },
        "inkling-2-mini": {
            "backend": "inkling",
            "description": "Inkling 2 Mini 26B looped MoE (SQ-10, ~37 tok/s, agentic coding)"
        }
    },
    "default_model": "qwen3.6-uncensored",
    "swap_timeout": 300,
    "idle_shutdown_sec": 0
}

# ─── Inkling GEN protocol engine ─────────────────────────────────────────────

class InklingEngine:
    """Manages strata-inkling.exe --serve subprocess with GEN protocol."""

    def __init__(self, exe, model, tokenizer_path, chat_template_path,
                 vision=None, max_context=8192, log_path="inkling_engine.log"):
        self.exe = exe
        self.model = model
        self.vision = vision
        self.max_context = max_context
        self.log_path = log_path
        self.proc = None
        self.ready = False
        self.lock = threading.Lock()
        self.token_queue = queue.Queue()
        self.reader_thread = None
        self._load_tokenizer(tokenizer_path, chat_template_path)

    def _load_tokenizer(self, tok_path, tmpl_path):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(tok_path)
        self.chat_template = ""
        if tmpl_path and os.path.exists(tmpl_path):
            self.chat_template = open(tmpl_path, encoding="utf-8").read()

    def render_prompt(self, messages, effort="0"):
        """Render chat messages into Inkling prompt format."""
        parts = []
        parts.append(f"<|message_system|><|content_text|>Thinking effort level: {effort}<|end_message|>")
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"<|message_system|><|content_text|>{content}<|end_message|>")
            elif role == "assistant":
                parts.append(f"<|message_model|><|content_text|>{content}<|end_message|>")
            else:
                parts.append(f"<|message_user|><|content_text|>{content}<|end_message|>")
        parts.append("<|message_model|>")
        return "".join(parts)

    def start(self):
        """Start engine subprocess, wait for READY."""
        if self.proc and self.proc.poll() is None:
            return self.ready
        cmd = [self.exe, "--serve", "--native", self.model,
               "--draft", "off", "--kv", "k8v8",
               "--max-context", str(self.max_context)]
        if self.vision:
            cmd += ["--vision", self.vision]
        logf = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=logf, bufsize=0
        )
        self.ready = False
        self.reader_thread = threading.Thread(target=self._reader, daemon=True)
        self.reader_thread.start()
        # wait for READY
        deadline = time.time() + 360
        while time.time() < deadline:
            if self.ready:
                return True
            if self.proc.poll() is not None:
                return False
            time.sleep(1)
        return False

    def _reader(self):
        """Read stdout lines, dispatch T/DONE/READY."""
        try:
            for raw in self.proc.stdout:
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("READY"):
                    self.ready = True
                elif line.startswith("T "):
                    self.token_queue.put(("T", int(line[2:])))
                elif line.startswith("DONE"):
                    self.token_queue.put(("DONE", line))
                elif line.startswith("PP"):
                    self.token_queue.put(("PP", line))
                elif line.startswith("ERR"):
                    self.token_queue.put(("ERR", line))
        except Exception:
            pass

    def generate(self, messages, max_tokens=512, temperature=0.7, effort="0"):
        """Send GEN command, collect tokens, return text."""
        with self.lock:
            if not self.ready:
                if not self.start():
                    raise RuntimeError("Inkling engine failed to start")
            prompt = self.render_prompt(messages, effort)
            ids = self.tokenizer.encode(prompt).ids
            ids_str = ",".join(str(i) for i in ids)
            cmd = f"GEN {max_tokens} temperature={temperature} top_p=0.9 {ids_str}\n"
            self.proc.stdin.write(cmd.encode())
            self.proc.stdin.flush()
            # collect
            tokens = []
            deadline = time.time() + 300
            while time.time() < deadline:
                try:
                    kind, val = self.token_queue.get(timeout=5)
                except queue.Empty:
                    if self.proc.poll() is not None:
                        raise RuntimeError("Inkling engine died")
                    continue
                if kind == "T":
                    tokens.append(val)
                elif kind == "DONE":
                    break
                elif kind == "ERR":
                    raise RuntimeError(f"Inkling ERR: {val}")
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            return text, len(tokens)

    def stop(self):
        """Kill engine subprocess, free VRAM."""
        self.ready = False
        if self.proc:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None

    @property
    def alive(self):
        return self.proc is not None and self.proc.poll() is None and self.ready


# ─── RustingStrata backend ────────────────────────────────────────────────────

class RSBackend:
    """Manages RustingStrata serve subprocess + hot-swap."""

    def __init__(self, exe, models_dir, addr="127.0.0.1:8084", ctx=4096, parallel=2,
                 log_path="rs_engine.log"):
        self.exe = exe
        self.models_dir = models_dir
        self.addr = addr
        self.ctx = ctx
        self.parallel = parallel
        self.log_path = log_path
        self.proc = None
        self.current_model = None
        self.base_url = f"http://{addr}"

    def start(self, gguf_filename):
        """Start RS serve with a specific model, or attach to existing instance."""
        # Check if already running externally (e.g. started by user)
        try:
            r = urllib.request.urlopen(f"{self.base_url}/v1/models", timeout=5)
            data = json.loads(r.read())
            if data.get("data"):
                loaded_id = data["data"][0]["id"]
                self.current_model = loaded_id
                target = gguf_filename.replace(".gguf", "")
                if loaded_id != target:
                    return self.swap(gguf_filename)
                return True
        except Exception:
            pass
        if self.proc and self.proc.poll() is None:
            return self.swap(gguf_filename)
        model_path = os.path.join(self.models_dir, gguf_filename)
        cmd = [self.exe, "serve", "--model", model_path,
               "--models-dir", self.models_dir,
               "--gpu", "--addr", self.addr,
               "--ctx", str(self.ctx), "--parallel", str(self.parallel)]
        logf = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=subprocess.DETACHED_PROCESS if sys.platform == "win32" else 0
        )
        # wait for ready
        deadline = time.time() + 180
        while time.time() < deadline:
            try:
                r = urllib.request.urlopen(f"{self.base_url}/v1/models", timeout=5)
                data = json.loads(r.read())
                if data.get("data"):
                    self.current_model = gguf_filename.replace(".gguf", "")
                    return True
            except Exception:
                pass
            if self.proc.poll() is not None:
                return False
            time.sleep(3)
        return False

    def swap(self, gguf_filename):
        """Hot-swap model via /api/load (~40s)."""
        body = json.dumps({"model": gguf_filename}).encode()
        req = urllib.request.Request(f"{self.base_url}/api/load",
                                     data=body, headers={"Content-Type": "application/json"})
        try:
            r = urllib.request.urlopen(req, timeout=300)
            resp = json.loads(r.read())
            self.current_model = resp.get("loaded", gguf_filename.replace(".gguf", ""))
            return True
        except Exception as e:
            return False

    def chat(self, model_id, messages, max_tokens=512, temperature=0.7, stream=False):
        """Forward chat request to RS."""
        body = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False}
        }
        req = urllib.request.Request(f"{self.base_url}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        r = urllib.request.urlopen(req, timeout=300)
        return json.loads(r.read())

    def stop(self):
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None
            self.current_model = None

    @property
    def alive(self):
        return self.proc is not None and self.proc.poll() is None


# ─── Scheduler ────────────────────────────────────────────────────────────────

class Scheduler:
    """Routes requests to the correct backend, manages VRAM exclusivity."""

    def __init__(self, config):
        self.config = config
        self.models = config["models"]
        self.lock = threading.Lock()
        self.active_backend = None  # "rs" or "inkling"
        self.active_model = None
        self.rs = None
        self.inkling = None
        self.swap_count = 0
        self.request_count = 0
        self._init_backends()

    def _init_backends(self):
        c = self.config
        if c.get("rs_exe"):
            self.rs = RSBackend(c["rs_exe"], c["rs_models_dir"], c["rs_addr"],
                                c["rs_ctx"], c["rs_parallel"],
                                log_path=os.path.join(os.path.dirname(c["rs_exe"]), "rs_scheduler.log"))
        if c.get("ink_exe"):
            self.inkling = InklingEngine(
                c["ink_exe"], c["ink_model"], c["ink_tokenizer"],
                c.get("ink_chat_template", ""),
                vision=c.get("ink_vision") or None,
                max_context=c["ink_max_context"],
                log_path=os.path.join(os.path.dirname(c["ink_exe"]), "ink_scheduler.log"))

    def _ensure_backend(self, model_name):
        """Switch to the backend needed for model_name. Blocks during swap."""
        info = self.models.get(model_name)
        if not info:
            raise ValueError(f"Unknown model: {model_name}")
        backend = info["backend"]

        if self.active_backend == backend:
            if backend == "rs":
                target_gguf = info["gguf"]
                if self.rs.current_model != target_gguf.replace(".gguf", ""):
                    self.rs.swap(target_gguf)
                    self.swap_count += 1
            return

        # Need to switch backends - free VRAM first
        if self.active_backend == "rs" and self.rs:
            self.rs.stop()
            time.sleep(3)  # let VRAM release
        elif self.active_backend == "inkling" and self.inkling:
            self.inkling.stop()
            time.sleep(3)

        # Start new backend
        if backend == "rs":
            ok = self.rs.start(info["gguf"])
            if not ok:
                raise RuntimeError("RustingStrata failed to start")
        elif backend == "inkling":
            ok = self.inkling.start()
            if not ok:
                raise RuntimeError("Inkling engine failed to start")

        self.active_backend = backend
        self.active_model = model_name
        self.swap_count += 1

    def chat(self, model_name, messages, max_tokens=512, temperature=0.7):
        """Main entry: route chat request."""
        with self.lock:
            self.request_count += 1
            self._ensure_backend(model_name)
            info = self.models[model_name]

            if info["backend"] == "rs":
                model_id = info["gguf"].replace(".gguf", "")
                return self.rs.chat(model_id, messages, max_tokens, temperature)
            else:
                text, ntok = self.inkling.generate(messages, max_tokens, temperature)
                # Build OpenAI-compatible response
                return {
                    "id": f"chatcmpl-ink-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": ntok, "total_tokens": ntok}
                }

    def status(self):
        return {
            "status": "ok",
            "active_backend": self.active_backend,
            "active_model": self.active_model,
            "swap_count": self.swap_count,
            "request_count": self.request_count,
            "models": list(self.models.keys())
        }


# ─── HTTP Handler ─────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    scheduler = None

    def log_message(self, fmt, *args):
        pass  # suppress default logging

    def _send_json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(self.scheduler.status())
        elif self.path == "/v1/models":
            models = []
            for name, info in self.scheduler.models.items():
                models.append({
                    "id": name, "object": "model", "created": 0,
                    "owned_by": "local-scheduler",
                    "description": info.get("description", "")
                })
            self._send_json({"object": "list", "data": models})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            model = body.get("model", self.scheduler.config["default_model"])
            messages = body.get("messages", [])
            max_tokens = body.get("max_tokens", 512)
            temperature = body.get("temperature", 0.7)
            try:
                result = self.scheduler.chat(model, messages, max_tokens, temperature)
                self._send_json(result)
            except Exception as e:
                self._send_json({"error": {"message": str(e), "type": "scheduler_error"}}, 500)
        else:
            self._send_json({"error": "not found"}, 404)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="qora-router scheduler bridge")
    parser.add_argument("--config", help="JSON config file")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--rs-exe", default="")
    parser.add_argument("--rs-models", default="")
    parser.add_argument("--ink-exe", default="")
    parser.add_argument("--ink-model", default="")
    parser.add_argument("--ink-tokenizer", default="")
    parser.add_argument("--ink-template", default="")
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    if args.config and os.path.exists(args.config):
        config.update(json.load(open(args.config, encoding="utf-8")))
    if args.rs_exe: config["rs_exe"] = args.rs_exe
    if args.rs_models: config["rs_models_dir"] = args.rs_models
    if args.ink_exe: config["ink_exe"] = args.ink_exe
    if args.ink_model: config["ink_model"] = args.ink_model
    if args.ink_tokenizer: config["ink_tokenizer"] = args.ink_tokenizer
    if args.ink_template: config["ink_chat_template"] = args.ink_template
    if args.port is not None: config["port"] = args.port
    if args.host is not None: config["host"] = args.host

    scheduler = Scheduler(config)
    Handler.scheduler = scheduler

    server = ThreadingHTTPServer((config["host"], config["port"]), Handler)
    print(f"qora-router listening on {config['host']}:{config['port']}")
    print(f"Models: {', '.join(config['models'].keys())}")
    print(f"Default: {config['default_model']}")
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if scheduler.rs: scheduler.rs.stop()
        if scheduler.inkling: scheduler.inkling.stop()
        server.server_close()


if __name__ == "__main__":
    main()
