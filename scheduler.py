"""
qora-router scheduler bridge
==================================
Single OpenAI-compatible endpoint that routes between multiple local LLM backends
on a single GPU. Only one backend is resident at a time; switching is automatic
and transparent to the client (request queues during swap).

Backends:
  - RustingStrata (GGUF models, native HTTP, hot-swap via /api/load ~40s)
  - Inkling (strata-inkling, GEN stdin/stdout protocol, cold start ~4min)
  - Any OpenAI-compatible HTTP server (llama.cpp, Ollama, vLLM, SGLang,
    LMDeploy, LM Studio, KoboldCpp, TGI, TensorRT-LLM, MLX, LocalAI, Jan,
    GPT4All, Mistral.rs, ...), managed or external-attach

Usage:
  python scheduler.py --config config.json
  python scheduler.py --list-engines
  # or CLI flags:
  python scheduler.py --rs-exe /path/to/strata \
                      --rs-models /path/to/gguf/models \
                      --port 8090

Client just picks model name:
  POST /v1/chat/completions {"model": "qwen3.6-uncensored", ...}
  POST /v1/chat/completions {"model": "inkling-2-mini", ...}
  GET /v1/models  -> lists all available models
  GET /health     -> current backend + status
  GET /engines    -> supported engine catalog
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

# Backend type aliases accepted in the "backends" config map.
BACKEND_TYPE_ALIASES = {
    "rs": "rs",
    "rustingstrata": "rs",
    "rusting-strata": "rs",
    "inkling": "inkling",
    "http": "http",
    "openai": "http",
    "openai-compatible": "http",
}

# Ready-to-copy documentation for every engine qora-router can drive.
# "type" is the backend class used by the scheduler; "exe"/"args" are only a
# launch hint (leave exe empty to attach to an already-running server).
ENGINE_CATALOG = {
    "llama.cpp": {
        "type": "http",
        "exe": "llama-server",
        "args": ["-m", "/path/to/model.gguf", "--host", "127.0.0.1",
                 "--port", "8085", "-ngl", "99", "-c", "4096"],
        "base_url": "http://127.0.0.1:8085",
        "health_path": "/health",
        "notes": "OpenAI API: /v1/chat/completions. NVIDIA CUDA, Apple Metal, "
                 "Vulkan or CPU. One model per process; changing the model "
                 "restarts llama-server unless you run one server per model.",
    },
    "ollama": {
        "type": "http",
        "exe": "ollama",
        "args": ["serve"],
        "base_url": "http://127.0.0.1:11434",
        "health_path": "/api/tags",
        "notes": "OpenAI API: /v1/chat/completions with ids like llama3.2:3b. "
                 "GPU auto-detected (CUDA / ROCm / Metal) or CPU. Ollama loads "
                 "and unloads models itself; keep_alive controls VRAM release.",
    },
    "vllm": {
        "type": "http",
        "exe": "vllm",
        "args": ["serve", "/path/to/model", "--host", "127.0.0.1",
                 "--port", "8000", "--gpu-memory-utilization", "0.90"],
        "base_url": "http://127.0.0.1:8000",
        "health_path": "/health",
        "notes": "OpenAI API: /v1/chat/completions. NVIDIA CUDA (ROCm on "
                 "Linux). One model per server; changing model restarts it.",
    },
    "sglang": {
        "type": "http",
        "exe": "python",
        "args": ["-m", "sglang.launch_server", "--model-path", "/path/to/model",
                 "--host", "127.0.0.1", "--port", "30000"],
        "base_url": "http://127.0.0.1:30000",
        "health_path": "/health",
        "notes": "OpenAI API: /v1/chat/completions. NVIDIA CUDA (Linux). "
                 "Restart the server to change model.",
    },
    "lmdeploy": {
        "type": "http",
        "exe": "lmdeploy",
        "args": ["serve", "api_server", "/path/to/model",
                 "--server-name", "127.0.0.1", "--server-port", "23333"],
        "base_url": "http://127.0.0.1:23333",
        "health_path": "/health",
        "notes": "OpenAI API: /v1/chat/completions. NVIDIA CUDA. Restart the "
                 "server to change model.",
    },
    "lm-studio": {
        "type": "http",
        "exe": "",
        "args": [],
        "base_url": "http://127.0.0.1:1234",
        "health_path": "/v1/models",
        "notes": "Start the local server in the LM Studio GUI. OpenAI API: "
                 "/v1/chat/completions; put the loaded model id in the "
                 "per-model request.model. External attach only (never "
                 "stopped by qora-router). CUDA / Metal / Vulkan builds.",
    },
    "koboldcpp": {
        "type": "http",
        "exe": "koboldcpp",
        "args": ["--model", "/path/to/model.gguf", "--host", "127.0.0.1",
                 "--port", "5001", "--usecublas"],
        "base_url": "http://127.0.0.1:5001",
        "health_path": "/api/v1/model",
        "notes": "OpenAI API: /v1/chat/completions. CPU, CUDA, ROCm or Vulkan. "
                 "Restart to change model.",
    },
    "tgi": {
        "type": "http",
        "exe": "text-generation-launcher",
        "args": ["--model-id", "/path/to/model", "--hostname", "127.0.0.1",
                 "--port", "8080"],
        "base_url": "http://127.0.0.1:8080",
        "health_path": "/health",
        "notes": "Text Generation Inference. OpenAI Messages API: "
                 "/v1/chat/completions. NVIDIA CUDA, AMD ROCm, Intel Gaudi or "
                 "CPU. Restart to change model.",
    },
    "tensorrt-llm": {
        "type": "http",
        "exe": "trtllm-serve",
        "args": ["/path/to/engine-or-hf-model", "--host", "127.0.0.1",
                 "--port", "8000"],
        "base_url": "http://127.0.0.1:8000",
        "health_path": "/health",
        "notes": "trtllm-serve. OpenAI API: /v1/chat/completions. NVIDIA "
                 "TensorRT-LLM (compiled engine or JIT build). Restart to "
                 "change model.",
    },
    "mlx": {
        "type": "http",
        "exe": "python",
        "args": ["-m", "mlx_lm.server", "--model",
                 "mlx-community/Llama-3.2-3B-Instruct-4bit",
                 "--host", "127.0.0.1", "--port", "8080"],
        "base_url": "http://127.0.0.1:8080",
        "health_path": "/v1/models",
        "notes": "OpenAI API: /v1/chat/completions. macOS Apple Silicon only. "
                 "Restart to change model.",
    },
    "localai": {
        "type": "http",
        "exe": "local-ai",
        "args": ["run", "--address", "127.0.0.1:8080"],
        "base_url": "http://127.0.0.1:8080",
        "health_path": "/readyz",
        "notes": "OpenAI API: /v1/chat/completions. CPU and GPU backends "
                 "(CUDA / ROCm / Intel). Models are declared in LocalAI's own "
                 "config; it can hot-load them.",
    },
    "jan": {
        "type": "http",
        "exe": "",
        "args": [],
        "base_url": "http://127.0.0.1:1337",
        "health_path": "/v1/models",
        "notes": "Start the Jan desktop app / Cortex server. OpenAI API: "
                 "/v1/chat/completions; put the loaded model id in the "
                 "per-model request.model. External attach only. "
                 "Metal / CUDA / Vulkan / CPU.",
    },
    "gpt4all": {
        "type": "http",
        "exe": "",
        "args": [],
        "base_url": "http://127.0.0.1:4891",
        "health_path": "/v1/models",
        "notes": "Enable the Local API Server in the GPT4All desktop app. "
                 "OpenAI API: /v1/chat/completions; model is chosen in the app. "
                 "External attach only. CPU and CUDA / Metal / Vulkan builds.",
    },
    "mistral.rs": {
        "type": "http",
        "exe": "mistralrs-server",
        "args": ["--port", "1234", "-m", "/path/to/model"],
        "base_url": "http://127.0.0.1:1234",
        "health_path": "/v1/models",
        "notes": "Rust engine for GGUF, plain and HF models. OpenAI API: "
                 "/v1/chat/completions. CUDA, Metal or CPU. Restart to change "
                 "model.",
    },
    "rustingstrata": {
        "type": "rs",
        "exe": "/path/to/strata",
        "args": ["serve", "--model", "/path/to/model.gguf",
                 "--models-dir", "/path/to/models", "--gpu",
                 "--addr", "127.0.0.1:8084", "--ctx", "4096", "--parallel", "2"],
        "base_url": "http://127.0.0.1:8084",
        "health_path": "/v1/models",
        "notes": "Native backend. GGUF only. Hot-swap via POST /api/load, "
                 "measured ~40s on RTX 3060 12GB. NVIDIA CUDA or CPU. OpenAI "
                 "API: /v1/chat/completions.",
    },
    "inkling": {
        "type": "inkling",
        "exe": "/path/to/strata-inkling",
        "args": ["--serve", "--native", "/path/to/model", "--draft", "off",
                 "--kv", "k8v8"],
        "base_url": "",
        "health_path": "",
        "notes": "Native GEN stdin/stdout protocol, not HTTP. Cold start "
                 "measured ~4 min on RTX 3060 12GB with expert streaming. "
                 "Requires the tokenizers Python package.",
    },
}


def format_engine_catalog():
    """Render ENGINE_CATALOG as plain text for --list-engines."""
    lines = ["qora-router supported engines", "=" * 64, ""]
    for name, info in ENGINE_CATALOG.items():
        lines.append(f"{name}  [type: {info.get('type', '')}]")
        exe = info.get("exe", "")
        lines.append(f"  exe      : {exe or '(external server)'}")
        if info.get("args"):
            lines.append("  args     : " + " ".join(str(a) for a in info["args"]))
        lines.append(f"  base_url : {info.get('base_url') or '(n/a)'}")
        if info.get("health_path"):
            lines.append(f"  health   : {info['health_path']}")
        if info.get("notes"):
            lines.append(f"  notes    : {info['notes']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

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
        self._tokenizer_path = tokenizer_path
        self._chat_template_path = chat_template_path
        self.tokenizer = None
        self.chat_template = ""
        if tokenizer_path and os.path.exists(tokenizer_path):
            self._load_tokenizer(tokenizer_path, chat_template_path)

    def _load_tokenizer(self, tok_path, tmpl_path):
        self.tokenizer = None
        self.chat_template = ""
        if tmpl_path and os.path.exists(tmpl_path):
            with open(tmpl_path, encoding="utf-8") as f:
                self.chat_template = f.read()
        if tok_path and os.path.exists(tok_path):
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(tok_path)

    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            self._load_tokenizer(self._tokenizer_path, self._chat_template_path)
        if self.tokenizer is None:
            raise RuntimeError(
                "Inkling tokenizer is not configured; set ink_tokenizer or "
                "backends.<name>.tokenizer")

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
            self._ensure_tokenizer()
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


# ─── Generic OpenAI-compatible HTTP backend ────────────────────────────────────

class HTTPBackend:
    """Manages any local engine that exposes an OpenAI-compatible HTTP API
    (llama.cpp llama-server, Ollama, vLLM, LM Studio, ...).

    Config per backend:
      exe / args        : optional; if set, the router launches the process
      base_url          : e.g. http://127.0.0.1:8085
      health_path       : polled until ready (default /v1/models)
      startup_timeout   : seconds to wait for readiness (default 180)
    Config per model:
      request           : dict merged into the chat request body
                          (e.g. {"model": "<server model id>"})
    If exe is empty the router attaches to an already-running server and
    never stops it (external lifecycle).
    """

    def __init__(self, name, cfg, log_path=None):
        self.name = name
        self.exe = cfg.get("exe", "")
        args = cfg.get("args", [])
        self.args = args.split() if isinstance(args, str) else list(args)
        self.base_url = cfg.get("base_url", "").rstrip("/")
        self.health_path = cfg.get("health_path", "/v1/models")
        self.startup_timeout = cfg.get("startup_timeout", 180)
        self.log_path = log_path or f"{name}_engine.log"
        self.proc = None
        self.external = not self.exe
        self.owned = False  # True only when this router launched the process
        self.current_model = None

    def _url(self, path):
        """Join base_url and path without duplicating a trailing /v1."""
        if not self.base_url:
            return path
        if self.base_url.endswith("/v1"):
            if path.startswith("/v1/"):
                return self.base_url + path[3:]
            return self.base_url[:-3] + path
        return self.base_url + path

    def _healthy(self):
        if not self.base_url:
            return False
        try:
            r = urllib.request.urlopen(self._url(self.health_path), timeout=5)
            return r.status == 200
        except Exception:
            return False

    def start(self, model_info=None):
        if self._healthy():
            return True
        if self.external:
            return False  # nothing to launch; server must be up already
        if not self.exe:
            return False
        cmd = [self.exe] + list(self.args)
        logf = open(self.log_path, "w", encoding="utf-8")
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=logf, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=(subprocess.DETACHED_PROCESS
                               if sys.platform == "win32" else 0))
            self.owned = True
        finally:
            logf.close()
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self._healthy():
                return True
            if self.proc.poll() is not None:
                return False
            time.sleep(0.5)
        return False

    def chat(self, model_info, messages, max_tokens=512, temperature=0.7,
             model_id=None):
        body = {"messages": messages, "max_tokens": max_tokens,
                "temperature": temperature}
        default_id = model_id or model_info.get("model")
        if default_id:
            body["model"] = default_id
        # Per-model request fields win, including the server-side model id.
        body.update(model_info.get("request") or {})
        if body.get("model"):
            self.current_model = body["model"]
        if not self.base_url:
            raise RuntimeError(f"HTTP backend '{self.name}' has no base_url")
        req = urllib.request.Request(
            self._url("/v1/chat/completions"),
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        r = urllib.request.urlopen(req, timeout=300)
        return json.loads(r.read())

    def stop(self):
        if not self.owned:
            return  # never kill a server this router did not start
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
        self.owned = False
        self.current_model = None

    @property
    def alive(self):
        if self.owned:
            return self.proc is not None and self.proc.poll() is None
        return self._healthy()


# ─── Scheduler ────────────────────────────────────────────────────────────────

class Scheduler:
    """Routes requests to the correct backend, manages VRAM exclusivity."""

    def __init__(self, config):
        self.config = config
        self.models = config["models"]
        self.lock = threading.Lock()
        self.active_backend = None  # configured backend name
        self.active_model = None
        self.rs = None
        self.inkling = None
        self.backends = {}
        self.backend_configs = {}
        self.backend_types = {}
        self.swap_count = 0
        self.request_count = 0
        self._init_backends()

    def _infer_backend_type(self, cfg):
        """Infer a backend type from a backend config entry (legacy compat)."""
        raw = cfg.get("type")
        if raw:
            return str(raw).lower()
        if cfg.get("gguf") or cfg.get("models_dir") or cfg.get("rs_exe"):
            return "rs"
        if (cfg.get("bridge_port") is not None or cfg.get("tokenizer")
                or cfg.get("chat_template")):
            return "inkling"
        if cfg.get("base_url") or cfg.get("exe"):
            return "http"
        return None

    def _build_backend(self, name, btype, cfg):
        exe = cfg.get("exe", "")
        log_path = cfg.get("log_path")
        if btype == "rs":
            exe = exe or self.config.get("rs_exe", "")
            models_dir = cfg.get("models_dir") or self.config.get("rs_models_dir", "")
            if not models_dir and cfg.get("gguf"):
                models_dir = os.path.dirname(cfg["gguf"])
            addr = cfg.get("addr")
            if not addr and cfg.get("base_url"):
                addr = cfg["base_url"].split("://", 1)[-1].rstrip("/")
                if addr.endswith("/v1"):
                    addr = addr[:-3]
            addr = addr or self.config.get("rs_addr", "127.0.0.1:8084")
            ctx = cfg.get("ctx", self.config.get("rs_ctx", 4096))
            parallel = cfg.get("parallel", self.config.get("rs_parallel", 2))
            if not log_path and exe:
                log_path = os.path.join(os.path.dirname(exe), "rs_scheduler.log")
            return RSBackend(exe, models_dir, addr, ctx, parallel,
                             log_path=log_path or "rs_engine.log")
        if btype == "inkling":
            exe = exe or self.config.get("ink_exe", "")
            model = cfg.get("model") or cfg.get("model_path") or self.config.get("ink_model", "")
            tokenizer = cfg.get("tokenizer") or self.config.get("ink_tokenizer", "")
            template = cfg.get("chat_template") or self.config.get("ink_chat_template", "")
            vision = cfg.get("vision") or self.config.get("ink_vision") or None
            max_context = cfg.get("max_context", self.config.get("ink_max_context", 8192))
            if not log_path and exe:
                log_path = os.path.join(os.path.dirname(exe), "ink_scheduler.log")
            return InklingEngine(exe, model, tokenizer, template, vision=vision,
                                 max_context=max_context,
                                 log_path=log_path or "inkling_engine.log")
        if btype == "http":
            if not log_path and exe:
                log_path = os.path.join(os.path.dirname(exe), f"{name}_scheduler.log")
            return HTTPBackend(name, cfg, log_path=log_path)
        raise ValueError(
            f"Unknown backend type '{btype}' for backend '{name}'; "
            f"expected 'rs', 'inkling' or 'http'")

    def _init_backends(self):
        c = self.config
        backends_cfg = c.get("backends") or c.get("engines") or {}
        if backends_cfg:
            if not isinstance(backends_cfg, dict):
                raise ValueError("'backends' must be a JSON object")
            for name, cfg in backends_cfg.items():
                if not isinstance(cfg, dict):
                    raise ValueError(f"Backend '{name}' config must be a JSON object")
                raw_type = cfg.get("type")
                btype = self._infer_backend_type(cfg)
                if btype is None:
                    raise ValueError(
                        f"Backend '{name}' has no 'type' and none could be inferred; "
                        f"set 'type' to 'rs', 'inkling' or 'http'")
                btype = BACKEND_TYPE_ALIASES.get(str(btype).lower(), str(btype).lower())
                if btype not in ("rs", "inkling", "http"):
                    raise ValueError(
                        f"Unknown backend type {raw_type!r} for backend '{name}'; "
                        f"expected 'rs', 'inkling' or 'http'")
                obj = self._build_backend(name, btype, cfg)
                self.backends[name] = obj
                self.backend_configs[name] = cfg
                self.backend_types[name] = btype
                if btype == "rs" and (name == "rs" or self.rs is None):
                    self.rs = obj
                elif btype == "inkling" and (name == "inkling" or self.inkling is None):
                    self.inkling = obj
            return

        # Single top-level backend config: {"type": "http", ...}.
        if c.get("type"):
            raw_type = c.get("type")
            btype = BACKEND_TYPE_ALIASES.get(str(raw_type).lower(), str(raw_type).lower())
            if btype not in ("rs", "inkling", "http"):
                raise ValueError(
                    f"Unknown backend type {raw_type!r}; expected 'rs', "
                    f"'inkling' or 'http'")
            name = str(c.get("backend_name") or c.get("name") or btype)
            obj = self._build_backend(name, btype, c)
            self.backends[name] = obj
            self.backend_configs[name] = c
            self.backend_types[name] = btype
            if btype == "rs":
                self.rs = obj
            elif btype == "inkling":
                self.inkling = obj
            return

        # Legacy flat config: rs_exe / ink_exe at the top level.
        if c.get("rs_exe"):
            obj = RSBackend(c["rs_exe"], c.get("rs_models_dir", ""),
                            c.get("rs_addr", "127.0.0.1:8084"),
                            c.get("rs_ctx", 4096), c.get("rs_parallel", 2),
                            log_path=os.path.join(os.path.dirname(c["rs_exe"]),
                                                  "rs_scheduler.log"))
            self.rs = obj
            self.backends["rs"] = obj
            self.backend_configs["rs"] = {"type": "rs", "exe": c["rs_exe"]}
            self.backend_types["rs"] = "rs"
        if c.get("ink_exe"):
            obj = InklingEngine(
                c["ink_exe"], c.get("ink_model", ""), c.get("ink_tokenizer", ""),
                c.get("ink_chat_template", ""),
                vision=c.get("ink_vision") or None,
                max_context=c.get("ink_max_context", 8192),
                log_path=os.path.join(os.path.dirname(c["ink_exe"]),
                                      "ink_scheduler.log"))
            self.inkling = obj
            self.backends["inkling"] = obj
            self.backend_configs["inkling"] = {"type": "inkling", "exe": c["ink_exe"]}
            self.backend_types["inkling"] = "inkling"

    def _backend_type(self, name):
        if name in self.backend_types:
            return self.backend_types[name]
        if name in ("rs", "inkling"):
            return name
        return None

    def _get_backend(self, name):
        if name in self.backends:
            return self.backends[name]
        if name == "rs":
            return self.rs
        if name == "inkling":
            return self.inkling
        return None

    def _default_http_backend(self):
        names = [n for n, t in self.backend_types.items() if t == "http"]
        return names[0] if len(names) == 1 else None

    def _resolve_model(self, model_name):
        """Return (backend_name, model_info) for a public model name."""
        info = self.models.get(model_name)
        if not info:
            raise ValueError(f"Unknown model: {model_name}")
        backend = info.get("backend")
        if not backend:
            if info.get("gguf"):
                backend = "rs"
            elif info.get("bridge_port") is not None:
                backend = "inkling"
            else:
                backend = self._default_http_backend()
            if not backend:
                raise ValueError(
                    f"Model '{model_name}' has no 'backend' and cannot be matched "
                    f"to one; set 'backend' to a configured backend name")
        if backend == "http" and "http" not in self.backends:
            http_names = [n for n, t in self.backend_types.items() if t == "http"]
            if len(http_names) > 1:
                raise ValueError(
                    f"Model '{model_name}' uses generic backend 'http' but multiple "
                    f"HTTP backends are configured ({', '.join(http_names)}); set "
                    f"'backend' to one of them")
            if not http_names:
                raise ValueError(
                    f"Model '{model_name}' resolves to 'http' but no HTTP backend "
                    f"is configured")
            backend = http_names[0]
        if backend not in self.backends and backend not in ("rs", "inkling"):
            raise ValueError(
                f"Unknown backend '{backend}' for model '{model_name}'; "
                f"configured backends: {', '.join(self.backends) or '(none)'}")
        return backend, info

    @staticmethod
    def _http_model_id(info, model_name):
        request = info.get("request") or {}
        return request.get("model") or model_name

    def ensure_active(self, model_name):
        """Make the backend/model for model_name resident. Blocks during a swap."""
        return self._ensure_backend(model_name)

    def _ensure_backend(self, model_name):
        """Switch to the backend needed for model_name. Blocks during swap."""
        backend_name, info = self._resolve_model(model_name)
        btype = self._backend_type(backend_name)
        backend = self._get_backend(backend_name)
        if backend is None:
            raise RuntimeError(
                f"Backend '{backend_name}' required by model '{model_name}' is not "
                f"configured (set its exe/base_url or add it under 'backends')")

        if self.active_backend == backend_name:
            if btype == "rs":
                target = info.get("gguf", "")
                target_id = target[:-5] if target.endswith(".gguf") else target
                if backend.current_model != target_id:
                    if not backend.swap(target):
                        raise RuntimeError(
                            f"Backend '{backend_name}' failed to load {target}")
                    self.swap_count += 1
            elif btype == "http":
                # Same server, different model id: zero-downtime request-body swap.
                request_id = self._http_model_id(info, model_name)
                if backend.current_model != request_id:
                    backend.current_model = request_id
                    self.swap_count += 1
            self.active_model = model_name
            return backend_name, info

        # Need to switch backends - free VRAM first.
        old_name = self.active_backend
        if old_name:
            old_backend = self._get_backend(old_name)
            old_type = self._backend_type(old_name)
            if old_backend is not None and old_backend is not backend:
                old_backend.stop()
                if old_type in ("rs", "inkling"):
                    time.sleep(3)  # let VRAM release

        # Start the new backend.
        if btype == "rs":
            gguf = info.get("gguf")
            if not gguf:
                raise ValueError(f"Model '{model_name}' uses an rs backend but has no 'gguf'")
            if not backend.start(gguf):
                raise RuntimeError(f"Backend '{backend_name}' failed to start")
            backend.current_model = gguf[:-5] if gguf.endswith(".gguf") else gguf
        elif btype == "inkling":
            if not backend.start():
                raise RuntimeError(f"Backend '{backend_name}' failed to start")
        elif btype == "http":
            if not backend.start(info):
                where = getattr(backend, "base_url", "") or "(no base_url)"
                hint = (" (external server must already be running)"
                        if getattr(backend, "external", False) else "")
                raise RuntimeError(
                    f"HTTP backend '{backend_name}' is not reachable at {where}{hint}")
            backend.current_model = self._http_model_id(info, model_name)
        else:
            raise ValueError(
                f"Unknown backend type '{btype}' for backend '{backend_name}'")

        self.active_backend = backend_name
        self.active_model = model_name
        self.swap_count += 1
        return backend_name, info

    def chat(self, model_name, messages, max_tokens=512, temperature=0.7):
        """Main entry: route chat request."""
        with self.lock:
            self.request_count += 1
            backend_name, info = self._ensure_backend(model_name)
            btype = self._backend_type(backend_name)
            backend = self._get_backend(backend_name)

            if btype == "rs":
                model_id = info["gguf"].replace(".gguf", "")
                return backend.chat(model_id, messages, max_tokens, temperature)
            if btype == "http":
                return backend.chat(
                    info, messages, max_tokens, temperature,
                    model_id=self._http_model_id(info, model_name))
            text, ntok = backend.generate(messages, max_tokens, temperature)
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
        backend_state = {}
        for name, btype in self.backend_types.items():
            backend = self._get_backend(name)
            backend_state[name] = {
                "type": btype,
                "current_model": getattr(backend, "current_model", None),
                "external": bool(getattr(backend, "external", False)),
            }
        return {
            "status": "ok",
            "active_backend": self.active_backend,
            "active_backend_type": (self._backend_type(self.active_backend)
                                    if self.active_backend else None),
            "active_model": self.active_model,
            "swap_count": self.swap_count,
            "request_count": self.request_count,
            "models": list(self.models.keys()),
            "backends": backend_state,
        }

    def stop_all(self):
        seen = set()
        for backend in list(self.backends.values()) + [self.rs, self.inkling]:
            if backend is None or id(backend) in seen:
                continue
            seen.add(id(backend))
            try:
                backend.stop()
            except Exception:
                pass


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
        elif self.path in ("/engines", "/v1/engines"):
            data = []
            for name, info in ENGINE_CATALOG.items():
                item = {"name": name}
                item.update(info)
                data.append(item)
            payload = dict(ENGINE_CATALOG)
            payload["object"] = "engine_catalog"
            payload["engines"] = ENGINE_CATALOG
            payload["data"] = data
            self._send_json(payload)
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
    parser.add_argument("--list-engines", action="store_true",
                        help="print the supported engine catalog and exit")
    args = parser.parse_args()

    if args.list_engines:
        sys.stdout.write(format_engine_catalog())
        return

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
        scheduler.stop_all()
        server.server_close()


if __name__ == "__main__":
    main()
